#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRIRA-DK v2: K-means distribuido con selección adaptativa de k (PySpark).

Un solo motor Lloyd (treeAggregate sobre bloques NumPy) para las trayectorias
propias, de modo que la ablación aísla el efecto de la inicialización y el del
criterio de selección.

Trayectorias k = 1..kmax (--inits):
  kmeansll        k-means|| independiente en cada k (réplica de la lógica de MLlib).
  warm            divisiva con arranque en caliente: se divide, a lo largo de su eje
                  principal, el clúster con mayor reducción esperada de SSE
                  R_j = n_j * lambda1_j * 2/pi, desplazamiento sqrt(2*lambda1_j/pi)*v1_j,
                  y luego Lloyd global. Si la semilla dividida no cumple
                  cost(semilla) <= W(k), se usa la semilla aditiva C_k U {c_j + delta}.
                  Garantía: W(k+1) <= W(k).
  incpp           incremental k-means++: C_{k+1} = C_k U {x}, x muestreado con prob.
                  proporcional a D^2(x, C_k); luego Lloyd global. Monótona.
  mllib_kmeans    pyspark.ml.clustering.KMeans en cada k (línea base externa).
  mllib_bisecting pyspark.ml.clustering.BisectingKMeans en cada k (línea base divisiva).
  Las dos últimas no están instrumentadas en pasadas (NaN); se mide tiempo y,
  vía REST, shuffle. Para comparabilidad, W(k) de todas las trayectorias se define
  como el costo de Voronoi de los centros devueltos (para bisecting difiere de
  trainingCost, que también se registra).

Criterios (--criteria):
  gap      Gap statistic: criterio de selección principal del método.
  crira    opcional, desde --crira-module (CRIRA-4 se evaluó con calibrar_crira4.py y no
           superó a Gap; no forma parte del método). Sin módulo no se puede pedir.
  lcurv    curvatura discreta de log W(k) (línea base de codo).
  kneedle  Kneedle (Satopää et al., 2011) sobre W(k) normalizada.
  ch       Calinski-Harabasz.
  sil      silueta euclidiana al cuadrado exacta (como ClusteringEvaluator de MLlib).
  sile     silueta euclidiana convencional (Rousseeuw, 1987), exacta sobre una muestra
           aleatoria fija de --sil-n puntos (como silhouette_score(sample_size=...) de
           scikit-learn). La muestra es la misma para todos los métodos y todos los k.
  gap      Gap statistic (Tibshirani et al., 2001), referencia uniforme en la caja
           envolvente, con la misma trayectoria que se evalúa.

Datos reales: las etiquetas solo se usan para ARI/NMI externos; k_true = vacío.

Tiempos (cada etapa se materializa con persist + count, porque Spark es perezoso):
  data_load_time_s, standardize_time_s, engine_prep_time_s   preparación
  path_time_s (= inicialización + Lloyd por k)               trayectoria
  crit_time_s                                                 criterio
  refine_time_s                                               refinamiento opcional
  total_time_s = trayectoria + criterio; total_time_refined_s = + refinamiento
  pipeline_time_s / pipeline_time_refined_s = preparación + total
Recursos: ejecutores activos, nodos físicos distintos (hosts) al inicio y al final de
cada trayectoria; resources_stable = 0 si cambiaron (corrida no válida para escalabilidad).

Salidas: <out>.csv, <out>_path.csv, <out>_mllib.csv, <out>_env.json (versiones, hardware,
configuración de Spark, sha256 de crira_dk.py y del módulo CRIRA, nodos YARN)
"""
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.request
import uuid

import numpy as np
from pyspark import StorageLevel
from pyspark.sql import SparkSession

CHUNK = 65536
NAN = float("nan")


# ----------------------------------------------------------------------------
# Núcleo numérico por bloque
# ----------------------------------------------------------------------------
def _add(a, b):
    return a + b


def _seed(*parts):
    out = []
    for p in parts:
        if isinstance(p, (list, tuple, np.ndarray)):
            out.extend(_seed(*p))
        else:
            out.append(int(p))
    return out


def _nearest(X, xn, C, Cn, chunk=CHUNK):
    m = X.shape[0]
    idx = np.empty(m, dtype=np.int64)
    d2 = np.empty(m, dtype=np.float64)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        D = xn[s:e, None] - 2.0 * (X[s:e] @ C.T) + Cn[None, :]
        j = D.argmin(axis=1)
        idx[s:e] = j
        d2[s:e] = np.maximum(D[np.arange(e - s), j], 0.0)
    return idx, d2


def _sqn(C):
    return np.einsum("ij,ij->i", C, C)


def _block(X, y):
    X = np.ascontiguousarray(X)
    xn = np.einsum("ij,ij->i", X, X, dtype=np.float64)
    return (X, y, xn)


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def make_synthetic(sc, n, d, k_true, sep, imbalance, std_jitter, P, seed, dtype):
    rng = np.random.default_rng(_seed(seed))
    centers = rng.normal(size=(k_true, d))
    D = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
    D[np.diag_indices(k_true)] = np.inf
    centers *= sep / D.min()
    w = np.exp(-imbalance * np.arange(k_true) / max(k_true - 1, 1))
    w /= w.sum()
    stds = 1.0 + std_jitter * (2.0 * rng.random(k_true) - 1.0)
    sizes = np.full(P, n // P, dtype=np.int64)
    sizes[: n % P] += 1

    def gen(pid, _it):
        r = np.random.default_rng(_seed(seed, 7919, pid))
        m = int(sizes[pid])
        y = r.choice(k_true, size=m, p=w).astype(np.int32)
        X = centers[y] + r.standard_normal((m, d)) * stds[y][:, None]
        yield _block(X.astype(dtype), y)

    return sc.parallelize(range(P), P).mapPartitionsWithIndex(gen)


def make_uniform(sc, n, lo, hi, P, seed, dtype):
    sizes = np.full(P, n // P, dtype=np.int64)
    sizes[: n % P] += 1
    d = lo.shape[0]

    def gen(pid, _it):
        r = np.random.default_rng(_seed(seed, 104729, pid))
        X = lo + r.random((int(sizes[pid]), d)) * (hi - lo)
        yield _block(X.astype(dtype), None)

    return sc.parallelize(range(P), P).mapPartitionsWithIndex(gen)


def make_gaussian(sc, n, mu, cov, P, seed, dtype):
    """Referencia nula de un solo clúster: N(mu, cov) con la media y covarianza de los datos."""
    d = mu.shape[0]
    jitter = 1e-9 * max(float(np.trace(cov)) / d, 1e-12)
    Lc = np.linalg.cholesky(cov + jitter * np.eye(d))
    sizes = np.full(P, n // P, dtype=np.int64)
    sizes[: n % P] += 1

    def gen(pid, _it):
        r = np.random.default_rng(_seed(seed, 15485863, pid))
        X = mu + r.standard_normal((int(sizes[pid]), d)) @ Lc.T
        yield _block(X.astype(dtype), None)

    return sc.parallelize(range(P), P).mapPartitionsWithIndex(gen)


def load_real(spark, a, dtype):
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType

    rd = spark.read.option("header", str(a.header).lower()).option("inferSchema", "true")
    df = rd.csv(a.path) if a.fmt == "csv" else rd.parquet(a.path)
    if a.drop_cols:
        df = df.drop(*a.drop_cols)
    if a.sample_fraction < 1.0:
        df = df.sample(False, a.sample_fraction, seed=a.data_seed)
    label = a.label_col
    if label and not isinstance(df.schema[label].dataType, NumericType):
        from pyspark.ml.feature import StringIndexer
        df = StringIndexer(inputCol=label, outputCol="__lbl").fit(df).transform(df).drop(label)
        label = "__lbl"
    feats = [f.name for f in df.schema.fields
             if f.name != label and isinstance(f.dataType, NumericType)]
    cols = [F.col(c).cast("double") for c in feats]
    if label:
        cols.append(F.col(label).cast("int").alias("__y"))
    df = df.select(*cols).dropna()
    d = len(feats)
    has_y = bool(label)

    def to_block(it, chunk=CHUNK):
        # Conversión por trozos: como máximo `chunk` objetos Row vivos a la vez; los trozos
        # ya convertidos se guardan como arreglos NumPy compactos y se concatenan al final
        # (un bloque por partición, que es lo que supone el motor).
        import itertools
        Xs, ys = [], []
        while True:
            rows = list(itertools.islice(it, chunk))
            if not rows:
                break
            A = np.asarray(rows, dtype=np.float64)
            del rows
            Xs.append(A[:, :d].astype(dtype))
            if has_y:
                ys.append(A[:, d].astype(np.int32))
            del A
        if not Xs:
            return
        X = np.concatenate(Xs) if len(Xs) > 1 else Xs[0]
        del Xs
        y = (np.concatenate(ys) if len(ys) > 1 else ys[0]) if has_y else None
        yield _block(X, y)

    return df.repartition(a.partitions).rdd.mapPartitions(to_block), feats


def standardize(rdd):
    def seq(acc, b):
        X = b[0].astype(np.float64)
        return (acc[0] + X.shape[0], acc[1] + X.sum(0), acc[2] + (X * X).sum(0))

    def comb(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    first = rdd.first()[0]
    d, dtype = first.shape[1], first.dtype
    n, s1, s2 = rdd.treeAggregate((0, np.zeros(d), np.zeros(d)), seq, comb)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu * mu, 1e-24))
    return rdd.map(lambda b: _block(((b[0] - mu) / sd).astype(dtype), b[1]))


# ----------------------------------------------------------------------------
# Métricas de Spark (REST de la UI) y entorno
# ----------------------------------------------------------------------------
STAGE_FIELDS = ["inputBytes", "shuffleReadBytes", "shuffleWriteBytes", "resultSize",
                "executorRunTime", "executorCpuTime", "jvmGcTime",
                "memoryBytesSpilled", "diskBytesSpilled", "peakExecutionMemory"]


class SparkMetrics:
    def __init__(self, sc):
        self.base = sc.uiWebUrl
        self.app = sc.applicationId
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _get(self, path):
        with self.opener.open(f"{self.base}/api/v1/applications/{self.app}{path}", timeout=30) as r:
            return json.loads(r.read().decode())

    def totals(self, groups):
        """Suma de métricas de etapas de los jobs cuyo jobGroup está en groups."""
        groups = {groups} if isinstance(groups, str) else set(groups)
        out = {f: NAN for f in STAGE_FIELDS}
        if not self.base:
            return out
        try:
            for _ in range(100):
                jobs = [j for j in self._get("/jobs") if j.get("jobGroup") in groups]
                if all(j.get("status") != "RUNNING" for j in jobs):
                    break
                time.sleep(0.2)
            time.sleep(0.5)
            sids = sorted({s for j in jobs for s in j.get("stageIds", [])})
            tot = {f: 0.0 for f in STAGE_FIELDS}
            for sid in sids:
                try:
                    atts = self._get(f"/stages/{sid}?details=false")
                except Exception:
                    continue
                for att in atts:
                    if att.get("status") == "SKIPPED":
                        continue
                    for f in STAGE_FIELDS:
                        v = att.get(f, 0) or 0
                        tot[f] = max(tot[f], float(v)) if f == "peakExecutionMemory" else tot[f] + float(v)
            return tot
        except Exception as e:
            print(f"[metrics] no disponibles: {e}", file=sys.stderr)
            return out

    def executors(self):
        """Ejecutores activos (sin driver), nodos físicos distintos que los alojan y memoria."""
        res = dict(executors=NAN, nodes=NAN, hosts="", cores_total=NAN, signature="",
                   storage_mem_used=NAN, peak_jvm_heap=NAN, peak_python_rss=NAN)
        if not self.base:
            return res
        try:
            ex = [e for e in self._get("/executors") if e.get("isActive", True)]
            workers = [e for e in ex if e.get("id") != "driver"]
            if workers:
                hosts = sorted({e.get("hostPort", "").rsplit(":", 1)[0] for e in workers})
                res["executors"] = len(workers)
                res["cores_total"] = int(sum(e.get("totalCores", 0) for e in workers))
            else:  # modo local: el driver ejecuta las tareas
                hosts = sorted({e.get("hostPort", "").rsplit(":", 1)[0] for e in ex})
                res["executors"] = 0
                res["cores_total"] = int(sum(e.get("totalCores", 0) for e in ex))
            res["nodes"] = len(hosts)
            res["hosts"] = ";".join(hosts)
            # firma: identificador@host#núcleos de cada ejecutor activo (detecta sustituciones)
            res["signature"] = ";".join(sorted(
                f"{e.get('id')}@{e.get('hostPort', '').rsplit(':', 1)[0]}#{e.get('totalCores', 0)}"
                for e in (workers or ex)))
            res["storage_mem_used"] = float(sum(e.get("memoryUsed", 0) for e in ex))
            pk = [e.get("peakMemoryMetrics") or {} for e in ex]
            res["peak_jvm_heap"] = float(max([p.get("JVMHeapMemory", 0) for p in pk] or [0]))
            res["peak_python_rss"] = float(max([p.get("ProcessTreePythonRSS", 0) for p in pk] or [0]))
        except Exception as e:
            print(f"[metrics] ejecutores no disponibles: {e}", file=sys.stderr)
        return res


def environment(spark, a):
    sc = spark.sparkContext

    def cmd(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""

    conf = dict(sc.getConf().getAll())
    keep = {k: v for k, v in conf.items() if k.startswith(("spark.executor", "spark.driver.memory",
                                                           "spark.master", "spark.dynamicAllocation",
                                                           "spark.default.parallelism"))}
    try:
        import pandas
        pd_v = pandas.__version__
    except Exception:
        pd_v = ""
    def sha(p):
        try:
            with open(p, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""

    return dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv,
        sha256_crira_dk=sha(os.path.abspath(__file__)),
        sha256_crira_module=sha(a.crira_module) if a.crira_module else "",
        executors_requested=conf.get("spark.executor.instances", ""),
        yarn_nodes=cmd("yarn node -list 2>/dev/null | tail -n +3"),
        spark=spark.version, python=platform.python_version(), numpy=np.__version__,
        pandas=pd_v, java=cmd("java -version 2>&1 | grep -i version | head -1"),
        os=platform.platform(),
        cpu=(cmd("lscpu | grep 'Model name' | head -1") or cmd("sysctl -n machdep.cpu.brand_string")
             or platform.processor()),
        logical_cpus=os.cpu_count(),
        performance_cores=cmd("sysctl -n hw.perflevel0.physicalcpu 2>/dev/null"),
        efficiency_cores=cmd("sysctl -n hw.perflevel1.physicalcpu 2>/dev/null"),
        mem_total=cmd("grep MemTotal /proc/meminfo") or cmd("sysctl -n hw.memsize"),
        default_parallelism=sc.defaultParallelism, master=sc.master, spark_conf=keep,
        args=vars(a))


# ----------------------------------------------------------------------------
# Motor distribuido
# ----------------------------------------------------------------------------
class Engine:
    def __init__(self, sc, rdd, depth, eps, conv="shift"):
        self.sc, self.rdd, self.depth, self.eps = sc, rdd, depth, eps
        self.conv = conv
        self.P = rdd.getNumPartitions()
        self.cat = "prep"
        self.passes, self.payload = {}, {}
        self._prep()

    @contextlib.contextmanager
    def category(self, cat):
        old = self.cat
        self.cat = cat
        try:
            yield
        finally:
            self.cat = old

    def _count(self, L, frac=1.0):
        self.passes[self.cat] = self.passes.get(self.cat, 0.0) + frac
        self.payload[self.cat] = self.payload.get(self.cat, 0.0) + frac * self.P * L * 8.0

    def snap(self, cat):
        return self.passes.get(cat, 0.0), self.payload.get(cat, 0.0)

    def _agg(self, seq, L):
        self._count(L)
        return self.rdd.treeAggregate(np.zeros(L), seq, _add, depth=self.depth)

    def _prep(self):
        self.sizes = np.array(self.rdd.map(lambda b: b[0].shape[0]).collect(), dtype=np.int64)
        self._count(1)
        d = self.rdd.first()[0].shape[1]
        self.d, self.n = d, int(self.sizes.sum())

        def seq(acc, b):
            X, y, _ = b
            ymax = int(y.max()) if (y is not None and y.size) else -1
            return (acc[0] + X.sum(0, dtype=np.float64), np.minimum(acc[1], X.min(0)),
                    np.maximum(acc[2], X.max(0)), max(acc[3], ymax), acc[4] or (y is not None))

        def comb(a, b):
            return (a[0] + b[0], np.minimum(a[1], b[1]), np.maximum(a[2], b[2]),
                    max(a[3], b[3]), a[4] or b[4])

        s, lo, hi, ymax, has_y = self.rdd.treeAggregate(
            (np.zeros(d), np.full(d, np.inf), np.full(d, -np.inf), -1, False), seq, comb,
            depth=self.depth)
        self._count(3 * d)
        self.mu = s / self.n
        self.lo, self.hi = lo.astype(np.float64), hi.astype(np.float64)
        self.has_y = bool(has_y)
        self.n_classes = ymax + 1 if self.has_y else 0
        mu = self.mu

        def seqT(acc, b):
            Z = b[0] - mu
            return acc + np.einsum("ij,ij->", Z, Z, dtype=np.float64)

        self.T = float(self.rdd.treeAggregate(0.0, seqT, _add, depth=self.depth))
        self._count(1)

    def covariance(self):
        """Covarianza de los datos (una pasada, se calcula una sola vez)."""
        if getattr(self, "_cov", None) is None:
            mu, d = self.mu, self.d

            def seq(acc, b):
                Z = b[0].astype(np.float64) - mu
                return acc + (Z.T @ Z).ravel()

            v = self._agg(seq, d * d)
            self._cov = v.reshape(d, d) / self.n
        return self._cov

    # --- pasada de asignación (estadísticos de Lloyd) ---------------------------
    def assign_stats(self, C):
        C = np.array(C, dtype=np.float64)
        k, d = C.shape
        L = k * d + 2 * k
        Cb = self.sc.broadcast(C)

        def seq(acc, blk, Cb=Cb, k=k, d=d, L=L):
            X, _y, xn = blk
            Cc = Cb.value
            idx, d2 = _nearest(X, xn, Cc, _sqn(Cc))
            v = np.empty(L)
            for j in range(d):
                v[j * k:(j + 1) * k] = np.bincount(idx, weights=X[:, j], minlength=k)
            v[k * d:k * d + k] = np.bincount(idx, minlength=k)
            v[k * d + k:] = np.bincount(idx, weights=d2, minlength=k)
            return acc + v

        v = self._agg(seq, L)
        Cb.unpersist()
        sse = v[k * d + k:]
        return dict(C=C, W=float(sse.sum()), sums=v[:k * d].reshape(d, k).T,
                    counts=v[k * d:k * d + k], sse=sse)

    def lloyd(self, C, max_iter, tol):
        C = np.array(C, dtype=np.float64)
        first, prev, it = None, np.inf, 0
        while True:
            st = self.assign_stats(C)
            cost = st["W"]
            it += 1
            if first is None:
                first = cost
            newC = C.copy()
            nz = st["counts"] > 0
            newC[nz] = st["sums"][nz] / st["counts"][nz, None]
            shift = float(np.sqrt(((newC - C) ** 2).sum(1).max()))
            # conv="shift": mismo criterio que MLlib (desplazamiento máximo de centros <= eps);
            # conv="both": además se detiene si la caída relativa de W es <= tol
            stop_cost = self.conv == "both" and (prev - cost) <= tol * max(cost, 1e-300)
            if self.conv == "sklearn":
                # criterio de scikit-learn: suma de desplazamientos al cuadrado <= tol * varianza media
                sk = float(((newC - C) ** 2).sum()) <= tol * (self.T / (self.n * self.d))
                done = it >= max_iter or sk
            else:
                done = it >= max_iter or shift <= self.eps or stop_cost
            if done:
                st.update(iters=it, first_cost=first)
                return st
            prev, C = cost, newC

    # --- k-means|| ---------------------------------------------------------------
    def random_point(self, rng):
        pid = int(rng.choice(self.P, p=self.sizes / self.sizes.sum()))
        row = int(rng.integers(self.sizes[pid]))

        def f(it, row=row):
            for b in it:
                yield b[0][row].astype(np.float64)

        self._count(self.d, frac=1.0 / self.P)
        return self.sc.runJob(self.rdd, f, partitions=[pid])[0]

    def cost(self, C):
        Cb = self.sc.broadcast(np.asarray(C, dtype=np.float64))

        def seq(acc, blk, Cb=Cb):
            X, _y, xn = blk
            Cc = Cb.value
            _, d2 = _nearest(X, xn, Cc, _sqn(Cc))
            return acc + np.array([d2.sum()])

        v = self._agg(seq, 1)
        Cb.unpersist()
        return float(v[0])

    def sample_d2(self, C, factor, seed):
        Cb = self.sc.broadcast(np.asarray(C, dtype=np.float64))

        def f(pid, it, Cb=Cb, factor=factor, seed=seed):
            for X, _y, xn in it:
                Cc = Cb.value
                _, d2 = _nearest(X, xn, Cc, _sqn(Cc))
                r = np.random.default_rng(_seed(seed, pid))
                mask = r.random(X.shape[0]) < np.minimum(1.0, factor * d2)
                if mask.any():
                    yield X[mask].astype(np.float64)

        self._count(self.d)
        parts = self.rdd.mapPartitionsWithIndex(f).collect()
        Cb.unpersist()
        return np.vstack(parts) if parts else np.empty((0, self.d))

    def weights(self, C):
        m = C.shape[0]
        Cb = self.sc.broadcast(np.asarray(C, dtype=np.float64))

        def seq(acc, blk, Cb=Cb, m=m):
            X, _y, xn = blk
            Cc = Cb.value
            idx, _ = _nearest(X, xn, Cc, _sqn(Cc))
            return acc + np.bincount(idx, minlength=m)

        v = self._agg(seq, m)
        Cb.unpersist()
        return v

    # --- incremental k-means++: un punto con prob. proporcional a D^2 --------------
    def d2_sample_one(self, C, rng):
        C = np.asarray(C, dtype=np.float64)
        Cb = self.sc.broadcast(C)

        def part_cost(pid, it, Cb=Cb):
            tot = 0.0
            for X, _y, xn in it:
                Cc = Cb.value
                _, d2 = _nearest(X, xn, Cc, _sqn(Cc))
                tot += d2.sum()
            yield (pid, tot)

        self._count(1)
        pc = dict(self.rdd.mapPartitionsWithIndex(part_cost).collect())
        costs = np.array([pc.get(i, 0.0) for i in range(self.P)])
        pid = int(rng.choice(self.P, p=costs / costs.sum()))
        s = int(rng.integers(2 ** 31))

        def pick(it, Cb=Cb, s=s):
            for X, _y, xn in it:
                Cc = Cb.value
                _, d2 = _nearest(X, xn, Cc, _sqn(Cc))
                r = np.random.default_rng(s)
                yield X[r.choice(X.shape[0], p=d2 / d2.sum())].astype(np.float64)

        self._count(self.d, frac=1.0 / self.P)
        x = self.sc.runJob(self.rdd, pick, partitions=[pid])[0]
        Cb.unpersist()
        return x

    # --- covarianzas de clústeres candidatos (división) ----------------------------
    def cand_cov(self, C, cands):
        C = np.asarray(C, dtype=np.float64)
        d, m = self.d, len(cands)
        L = m * d * d
        Cb = self.sc.broadcast(C)
        cands = np.asarray(cands)

        def seq(acc, blk, Cb=Cb, cands=cands, d=d, L=L):
            X, _y, xn = blk
            Cc = Cb.value
            idx, _ = _nearest(X, xn, Cc, _sqn(Cc))
            v = np.zeros(L)
            for t, j in enumerate(cands):
                Z = X[idx == j].astype(np.float64) - Cc[j]
                if Z.shape[0]:
                    v[t * d * d:(t + 1) * d * d] = (Z.T @ Z).ravel()
            return acc + v

        v = self._agg(seq, L)
        Cb.unpersist()
        return v.reshape(m, d, d)

    # --- criterios y evaluación ----------------------------------------------------
    def silhouette(self, res):
        C, counts, S = res["C"], res["counts"], res["sums"]
        Q = res["sse"] + 2.0 * np.einsum("ij,ij->i", C, S) - counts * _sqn(C)
        Cb = self.sc.broadcast((C, counts, S, Q))

        def seq(acc, blk, Cb=Cb):
            np.seterr(invalid="ignore", divide="ignore")
            X, _y, xn = blk
            Cc, cnt, Sc, Qc = Cb.value
            idx, _ = _nearest(X, xn, Cc, _sqn(Cc))
            safe = np.where(cnt > 0, cnt, 1.0)
            tot = 0.0
            for s in range(0, X.shape[0], CHUNK):
                e = min(s + CHUNK, X.shape[0])
                A = (xn[s:e, None] * cnt[None, :] - 2.0 * (X[s:e] @ Sc.T) + Qc[None, :]) / safe[None, :]
                A[:, cnt <= 0] = np.inf
                own = idx[s:e]
                r = np.arange(e - s)
                a = A[r, own].copy()
                A[r, own] = np.inf
                b = A.min(axis=1)
                den = np.maximum(a, b)
                sv = np.where(den > 0, (b - a) / np.where(den > 0, den, 1.0), 0.0)
                sv[(cnt[own] <= 1) | ~np.isfinite(b)] = 0.0
                tot += sv.sum()
            return acc + np.array([tot])

        v = self._agg(seq, 1)
        Cb.unpersist()
        return float(v[0]) / self.n

    def sample_rows(self, m, seed):
        """Muestra aleatoria simple (sin reemplazo) de ~m filas; una pasada sobre los datos."""
        frac = min(1.0, m / max(self.n, 1))
        self._count(1)

        def take(i, it, frac=frac, seed=seed):
            rng = np.random.default_rng(_seed([seed, i]))
            for X, _y, _xn in it:
                keep = rng.random(X.shape[0]) < frac
                if keep.any():
                    yield np.asarray(X[keep], dtype=np.float64)

        parts = self.rdd.mapPartitionsWithIndex(take).collect()
        Z = np.vstack(parts) if parts else np.zeros((0, self.d))
        if Z.shape[0] > m:
            Z = Z[np.random.default_rng(_seed([seed, 7919])).choice(Z.shape[0], m, replace=False)]
        return Z

    @staticmethod
    def silhouette_euclid(Z, C, chunk=1000):
        """Silueta euclidiana exacta de la muestra Z con etiquetas = centro más cercano."""
        lab, _ = _nearest(Z, _sqn(Z), C, _sqn(C))
        k, m = C.shape[0], Z.shape[0]
        cnt = np.bincount(lab, minlength=k).astype(np.float64)
        H = np.zeros((m, k))
        H[np.arange(m), lab] = 1.0
        zn = _sqn(Z)
        tot = 0.0
        for s in range(0, m, chunk):
            e = min(s + chunk, m)
            D = np.sqrt(np.maximum(zn[s:e, None] - 2.0 * (Z[s:e] @ Z.T) + zn[None, :], 0.0))
            D[np.arange(e - s), np.arange(s, e)] = 0.0
            S = D @ H                      # suma de distancias a cada cluster
            own = lab[s:e]
            r = np.arange(e - s)
            n_own = cnt[own] - 1.0
            a = np.where(n_own > 0, S[r, own] / np.where(n_own > 0, n_own, 1.0), 0.0)
            M = S / np.where(cnt > 0, cnt, 1.0)[None, :]
            M[:, cnt <= 0] = np.inf
            M[r, own] = np.inf
            b = M.min(axis=1)
            den = np.maximum(a, b)
            sv = np.where((den > 0) & np.isfinite(b), (b - a) / np.where(den > 0, den, 1.0), 0.0)
            sv[n_own <= 0] = 0.0
            tot += sv.sum()
        return tot / max(m, 1)

    def contingency(self, C):
        K, k = self.n_classes, C.shape[0]
        Cb = self.sc.broadcast(np.asarray(C, dtype=np.float64))

        def seq(acc, blk, Cb=Cb, k=k, K=K):
            X, y, xn = blk
            Cc = Cb.value
            idx, _ = _nearest(X, xn, Cc, _sqn(Cc))
            return acc + np.bincount(idx * K + y, minlength=k * K)

        v = self._agg(seq, k * K)
        Cb.unpersist()
        return v.reshape(k, K)


# ----------------------------------------------------------------------------
# Inicializaciones y trayectorias
# ----------------------------------------------------------------------------
def local_kmeanspp(Pts, w, k, rng, iters=30):
    n = Pts.shape[0]
    w = np.asarray(w, dtype=np.float64)
    if n <= k:
        extra = Pts[rng.integers(n, size=k - n)] if n else np.zeros((k - n, Pts.shape[1]))
        return np.vstack([Pts, extra])
    C = [Pts[rng.choice(n, p=w / w.sum())]]
    d2 = ((Pts - C[0]) ** 2).sum(1)
    for _ in range(1, k):
        pr = w * d2
        j = rng.choice(n, p=pr / pr.sum()) if pr.sum() > 0 else int(rng.integers(n))
        C.append(Pts[j])
        d2 = np.minimum(d2, ((Pts - Pts[j]) ** 2).sum(1))
    C = np.array(C)
    pn = _sqn(Pts)
    for _ in range(iters):
        idx, _ = _nearest(Pts, pn, C, _sqn(C))
        newC = C.copy()
        for j in range(k):
            m = idx == j
            ws = w[m].sum()
            if ws > 0:
                newC[j] = (w[m, None] * Pts[m]).sum(0) / ws
        if np.allclose(newC, C):
            break
        C = newC
    return C


def kmeans_parallel_init(eng, k, seed, rounds, oversample):
    rng = np.random.default_rng(_seed(seed))
    C = eng.random_point(rng)[None, :]
    for r in range(rounds):
        phi = eng.cost(C)
        if phi <= 0:
            break
        new = eng.sample_d2(C, oversample * k / phi, seed=_seed(seed, r))
        if new.shape[0]:
            C = np.vstack([C, new])
    return local_kmeanspp(C, eng.weights(C), k, rng)


def split_seeds(eng, prev, n_cands, timing):
    """Devuelve (semilla dividida, semilla aditiva). Registra tiempos en timing."""
    C, counts, sse = prev["C"], prev["counts"], prev["sse"]
    elig = np.where(counts >= 2)[0]
    order = elig[np.argsort(-sse[elig])][:max(1, n_cands)]
    t0 = time.perf_counter()
    covs = eng.cand_cov(C, order)
    t1 = time.perf_counter()
    best, best_score, best_delta = None, -1.0, None
    for t, j in enumerate(order):
        lam, V = np.linalg.eigh(covs[t] / counts[j])
        l1 = max(float(lam[-1]), 0.0)
        score = counts[j] * l1 * 2.0 / math.pi
        if score > best_score:
            best, best_score, best_delta = j, score, math.sqrt(2.0 * l1 / math.pi) * V[:, -1]
    t2 = time.perf_counter()
    c = C[best]
    Cs = C.copy()
    Cs[best] = c + best_delta
    Cs = np.vstack([Cs, c - best_delta])
    Ca = np.vstack([C, c + best_delta])
    t3 = time.perf_counter()
    timing["split_cov_s"] += t1 - t0
    timing["split_select_s"] += t2 - t1
    timing["split_build_s"] += t3 - t2
    return Cs, Ca


def k1_solution(eng):
    return dict(C=eng.mu[None, :].copy(), W=eng.T, counts=np.array([float(eng.n)]),
                sums=eng.mu[None, :] * eng.n, sse=np.array([eng.T]), iters=0, first_cost=eng.T,
                fallback=0, time_s=0.0, init_time_s=0.0, lloyd_time_s=0.0, passes=0.0,
                init_passes=0.0, lloyd_passes=0.0, mllib_cost=NAN)


def run_path(eng, init, kmax, seed, a, cat="path", mllib_df=None):
    path = {1: k1_solution(eng)}
    timing = dict(split_cov_s=0.0, split_select_s=0.0, split_build_s=0.0)
    with eng.category(cat):
        for k in range(2, kmax + 1):
            t0 = time.perf_counter()
            p0, _ = eng.snap(cat)
            fb = 0
            mcost = NAN
            if init in ("mllib_kmeans", "mllib_bisecting"):
                from pyspark.ml.clustering import BisectingKMeans, KMeans
                est = (KMeans(k=k, seed=int(seed) * 1000 + k, maxIter=a.max_iter, tol=a.eps,
                              initSteps=a.init_steps) if init == "mllib_kmeans"
                       else BisectingKMeans(k=k, seed=int(seed) * 1000 + k, maxIter=a.max_iter))
                model = est.fit(mllib_df)
                t1 = time.perf_counter()
                Cm = np.array(model.clusterCenters(), dtype=np.float64)
                mcost = float(model.summary.trainingCost)
                iters = int(model.summary.numIter)
                with eng.category("eval"):
                    res = eng.assign_stats(Cm)
                res.update(iters=iters, first_cost=NAN)
                t_init, t_lloyd = NAN, t1 - t0
                pi_, pl_ = NAN, NAN
                t_end = t1
            else:
                if init == "kmeansll":
                    C0 = kmeans_parallel_init(eng, k, [seed, k], a.init_steps, a.oversample)
                    seeds = [C0]
                    bound = None
                elif init == "incpp":
                    prev = path[k - 1]
                    rng = np.random.default_rng(_seed(seed, k, 31337))
                    C0 = np.vstack([prev["C"], eng.d2_sample_one(prev["C"], rng)])
                    seeds = [C0]
                    bound = None
                else:  # warm
                    prev = path[k - 1]
                    Cs, Ca = split_seeds(eng, prev, a.split_cands, timing)
                    seeds = [Cs, Ca]
                    bound = prev["W"]
                t1 = time.perf_counter()
                p1, _ = eng.snap(cat)
                res = eng.lloyd(seeds[0], a.max_iter, a.tol)
                if bound is not None and res["first_cost"] > bound:
                    res = eng.lloyd(seeds[1], a.max_iter, a.tol)
                    fb = 1
                t_end = time.perf_counter()
                p2, _ = eng.snap(cat)
                t_init, t_lloyd = t1 - t0, t_end - t1
                pi_, pl_ = p1 - p0, p2 - p1
            res.update(fallback=fb, time_s=t_end - t0, init_time_s=t_init, lloyd_time_s=t_lloyd,
                       passes=eng.snap(cat)[0] - p0 if init not in ("mllib_kmeans", "mllib_bisecting") else NAN,
                       init_passes=pi_, lloyd_passes=pl_, mllib_cost=mcost)
            path[k] = res
    return path, timing


# ----------------------------------------------------------------------------
# Criterios de selección
# ----------------------------------------------------------------------------
def load_crira(path):
    if not path:
        return None
    spec = importlib.util.spec_from_file_location("crira_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "select"):
        raise SystemExit(f"{path} debe definir select(ks, W, ctx) -> int")
    return mod


def validate_k(v, ks, name):
    """El criterio debe devolver un entero finito que pertenezca a los candidatos ks."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise SystemExit(f"{name}.select devolvió {v!r}: no es numérico")
    if not math.isfinite(f) or f != int(f):
        raise SystemExit(f"{name}.select devolvió {v!r}: no es un entero finito")
    k = int(f)
    if k not in set(int(x) for x in ks):
        raise SystemExit(f"{name}.select devolvió k={k}, fuera de los candidatos {int(ks[0])}..{int(ks[-1])}")
    return k


def lcurv_select(ks, W):
    L = np.log(np.asarray(W, dtype=np.float64))
    best_k, best_s = int(ks[1]), -np.inf
    for i in range(1, len(ks) - 1):
        s = (L[i - 1] - L[i]) - (L[i] - L[i + 1])
        if s > best_s:
            best_k, best_s = int(ks[i]), s
    return best_k


def kneedle_select(ks, W):
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (np.asarray(W) - W[-1]) / max(W[0] - W[-1], 1e-300)
    diff = (1.0 - x) - y
    return int(ks[int(np.argmax(diff[1:])) + 1])


def ch_select(ks, W, T, n):
    best_k, best = None, -np.inf
    for k, w in zip(ks, W):
        if k < 2 or w <= 0:
            continue
        v = ((T - w) / (k - 1)) / (w / (n - k))
        if v > best:
            best_k, best = int(k), v
    return best_k


def gap_select(ks, W, refW):
    logW, lr = np.log(W), np.log(refW)
    B = lr.shape[0]
    gap = lr.mean(0) - logW
    sk = lr.std(0, ddof=0) * math.sqrt(1.0 + 1.0 / B)
    for i in range(len(ks) - 1):
        if gap[i] >= gap[i + 1] - sk[i + 1]:
            return int(ks[i])
    return int(ks[-1])


def reference_paths(sc, eng, init, ks, seed, a, dtype, B, mllib_df_builder, kind="uniform", n_ref=None):
    """
    Trayectorias W*(k) sobre B referencias nulas generadas con la misma trayectoria `init`.
      kind="uniform": uniforme en la caja envolvente de los datos.
      kind="gauss":   un solo clúster gaussiano con la media y covarianza de los datos.
    Devuelve (B x |ks|) escalado por n/n_ref y las pasadas equivalentes sobre los datos.
    """
    n_ref = n_ref or (a.gap_ref_n if a.gap_ref_n > 0 else eng.n)
    refW = np.zeros((B, len(ks)))
    extra = 0.0
    p0 = eng.snap(eng.cat)[0]
    cov = eng.covariance() if kind == "gauss" else None
    extra += eng.snap(eng.cat)[0] - p0
    off = {"uniform": 0, "gauss": 7777}[kind]
    for b in range(B):
        if kind == "uniform":
            rr = make_uniform(sc, n_ref, eng.lo, eng.hi, eng.P, [a.data_seed, seed, b], dtype)
        else:
            rr = make_gaussian(sc, n_ref, eng.mu, cov, eng.P, [a.data_seed, seed, b, off], dtype)
        rr = rr.persist(StorageLevel.MEMORY_AND_DISK)
        reng = Engine(sc, rr, a.tree_depth, a.eps, getattr(a, "conv", "shift"))
        mdf = mllib_df_builder(rr) if init.startswith("mllib") else None
        rpath, _ = run_path(reng, init, int(ks[-1]), seed * 1000 + b + off, a, mllib_df=mdf)
        refW[b] = [rpath[k]["W"] for k in ks]
        extra += sum(v for c, v in reng.passes.items() if c != "eval") * n_ref / eng.n
        if mdf is not None:
            mdf.unpersist()
        rr.unpersist()
    return refW * (eng.n / n_ref), extra


# ----------------------------------------------------------------------------
# Calidad
# ----------------------------------------------------------------------------
def ari_nmi(M):
    M = M.astype(np.float64)
    n = M.sum()

    def c2(x):
        return x * (x - 1.0) / 2.0

    sij, a, b = c2(M).sum(), c2(M.sum(1)).sum(), c2(M.sum(0)).sum()
    exp = a * b / c2(n)
    mx = 0.5 * (a + b)
    ari = (sij - exp) / (mx - exp) if mx != exp else 1.0
    Pm = M / n
    pi, pj = Pm.sum(1), Pm.sum(0)
    nz = Pm > 0
    outer = pi[:, None] * pj[None, :]
    mi = float((Pm[nz] * np.log(Pm[nz] / outer[nz])).sum())
    hi = -float((pi[pi > 0] * np.log(pi[pi > 0])).sum())
    hj = -float((pj[pj > 0] * np.log(pj[pj > 0])).sum())
    nmi = mi / (0.5 * (hi + hj)) if (hi + hj) > 0 else 1.0
    return float(ari), float(nmi)


# ----------------------------------------------------------------------------
# E/S
# ----------------------------------------------------------------------------
def append_rows(path, rows):
    """Agrega filas; si el archivo existe con otras columnas, reescribe con la unión de columnas."""
    if not rows:
        return
    cols = list(rows[0].keys())
    for r in rows[1:]:
        cols += [c for c in r if c not in cols]
    if os.path.exists(path):
        with open(path, newline="") as f:
            old_cols = next(csv.reader(f), [])
        if not set(cols) <= set(old_cols):
            with open(path, newline="") as f:
                old_rows = list(csv.DictReader(f))
            old_cols = old_cols + [c for c in cols if c not in old_cols]
            tmp = path + ".tmp"
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=old_cols, restval="")
                w.writeheader()
                w.writerows(old_rows)
            os.replace(tmp, path)
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=old_cols, restval="").writerows(rows)
    else:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(rows)


# Parámetros que definen una configuración experimental (todo excepto data_seed, seed, inits,
# repetición, salida y política de duplicados). config_hash identifica el escenario; junto con
# data_seed, seed, init y repeat forma la clave única de una trayectoria.
CONFIG_KEYS = ["data", "n", "d", "k_true", "sep", "imbalance", "std_jitter", "path", "fmt", "header",
               "label_col", "drop_cols", "sample_fraction", "no_standardize", "dataset_name", "kmax",
               "criteria", "max_iter", "tol", "eps", "init_steps", "oversample",
               "split_cands", "final_restarts", "gap_B", "gap_ref_n", "crira_B", "crira_ref_n", "crira_tau", "crira_rule", "master", "partitions",
               "tree_depth", "dtype", "no_cache", "tag", "conv"]


def config_hash(a, sc, code_sha, crira_sha):
    cfg = {k: getattr(a, k) for k in CONFIG_KEYS}
    cfg["parallelism"] = sc.defaultParallelism
    cfg["executors_requested"] = sc.getConf().get("spark.executor.instances", "")
    cfg["code_sha"] = code_sha
    cfg["crira_sha"] = crira_sha
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def existing_keys(path, code_sha):
    """Claves (config_hash, data_seed, seed, init, repeat) ya presentes. Rechaza mezclar versiones."""
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("code_sha") and r["code_sha"] != code_sha:
                raise SystemExit(
                    f"{path} contiene resultados de otra versión de crira_dk.py "
                    f"(sha {r['code_sha'][:12]} != {code_sha[:12]}). Use un directorio de salida nuevo.")
            keys.add((r.get("config_hash"), r.get("data_seed"), r.get("seed"), r.get("init"),
                      r.get("repeat")))
    return keys


def file_sha(p):
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return ""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def blocks_to_df(spark, rdd):
    from pyspark.ml.linalg import Vectors

    def rows(b):
        X, y, _ = b
        yy = y if y is not None else np.full(X.shape[0], -1, dtype=np.int32)
        return [(Vectors.dense(X[i].astype(float)), int(yy[i])) for i in range(X.shape[0])]

    df = spark.createDataFrame(rdd.flatMap(rows), ["features", "label"])
    df = df.persist(StorageLevel.MEMORY_AND_DISK)
    df.count()
    return df


def mllib_check(eng, df, k, seed, a):
    from pyspark.ml.clustering import KMeans
    t0 = time.perf_counter()
    model = KMeans(k=k, seed=seed, maxIter=a.max_iter, tol=a.eps, initSteps=a.init_steps).fit(df)
    t_ml = time.perf_counter() - t0
    with eng.category("eval"):
        W_ml_voronoi = eng.assign_stats(np.array(model.clusterCenters()))["W"]
        t0 = time.perf_counter()
        res = eng.lloyd(kmeans_parallel_init(eng, k, [seed, k], a.init_steps, a.oversample),
                        a.max_iter, a.tol)
        t_own = time.perf_counter() - t0
    W_ml = float(model.summary.trainingCost)
    return dict(k=k, seed=seed, W_mllib=W_ml, W_mllib_voronoi=W_ml_voronoi,
                iters_mllib=int(model.summary.numIter), time_mllib_s=t_ml, W_engine=res["W"],
                iters_engine=res["iters"], time_engine_s=t_own, rel_diff=(res["W"] - W_ml) / W_ml)


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
INITS = ["kmeansll", "warm", "incpp", "mllib_kmeans", "mllib_bisecting"]
CRITS = ["crira", "lcurv", "kneedle", "ch", "sil", "sile", "gap"]


def parse_args():
    p = argparse.ArgumentParser(description="CRIRA-DK v2: experimentos en PySpark")
    p.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--n", type=int, default=1_000_000)
    p.add_argument("--d", type=int, default=10)
    p.add_argument("--k-true", type=int, default=8)
    p.add_argument("--sep", type=float, default=4.0,
                   help="distancia mínima entre centros, en desviaciones estándar")
    p.add_argument("--imbalance", type=float, default=0.0)
    p.add_argument("--std-jitter", type=float, default=0.0)
    p.add_argument("--path")
    p.add_argument("--fmt", choices=["csv", "parquet"], default="csv")
    p.add_argument("--header", action="store_true")
    p.add_argument("--label-col")
    p.add_argument("--drop-cols", nargs="*", default=[])
    p.add_argument("--sample-fraction", type=float, default=1.0)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--dataset-name", default="real")
    p.add_argument("--kmax", type=int, default=20)
    p.add_argument("--inits", nargs="+", default=["kmeansll", "warm", "incpp"], choices=INITS)
    p.add_argument("--criteria", nargs="+", default=["gap", "ch", "sil", "lcurv", "kneedle"],
                   choices=CRITS)
    p.add_argument("--crira-module", help="archivo .py con select(ks, W, ctx) -> int (CRIRA-4)")
    p.add_argument("--crira-B", type=int, default=5, help="referencias por tipo para CRIRA-4")
    p.add_argument("--crira-ref-n", type=int, default=200000,
                   help="tamaño de cada referencia de CRIRA-4 (0 = n); la curvatura no depende de la escala")
    p.add_argument("--crira-tau", type=float, default=2.0, help="umbral de significancia de CRIRA-4")
    p.add_argument("--crira-rule", choices=["argmax", "largest"], default="argmax",
                   help="regla de decisión de CRIRA-4 (fijar con calibrar_crira4.py)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--data-seed", type=int, default=12345)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--init-steps", type=int, default=2)
    p.add_argument("--oversample", type=float, default=2.0)
    p.add_argument("--split-cands", type=int, default=3)
    p.add_argument("--final-restarts", type=int, default=0)
    p.add_argument("--sil-n", type=int, default=10000,
                   help="tamaño de la muestra fija para la silueta euclidiana convencional (sile)")
    p.add_argument("--gap-B", type=int, default=10)
    p.add_argument("--gap-ref-n", type=int, default=0)
    p.add_argument("--master", default=None,
                   help="omitir con spark-submit (usa el master de la línea de comandos)")
    p.add_argument("--partitions", type=int, default=64)
    p.add_argument("--tree-depth", type=int, default=2)
    p.add_argument("--driver-memory", default=None)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--mllib-check", action="store_true")
    p.add_argument("--conv", choices=["shift", "both", "sklearn"], default="sklearn",
                   help="criterio de paro de Lloyd, igual para todos los métodos del motor: "
                        "shift = MLlib (desplazamiento máximo <= eps); both = shift o caída relativa "
                        "de W <= tol; sklearn = suma de desplazamientos^2 <= tol * varianza media")
    p.add_argument("--mllib-once", action="store_true",
                   help="mide UN ajuste de MLlib KMeans y BisectingKMeans con k = kmax "
                        "(cota inferior del costo de bisecting: un ajuste construye toda la jerarquía)")
    p.add_argument("--tag", default="")
    p.add_argument("--repeat", type=int, default=0,
                   help="índice de repetición de medición de tiempo con las mismas semillas; "
                        "el análisis de calidad solo usa repeat=0")
    p.add_argument("--on-duplicate", choices=["skip", "error"], default="skip",
                   help="si la trayectoria (configuración, data_seed, seed, init, repeat) ya está "
                        "en el CSV de salida: omitirla (permite reanudar) o detenerse")
    p.add_argument("--out", default="resultados")
    a = p.parse_args()
    if "crira" in a.criteria and not a.crira_module:
        p.error("el criterio 'crira' requiere --crira-module con la definición de CRIRA-4 "
                "(o quítalo de --criteria)")
    return a


def main():
    a = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    dtype = np.float32 if a.dtype == "float32" else np.float64
    crira = load_crira(a.crira_module)
    b = (SparkSession.builder.appName("CRIRA-DK")
         .config("spark.ui.retainedJobs", "1000000")
         .config("spark.ui.retainedStages", "1000000")
         .config("spark.ui.showConsoleProgress", "false")
         .config("spark.executor.processTreeMetrics.enabled", "true"))
    if a.master:
        b = b.master(a.master)
    if a.driver_memory:
        b = b.config("spark.driver.memory", a.driver_memory)
    spark = b.getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")
    metrics = SparkMetrics(sc)
    code_sha = file_sha(os.path.abspath(__file__))
    crira_sha = file_sha(a.crira_module) if a.crira_module else ""
    chash = config_hash(a, sc, code_sha, crira_sha)
    done = existing_keys(f"{a.out}.csv", code_sha)
    with open(f"{a.out}_env_{chash}.json", "w") as f:
        json.dump(dict(environment(spark, a), config_hash=chash), f, indent=2, default=str)

    # Spark es perezoso: cada etapa se materializa con persist + count para medir su tiempo.
    t0 = time.perf_counter()
    if a.data == "synthetic":
        rdd = make_synthetic(sc, a.n, a.d, a.k_true, a.sep, a.imbalance, a.std_jitter,
                             a.partitions, a.data_seed, dtype)
        name, k_true = "synthetic", a.k_true
    else:
        rdd, _ = load_real(spark, a, dtype)
        # En datos reales las etiquetas solo sirven para evaluación externa (ARI/NMI);
        # el número de clases no se toma como número verdadero de clústeres.
        name, k_true = a.dataset_name, None
    if not (a.data == "synthetic" and a.no_cache):
        rdd = rdd.persist(StorageLevel.MEMORY_AND_DISK)
        rdd.count()
    data_load_time = time.perf_counter() - t0
    standardize_time = 0.0
    if a.data == "real" and not a.no_standardize:
        t0 = time.perf_counter()
        raw = rdd
        rdd = standardize(raw).persist(StorageLevel.MEMORY_AND_DISK)
        rdd.count()
        raw.unpersist()
        standardize_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    eng = Engine(sc, rdd, a.tree_depth, a.eps, a.conv)
    engine_prep_time = time.perf_counter() - t0
    exinfo = metrics.executors()
    log(f"datos: n={eng.n:,} d={eng.d} P={eng.P} paralelismo={sc.defaultParallelism} "
        f"ejecutores={exinfo['executors']} nodos={exinfo['nodes']} clases={eng.n_classes} | "
        f"carga {data_load_time:.1f} s, estandarización {standardize_time:.1f} s, "
        f"preparación {engine_prep_time:.1f} s")
    if a.no_cache and a.data == "synthetic":
        log("aviso: --no-cache regenera los datos en cada pasada; ese costo queda dentro de path_time_s")

    need_df = any(i.startswith("mllib") for i in a.inits) or a.mllib_check or a.mllib_once
    mdf = blocks_to_df(spark, rdd) if need_df else None

    base = dict(dataset=name, synthetic=int(a.data == "synthetic"), n=eng.n, d=eng.d,
                k_true=k_true if k_true else "",
                n_classes=eng.n_classes if eng.has_y else "",
                sep=a.sep if a.data == "synthetic" else "",
                imbalance=a.imbalance if a.data == "synthetic" else "",
                std_jitter=a.std_jitter if a.data == "synthetic" else "",
                partitions=eng.P, parallelism=sc.defaultParallelism,
                executors_requested=sc.getConf().get("spark.executor.instances", ""),
                executors=exinfo["executors"], nodes=exinfo["nodes"], cores_total=exinfo["cores_total"],
                hosts=exinfo["hosts"], master=sc.master, tree_depth=a.tree_depth,
                data_load_time_s=data_load_time, standardize_time_s=standardize_time,
                engine_prep_time_s=engine_prep_time,
                dtype=a.dtype, kmax=a.kmax, data_seed=a.data_seed, tag=a.tag,
                config_hash=chash, code_sha=code_sha, crira_sha=crira_sha, repeat=a.repeat)
    out_rows, out_path, out_ml = f"{a.out}.csv", f"{a.out}_path.csv", f"{a.out}_mllib.csv"

    if a.mllib_once:
        from pyspark.ml.clustering import BisectingKMeans, KMeans
        rows = []
        for name, est in (("mllib_kmeans_kmax", KMeans(k=a.kmax, seed=a.seeds[0], maxIter=a.max_iter,
                                                        tol=a.eps, initSteps=a.init_steps)),
                          ("mllib_bisecting_kmax", BisectingKMeans(k=a.kmax, seed=a.seeds[0],
                                                                   maxIter=a.max_iter))):
            log(f"un solo ajuste: {name}")
            t0 = time.perf_counter()
            model = est.fit(mdf)
            t_fit = time.perf_counter() - t0
            with eng.category("eval"):
                Wv = eng.assign_stats(np.array(model.clusterCenters(), dtype=np.float64))["W"]
            rows.append(dict(base, method=name, k=a.kmax, seed=a.seeds[0], fit_time_s=t_fit,
                             training_cost=float(model.summary.trainingCost), W_voronoi=Wv,
                             iters=int(model.summary.numIter)))
        append_rows(f"{a.out}_mllib_once.csv", rows)

    if a.mllib_check and k_true:
        log("verificación del motor contra MLlib")
        append_rows(out_ml, [dict(base, **mllib_check(eng, mdf, k_true, a.seeds[0], a))])

    ks = np.arange(1, a.kmax + 1)
    for seed in a.seeds:
        for init in a.inits:
            key = (chash, str(a.data_seed), str(seed), init, str(a.repeat))
            if key in done:
                if a.on_duplicate == "error":
                    raise SystemExit(f"trayectoria duplicada {key} en {a.out}.csv")
                log(f"omitida (ya existe): init={init} seed={seed} repeat={a.repeat}")
                continue
            run_id = uuid.uuid4().hex[:12]
            ex_start = metrics.executors()
            gpath = f"path-{run_id}"
            sc.setJobGroup(gpath, f"{init} seed={seed}")
            log(f"trayectoria init={init} seed={seed}")
            p0, by0 = eng.snap("path")
            t0 = time.perf_counter()
            path, split_t = run_path(eng, init, a.kmax, seed, a, mllib_df=mdf)
            path_time = time.perf_counter() - t0
            p1, by1 = eng.snap("path")
            instrumented = not init.startswith("mllib")
            sm = metrics.totals(gpath)
            W = np.array([path[k]["W"] for k in ks])
            viol = int(np.sum(W[1:] > W[:-1] * (1.0 + 1e-12)))
            fbs = int(sum(path[k]["fallback"] for k in ks))
            path_rows = [dict(base, run_id=run_id, seed=seed, init=init, k=int(k),
                                        W=path[k]["W"], mllib_training_cost=path[k]["mllib_cost"],
                                        iters=path[k]["iters"], time_s=path[k]["time_s"],
                                        init_time_s=path[k]["init_time_s"],
                                        lloyd_time_s=path[k]["lloyd_time_s"],
                                        passes=path[k]["passes"], init_passes=path[k]["init_passes"],
                                        lloyd_passes=path[k]["lloyd_passes"],
                                        first_cost=path[k]["first_cost"],
                                        fallback=path[k]["fallback"]) for k in ks]

            # --- criterios: costo marginal de cada uno, medido por separado ---
            sel, cstat = {}, {}
            for crit in a.criteria:
                gcrit = f"crit-{run_id}-{crit}"
                sc.setJobGroup(gcrit, crit)
                t0 = time.perf_counter()
                extra = 0.0
                with eng.category("crit"):
                    pc0, _ = eng.snap("crit")
                    if crit == "crira":
                        cache = {}

                        def ref_W(B, kind="uniform", n_ref=None, _init=init, _seed_=seed, _cache=cache):
                            n_ref = n_ref or (min(a.crira_ref_n, eng.n) if a.crira_ref_n > 0 else eng.n)
                            key = (B, kind, n_ref)
                            if key not in _cache:
                                _cache[key] = reference_paths(sc, eng, _init, ks, _seed_, a, dtype, B,
                                                              lambda r: blocks_to_df(spark, r),
                                                              kind=kind, n_ref=n_ref)
                            return _cache[key][0]

                        ctx = dict(n=eng.n, d=eng.d, T=eng.T, ref_W=ref_W, B=a.crira_B,
                                   tau=a.crira_tau, rule=a.crira_rule)
                        sel[crit] = validate_k(crira.select(ks.copy(), W.copy(), ctx), ks, "crira")
                        extra = sum(v[1] for v in cache.values())
                    elif crit == "lcurv":
                        sel[crit] = validate_k(lcurv_select(ks, W), ks, crit)
                    elif crit == "kneedle":
                        sel[crit] = validate_k(kneedle_select(ks, W), ks, crit)
                    elif crit == "ch":
                        sel[crit] = validate_k(ch_select(ks, W, eng.T, eng.n), ks, crit)
                    elif crit == "sil":
                        best_k, best = None, -np.inf
                        for k in ks[1:]:
                            s = eng.silhouette(path[k])
                            if s > best:
                                best_k, best = int(k), s
                        sel[crit] = validate_k(best_k, ks, crit)
                    elif crit == "sile":
                        if getattr(eng, "_sil_Z", None) is None:
                            eng._sil_Z = eng.sample_rows(a.sil_n, [a.data_seed, 424242])
                        best_k, best = None, -np.inf
                        for k in ks[1:]:
                            s_ = eng.silhouette_euclid(eng._sil_Z, path[k]["C"])
                            if s_ > best:
                                best_k, best = int(k), s_
                        sel[crit] = validate_k(best_k, ks, crit)
                    elif crit == "gap":
                        refW, extra = reference_paths(sc, eng, init, ks, seed, a, dtype, a.gap_B,
                                                      lambda r: blocks_to_df(spark, r))
                        sel[crit] = validate_k(gap_select(ks, W, refW), ks, crit)
                    pc1, _ = eng.snap("crit")
                ct = time.perf_counter() - t0
                cm = metrics.totals(gcrit)
                cp = pc1 - pc0 + extra
                if init.startswith("mllib") and crit in ("gap", "crira") and extra > 0:
                    cp = NAN  # las trayectorias de referencia de MLlib no están instrumentadas
                cstat[crit] = dict(crit_time_s=ct, crit_passes=cp,
                                   crit_shuffle_read=cm["shuffleReadBytes"],
                                   crit_shuffle_write=cm["shuffleWriteBytes"],
                                   crit_result_bytes=cm["resultSize"],
                                   crit_executor_run_ms=cm["executorRunTime"])

            # --- refinamiento opcional en el k elegido ---
            refined, rstat = {}, {}
            if a.final_restarts > 0:
                for kh in sorted(set(sel.values())):
                    sc.setJobGroup(f"ref-{run_id}-{kh}", "refine")
                    t0 = time.perf_counter()
                    with eng.category("refine"):
                        pr0, _ = eng.snap("refine")
                        best = path[kh]
                        for r in range(a.final_restarts):
                            res = eng.lloyd(kmeans_parallel_init(eng, kh, [seed, kh, 9973 + r],
                                                                 a.init_steps, a.oversample),
                                            a.max_iter, a.tol)
                            if res["W"] < best["W"]:
                                best = res
                        refined[kh] = best
                        pr1, _ = eng.snap("refine")
                    rstat[kh] = dict(refine_time_s=time.perf_counter() - t0, refine_passes=pr1 - pr0)

            # --- evaluación externa (no contabilizada como costo del algoritmo) ---
            evals = {}

            def quality(tag, k, C):
                if (tag, k) not in evals:
                    if eng.has_y:
                        with eng.category("eval"):
                            evals[(tag, k)] = ari_nmi(eng.contingency(C))
                    else:
                        evals[(tag, k)] = (NAN, NAN)
                return evals[(tag, k)]

            sc.setJobGroup(f"eval-{run_id}", "eval")
            kt = k_true if (k_true and k_true <= a.kmax) else None
            ari_t, nmi_t = quality("path", kt, path[kt]["C"]) if kt else (NAN, NAN)
            mem = metrics.executors()
            # la corrida solo es válida para escalabilidad si los recursos no cambiaron
            run_valid = int(bool(ex_start["signature"]) and mem["signature"] == ex_start["signature"])
            sumk = lambda f: float(np.nansum([path[k][f] for k in ks[1:]]))
            pathrow = dict(
                path_time_s=path_time, path_init_time_s=sumk("init_time_s") if instrumented else NAN,
                path_lloyd_time_s=sumk("lloyd_time_s"),
                split_cov_time_s=split_t["split_cov_s"], split_select_time_s=split_t["split_select_s"],
                split_build_time_s=split_t["split_build_s"],
                path_passes=(p1 - p0) if instrumented else NAN,
                path_init_passes=sumk("init_passes") if instrumented else NAN,
                path_lloyd_passes=sumk("lloyd_passes") if instrumented else NAN,
                path_lloyd_iters=int(sum(path[k]["iters"] for k in ks)),
                path_payload_bytes=(by1 - by0) if instrumented else NAN,
                path_input_bytes=sm["inputBytes"], path_shuffle_read=sm["shuffleReadBytes"],
                path_shuffle_write=sm["shuffleWriteBytes"], path_result_bytes=sm["resultSize"],
                path_executor_run_ms=sm["executorRunTime"], path_executor_cpu_ns=sm["executorCpuTime"],
                path_spill_mem=sm["memoryBytesSpilled"], path_spill_disk=sm["diskBytesSpilled"],
                path_peak_exec_mem=sm["peakExecutionMemory"],
                storage_mem_used=mem["storage_mem_used"], peak_jvm_heap_cum=mem["peak_jvm_heap"],
                peak_python_rss_cum=mem["peak_python_rss"],
                executors_start=ex_start["executors"], nodes_start=ex_start["nodes"],
                hosts_start=ex_start["hosts"], cores_start=ex_start["cores_total"],
                executor_signature_start=ex_start["signature"],
                executors_end=mem["executors"], nodes_end=mem["nodes"], hosts_end=mem["hosts"],
                cores_end=mem["cores_total"], executor_signature_end=mem["signature"],
                resources_stable=run_valid,
                monotone_violations=viol, fallbacks=fbs)
            rows = []
            for crit in a.criteria:
                kh = sel[crit]
                ari_h, nmi_h = quality("path", kh, path[kh]["C"])
                row = dict(base, run_id=run_id, seed=seed, init=init, criterion=crit, k_hat=kh,
                           k_correct=int(kh == k_true) if k_true else "",
                           k_error=(kh - k_true) if k_true else "",
                           W_khat=path[kh]["W"], ari_khat=ari_h, nmi_khat=nmi_h,
                           W_ktrue=path[kt]["W"] if kt else "",
                           ari_ktrue=ari_t if kt else "", nmi_ktrue=nmi_t if kt else "",
                           **pathrow, **cstat[crit])
                prep = data_load_time + standardize_time + engine_prep_time
                row["total_time_s"] = path_time + cstat[crit]["crit_time_s"]
                row["total_passes"] = pathrow["path_passes"] + cstat[crit]["crit_passes"]
                row["pipeline_time_s"] = prep + row["total_time_s"]
                if a.final_restarts > 0:
                    ra, rn = quality("refined", kh, refined[kh]["C"])
                    row.update(refined_W_khat=refined[kh]["W"], refined_ari_khat=ra,
                               refined_nmi_khat=rn, **rstat[kh])
                    row["total_time_refined_s"] = row["total_time_s"] + rstat[kh]["refine_time_s"]
                    row["total_passes_refined"] = row["total_passes"] + rstat[kh]["refine_passes"]
                    row["pipeline_time_refined_s"] = prep + row["total_time_refined_s"]
                rows.append(row)
            # trayectoria y resultados se escriben juntos al terminar: una corrida interrumpida
            # no deja filas huérfanas y al reanudar no se duplican
            append_rows(out_path, path_rows)
            append_rows(out_rows, rows)
            done.add(key)
            log("  " + "  ".join(f"{c}:k={sel[c]}" for c in a.criteria)
                + f"  | {path_time:.1f} s, pasadas={pathrow['path_passes']}, viol={viol}, fb={fbs}")
            sc.setJobGroup("idle", "idle")

    if mdf is not None:
        mdf.unpersist()
    rdd.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
