#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de resultados de crira_dk.py (v2).

Uso:
  python analizar_crira_dk.py --res res/calidad.csv --path res/calidad_path.csv \
      --outdir res/analisis_calidad --ref-init kmeansll --margin-ari 0.01 --margin-inertia 0.01

Unidad de remuestreo: bootstrap jerárquico de dos niveles
  nivel 1: conjuntos de datos (data_seed) con reemplazo, dentro de cada escenario;
  nivel 2: semillas de inicialización (seed) con reemplazo, dentro de cada data_seed.
Estimador de dos etapas: estadístico por conjunto y media entre conjuntos (pesos iguales).
No inferioridad (márgenes fijados antes de correr):
  ARI:     cota inf. 95 % de (ARI_nuevo - ARI_ref) > -margin_ari
  inercia: cota sup. 95 % de (exceso_nuevo - exceso_ref) < margin_inertia,
           exceso = W / W_mejor_conocido(k) - 1 en el mismo conjunto de datos.
Las comparaciones entre inicializaciones se hacen pareadas en (data_seed, seed, criterio).

Datos sintéticos (k verdadero conocido) y reales (sin k verdadero) se analizan por separado.

Salidas en --outdir:
  sint_seleccion_k.csv     recuperación exacta de k, error k_hat - k_true (media, |error|), IC jerárquico
  sint_calidad.csv         ARI, NMI y exceso de inercia en k_hat y en k_true, IC jerárquico
  sint_ablacion.csv        2x2 (init x {CRIRA, convencional}) con término de interacción e IC
  sint_no_inferioridad.csv cada init frente a --ref-init: IC unilateral 95 % jerárquico
  real_calidad.csv         ARI/NMI contra etiquetas, exceso de inercia, estabilidad de k_hat
  eficiencia.csv           tiempo total, inicialización vs Lloyd, pasadas, iteraciones, shuffle
  costo_criterios.csv      costo marginal de cada criterio (tiempo, pasadas, shuffle)
  monotonia.csv            violaciones W(k+1) > W(k) y recurso a la semilla aditiva
  escalabilidad.csv        eficiencia fuerte/débil (tags strong*/weak*), local y clúster
"""
import argparse
import os

import numpy as np
import pandas as pd

SCEN = ["config_hash", "dataset", "n", "d", "k_true", "sep", "imbalance", "std_jitter", "partitions",
        "parallelism", "executors", "tree_depth", "dtype", "kmax", "tag"]
CONVENTIONAL = ["lcurv", "kneedle", "ch", "sil", "gap"]


def scen_cols(df):
    return [c for c in SCEN if c in df.columns]


# ----------------------------------------------------------------------------
# Bootstrap jerárquico
# ----------------------------------------------------------------------------
def hboot(df, col, stat="mean", B=5000, seed=0, outer="data_seed", inner="seed"):
    """
    Estimador de dos etapas y su IC por bootstrap jerárquico.
      estimador: theta = media sobre conjuntos de datos (outer) de f(corridas del conjunto),
                 con f = media (stat="mean") o mediana (stat="median") dentro del conjunto.
      réplica:   se remuestrean conjuntos con reemplazo; dentro de cada uno, corridas (inner)
                 con reemplazo; se aplica exactamente el mismo estimador.
    Cada conjunto pesa lo mismo aunque falten corridas. Devuelve (theta, cota inf. 95 %
    unilateral, cota sup. 95 % unilateral), es decir, los cuantiles 5 % y 95 %.
    """
    d = df[[outer, inner, col]].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[np.isfinite(d[col])]
    if d.empty:
        return np.nan, np.nan, np.nan
    groups = [g[col].to_numpy() for _, g in d.groupby(outer)]
    f = np.mean if stat == "mean" else np.median
    point = float(np.mean([f(x) for x in groups]))
    rng = np.random.default_rng(seed)
    G = len(groups)
    reps = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, G, size=G)
        reps[b] = np.mean([f(groups[i][rng.integers(0, groups[i].size, size=groups[i].size)])
                           for i in pick])
    return point, float(np.quantile(reps, 0.05)), float(np.quantile(reps, 0.95))


def add_ci(row, prefix, df, col, **kw):
    m, lo, hi = hboot(df, col, **kw)
    row[f"{prefix}"] = m
    row[f"{prefix}_lb95"] = lo
    row[f"{prefix}_ub95"] = hi


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--outdir", default="analisis")
    p.add_argument("--ref-init", default="kmeansll")
    p.add_argument("--margin-ari", type=float, default=0.01)
    p.add_argument("--margin-inertia", type=float, default=0.01)
    p.add_argument("--B", type=int, default=5000)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    res = pd.read_csv(a.res)
    path = pd.read_csv(a.path)

    # --- integridad: una sola versión del código y sin observaciones duplicadas
    if "code_sha" in res.columns and res["code_sha"].nunique() > 1:
        raise SystemExit(f"{a.res} mezcla {res['code_sha'].nunique()} versiones de crira_dk.py")
    if "config_hash" in res.columns:
        if "repeat" not in res.columns:
            res["repeat"] = 0
        kcols = ["config_hash", "data_seed", "seed", "init", "criterion", "repeat"]
        dup = res.duplicated(kcols, keep=False)
        if dup.any():
            raise SystemExit(f"{int(dup.sum())} filas duplicadas en {kcols}; revise la campaña")
        pdup = path.duplicated(["config_hash", "data_seed", "seed", "init", "k", "repeat"], keep=False)
        if pdup.any():
            raise SystemExit(f"{int(pdup.sum())} filas duplicadas en {a.path}")
    else:
        res["repeat"] = 0
    for df in (res, path):
        for c in SCEN:
            if c in df.columns and df[c].dtype == object:
                df[c] = df[c].fillna("")
            elif c in df.columns:
                df[c] = df[c].fillna(-1)
    S = scen_cols(res)
    Sd = S + ["data_seed"]

    # --- mejor W(k) conocido por escenario y conjunto de datos (referencia de inercia)
    wb = path.groupby(Sd + ["k"])["W"].min().rename("W_best").reset_index()
    if "refined_W_khat" in res.columns:
        rb = (res.rename(columns={"k_hat": "k"}).groupby(Sd + ["k"])["refined_W_khat"].min()
                 .rename("W_ref").reset_index())
        wb = wb.merge(rb, on=Sd + ["k"], how="left")
        wb["W_best"] = wb[["W_best", "W_ref"]].min(axis=1)
        wb = wb.drop(columns="W_ref")
    r = res.merge(wb.rename(columns={"k": "k_hat"}), on=Sd + ["k_hat"], how="left")
    r["excess_khat"] = r["W_khat"] / r["W_best"] - 1.0
    r = r.drop(columns="W_best")
    if "refined_W_khat" in r.columns:
        r = r.merge(wb.rename(columns={"k": "k_hat"}), on=Sd + ["k_hat"], how="left")
        r["excess_refined_khat"] = r["refined_W_khat"] / r["W_best"] - 1.0
        r = r.drop(columns="W_best")

    # Las repeticiones (repeat > 0) repiten semillas: sirven para medir tiempo, no son
    # réplicas de calidad. Los análisis de calidad usan solo repeat == 0.
    rq = r[pd.to_numeric(r["repeat"], errors="coerce").fillna(0) == 0]
    syn = rq[rq["synthetic"] == 1].copy()
    real = rq[rq["synthetic"] == 0].copy()

    if not syn.empty:
        syn["k_true_num"] = pd.to_numeric(syn["k_true"], errors="coerce")
        kt = wb.rename(columns={"k": "k_true_num", "W_best": "W_best_ktrue"})
        kt = kt[kt["k_true_num"].isin(syn["k_true_num"].unique())]
        syn = syn.merge(kt, on=[c for c in Sd if c != "k_true"] + ["k_true_num"], how="left",
                        suffixes=("", "_y"))
        syn["excess_ktrue"] = pd.to_numeric(syn["W_ktrue"], errors="coerce") / syn["W_best_ktrue"] - 1.0
        for c in ("k_correct", "k_error", "ari_ktrue", "nmi_ktrue"):
            syn[c] = pd.to_numeric(syn[c], errors="coerce")
        syn["abs_k_error"] = syn["k_error"].abs()

        # 1. selección de k
        rows = []
        for key, g in syn.groupby(S + ["init", "criterion"]):
            row = dict(zip(S + ["init", "criterion"], key))
            row["corridas"] = len(g)
            row["conjuntos"] = g["data_seed"].nunique()
            add_ci(row, "recuperacion_k", g, "k_correct", B=a.B)
            add_ci(row, "error_k", g, "k_error", B=a.B)
            add_ci(row, "abs_error_k", g, "abs_k_error", B=a.B)
            rows.append(row)
        sel_df = pd.DataFrame(rows)
        sel_df.to_csv(f"{a.outdir}/sint_seleccion_k.csv", index=False)

        # 2. calidad de agrupamiento (independiente de acertar k)
        rows = []
        for key, g in syn.groupby(S + ["init", "criterion"]):
            row = dict(zip(S + ["init", "criterion"], key))
            for c in ["ari_khat", "nmi_khat", "excess_khat", "ari_ktrue", "nmi_ktrue", "excess_ktrue",
                      "excess_refined_khat", "refined_ari_khat"]:
                if c in g.columns:
                    add_ci(row, c, g, c, B=a.B)
            rows.append(row)
        pd.DataFrame(rows).to_csv(f"{a.outdir}/sint_calidad.csv", index=False)

        # 3. ablación 2x2 con interacción, pareada por (data_seed, seed)
        rows = []
        inits = sorted(set(syn["init"]) - {a.ref_init})
        if "crira" in set(syn["criterion"]):
            for ini in inits:
                for conv in sorted(set(syn["criterion"]) & set(CONVENTIONAL)):
                    for key, g in syn.groupby(S):
                        key = key if isinstance(key, tuple) else (key,)
                        piv = {}
                        for m in ("k_correct", "ari_khat"):
                            t = g[g["init"].isin([ini, a.ref_init]) & g["criterion"].isin(["crira", conv])]
                            t = t.pivot_table(index=["data_seed", "seed"], columns=["init", "criterion"],
                                              values=m)
                            need = [(ini, "crira"), (a.ref_init, "crira"), (ini, conv), (a.ref_init, conv)]
                            if not all(c in t.columns for c in need):
                                continue
                            t = t.dropna(subset=need)
                            inter = ((t[(ini, "crira")] - t[(a.ref_init, "crira")])
                                     - (t[(ini, conv)] - t[(a.ref_init, conv)]))
                            piv[m] = pd.DataFrame({"interaction": inter.values,
                                                   "data_seed": t.index.get_level_values(0),
                                                   "seed": t.index.get_level_values(1)})
                            for c in need:
                                piv[f"{m}_{c[0]}_{c[1]}"] = float(t[c].mean())
                        if not piv:
                            continue
                        row = dict(zip(S, key), init=ini, reference=a.ref_init, conventional=conv)
                        for m in ("k_correct", "ari_khat"):
                            if m in piv:
                                add_ci(row, f"interaccion_{m}", piv[m], "interaction", B=a.B)
                        row.update({k: v for k, v in piv.items() if not isinstance(v, pd.DataFrame)})
                        rows.append(row)
        pd.DataFrame(rows).to_csv(f"{a.outdir}/sint_ablacion.csv", index=False)

        # 4. no inferioridad de cada init frente a la referencia
        rows = []
        key = Sd + ["seed", "criterion"]
        cols = ["ari_khat", "k_correct", "excess_khat", "excess_ktrue", "ari_ktrue"]
        ref = syn[syn["init"] == a.ref_init].set_index(key)[cols]
        for ini in inits:
            cur = syn[syn["init"] == ini].set_index(key)[cols]
            j = cur.join(ref, lsuffix="_new", rsuffix="_ref", how="inner").reset_index()
            for c in ("ari_khat", "k_correct", "ari_ktrue", "excess_khat", "excess_ktrue"):
                j[f"d_{c}"] = j[f"{c}_new"] - j[f"{c}_ref"]
            for gk, g in j.groupby(S + ["criterion"]):
                row = dict(zip(S + ["criterion"], gk), init=ini, reference=a.ref_init, pares=len(g),
                           conjuntos=g["data_seed"].nunique())
                add_ci(row, "delta_ari_khat", g, "d_ari_khat", B=a.B)
                add_ci(row, "delta_ari_ktrue", g, "d_ari_ktrue", B=a.B)
                add_ci(row, "delta_recuperacion", g, "d_k_correct", B=a.B)
                # diferencia pareada de exceso relativo de inercia (nuevo - referencia)
                add_ci(row, "delta_exceso_inercia_ktrue", g, "d_excess_ktrue", B=a.B)
                add_ci(row, "delta_exceso_inercia_khat", g, "d_excess_khat", B=a.B)
                # nivel absoluto, solo descriptivo
                add_ci(row, "exceso_inercia_ktrue_nuevo", g, "excess_ktrue_new", B=a.B)
                add_ci(row, "exceso_inercia_ktrue_ref", g, "excess_ktrue_ref", B=a.B)
                # no inferioridad con márgenes fijados antes de correr los experimentos
                row["NI_ari_khat"] = bool(row["delta_ari_khat_lb95"] > -a.margin_ari)
                row["NI_ari_ktrue"] = bool(row["delta_ari_ktrue_lb95"] > -a.margin_ari)
                row["NI_inercia_ktrue"] = bool(row["delta_exceso_inercia_ktrue_ub95"] < a.margin_inertia)
                row["NI_inercia_khat"] = bool(row["delta_exceso_inercia_khat_ub95"] < a.margin_inertia)
                rows.append(row)
        ni_df = pd.DataFrame(rows)
        ni_df.to_csv(f"{a.outdir}/sint_no_inferioridad.csv", index=False)

    # 5. datos reales: sin k verdadero
    if not real.empty:
        rows = []
        for key, g in real.groupby(S + ["init", "criterion"]):
            row = dict(zip(S + ["init", "criterion"], key), corridas=len(g))
            vc = g["k_hat"].value_counts()
            row["k_hat_moda"] = int(vc.index[0])
            row["k_hat_frec_moda"] = float(vc.iloc[0] / len(g))
            row["k_hat_min"] = int(g["k_hat"].min())
            row["k_hat_max"] = int(g["k_hat"].max())
            row["k_hat_iqr"] = float(g["k_hat"].quantile(0.75) - g["k_hat"].quantile(0.25))
            for c in ["ari_khat", "nmi_khat", "excess_khat", "excess_refined_khat", "total_time_s"]:
                if c in g.columns:
                    add_ci(row, c, g, c, B=a.B)
            rows.append(row)
        pd.DataFrame(rows).to_csv(f"{a.outdir}/real_calidad.csv", index=False)

    # 6. eficiencia de extremo a extremo (por trayectoria)
    pth = r.drop_duplicates("run_id").copy()
    pth["seed_rep"] = pth["seed"].astype(str) + "_" + pth["repeat"].astype(str)
    # columnas que dependen solo de la trayectoria; los totales por criterio están en costo_criterios.csv
    ecols = ["data_load_time_s", "standardize_time_s", "engine_prep_time_s",
             "path_time_s", "path_init_time_s", "path_lloyd_time_s", "split_cov_time_s",
             "split_select_time_s", "split_build_time_s", "path_passes", "path_init_passes",
             "path_lloyd_passes", "path_lloyd_iters", "path_payload_bytes", "path_shuffle_read",
             "path_shuffle_write", "path_result_bytes", "path_executor_run_ms",
             "path_spill_disk", "path_peak_exec_mem", "peak_python_rss_cum"]
    ecols = [c for c in ecols if c in pth.columns]
    rows = []
    for key, g in pth.groupby(S + ["init"]):
        row = dict(zip(S + ["init"], key), corridas=len(g))
        for c in ecols:
            row[f"{c}_mediana"] = float(pd.to_numeric(g[c], errors="coerce").median())
        rows.append(row)
    eff = pd.DataFrame(rows)
    ref_p = pth[pth["init"] == a.ref_init].set_index(Sd + ["seed", "repeat"])
    for ini in sorted(set(pth["init"]) - {a.ref_init}):
        cur = pth[pth["init"] == ini].set_index(Sd + ["seed", "repeat"])
        j = ref_p[["path_time_s", "path_lloyd_iters"]].join(
            cur[["path_time_s", "path_lloyd_iters"]], lsuffix="_ref", rsuffix="_new", how="inner").reset_index()
        j["speedup"] = j["path_time_s_ref"] / j["path_time_s_new"]
        j["ratio_iters"] = j["path_lloyd_iters_ref"] / j["path_lloyd_iters_new"]
        for key, g in j.groupby(S):
            key = key if isinstance(key, tuple) else (key,)
            m = (eff["init"] == ini)
            for c, v in zip(S, key):
                m &= eff[c] == v
            for col, lab in (("speedup", "speedup_vs_ref"), ("ratio_iters", "ratio_iters_vs_ref")):
                g = g.assign(seed_rep=g["seed"].astype(str) + "_" + g["repeat"].astype(str))
                pt, lo, hi = hboot(g, col, stat="median", B=a.B, inner="seed_rep")
                eff.loc[m, lab] = pt
                eff.loc[m, f"{lab}_lb95"] = lo
                eff.loc[m, f"{lab}_ub95"] = hi
    eff.to_csv(f"{a.outdir}/eficiencia.csv", index=False)

    # 7. costo marginal de cada criterio
    ccols = [c for c in ["crit_time_s", "crit_passes", "crit_shuffle_read", "crit_shuffle_write",
                         "crit_result_bytes", "crit_executor_run_ms", "total_time_s", "total_passes",
                         "refine_time_s", "refine_passes", "total_time_refined_s",
                         "total_passes_refined", "pipeline_time_s", "pipeline_time_refined_s"]
             if c in r.columns]
    cc = r.groupby(S + ["init", "criterion"])[ccols].median().reset_index()
    cc.to_csv(f"{a.outdir}/costo_criterios.csv", index=False)

    # 8. monotonía
    mono = pth.groupby(S + ["init"]).agg(
        corridas=("run_id", "count"),
        corridas_con_violacion=("monotone_violations", lambda x: int((x > 0).sum())),
        violaciones=("monotone_violations", "sum"),
        semillas_aditivas=("fallbacks", "sum")).reset_index()
    mono.to_csv(f"{a.outdir}/monotonia.csv", index=False)

    # 9. escalabilidad. Eje = recursos ACTIVOS medidos, no los solicitados:
    #    clúster -> nodos físicos distintos (y se reporta ejecutores/nodo); local -> hilos.
    rows, alerts = [], []
    for tagpref in ("strong", "weak"):
        g = pth[pth["tag"].astype(str).str.startswith(tagpref)].copy()
        if g.empty:
            continue
        for c in ("executors", "nodes", "parallelism", "executors_requested", "resources_stable"):
            if c in g.columns:
                g[c] = pd.to_numeric(g[c], errors="coerce")
        bad = g[g["resources_stable"] == 0] if "resources_stable" in g.columns else g.iloc[0:0]
        if len(bad):
            alerts.append(f"{tagpref}: {len(bad)} corridas excluidas (cambiaron ejecutores o nodos durante la corrida)")
            g = g[g["resources_stable"] != 0]
        cl = g["executors"].fillna(0) > 0
        if "executors_requested" in g.columns:
            short = g[cl & (g["executors"] < g["executors_requested"])]
            if len(short):
                alerts.append(f"{tagpref}: {len(short)} corridas con menos ejecutores activos que solicitados")
        packed = g[cl & (g["executors"] > g["nodes"])]
        if len(packed):
            alerts.append(f"{tagpref}: {len(packed)} corridas con más de un ejecutor por nodo")
        g["modo"] = np.where(cl, "cluster", "local")
        g["recurso"] = np.where(cl, g["nodes"], g["parallelism"])
        g["eje"] = np.where(cl, "nodos_activos", "hilos_locales")
        for (tag, modo, ini), gi in g.groupby(["tag", "modo", "init"]):
            t = gi.groupby("recurso").agg(
                tiempo=("path_time_s", "median"), tiempo_total=("total_time_s", "median"),
                n=("n", "first"), ejecutores=("executors", "median"),
                shuffle=("path_shuffle_read", "median"), corridas=("run_id", "count")).reset_index()
            t = t.sort_values("recurso")
            c0, t0 = t.iloc[0]["recurso"], t.iloc[0]["tiempo"]
            for _, rr in t.iterrows():
                eff_ = ((t0 * c0) / (rr["recurso"] * rr["tiempo"]) if tagpref == "strong"
                        else t0 / rr["tiempo"])
                rows.append(dict(tipo=tagpref, tag=tag, modo=modo, eje=gi["eje"].iloc[0], init=ini,
                                 recurso=rr["recurso"], ejecutores_mediana=rr["ejecutores"],
                                 n=rr["n"], corridas=rr["corridas"],
                                 tiempo_trayectoria_mediana_s=rr["tiempo"],
                                 tiempo_total_mediana_s=rr["tiempo_total"], speedup=t0 / rr["tiempo"],
                                 eficiencia=eff_, shuffle_read_mediana=rr["shuffle"]))
    if rows:
        pd.DataFrame(rows).to_csv(f"{a.outdir}/escalabilidad.csv", index=False)
    for m in alerts:
        print("AVISO:", m)

    # resumen en consola
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    if not syn.empty:
        print("\n== Sintéticos: selección de k ==")
        print(sel_df[["init", "criterion", "recuperacion_k", "recuperacion_k_lb95",
                      "recuperacion_k_ub95", "abs_error_k"]].to_string(index=False))
        if len(ni_df):
            print(f"\n== No inferioridad frente a {a.ref_init} ==")
            print(ni_df[["init", "criterion", "pares", "delta_ari_khat", "delta_ari_khat_lb95",
                         "NI_ari_khat", "delta_exceso_inercia_ktrue_ub95", "NI_inercia_ktrue"]].to_string(index=False))
    print("\n== Eficiencia (medianas) ==")
    show = [c for c in ["init", "path_time_s_mediana", "path_init_time_s_mediana",
                        "path_lloyd_time_s_mediana", "path_passes_mediana", "path_lloyd_iters_mediana",
                        "speedup_vs_ref"] if c in eff.columns]
    print(eff[show].to_string(index=False))
    print("\n== Monotonía ==")
    print(mono[["init", "corridas", "corridas_con_violacion", "semillas_aditivas"]].to_string(index=False))
    print(f"\nTablas en {a.outdir}/")


if __name__ == "__main__":
    main()
