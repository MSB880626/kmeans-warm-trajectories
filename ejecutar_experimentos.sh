#!/usr/bin/env bash
# K-means distribuido con trayectoria monótona (warm) — experimentos en UNA máquina (modo local de Spark).
# Lo que se mide aquí es paralelismo local (hilos de un solo proceso ejecutor),
# no escalabilidad multinodo; esa está en ejecutar_cluster.sh.
#
# Uso: bash ejecutar_experimentos.sh [piloto|calidad|escala_n|memoria|particiones|fuerte|debil|criterios|reales|analizar|todo]
#      Equipo con 8 GB (p. ej. Mac M1): MEM=4g bash ejecutar_experimentos.sh [mac_prueba|mac_piloto|mac_calidad|mac_escala|mac_fuerte|mac_debil|mac_criterios|mac_reales|mac_todo|analizar]
set -euo pipefail
PY=${PY:-python3}
S=crira_dk.py
if [ "$(uname)" = "Darwin" ]; then MEM=${MEM:-4g}; else MEM=${MEM:-12g}; fi
if [ "$(uname)" = "Darwin" ]; then
  # Apple Silicon: solo núcleos de rendimiento; los de eficiencia son más lentos y
  # distorsionan las mediciones de escalabilidad.
  CORES=${CORES:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)}
  TIMECMD="/usr/bin/time -l"
  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
  export SPARK_LOCAL_IP=${SPARK_LOCAL_IP:-127.0.0.1}
else
  CORES=${CORES:-$(nproc)}
  TIMECMD="/usr/bin/time -v"
fi
# si el sistema no tiene /usr/bin/time (p. ej. Kaggle), se omite: crira_dk.py ya mide los tiempos
[ -x /usr/bin/time ] || TIMECMD=""
export PYSPARK_PYTHON=${PYSPARK_PYTHON:-$(command -v $PY)}
# Cada campaña va a un directorio nuevo. Para REANUDAR una campaña interrumpida, pase el mismo
# OUT: las trayectorias ya registradas (misma configuración y semillas) se omiten, no se duplican.
OUT=${OUT:-res_local/campana_$(date +%Y%m%d_%H%M%S)}
REPS=${REPS:-1}   # repeticiones de medición de tiempo (mismas semillas) en bloques de escalabilidad
# Gap con referencias de tamaño fijo: su costo no crece con n
if [ "$(uname)" = "Darwin" ]; then GAPREF=${GAPREF:-200000}; else GAPREF=${GAPREF:-1000000}; fi
INITS=${INITS:-"kmeansll warm incpp mllib_kmeans mllib_bisecting"}
mkdir -p "$OUT"
echo "campaña: $OUT"
COMMON="--master local[$CORES] --driver-memory $MEM"

piloto() {
  # Una configuración: medir tiempo antes de lanzar la matriz.
  $TIMECMD $PY $S $COMMON --n 1000000 --d 10 --k-true 8 --sep 3.5 --kmax 25 \
    --data-seed 101 --seeds 0 --gap-B 5 --gap-ref-n 200000 --final-restarts 3 \
    --inits $INITS --partitions $((CORES*4)) --mllib-check --tag piloto --out "$OUT/piloto"
}

calidad() {
  # Factorial de dificultad en 1M. Unidad de réplica: data-seed (5 conjuntos) x seed (3).
  for ds in 101 102 103 104 105; do
    for sep in 2.5 3.5 5; do
      for imb in 0 2; do
        for kt in 4 8 16; do
          for d in 10 50; do
            $PY $S $COMMON --n 1000000 --d $d --k-true $kt --sep $sep --imbalance $imb \
              --std-jitter 0.3 --kmax 25 --data-seed $ds --seeds 0 1 2 --gap-B 5 \
              --gap-ref-n 200000 --final-restarts 3 --inits $INITS \
              --partitions $((CORES*4)) --tag calidad --out "$OUT/calidad"
          done
        done
      done
    done
  done
}

escala_n() {
  # Crecimiento de n con P FIJO (aísla n) y con tamaño de bloque fijo (250k filas/partición).
  for rep in $(seq 0 $((REPS-1))); do
  for n in 1000000 5000000 10000000 50000000 100000000; do
    crit="gap ch sil lcurv kneedle"
    $PY $S $COMMON --n $n --d 10 --k-true 8 --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 1 2 \
      --criteria $crit --gap-B 5 --gap-ref-n 1000000 --inits kmeansll warm incpp \
      --dtype float32 --partitions 256 --tag escala_n_Pfijo --repeat $rep --out "$OUT/escala_n"
    $PY $S $COMMON --n $n --d 10 --k-true 8 --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 1 2 \
      --criteria $crit --gap-B 5 --gap-ref-n 1000000 --inits kmeansll warm incpp \
      --dtype float32 --partitions $(( n/250000 > 4 ? n/250000 : 4 )) --tag escala_n_bloquefijo --repeat $rep \
      --out "$OUT/escala_n"
  done
  done
}

memoria() {
  # Escalabilidad de memoria: n x d con P fijo. Registra derrames a disco y picos de memoria.
  for d in 10 50 100; do
    for n in 1000000 10000000 50000000; do
      $PY $S $COMMON --n $n --d $d --k-true 8 --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 \
        --criteria gap ch --gap-B 5 --gap-ref-n $GAPREF --inits kmeansll warm incpp --dtype float32 --partitions 256 \
        --tag memoria --out "$OUT/memoria"
    done
  done
}

particiones() {
  # Sensibilidad a P y a la profundidad del árbol de agregación; todo lo demás fijo.
  for rep in $(seq 0 $((REPS-1))); do
  for P in 16 64 256 1024; do
    for depth in 2 3; do
      $PY $S $COMMON --n 10000000 --d 10 --k-true 8 --sep 3.5 --kmax 20 --data-seed 101 \
        --seeds 0 1 2 --criteria gap ch --gap-B 5 --gap-ref-n $GAPREF --inits kmeansll warm incpp --dtype float32 \
        --partitions $P --tree-depth $depth --tag particiones --repeat $rep --out "$OUT/particiones"
    done
  done
  done
}

fuerte() {
  # Paralelismo local fuerte: n y P fijos, hilos variables.
  for rep in $(seq 0 $((REPS-1))); do
  for c in 1 2 4 8 16 32; do
    [ $c -le $CORES ] || continue
    $PY $S --master "local[$c]" --driver-memory $MEM --n 10000000 --d 10 \
      --k-true 8 --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 1 2 --criteria gap ch --gap-B 5 --gap-ref-n $GAPREF \
      --inits kmeansll warm incpp --dtype float32 --partitions 64 --tag strong_local --repeat $rep --out "$OUT/fuerte"
  done
  done
}

debil() {
  # Paralelismo local débil: 2.5M filas por hilo, 8 particiones por hilo.
  for rep in $(seq 0 $((REPS-1))); do
  for c in 1 2 4 8 16 32; do
    [ $c -le $CORES ] || continue
    $PY $S --master "local[$c]" --driver-memory $MEM --n $((c*2500000)) \
      --d 10 --k-true 8 --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 1 2 --criteria gap ch --gap-B 5 --gap-ref-n $GAPREF \
      --inits kmeansll warm incpp --dtype float32 --partitions $((c*8)) --tag weak_local --repeat $rep --out "$OUT/debil"
  done
  done
}

criterios() {
  # Costo marginal de cada criterio sobre la misma trayectoria (tiempo, pasadas, shuffle).
  for n in 1000000 10000000; do
    $PY $S $COMMON --n $n --d 10 --k-true 8 --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 1 2 \
      --criteria gap ch sil lcurv kneedle --gap-B 10 --gap-ref-n 0 --inits kmeansll warm incpp \
      --dtype float32 --partitions 64 --tag criterios --out "$OUT/criterios"
  done
}

reales() {
  # Archivos esperados en datos/ (UCI): USCensus1990.data.txt, HIGGS.csv.gz, SUSY.csv.gz,
  # covtype.data.gz, kddcup.data.gz. Las etiquetas solo se usan para ARI/NMI.
  R="--kmax 30 --seeds 0 1 2 3 4 --gap-B 5 --gap-ref-n 1000000 --final-restarts 3 --inits $INITS --tag real"
  $PY $S $COMMON --data real --path datos/USCensus1990.data.txt --header --drop-cols caseid \
    --dataset-name census1990 --partitions 64 $R --out "$OUT/reales"
  $PY $S $COMMON --data real --path datos/HIGGS.csv.gz --label-col _c0 --dataset-name higgs \
    --dtype float32 --partitions 128 $R --out "$OUT/reales"
  $PY $S $COMMON --data real --path datos/SUSY.csv.gz --label-col _c0 --dataset-name susy \
    --partitions 64 $R --out "$OUT/reales"
  $PY $S $COMMON --data real --path datos/covtype.data.gz --label-col _c54 --dataset-name covertype \
    --partitions 32 $R --out "$OUT/reales"
  $PY $S $COMMON --data real --path datos/kddcup.data.gz --label-col _c41 --drop-cols _c1 _c2 _c3 \
    --dataset-name kdd99 --partitions 64 $R --out "$OUT/reales"
}


# ----------------------------------------------------------------------------
# Matriz reducida para un equipo de 8 GB de RAM (Mac M1). Límites:
#  - n <= 10M con d=10 en float32 (datos ~0.4 GB, más copias de serialización JVM/Python);
#  - líneas base de MLlib solo hasta 1M (convierten cada fila en un objeto Vector);
#  - Gap con referencias de 50k-200k puntos;
#  - escalabilidad solo con núcleos de rendimiento. 50M-100M y multinodo: ejecutar_cluster.sh.
# La calidad (recuperación de k, ARI, no inferioridad) no depende de que n sea enorme:
# esa parte del artículo sí puede salir completa de este equipo.
# ----------------------------------------------------------------------------
MACX="--dtype float32 --driver-memory $MEM"

mac_prueba() {
  # 2-3 minutos: comprueba instalación y salida.
  $PY $S --master "local[$CORES]" $MACX --n 100000 --d 5 --k-true 5 --sep 4 --kmax 8 \
    --data-seed 1 --seeds 0 --criteria lcurv ch sil --inits kmeansll warm incpp \
    --partitions 8 --tag prueba --out "$OUT/prueba"
}

mac_piloto() {
  # Una configuración de la matriz de calidad, para estimar el tiempo total antes de lanzarla.
  $TIMECMD $PY $S --master "local[$CORES]" $MACX --n 200000 --d 10 \
    --k-true 8 --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 --gap-B 5 --gap-ref-n 50000 \
    --final-restarts 3 --inits kmeansll warm incpp mllib_kmeans mllib_bisecting \
    --partitions 16 --mllib-check --tag piloto --out "$OUT/piloto"
}

mac_calidad() {
  # Factorial de dificultad con n = 200k. 5 conjuntos x 3 semillas por celda.
  # Las líneas base de MLlib se corren solo en un subconjunto (sep=3.5, imb=0) por costo.
  for ds in 101 102 103 104 105; do
    for sep in 2.5 3.5 5; do
      for imb in 0 2; do
        for kt in 4 8 16; do
          for d in 10 50; do
            ini="kmeansll warm incpp"
            [ "$sep" = "3.5" ] && [ "$imb" = "0" ] && ini="kmeansll warm incpp mllib_kmeans mllib_bisecting"
            $PY $S --master "local[$CORES]" $MACX --n 200000 --d $d \
              --k-true $kt --sep $sep --imbalance $imb --std-jitter 0.3 --kmax 25 \
              --data-seed $ds --seeds 0 1 2 --gap-B 5 --gap-ref-n 50000 --final-restarts 3 \
              --inits $ini --partitions 16 --tag calidad_200k --out "$OUT/calidad"
          done
        done
      done
    done
  done
}

mac_escala() {
  # Crecimiento de n hasta 10M, P fijo y bloque fijo. Gap con referencias de tamaño fijo en todos los n.
  for rep in $(seq 0 $((REPS-1))); do
    for n in 250000 1000000 2500000 5000000 10000000; do
      crit="gap ch sil lcurv kneedle"
      $PY $S --master "local[$CORES]" $MACX --n $n --d 10 --k-true 8 \
        --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 1 2 --criteria $crit --gap-B 5 \
        --gap-ref-n 200000 --inits kmeansll warm incpp --partitions 32 \
        --tag escala_n_Pfijo --repeat $rep --out "$OUT/escala_n"
      $PY $S --master "local[$CORES]" $MACX --n $n --d 10 --k-true 8 \
        --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 1 2 --criteria $crit --gap-B 5 \
        --gap-ref-n 200000 --inits kmeansll warm incpp \
        --partitions $(( n/125000 > 2 ? n/125000 : 2 )) --tag escala_n_bloquefijo --repeat $rep \
        --out "$OUT/escala_n"
    done
  done
}

mac_fuerte() {
  # Paralelismo local fuerte con 1..CORES núcleos de rendimiento; n = 5M fijo.
  for rep in $(seq 0 $((REPS-1))); do
    for c in 1 2 3 4 6 8; do
      [ $c -le $CORES ] || continue
      $PY $S --master "local[$c]" $MACX --n 5000000 --d 10 --k-true 8 \
        --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 1 2 --criteria gap ch --gap-B 5 --gap-ref-n $GAPREF \
        --inits kmeansll warm incpp --partitions 32 --tag strong_local --repeat $rep --out "$OUT/fuerte"
    done
  done
}

mac_debil() {
  # Paralelismo local débil: 2M filas por núcleo.
  for rep in $(seq 0 $((REPS-1))); do
    for c in 1 2 3 4 6 8; do
      [ $c -le $CORES ] || continue
      $PY $S --master "local[$c]" $MACX --n $((c*2000000)) --d 10 \
        --k-true 8 --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 1 2 --criteria gap ch --gap-B 5 --gap-ref-n $GAPREF \
        --inits kmeansll warm incpp --partitions $((c*8)) --tag weak_local --repeat $rep --out "$OUT/debil"
    done
  done
}

mac_criterios() {
  # Costo marginal de cada criterio sobre la misma trayectoria, n = 1M, Gap con referencia completa.
  $PY $S --master "local[$CORES]" $MACX --n 1000000 --d 10 --k-true 8 \
    --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 1 2 --criteria gap ch sil lcurv kneedle \
    --gap-B 10 --gap-ref-n 0 --inits kmeansll warm incpp --partitions 32 \
    --tag criterios --out "$OUT/criterios"
}

mac_reales() {
  # Covertype completo (581k); SUSY y HIGGS submuestreados para caber en 8 GB.
  R="--kmax 30 --seeds 0 1 2 3 4 --gap-B 5 --gap-ref-n 100000 --final-restarts 3 --tag real"
  $PY $S --master "local[$CORES]" $MACX --data real \
    --path datos/covtype.data.gz --label-col _c54 --dataset-name covertype --partitions 16 \
    --inits kmeansll warm incpp mllib_kmeans mllib_bisecting $R --out "$OUT/reales"
  $PY $S --master "local[$CORES]" $MACX --data real \
    --path datos/SUSY.csv.gz --label-col _c0 --dataset-name susy_40pct --sample-fraction 0.4 \
    --partitions 32 --inits kmeansll warm incpp $R --out "$OUT/reales"
  $PY $S --master "local[$CORES]" $MACX --data real \
    --path datos/HIGGS.csv.gz --label-col _c0 --dataset-name higgs_20pct --sample-fraction 0.2 \
    --partitions 32 --inits kmeansll warm incpp $R --out "$OUT/reales"
}

analizar() {
  for f in prueba piloto calidad ablacion mllib sensibilidad escala_n memoria particiones fuerte debil criterios reales; do
    if [ -f "$OUT/$f.csv" ]; then $PY analizar_crira_dk.py --res "$OUT/$f.csv" --path "$OUT/${f}_path.csv" \
      --outdir "$OUT/analisis_$f"; fi
  done
}

# ----------------------------------------------------------------------------
# Bloques para Kaggle (4 núcleos, ~30 GB, sesiones de 12 h). Cada uno cabe en una sesión.
# Criterio de paro de Lloyd: scikit-learn (--conv sklearn, valor por defecto); la sensibilidad a shift/both está en kaggle_sensibilidad.
# ----------------------------------------------------------------------------
KAG="--master local[4] --driver-memory ${MEM}"
# Freno de tiempo común: si se acerca al límite de 12 h de Kaggle, no inicia otra corrida
# y termina limpio para que Kaggle guarde la salida (BUDGET_H, por defecto 10.5 h).
T0_KAG=$(date +%s)
presupuesto_ok() { [ $(( $(date +%s) - T0_KAG )) -lt $(awk "BEGIN{print int(${BUDGET_H:-10.5}*3600)}") ]; }
DS=${DS:-"101 102 103 104 105"}

kaggle_calidad() {
  # 12 configuraciones por data_seed (~8 h por data_seed en Kaggle). Una sesión por data_seed:
  #   DS="101", luego DS="102", luego DS="103". El incremental va aparte (kaggle_ablacion).
  # Presupuesto de tiempo: si se acerca al límite de 12 h de Kaggle, termina limpiamente
  # para que Kaggle guarde la salida (BUDGET_H, por defecto 10.5 h).
  local START=$(date +%s)
  local BUDGET_S=$(awk "BEGIN{print int(${BUDGET_H:-10.5}*3600)}")
  for ds in $DS; do
    for sep in ${SEPS:-2.5 4}; do
      for imb in 0 2; do
        for kt in 4 8 16; do
          if [ $(( $(date +%s) - START )) -gt $BUDGET_S ]; then
            echo "presupuesto de tiempo agotado: se detiene antes de ds=$ds sep=$sep imb=$imb k=$kt"
            return 0
          fi
          $PY $S $KAG --n 200000 --d 10 --k-true $kt --sep $sep --imbalance $imb --std-jitter 0.3 \
            --kmax 20 --data-seed $ds --seeds 0 --inits kmeansll warm --criteria ${CRIT_C:-gap ch sil} \
            --gap-B ${GAPB:-3} --gap-ref-n 20000 --partitions 16 \
            --tag ${TAG_C:-calidad_kaggle} --out "$OUT/${DIR_C:-calidad}"
        done
      done
    done
  done
}

kaggle_mllib() {
  # Comparación con MLlib en 3 conjuntos de 1M (~30 min cada uno), con un solo ajuste
  # de BisectingKMeans y KMeans en k = kmax como cota inferior de su costo.
  for ds in 101 102 103; do
    presupuesto_ok || return 0
    $PY $S $KAG --n 1000000 --d 10 --k-true 8 --sep 3.5 --kmax 25 --data-seed $ds --seeds 0 \
      --inits warm kmeansll mllib_kmeans --criteria ch --mllib-once --mllib-check \
      --partitions 16 --tag mllib --out "$OUT/mllib"
  done
}

kaggle_escala() {
  # Crecimiento de n con P y recursos fijos. Sin Gap: aquí se mide tiempo.
  # Medido: ~11 min con 1M; crece casi lineal -> 1, 2, 5 y 10 M en ~3.3 h.
  for n in 1000000 2000000 5000000 10000000; do
    presupuesto_ok || return 0
    $PY $S $KAG --n $n --d 10 --k-true 8 --sep 3.5 --kmax 20 --data-seed 101 --seeds 0 \
      --inits kmeansll warm --criteria ch --dtype float32 --partitions 64 \
      --tag escala_n --out "$OUT/escala_n"
  done
}

kaggle_fuerte() {
  # Paralelismo local fuerte: 1, 2 y 4 hilos con n = 5M (~3 h).
  for c in 1 2 4; do
    $PY $S --master "local[$c]" --driver-memory ${MEM} --n 5000000 --d 10 --k-true 8 --sep 3.5 \
      --kmax 20 --data-seed 101 --seeds 0 --inits kmeansll warm --criteria ch --dtype float32 \
      --partitions 32 --tag strong_local --out "$OUT/fuerte"
  done
}

kaggle_sensibilidad() {
  # Misma comparación warm vs k-means|| con los tres criterios de paro (~3 h).
  for conv in sklearn both shift; do
    for ds in 101 102 103; do
      presupuesto_ok || return 0
      $PY $S $KAG --n 1000000 --d 10 --k-true 8 --sep 3.5 --kmax 25 --data-seed $ds --seeds 0 \
        --inits warm kmeansll --criteria ch --conv $conv --partitions 16 \
        --tag conv_$conv --out "$OUT/sensibilidad"
    done
  done
}

kaggle_ablacion() {
  # warm frente a k-means++ incremental (incpp, también monótono) y k-means||, n = 200,000,
  # k = 8, 3 conjuntos x 2 separaciones, semillas 0 1 2 (warm es determinista; incpp y
  # k-means|| no). Medido con las corridas previas: ~2.5 h.
  for ds in 101 102 103; do
    for sep in 2.5 4; do
      presupuesto_ok || return 0
      $PY $S $KAG --n 200000 --d 10 --k-true 8 --sep $sep --std-jitter 0.3 --kmax 20 \
        --data-seed $ds --seeds 0 1 2 --inits warm incpp kmeansll --criteria ch sil sile \
        --partitions 16 --tag ablacion --out "$OUT/ablacion"
    done
  done
}

kaggle_reales() {
  # Datos reales (UCI). Se descargan en /kaggle/tmp (no ocupa la salida). Las etiquetas solo
  # sirven para ARI/NMI; no hay k verdadero. Orden: Covertype, SUSY (muestra), HIGGS (muestra).
  # Freno de tiempo igual que kaggle_calidad (BUDGET_H, por defecto 10.5 h).
  local START=$(date +%s)
  local BUDGET_S=$(awk "BEGIN{print int(${BUDGET_H:-10.5}*3600)}")
  local D=${DATOS:-/kaggle/tmp/datos}
  local U=https://archive.ics.uci.edu/ml/machine-learning-databases
  mkdir -p "$D"
  bajar() {  # bajar <url> <archivo>
    [ -s "$D/$2" ] && return 0
    echo "descargando $2 ..."
    if ! wget -q -O "$D/$2" "$1"; then echo "AVISO: no se pudo descargar $2"; rm -f "$D/$2"; return 1; fi
    ls -lh "$D/$2"
  }
  tiempo_ok() { [ $(( $(date +%s) - START )) -lt $BUDGET_S ]; }
  R="--kmax 20 --seeds ${SEEDS_R:-0} --inits kmeansll warm --criteria ${CRIT_R:-gap ch sil} --gap-B 3 --gap-ref-n 20000 --tag real"

  if [ "${SIN_COVERTYPE:-0}" = 0 ] && bajar "$U/covtype/covtype.data.gz" covtype.data.gz && tiempo_ok; then
    $PY $S $KAG --data real --path "$D/covtype.data.gz" --label-col _c54 --dataset-name covertype \
      --partitions 16 $R --out "$OUT/reales"
  fi
  if bajar "$U/00279/SUSY.csv.gz" SUSY.csv.gz && tiempo_ok; then
    $PY $S $KAG --data real --path "$D/SUSY.csv.gz" --label-col _c0 --dataset-name susy_40pct \
      --sample-fraction 0.4 --data-seed 7 --dtype float32 --partitions 32 $R --out "$OUT/reales"
  fi
  if bajar "$U/00280/HIGGS.csv.gz" HIGGS.csv.gz && tiempo_ok; then
    $PY $S $KAG --data real --path "$D/HIGGS.csv.gz" --label-col _c0 --dataset-name higgs_10pct \
      --sample-fraction 0.1 --data-seed 7 --dtype float32 --partitions 32 $R --out "$OUT/reales"
  fi
  tiempo_ok || echo "presupuesto de tiempo agotado"
}

kaggle_sesionA() {
  # Una sesión de Kaggle (~6.2 h), todo con el código final, en orden de prioridad:
  # ablación (~2.5 h), sensibilidad al criterio de paro (~1.6 h) y MLlib (~2.1 h).
  kaggle_ablacion; kaggle_sensibilidad; kaggle_mllib
  presupuesto_ok || echo "presupuesto de tiempo agotado"
}

kaggle_sesionB() {
  # Una sesión de Kaggle (~3.3 h): escalamiento con n.
  kaggle_escala
  presupuesto_ok || echo "presupuesto de tiempo agotado"
}

kaggle_semillas() {
  # Datos reales con las semillas 1 y 2 (la 0 ya se corrió) y la silueta euclidiana
  # convencional (sile). Medido: ~5.5 h en Kaggle (Covertype ~50 min, SUSY ~2.3 h, HIGGS ~2.2 h).
  SEEDS_R="1 2" CRIT_R="gap ch sil sile" kaggle_reales
}

kaggle_semillas2() {
  # Igual que kaggle_semillas pero sin Covertype (ya terminado): SUSY y HIGGS, semillas 1 y 2.
  # Medido: ~4.6 h en Kaggle.
  SIN_COVERTYPE=1 SEEDS_R="1 2" CRIT_R="gap ch sil sile" kaggle_reales
}

kaggle_gap10() {
  # Datos sintéticos con Gap B=10 y silueta convencional, en DOS sesiones de Kaggle:
  #   SEPS=2.5 bash ejecutar_experimentos.sh kaggle_gap10   (~7.5 h)
  #   SEPS=4   bash ejecutar_experimentos.sh kaggle_gap10   (~7.5 h)
  DS=${DS:-101} GAPB=10 CRIT_C="gap ch sil sile" TAG_C=calidad_gap10 DIR_C=calidad_gap10 kaggle_calidad
}

case "${1:-piloto}" in
  piloto) piloto ;; calidad) calidad ;; escala_n) escala_n ;; memoria) memoria ;;
  particiones) particiones ;; fuerte) fuerte ;; debil) debil ;; criterios) criterios ;;
  reales) reales ;; analizar) analizar ;;
  kaggle_calidad) kaggle_calidad ;; kaggle_mllib) kaggle_mllib ;;
  kaggle_escala) kaggle_escala ;; kaggle_fuerte) kaggle_fuerte ;;
  kaggle_sensibilidad) kaggle_sensibilidad ;; kaggle_ablacion) kaggle_ablacion ;;
  kaggle_reales) kaggle_reales ;;
  kaggle_semillas) kaggle_semillas ;; kaggle_semillas2) kaggle_semillas2 ;;
  kaggle_sesionA) kaggle_sesionA ;; kaggle_sesionB) kaggle_sesionB ;; kaggle_gap10) kaggle_gap10 ;;
  mac_prueba) mac_prueba ;; mac_piloto) mac_piloto ;; mac_calidad) mac_calidad ;;
  mac_escala) mac_escala ;; mac_fuerte) mac_fuerte ;; mac_debil) mac_debil ;;
  mac_criterios) mac_criterios ;; mac_reales) mac_reales ;;
  mac_todo) mac_piloto; mac_calidad; mac_criterios; mac_escala; mac_fuerte; mac_debil; mac_reales; analizar ;;
  todo) piloto; calidad; escala_n; memoria; particiones; fuerte; debil; criterios; reales; analizar ;;
  *) echo "opción no válida"; exit 1 ;;
esac
