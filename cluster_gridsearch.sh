nohup bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

FEATURES="sincosphi_sincospsi_tco_hbondflags"
DATA_VER="v5"
LOG="cluster_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== START $(date) ==="
  echo "FEATURES=$FEATURES  DATA_VER=$DATA_VER"

  # -------------------------
  # HDBSCAN sweeps
  # -------------------------
  # Try a range of cluster scales + optional min_samples + both selection methods
  for MCS in 25 50 100 200 400; do
    for MS in none 5 10 25 50 100; do
      for CSM in eom; do
        HDB_ARGS=(--algo hdbscan
                  --features_desc "$FEATURES"
                  --data_version "$DATA_VER"
                  --hdb_min_cluster_size "$MCS"
                  --hdb_cluster_selection_method "$CSM")
        if [[ "$MS" != "none" ]]; then
          HDB_ARGS+=(--hdb_min_samples "$MS")
        fi
        echo ""
        echo "*** HDBSCAN: MCS=$MCS  MS=$MS  CSM=$CSM"
        python3 -u scripts/cluster.py "${HDB_ARGS[@]}"
      done
    done
  done

  # -------------------------
  # Agglomerative sweeps
  # -------------------------
  # Sweep common linkages and a wide range of distance thresholds
  for LINK in ward complete average single; do
    for THR in 10 15 20 25 30 35 40 50 60 80 100; do
      echo ""
      echo "*** AGGLO: LINK=$LINK  THR=$THR"
      python3 -u scripts/cluster.py \
        --algo agglomerative \
        --features_desc "$FEATURES" \
        --data_version "$DATA_VER" \
        --agg_linkage "$LINK" \
        --agg_distance_threshold "$THR"
    done
  done

  echo "=== END $(date) ==="
} 2>&1 | tee "$LOG"
' >/dev/null 2>&1 & echo $! > cluster_jobs.pid
