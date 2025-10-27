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
  # HDBSCAN sweeps (safe)
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

        # Temporarily disable -e so a failure here doesn’t kill the whole script
        set +e
        python3 -u scripts/cluster.py "${HDB_ARGS[@]}"
        status=$?
        set -e

        if [ "$status" -ne 0 ]; then
          echo "!!! FAILED (exit=$status) for HDBSCAN MCS=$MCS MS=$MS CSM=$CSM at $(date)"
          continue
        else
          echo "--- OK for HDBSCAN MCS=$MCS MS=$MS CSM=$CSM at $(date)"
        fi
      done
    done
  done

  # -------------------------
  # Agglomerative sweeps
  # -------------------------
  for LINK in average single; do
    for THR in 30 40 50 60 80; do
      echo ""
      echo "*** AGGLO: LINK=$LINK  THR=$THR"

      # Temporarily disable -e so a failure here doesn’t kill the whole script
      set +e
      python3 -u scripts/cluster.py \
        --algo agglomerative \
        --features_desc "$FEATURES" \
        --data_version "$DATA_VER" \
        --agg_linkage "$LINK" \
        --agg_distance_threshold "$THR"
      status=$?
      set -e

      if [ "$status" -ne 0 ]; then
        echo "!!! FAILED (exit=$status) for LINK=$LINK THR=$THR at $(date)"
        # continue to next combo
        continue
      else
        echo "--- OK for LINK=$LINK THR=$THR at $(date)"
      fi
    done
  done

  echo "=== END $(date) ==="
} 2>&1 | tee "$LOG"
' >/dev/null 2>&1 & echo $! > cluster_jobs.pid
