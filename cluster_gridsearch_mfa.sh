nohup bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

FEATURES="sincosphi_sincospsi_tco_hbondflags"
DATA_VER="v5"
LOG="mfa_grid_$(date +%Y%m%d_%H%M%S).log"

# Search ranges
Q_LIST=(3 5 8 12)
THR_LIST=(40 50 60 80)
TOL_LIST=(1e-2 1e-3 1e-4)

{
  echo "=== START $(date) ==="
  echo "FEATURES=$FEATURES  DATA_VER=$DATA_VER"
  echo "Q_LIST=${Q_LIST[*]}  THR_LIST=${THR_LIST[*]}  TOL_LIST=${TOL_LIST[*]}"

  for Q in "${Q_LIST[@]}"; do
    for THR in "${THR_LIST[@]}"; do
      for TOL in "${TOL_LIST[@]}"; do
        echo ""
        echo "*** MFA: q=$Q  thr=$THR  tol=$TOL"

        # Temporarily disable -e so a failure here doesn’t kill the whole sweep
        set +e
        python3 -u scripts/cluster_subspace.py \
          --algo mfa \
          --features_desc "$FEATURES" \
          --data_version "$DATA_VER" \
          --mfa_q "$Q" \
          --agg_distance_threshold "$THR" \
          --fa_tol "$TOL"
        status=$?
        set -e

        if [ "$status" -ne 0 ]; then
          echo "!!! FAILED (exit=$status) for q=$Q thr=$THR tol=$TOL at $(date)"
          continue
        else
          echo "--- OK for q=$Q thr=$THR tol=$TOL at $(date)"
        fi
      done
    done
  done

  echo "=== END $(date) ==="
} 2>&1 | tee "$LOG"
' >/dev/null 2>&1 & echo $! > mfa_grid_jobs.pid
