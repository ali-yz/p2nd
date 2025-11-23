#!/usr/bin/env bash

LOGFILE="profile_clusters_main_versions_data_fixed_dropped_v2.log"

echo "=== Profile job started at $(date) ===" >> "$LOGFILE"

run_cmd () {
    echo "Running: $*" | tee -a "$LOGFILE"
    $* >> "$LOGFILE" 2>&1
    echo "--- Completed at $(date) ---" >> "$LOGFILE"
    echo "" >> "$LOGFILE"
}

# Versions and corresponding feature descriptions
declare -A descs
descs[v3.1]="sincosphi_sincospsi"
descs[v4.1]="sincosphi_sincospsi_tco"
descs[v5]="sincosphi_sincospsi_tco_hbondflags"
descs[v6]="sincosphi_sincospsi_sincosalpha_hbondflags"
descs[v7]="sincosphi_sincospsi_hbondflags"

algos=("hdbscan" "agglomerative" "kmeans")

for version in v3.1 v4.1 v5 v6 v7; do
    desc="${descs[$version]}"
    for algo in "${algos[@]}"; do
        run_cmd python scripts/profile_clusters_basic.py --data-version "$version" --desc "$desc" --algo "$algo" --only-combined-output
    done
done

echo "=== Profile job finished at $(date) ===" >> "$LOGFILE"
