#!/usr/bin/env bash

LOGFILE="main_versions_run.log"

echo "=== Cluster job started at $(date) ===" >> "$LOGFILE"

run_cmd () {
    echo "Running: $*" | tee -a "$LOGFILE"
    $* >> "$LOGFILE" 2>&1
    echo "--- Completed at $(date) ---" >> "$LOGFILE"
    echo "" >> "$LOGFILE"
}

run_cmd python scripts/cluster.py --algo hdbscan --features_desc sincosphi_sincospsi_tco_hbondflags --data_version v5
run_cmd python scripts/cluster.py --algo agglomerative --features_desc sincosphi_sincospsi_tco_hbondflags --data_version v5
run_cmd python scripts/cluster.py --algo kmeans --features_desc sincosphi_sincospsi_tco_hbondflags --data_version v5

run_cmd python scripts/cluster.py --algo hdbscan --features_desc sincosphi_sincospsi_sincosalpha_hbondflags --data_version v6
run_cmd python scripts/cluster.py --algo agglomerative --features_desc sincosphi_sincospsi_sincosalpha_hbondflags --data_version v6
run_cmd python scripts/cluster.py --algo kmeans --features_desc sincosphi_sincospsi_sincosalpha_hbondflags --data_version v6

run_cmd python scripts/cluster.py --algo hdbscan --features_desc sincosphi_sincospsi_hbondflags --data_version v7
run_cmd python scripts/cluster.py --algo agglomerative --features_desc sincosphi_sincospsi_hbondflags --data_version v7
run_cmd python scripts/cluster.py --algo kmeans --features_desc sincosphi_sincospsi_hbondflags --data_version v7

echo "=== Cluster job finished at $(date) ===" >> "$LOGFILE"
