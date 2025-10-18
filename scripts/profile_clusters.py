#!/usr/bin/env python3
# profile_clusters.py
"""
Profile clustering results: per-cluster stats, z-scored heatmap, violin plots.

python scripts/profile_clusters.py \
--algo hdbscan \
--features_desc sincosphi_sincospsi_tco_hbondflags \
--data_version v5 \
--max_violins 30 \
--violin_sample 100000

python scripts/profile_clusters.py \
--algo agglomerative \
--features_desc sincosphi_sincospsi_tco_hbondflags \
--data_version v5 \
--max_violins 30 \
--violin_sample 100000

Outputs saved to:
  data/output/pc20_{data_version}/{features_desc}/{algo}/profile/
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import load

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def setup_fonts():
    preferred = ["Roboto", "Roboto Regular", "Roboto Condensed", "Roboto Slab", "Roboto Mono"]
    available = {f.name for f in fm.fontManager.ttflist}
    for fam in preferred:
        if fam in available:
            mpl.rcParams["font.family"] = fam
            break
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["mathtext.fontset"] = "stixsans"


def parse_args():
    p = argparse.ArgumentParser(description="Cluster profiling: stats & violins")
    p.add_argument("--data_version", required=True, help="e.g., v5")
    p.add_argument("--features_desc", required=True, help="feature descriptor used in clustering")
    p.add_argument("--algo", required=True, choices=["agglomerative", "hdbscan"], help="clustering algo used")
    p.add_argument("--drop_noise", action="store_true",
                   help="drop cluster -1 (noise) from profiling if present")
    p.add_argument("--max_violins", type=int, default=30,
                   help="max number of features to include in violins.pdf")
    p.add_argument("--violin_sample", type=int, default=200_000,
                   help="optional row downsample for violins for speed")
    p.add_argument("--features_limit", type=int, default=None,
                   help="optional cap on number of features (first N columns)")
    p.add_argument("--title_suffix", default="",
                   help="optional text appended to plot titles")
    return p.parse_args()


def to_py(o):
    # helper to serialize numpy types to JSON
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return o


def main():
    args = parse_args()
    setup_fonts()

    # Paths
    base_dir = f"/home/ubuntu/p2nd/data/output/pc20_{args.data_version}"
    algo_dir = os.path.join(base_dir, args.features_desc, args.algo)
    profile_dir = os.path.join(algo_dir, "profile")
    os.makedirs(profile_dir, exist_ok=True)

    path_X = os.path.join(base_dir, "dssp_dataset_transformed_X.parquet")
    path_clusters = os.path.join(algo_dir, "clusters.parquet")
    path_scaler = os.path.join(algo_dir, "scaler.joblib")
    path_meta = os.path.join(algo_dir, "metadata.json")

    # Load data
    Xdf = pd.read_parquet(path_X)  # keep columns for feature names
    feature_names = Xdf.columns.tolist()
    if args.features_limit is not None:
        feature_names = feature_names[: args.features_limit]
        Xdf = Xdf[feature_names]

    clusters = pd.read_parquet(path_clusters)["cluster"].to_numpy()
    scaler = load(path_scaler)

    # Align lengths defensively (should normally match)
    n = min(len(Xdf), len(clusters))
    if n < len(Xdf):
        Xdf = Xdf.iloc[:n].copy()
    if n < len(clusters):
        clusters = clusters[:n]

    # Transform to scaled space (Xs_all)
    Xs_all = scaler.transform(Xdf.to_numpy())

    # Build profiling dataframe
    df_features = pd.DataFrame(Xs_all, columns=feature_names)
    df_features["cluster"] = clusters

    # Optionally drop noise
    if args.drop_noise and (-1 in df_features["cluster"].unique()):
        df_features = df_features[df_features["cluster"] != -1].copy()

    # 1) Basic descriptive statistics per cluster
    # mean, std, min, max computed per cluster per feature
    cluster_summary = df_features.groupby("cluster")[feature_names].agg(["mean", "std", "min", "max"])
    cluster_summary.to_csv(os.path.join(profile_dir, "cluster_summary.csv"))

    # z-score normalize the cluster mean profile across clusters for each feature
    means_only = cluster_summary.xs("mean", axis=1, level=1)
    cluster_z = means_only.subtract(means_only.mean(axis=0), axis=1).divide(means_only.std(axis=0).replace(0, np.nan), axis=1)
    cluster_z = cluster_z.fillna(0.0)
    cluster_z.to_csv(os.path.join(profile_dir, "cluster_z.csv"))

    # Heatmap of z-scored means (center=0 shows extremes)
    plt.figure(figsize=(max(8, len(feature_names) * 0.25), max(4, cluster_z.shape[0] * 0.4)))
    ax = sns.heatmap(cluster_z, cmap="coolwarm", center=0, cbar=True)
    algo_name = "AgglomerativeClustering" if args.algo == "agglomerative" else "HDBSCAN"
    title = f"Cluster mean (z-scored across clusters) — {algo_name} — features={args.features_desc} — data=pc20_{args.data_version}"
    if args.title_suffix:
        title += f" — {args.title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("Feature (scaled space)")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    heatmap_path = os.path.join(profile_dir, "cluster_z_heatmap.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close()

    # 2) Pairwise feature comparisons (violin plots)
    # Downsample rows for performance if needed
    df_v = df_features
    if (args.violin_sample is not None) and (len(df_features) > args.violin_sample):
        df_v = df_features.sample(n=args.violin_sample, random_state=42)

    # Limit number of features for violins
    violin_features = feature_names[: args.max_violins] if args.max_violins else feature_names

    violins_pdf_path = os.path.join(profile_dir, "violins.pdf")
    with PdfPages(violins_pdf_path) as pdf:
        for feat in violin_features:
            plt.figure(figsize=(9, 4.5))
            sns.violinplot(data=df_v, x="cluster", y=feat, inner="quartile", cut=0, scale="area")
            plt.title(f"Feature vs Cluster — {feat}")
            plt.xlabel("Cluster")
            plt.ylabel(f"{feat} (scaled)")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    # Save a tiny profiling metadata for reproducibility
    profile_meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data_version": args.data_version,
        "features_desc": args.features_desc,
        "algo": args.algo,
        "n_rows_profiled": int(len(df_features)),
        "n_features": int(len(feature_names)),
        "dropped_noise": bool(args.drop_noise),
        "artifacts": {
            "cluster_summary_csv": os.path.join(profile_dir, "cluster_summary.csv"),
            "cluster_z_csv": os.path.join(profile_dir, "cluster_z.csv"),
            "cluster_z_heatmap_png": heatmap_path,
            "violins_pdf": violins_pdf_path,
        }
    }
    with open(os.path.join(profile_dir, "profile_metadata.json"), "w") as f:
        json.dump(profile_meta, f, indent=2, default=to_py)

    print(f"[OK] Profiling complete. Outputs in: {profile_dir}")


if __name__ == "__main__":
    main()
