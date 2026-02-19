#!/usr/bin/env python3
# cluster_dendrogram.py

"""
Agglomerative clustering with dendrogram + clustermap visualizations.

python scripts/cluster_dendrogram.py \
    --features_desc sincosphi_sincospsi_tco_hbondflags \
    --data_version v5

python scripts/cluster_dendrogram.py \
   --features_desc sincosphi_sincospsi_sincosalpha_hbondflags \
   --data_version v6

python scripts/cluster_dendrogram.py \
    --features_desc sincosphi_sincospsi_hbondflags \
    --data_version v7
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, adjusted_mutual_info_score, silhouette_score, homogeneity_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import logging

import argparse, os, json
from datetime import datetime

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# argparse — agglomerative only
parser = argparse.ArgumentParser(description="Agglomerative clustering with dendrogram visualizations.")
parser.add_argument("--features_desc", required=True,
                    help="Short descriptor for features used (goes into output paths).")
parser.add_argument("--data_version", required=True,
                    help="Data version tag, e.g. v5.")
parser.add_argument("--downsample", action="store_true",
                    help="Optional: enable downsampling for quick checks.")
parser.add_argument("--downsample_size", type=int, default=100_000,
                    help="Downsample size if --downsample is set.")
parser.add_argument("--agg_distance_threshold", type=float, default=50.0,
                    help="Agglomerative distance_threshold.")
parser.add_argument("--agg_linkage", type=str, default="ward",
                    help="Agglomerative linkage method. ward, complete, average, single. default=ward.")
args = parser.parse_args()

DATA_VERSION = args.data_version
FEATURES_DESC = args.features_desc
DOWNSAMPLE = args.downsample
DOWNSAMPLE_SIZE = args.downsample_size
AGGLOMERATIVE_DISTANCE_THRESHOLD = args.agg_distance_threshold
AGGLOMERATIVE_LINKAGE = args.agg_linkage
CLASS_CAP = 6_000

# derive IO paths — hardcoded agglomerative subdir
BASE_DIR = f"/home/ubuntu/p2nd/data/output/pc20_{DATA_VERSION}"
ALGO_DIR = os.path.join(BASE_DIR, FEATURES_DESC, "agglomerative")
os.makedirs(ALGO_DIR, exist_ok=True)

TRANSFORMED_PATH_X = f"/home/ubuntu/p2nd/data/output/pc20_{DATA_VERSION}/dssp_dataset_transformed_X.parquet"
TRANSFORMED_PATH_Y = f"/home/ubuntu/p2nd/data/output/pc20_{DATA_VERSION}/dssp_dataset_transformed_Y.parquet"

PLOT_TITLE = f"Cluster - DSSP Overlap : AgglomerativeClustering : features={FEATURES_DESC} : data=pc20_{DATA_VERSION}"
PLOT_PATH = os.path.join(
    ALGO_DIR,
    f"{FEATURES_DESC}_agglomerative{'_' + str(DOWNSAMPLE_SIZE) if DOWNSAMPLE else ''}_pc20.png"
)
PLOT_PATH_DENDROGRAM = os.path.join(
    ALGO_DIR,
    f"{FEATURES_DESC}_agglomerative{'_' + str(DOWNSAMPLE_SIZE) if DOWNSAMPLE else ''}_dendrogram.png"
)

PLOT_XLABEL = "DSSP label"
PLOT_YLABEL = "Cluster (balanced-core + medoid assignment)"
PLOT_COL_ORDER = ["C", "B", "E", "G", "H", "I", "P", "S", "T"]  # desired order of DSSP columns in the heatmap

## SET FONTS
preferred = ["Roboto", "Roboto Regular", "Roboto Condensed", "Roboto Slab", "Roboto Mono"]
available = {f.name for f in fm.fontManager.ttflist}
for fam in preferred:
    if fam in available:
        mpl.rcParams["font.family"] = fam
        break
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["mathtext.fontset"] = "stixsans"  # STIX sans pairs reasonably with Roboto

features = pd.read_parquet(TRANSFORMED_PATH_X).to_numpy()
labels = pd.read_parquet(TRANSFORMED_PATH_Y)["DSSP_label"].tolist()

rng = np.random.default_rng(42)

X = features
y = np.array(labels)
N = X.shape[0]
logger.info(f"Clustering: N={N:,}, features={X.shape[1]}")

# sub sample to make it faster for a quick check
if N > DOWNSAMPLE_SIZE and DOWNSAMPLE:
    sel = rng.choice(N, size=DOWNSAMPLE_SIZE, replace=False)
    X = X[sel]
    y = y[sel]
    N = X.shape[0]
    logger.info(f"Clustering: downsampled to N={N:,}, features={X.shape[1]}")

classes, counts = np.unique(y, return_counts=True)
logger.info(f"Clustering: class distribution: {dict(zip(classes, counts))}")
count_map = dict(zip(classes, counts))

core_idx = []
for cls in classes:
    idx_cls = np.where(y == cls)[0]
    cap     = CLASS_CAP
    k       = min(len(idx_cls), cap)
    if len(idx_cls) > k:
        sel = rng.choice(idx_cls, size=k, replace=False)
    else:
        sel = idx_cls
    core_idx.append(sel)

core_idx = np.concatenate(core_idx)
rest_idx = np.setdiff1d(np.arange(N), core_idx, assume_unique=False)

logger.info(f"Clustering: balanced core size={len(core_idx):,}, rest size={len(rest_idx):,}")
logger.info(f"Clustering: balanced core class distribution: {dict(zip(*np.unique(y[core_idx], return_counts=True)))}")

# Scale using ONLY the balanced core
scaler   = StandardScaler().fit(X[core_idx])
Xs_core  = scaler.transform(X[core_idx])
Xs_rest  = scaler.transform(X[rest_idx])
Xs_all   = scaler.transform(X)  # for later use

# Clustering: scipy linkage + fcluster
logger.info(f"Clustering algorithm: AgglomerativeClustering (linkage={AGGLOMERATIVE_LINKAGE}, distance_threshold={AGGLOMERATIVE_DISTANCE_THRESHOLD})")
Z_core = linkage(Xs_core, method=AGGLOMERATIVE_LINKAGE, metric='euclidean')
core_labels = fcluster(Z_core, t=AGGLOMERATIVE_DISTANCE_THRESHOLD, criterion='distance')
core_labels = core_labels - 1  # fcluster is 1-based, shift to 0-based

# Map cluster labels to 0..K-1 for clean indexing
uniq = np.unique(core_labels)
label_to_compact = {lab:i for i, lab in enumerate(uniq)}
core_labels_compact = np.array([label_to_compact[lab] for lab in core_labels])
K = len(uniq)
logger.info(f"Found {K} clusters in the core dataset.")

# Compute External metric on the core (AMI)
try:
    logger.info("Computing AMI_core...")
    AMI_core = float(adjusted_mutual_info_score(y[core_idx], core_labels_compact))
except Exception as e:
    logger.warning(f"AMI_core computation failed: {e}")
    AMI_core = None

# Compute External metric on the core (Homogeneity)
try:
    logger.info("Computing Homogeneity_core...")
    Homogeneity_core = float(homogeneity_score(y[core_idx], core_labels_compact))
except Exception as e:
    logger.warning(f"Homogeneity_core computation failed: {e}")
    Homogeneity_core = None

# Compute Internal metric on the core (Silhouette)
try:
    logger.info("Computing Silhouette_core (sampled)...")
    SIL_SAMPLE_SIZE = min(27000, Xs_core.shape[0])
    logger.info(f"Silhouette_core (sampled) N = {SIL_SAMPLE_SIZE}")
    Silhouette_core = float(silhouette_score(
        Xs_core, core_labels_compact,
        metric='euclidean',
        sample_size=SIL_SAMPLE_SIZE,
        random_state=42
    ))
except Exception as e:
    logger.warning(f"Silhouette_core computation failed: {e}")
    Silhouette_core = None

def cluster_medoid(Xs, idxs):
    # compute distances within cluster (avoid full NxN by slicing per-cluster)
    Xi  = Xs[idxs]
    D   = pairwise_distances(Xi, Xi, metric='euclidean')
    med = idxs[np.argmin(D.sum(axis=1))]
    return med

core_indices_by_k = [np.where(core_labels_compact == k)[0] for k in range(K)]
medoid_core_local = [cluster_medoid(Xs_core, idxs) for idxs in core_indices_by_k]
medoid_global_idx = [core_idx[i] for i in medoid_core_local]
medoids_Xs        = Xs_all[medoid_global_idx]

# Per-cluster assignment radius
radii = []
for k in range(K):
    members_global = core_idx[core_indices_by_k[k]]
    Dk = pairwise_distances(Xs_all[members_global], medoids_Xs[k].reshape(1, -1))
    radii.append(np.percentile(Dk, 95))  # tweak percentile

# Assign REST
D_rest = pairwise_distances(Xs_rest, medoids_Xs)
nearest_k = D_rest.argmin(axis=1)
nearest_d = D_rest.min(axis=1)

assigned_rest = np.full(len(rest_idx), -1)
for i in range(len(rest_idx)):
    k = nearest_k[i]
    if nearest_d[i] <= radii[k]:
        assigned_rest[i] = k
    else:
        assigned_rest[i] = -1  # noise/unassigned

# Stitch full labels
full_labels = np.full(N, -1)
full_labels[core_idx] = core_labels_compact
full_labels[rest_idx] = assigned_rest

# inverse-frequency weights per DSSP class
w_map = {c: 1.0/count_map[c] for c in classes}
w_map_core = {c: 1.0/CLASS_CAP for c in classes}  # balanced core has uniform class counts
# weights for full set
w = np.array([w_map[lab] for lab in y])
w_core = np.array([w_map_core[lab] for lab in y])

df = pd.DataFrame({"cluster": full_labels, "dssp": y, "w": w})

# weighted crosstab (row-normalized)
ct_w = df.pivot_table(index="cluster", columns="dssp", values="w", aggfunc="sum", fill_value=0.0)
ct_w = ct_w.div(ct_w.sum(axis=1), axis=0)

# sort rows by their dominant DSSP column for readability
ct_w = ct_w.reindex(ct_w.idxmax(axis=1).sort_values().index)

desired_cols = [c for c in PLOT_COL_ORDER if c in ct_w.columns]
ct_w = ct_w.reindex(columns=desired_cols)

# absolute counts for annotations (full set)
ct_counts = df.pivot_table(index="cluster", columns="dssp", values="w", aggfunc="count", fill_value=0)
ct_counts = ct_counts.loc[ct_w.index, ct_w.columns].astype(int)

label_map = {
    "B": "β-bridge",
    "C": "coil/other",
    "E": "β-strand",
    "G": "3₁₀ helix",
    "H": "α-helix",
    "I": "π-helix",
    "P": "PPII helix",
    "S": "bend",
    "T": "turn",
}

# DSSP labels with totals (full set)
totals_per_dssp_full = df["dssp"].value_counts()
xlabels = [
    f"{c}:{label_map.get(c)} (n={int(totals_per_dssp_full.get(c, 0))})"
    for c in ct_w.columns
]

# Add cluster sizes to ylabels
cluster_sizes = df.groupby("cluster").size()
ylabels = [f"Cluster {k} (n={cluster_sizes[k]})" for k in ct_w.index]

# ============================================================
# STANDALONE DENDROGRAM PLOT
# ============================================================
logger.info("Plotting standalone dendrogram...")
fig_dend, ax_dend = plt.subplots(figsize=(12, 6))
dendrogram(
    Z_core,
    truncate_mode='lastp',
    p=K,
    color_threshold=AGGLOMERATIVE_DISTANCE_THRESHOLD,
    show_contracted=True,
    ax=ax_dend,
)
ax_dend.axhline(y=AGGLOMERATIVE_DISTANCE_THRESHOLD, color='red', linestyle='--', linewidth=1,
                label=f'distance threshold = {AGGLOMERATIVE_DISTANCE_THRESHOLD}')
ax_dend.legend(loc='best')
ax_dend.set_title(f"Dendrogram (truncated to {K} leaves) : {FEATURES_DESC} : pc20_{DATA_VERSION}")
ax_dend.set_xlabel("Cluster (sample count)")
ax_dend.set_ylabel("Distance")
fig_dend.tight_layout()
fig_dend.savefig(PLOT_PATH_DENDROGRAM, dpi=200)
plt.close(fig_dend)
logger.info(f"Saved dendrogram to {PLOT_PATH_DENDROGRAM}")

# ============================================================
# FULL-SET CLUSTERMAP (heatmap + row dendrogram)
# ============================================================
logger.info("Plotting full-set clustermap...")

# Drop noise cluster (-1) if present — it would cause a dimension mismatch with the linkage matrix
ct_w_hm = ct_w.drop(index=-1, errors='ignore')
ct_counts_hm = ct_counts.drop(index=-1, errors='ignore')
ylabels_hm = [f"Cluster {k} (n={cluster_sizes[k]})" for k in ct_w_hm.index]

# Compute cluster centroids in feature space for the row linkage
centroids = np.array([Xs_core[core_labels_compact == k].mean(axis=0) for k in ct_w_hm.index])
Z_clusters = linkage(centroids, method='ward')

# Set index to ylabels so clustermap preserves label alignment
ct_w_hm.index = ylabels_hm
ct_counts_hm.index = ylabels_hm
ct_annot = ct_counts_hm.copy()  # DataFrame so annotations reorder with rows

g = sns.clustermap(
    ct_w_hm,
    row_linkage=Z_clusters,
    col_cluster=False,
    cmap="viridis",
    annot=ct_annot,
    fmt="d",
    xticklabels=xlabels,
    figsize=(12, 8),
    dendrogram_ratio=(0.15, 0.0),
    cbar_pos=None,
)
g.ax_heatmap.set_xlabel(PLOT_XLABEL)
g.ax_heatmap.set_ylabel(PLOT_YLABEL)
g.ax_heatmap.tick_params(axis='x', rotation=45)
g.fig.suptitle(PLOT_TITLE, y=1.02, fontsize=11)
g.fig.savefig(PLOT_PATH, dpi=200, bbox_inches='tight')
plt.close(g.fig)
logger.info(f"Saved cluster vs DSSP clustermap to {PLOT_PATH}")

# ============================================================
# CORE-ONLY CLUSTERMAP
# ============================================================
PLOT_PATH_CORE = os.path.join(
    ALGO_DIR,
    f"{FEATURES_DESC}_agglomerative{'_' + str(DOWNSAMPLE_SIZE) if DOWNSAMPLE else ''}_pc20_core.png"
)
PLOT_TITLE_CORE = PLOT_TITLE + " — CORE ONLY"

# Build a core-only dataframe using the same inverse-frequency weights (w) computed on the full set
core_df = pd.DataFrame({
    "cluster": core_labels_compact,     # clusters are defined from the core
    "dssp":    y[core_idx],
    "w":       w_core[core_idx]
})

# Weighted crosstab (row-normalized) for core only
ct_w_core = core_df.pivot_table(
    index="cluster",
    columns="dssp",
    values="w",
    aggfunc="sum",
    fill_value=0.0
)
ct_w_core = ct_w_core.div(ct_w_core.sum(axis=1), axis=0)

# Sort rows by dominant DSSP column for readability
ct_w_core = ct_w_core.reindex(ct_w_core.idxmax(axis=1).sort_values().index)

desired_cols = [c for c in PLOT_COL_ORDER if c in ct_w_core.columns]
ct_w_core = ct_w_core.reindex(columns=desired_cols)

# absolute counts for annotations (core only)
ct_counts_core = core_df.pivot_table(index="cluster", columns="dssp", values="w", aggfunc="count", fill_value=0)
ct_counts_core = ct_counts_core.loc[ct_w_core.index, ct_w_core.columns].astype(int)

# DSSP labels with totals (core only)
totals_per_dssp_core = core_df["dssp"].value_counts()
xlabels_core = [
    f"{c}:{label_map.get(c)} (n={int(totals_per_dssp_core.get(c, 0))})"
    for c in ct_w_core.columns
]

# Add core cluster sizes to Y labels
cluster_sizes_core = core_df.groupby("cluster").size()
ylabels_core = [f"Core cluster {k} (n={cluster_sizes_core[k]})" for k in ct_w_core.index]

logger.info("Plotting core-only clustermap...")

# Drop noise cluster (-1) if present
ct_w_core_hm = ct_w_core.drop(index=-1, errors='ignore')
ct_counts_core_hm = ct_counts_core.drop(index=-1, errors='ignore')
ylabels_core_hm = [f"Core cluster {k} (n={cluster_sizes_core[k]})" for k in ct_w_core_hm.index]

# Centroids for core-only linkage
centroids_core = np.array([Xs_core[core_labels_compact == k].mean(axis=0) for k in ct_w_core_hm.index])
Z_clusters_core = linkage(centroids_core, method='ward')

ct_w_core_hm.index = ylabels_core_hm
ct_counts_core_hm.index = ylabels_core_hm
ct_annot_core = ct_counts_core_hm.copy()

g_core = sns.clustermap(
    ct_w_core_hm,
    row_linkage=Z_clusters_core,
    col_cluster=False,
    cmap="viridis",
    annot=ct_annot_core,
    fmt="d",
    xticklabels=xlabels_core,
    figsize=(12, 8),
    dendrogram_ratio=(0.15, 0.0),
    cbar_pos=None,
)
g_core.ax_heatmap.set_xlabel(PLOT_XLABEL)
g_core.ax_heatmap.set_ylabel("Cluster (balanced-core only)")
g_core.ax_heatmap.tick_params(axis='x', rotation=45)
g_core.fig.suptitle(PLOT_TITLE_CORE, y=1.02, fontsize=11)
g_core.fig.savefig(PLOT_PATH_CORE, dpi=200, bbox_inches='tight')
plt.close(g_core.fig)
logger.info(f"Saved CORE-ONLY clustermap to {PLOT_PATH_CORE}")

# Persist minimal artifacts for downstream profiling
# 1) Cluster labels aligned to original row order in this run
clusters_path = os.path.join(ALGO_DIR, "clusters.parquet")
pd.DataFrame({"cluster": full_labels}).to_parquet(clusters_path, index=False)

# 2) Scaler so profiling can reproduce the exact transform
scaler_path = os.path.join(ALGO_DIR, "scaler.joblib")
# dump(scaler, scaler_path)

logger.info(f"Scores — AMI_core={AMI_core}, Homogeneity_core={Homogeneity_core}, Silhouette_core={Silhouette_core}: ARGS: {args}")

# 3) Metadata for reproducibility
meta = {
    "timestamp": datetime.now().isoformat(),
    "data_version": DATA_VERSION,
    "features_desc": FEATURES_DESC,
    "algo": "agglomerative",
    "params": {
        "agglomerative": {
            "distance_threshold": AGGLOMERATIVE_DISTANCE_THRESHOLD,
            "linkage": AGGLOMERATIVE_LINKAGE
        },
        "class_cap": CLASS_CAP,
        "downsample": DOWNSAMPLE,
        "downsample_size": DOWNSAMPLE_SIZE
    },
    "shapes": {
        "N": int(N),
        "n_features": int(X.shape[1]),
        "n_clusters_in_core": int(K)
    },
    "scores": {
        "AMI_core": AMI_core,
        "Homogeneity_core": Homogeneity_core,
        "Silhouette_core": Silhouette_core,
    },
    "class_counts_core": dict(zip(*np.unique(y[core_idx], return_counts=True))),
    "class_counts_full": dict(zip(*np.unique(y, return_counts=True))),
    "plots": {
        "full": PLOT_PATH,
        "core": PLOT_PATH_CORE,
        "dendrogram": PLOT_PATH_DENDROGRAM
    },
    "artifacts": {
        "clusters_parquet": clusters_path,
        "scaler_joblib": scaler_path
    }
}

def _to_py(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return o

with open(os.path.join(ALGO_DIR, f"metadata_{datetime.now().isoformat()}.json"), "w") as f:
    json.dump(meta, f, indent=2, default=_to_py)

logger.info(f"Saved clustering artifacts to {ALGO_DIR}")
