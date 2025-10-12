import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import logging

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DATA_VERSION = "v5"
FEATURES_DESC = "sincosphi_sincospsi_tco_hbondflags"
CLASS_CAP = 6_000
TRANSFORMED_PATH_X = f"/home/ubuntu/p2nd/data/output/pc20_{DATA_VERSION}/dssp_dataset_transformed_X.parquet"
TRANSFORMED_PATH_Y = f"/home/ubuntu/p2nd/data/output/pc20_{DATA_VERSION}/dssp_dataset_transformed_Y.parquet"
PLOT_XLABEL = "DSSP label"
PLOT_YLABEL = "Cluster (balanced-core + medoid assignment)"
DOWNSAMPLE = False
DOWNSAMPLE_SIZE = 100_000

CLUSTERING_ALGO = "hdbscan"  # "agglomerative" or "hdbscan"
HDBSCAN_MIN_CLUSTER_SIZE = 100
HDBSCAN_MIN_SAMPLES = None
AGGLOMERATIVE_DISTANCE_THRESHOLD = 40.0
PLOT_TITLE = f"Cluster - DSSP Overlap : {"AgglomerativeClustering" if CLUSTERING_ALGO=='agglomerative' else 'HDBSCAN'} : features={FEATURES_DESC} : data=pc20_{DATA_VERSION}"
PLOT_PATH = f"/home/ubuntu/p2nd/data/output/pc20_{DATA_VERSION}/{FEATURES_DESC}_{CLUSTERING_ALGO}{'_' + DOWNSAMPLE_SIZE if DOWNSAMPLE else ''}_pc20.png"
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

# Clustering step with selectable algorithm
if CLUSTERING_ALGO.lower() == "agglomerative":
    logger.info(f"Clustering algorithm: AgglomerativeClustering (ward, distance_threshold={AGGLOMERATIVE_DISTANCE_THRESHOLD})")
    agg = AgglomerativeClustering(
        n_clusters=None,           # let threshold decide
        distance_threshold=AGGLOMERATIVE_DISTANCE_THRESHOLD,    # tune this!
        linkage='ward'
    )
    core_labels = agg.fit_predict(Xs_core)
elif CLUSTERING_ALGO.lower() == "hdbscan":
    logger.info(f"Clustering algorithm: HDBSCAN (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={HDBSCAN_MIN_SAMPLES})")
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric='euclidean'
    )
    core_labels = hdb.fit_predict(Xs_core)  # labels: -1 for noise, 0..K-1 otherwise
else:
    raise ValueError("CLUSTERING_ALGO must be either 'agglomerative' or 'hdbscan'.")

# Map cluster labels to 0..K-1 for clean indexing
uniq = np.unique(core_labels)
label_to_compact = {lab:i for i, lab in enumerate(uniq)}
core_labels_compact = np.array([label_to_compact[lab] for lab in core_labels])
K = len(uniq)

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
ct_counts = ct_counts.loc[ct_w.index, ct_w.columns].astype(int).to_numpy()

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

plt.figure(figsize=(10, 6))
ax = sns.heatmap(ct_w, cmap="viridis", xticklabels=xlabels, yticklabels=ylabels, annot=ct_counts, fmt="d")
ax.set_title(PLOT_TITLE)
ax.set_xlabel(PLOT_XLABEL)
ax.set_ylabel(PLOT_YLABEL)
ax.tick_params(axis='x', rotation=45, labelrotation=45)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=200)
logger.info(f"Saved cluster vs DSSP overlap plot to {PLOT_PATH}")

# === Core-only plot (balanced core subset) ===
PLOT_PATH_CORE = PLOT_PATH.replace(".png", "_core.png")
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
ct_counts_core = ct_counts_core.loc[ct_w_core.index, ct_w_core.columns].astype(int).to_numpy()

# DSSP labels with totals (core only)
totals_per_dssp_core = core_df["dssp"].value_counts()
xlabels_core = [
    f"{c}:{label_map.get(c)} (n={int(totals_per_dssp_core.get(c, 0))})"
    for c in ct_w_core.columns
]

# Add core cluster sizes to Y labels
cluster_sizes_core = core_df.groupby("cluster").size()
ylabels_core = [f"Core cluster {k} (n={cluster_sizes_core[k]})" for k in ct_w_core.index]

plt.figure(figsize=(10, 6))
ax = sns.heatmap(ct_w_core, cmap="viridis", xticklabels=xlabels_core, yticklabels=ylabels_core, annot=ct_counts_core, fmt="d")
ax.set_title(PLOT_TITLE_CORE)
ax.set_xlabel(PLOT_XLABEL)
ax.set_ylabel("Cluster (balanced-core only)")
ax.tick_params(axis='x', rotation=45, labelrotation=45)
plt.tight_layout()
plt.savefig(PLOT_PATH_CORE, dpi=200)
logger.info(f"Saved CORE-ONLY cluster vs DSSP overlap plot to {PLOT_PATH_CORE}")
