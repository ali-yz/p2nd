#!/usr/bin/env python3
# cluster.py

"""
Cluster DSSP features using specified algorithm and persist results.
python scripts/cluster.py \
   --algo hdbscan \
   --features_desc sincosphi_sincospsi_tco_hbondflags \
   --data_version v5

python scripts/cluster.py \
    --algo agglomerative \
    --features_desc sincosphi_sincospsi_tco_hbondflags \
    --data_version v5

python scripts/cluster.py \
    --algo mfa \
    --features_desc sincosphi_sincospsi_tco_hbondflags \
    --data_version v5

python scripts/cluster.py \
    --algo proclus \
    --features_desc sincosphi_sincospsi_tco_hbondflags \
    --data_version v5 \
    --proclus_k 30 --proclus_l 0 --proclus_iters 5
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import FactorAnalysis  # <<< ADDED
from sklearn.metrics import adjusted_mutual_info_score  # <<< ADDED (metrics)
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import logging

import argparse, os, json
from datetime import datetime
from joblib import dump

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# >>> CHANGED: argparse to receive algo, feature desc, and data version
parser = argparse.ArgumentParser(description="Cluster DSSP features and persist results.")
parser.add_argument("--algo", choices=["agglomerative", "hdbscan", "mfa", "proclus"], required=True,  # <<< CHANGED
                    help="Clustering algorithm to use.")
parser.add_argument("--features_desc", required=True,
                    help="Short descriptor for features used (goes into output paths).")
parser.add_argument("--data_version", required=True,
                    help="Data version tag, e.g. v5.")
parser.add_argument("--downsample", action="store_true",
                    help="Optional: enable downsampling for quick checks.")
parser.add_argument("--downsample_size", type=int, default=100_000,
                    help="Downsample size if --downsample is set.")
parser.add_argument("--hdb_min_cluster_size", type=int, default=100,
                    help="HDBSCAN min_cluster_size.")
parser.add_argument("--hdb_min_samples", type=int, default=None,
                    help="HDBSCAN min_samples (None defaults to min_cluster_size).")
parser.add_argument("--agg_distance_threshold", type=float, default=40.0,
                    help="Agglomerative distance_threshold.")
parser.add_argument("--mfa_q", type=int, default=8,                    # <<< ADDED
                    help="MFA latent dimensionality per component (q).")  # <<< ADDED
parser.add_argument("--proclus_k", type=int, default=30,               # <<< ADDED
                    help="PROCLUS number of clusters (medoids).")      # <<< ADDED
parser.add_argument("--proclus_l", type=int, default=0,                # <<< ADDED
                    help="PROCLUS average dimensions per cluster; 0 = auto (ceil(0.1*d)).")  # <<< ADDED
parser.add_argument("--proclus_iters", type=int, default=5,            # <<< ADDED
                    help="PROCLUS refinement iterations.")             # <<< ADDED
args = parser.parse_args()

DATA_VERSION = args.data_version
FEATURES_DESC = args.features_desc
CLUSTERING_ALGO = args.algo
DOWNSAMPLE = args.downsample
DOWNSAMPLE_SIZE = args.downsample_size
HDBSCAN_MIN_CLUSTER_SIZE = args.hdb_min_cluster_size
HDBSCAN_MIN_SAMPLES = args.hdb_min_samples
AGGLOMERATIVE_DISTANCE_THRESHOLD = args.agg_distance_threshold
MFA_Q = args.mfa_q  # <<< ADDED
PROCLUS_K = args.proclus_k           # <<< ADDED
PROCLUS_L = args.proclus_l           # <<< ADDED
PROCLUS_ITERS = args.proclus_iters   # <<< ADDED
CLASS_CAP = 6_000

# derive IO paths and per-algo subdir
BASE_DIR = f"/home/ubuntu/p2nd/data/output/pc20_{DATA_VERSION}"
ALGO_DIR = os.path.join(BASE_DIR, FEATURES_DESC, CLUSTERING_ALGO)
os.makedirs(ALGO_DIR, exist_ok=True)

TRANSFORMED_PATH_X = f"/home/ubuntu/p2nd/data/output/pc20_{DATA_VERSION}/dssp_dataset_transformed_X.parquet"
TRANSFORMED_PATH_Y = f"/home/ubuntu/p2nd/data/output/pc20_{DATA_VERSION}/dssp_dataset_transformed_Y.parquet"

# <<< CHANGED: include MFA/PROCLUS in title mapping
algo_name_for_title = (
    "AgglomerativeClustering" if CLUSTERING_ALGO == "agglomerative"
    else ("HDBSCAN" if CLUSTERING_ALGO == "hdbscan"
          else ("Mixture of Factor Analyzers" if CLUSTERING_ALGO == "mfa"
                else "PROCLUS (Projected k-medoids)"))
)
PLOT_TITLE = f"Cluster - DSSP Overlap : {algo_name_for_title} : features={FEATURES_DESC} : data=pc20_{DATA_VERSION}"
PLOT_PATH = os.path.join(
    ALGO_DIR,
    f"{FEATURES_DESC}_{CLUSTERING_ALGO}{'_' + str(DOWNSAMPLE_SIZE) if DOWNSAMPLE else ''}_pc20.png"
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

# ----------------------------------------
# Helper for MFA log-likelihood (Woodbury)
# ----------------------------------------  # <<< ADDED
def mfa_logpdf_batch(Xbatch, mu, L, psi):
    """
    Xbatch: (n, d)
    mu: (d,)
    L: (d, q)
    psi: (d,)  diagonal noise (positive)
    returns: (n,) log N(x | mu, L L^T + diag(psi))
    """
    Xc = Xbatch - mu
    d = Xc.shape[1]
    invPsi = 1.0 / np.clip(psi, 1e-7, None)  # numerical safety
    Lt_invPsi = (L.T * invPsi)          # (q, d)
    A = np.eye(L.shape[1]) + Lt_invPsi @ L  # (q, q)
    signA, logdetA = np.linalg.slogdet(A)
    if signA <= 0:
        logdetA = np.log(np.linalg.det(A + 1e-6*np.eye(A.shape[0])))
    logdetSigma = np.sum(np.log(np.clip(psi, 1e-12, None))) + logdetA
    invPsi_Xc = Xc * invPsi
    Lt_invPsi_Xc_T = Lt_invPsi @ Xc.T            # (q, n)
    sol = np.linalg.solve(A, Lt_invPsi_Xc_T)     # (q, n)
    quad = np.einsum("ij,ij->i", Xc, invPsi_Xc) - np.einsum("ij,ij->j", Lt_invPsi_Xc_T, sol)
    return -0.5 * (d * np.log(2.0*np.pi) + logdetSigma + quad)

# ----------------------------------------
# Minimal PROCLUS implementation (projected k-medoids)
# ----------------------------------------  # <<< ADDED
def _manhattan_dist(a, b):
    return np.sum(np.abs(a - b), axis=1)

def _init_medoids_kpp(X, k, rng):
    n = X.shape[0]
    medoids = []
    # pick first medoid randomly
    medoids.append(rng.integers(0, n))
    # k-medoids++ style init using Manhattan distances
    dists = np.full(n, np.inf)
    for _ in range(1, k):
        m = medoids[-1]
        dnew = _manhattan_dist(X, X[m])
        dists = np.minimum(dists, dnew)
        probs = dists / np.sum(dists)
        medoids.append(rng.choice(n, p=probs))
    return np.array(medoids, dtype=int)

def proclus_fit_predict(X, k, l, iters, rng):
    """
    X: (n_core, d), standardized
    k: clusters
    l: avg relevant dims per cluster (we use exactly l per cluster here, clipped)
    iters: refinement steps
    Returns: labels in 0..k-1
    """
    n, d = X.shape
    k = max(2, min(k, n))
    l = max(1, min(l, d))
    # init medoids
    medoid_idx = _init_medoids_kpp(X, k, rng)
    # start with all dims relevant
    rel_dims = [np.arange(d, dtype=int) for _ in range(k)]
    labels = np.zeros(n, dtype=int)
    for it in range(max(1, iters)):
        # assign using projected Manhattan distance
        D = np.empty((n, k))
        for j in range(k):
            idxs = rel_dims[j]
            Dj = np.sum(np.abs(X[:, idxs] - X[medoid_idx[j], idxs]), axis=1)
            D[:, j] = Dj
        new_labels = D.argmin(axis=1)
        # handle empty clusters by re-seeding
        for j in range(k):
            if not np.any(new_labels == j):
                # pick farthest point as new medoid
                far = np.argmax(D.min(axis=1))
                medoid_idx[j] = far
                new_labels[far] = j
        labels = new_labels
        # update medoids per cluster (true medoid under projected L1)
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                continue
            idxs = rel_dims[j]
            # compute L1 distance matrix to choose medoid
            Xm = X[members][:, idxs]
            # sum L1 to all others
            # (|Xm_i - Xm|).sum(axis=2).sum(axis=1) – compute efficiently
            # pairwise L1 via broadcasting may be heavy; use candidate medoid search
            # compute distance to each candidate medoid
            sums = []
            for ii in range(len(members)):
                sums.append(np.sum(np.abs(Xm[ii] - Xm)))
            sums = np.sum(np.vstack(sums), axis=1)
            med_rel = members[np.argmin(sums)]
            medoid_idx[j] = med_rel
        # select relevant dimensions per cluster (smallest mean absolute deviation)
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                rel_dims[j] = np.arange(d, dtype=int)
                continue
            Xm = X[members]
            center = X[medoid_idx[j]]
            mad = np.mean(np.abs(Xm - center), axis=0)
            rel_dims[j] = np.argsort(mad)[:l]
    # final assignment with learned rel dims
    D = np.empty((n, k))
    for j in range(k):
        idxs = rel_dims[j]
        D[:, j] = np.sum(np.abs(X[:, idxs] - X[medoid_idx[j], idxs]), axis=1)
    labels = D.argmin(axis=1)
    return labels

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
elif CLUSTERING_ALGO.lower() == "mfa":  # <<< ADDED
    # Step 1) bootstrap components using your existing agglomerative threshold on the core
    logger.info(f"Clustering algorithm: MFA (bootstrap via Agglomerative distance_threshold={AGGLOMERATIVE_DISTANCE_THRESHOLD}, q={MFA_Q})")
    boot = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=AGGLOMERATIVE_DISTANCE_THRESHOLD,
        linkage='ward'
    )
    boot_labels = boot.fit_predict(Xs_core)
    uniq_boot = np.unique(boot_labels)
    # Prepare MFA parameters per bootstrap cluster
    comps = []  # list of dicts with {mu, L, psi, pi}
    n_core = Xs_core.shape[0]
    for k in uniq_boot:
        idxs = np.where(boot_labels == k)[0]
        Xk = Xs_core[idxs]
        nk = Xk.shape[0]
        if nk < 3:
            mu_k = Xk.mean(axis=0) if nk > 0 else np.zeros(Xs_core.shape[1])
            var_k = Xk.var(axis=0) + 1e-3
            q_eff = 1
            L_k = np.zeros((Xs_core.shape[1], q_eff))
            psi_k = var_k
        else:
            d = Xk.shape[1]
            q_eff = max(1, min(MFA_Q, d-1, nk-1))
            mu_k = Xk.mean(axis=0)
            Xk_c = Xk - mu_k
            fa = FactorAnalysis(n_components=q_eff, random_state=42)
            fa.fit(Xk_c)
            L_k = fa.components_.T
            psi_k = np.maximum(fa.noise_variance_, 1e-6)
        pi_k = nk / float(n_core)
        comps.append({"mu": mu_k, "L": L_k, "psi": psi_k, "pi": max(pi_k, 1e-12)})

    # Step 2) one-shot E-step style assignment by max posterior (log pi + log N_MFA)
    log_probs = np.zeros((n_core, len(comps)))
    for j, c in enumerate(comps):
        log_probs[:, j] = np.log(c["pi"]) + mfa_logpdf_batch(Xs_core, c["mu"], c["L"], c["psi"])
    core_labels = log_probs.argmax(axis=1)
elif CLUSTERING_ALGO.lower() == "proclus":  # <<< ADDED
    d = Xs_core.shape[1]
    l_auto = int(np.ceil(0.1 * d))
    l_use = PROCLUS_L if PROCLUS_L > 0 else max(2, min(d, l_auto))
    k_use = max(2, min(PROCLUS_K, Xs_core.shape[0]))
    iters = max(1, PROCLUS_ITERS)
    logger.info(f"Clustering algorithm: PROCLUS (k={k_use}, l={l_use}, iters={iters}, metric=Manhattan in projected subspaces)")
    core_labels = proclus_fit_predict(Xs_core, k=k_use, l=l_use, iters=iters, rng=rng)
else:
    raise ValueError("CLUSTERING_ALGO must be either 'agglomerative', 'hdbscan', 'mfa', or 'proclus'.")

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

# ============ METRICS (AMI_core, DBCV_full) ============  # <<< ADDED (metrics)
try:
    logger.info("Computing AMI_core...")
    ami_core = float(adjusted_mutual_info_score(y[core_idx], core_labels_compact, average_method='arithmetic'))
except Exception as e:
    logger.warning(f"AMI_core computation failed: {e}")
    ami_core = None

# DBCV requires at least two non-noise clusters; guard for degenerate cases
try:
    logger.info("Computing DBCV_full...")
    non_noise = full_labels[full_labels != -1]
    dbcv_full = float(hdbscan.validity_index(Xs_all, full_labels, metric='euclidean')) \
        if (len(np.unique(non_noise)) >= 2) else None
except Exception as e:
    logger.warning(f"DBCV_full computation failed: {e}")
    dbcv_full = None

logger.info(f"Metrics — AMI_core: {ami_core}, DBCV_full: {dbcv_full}")  # <<< ADDED (metrics)
# ========================================================  # <<< ADDED (metrics)

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

plt.figure(figsize=(12, 8))
ax = sns.heatmap(ct_w, cmap="viridis", xticklabels=xlabels, yticklabels=ylabels, annot=ct_counts, fmt="d")
ax.set_title(PLOT_TITLE)
ax.set_xlabel(PLOT_XLABEL)
ax.set_ylabel(PLOT_YLABEL)
ax.tick_params(axis='x', rotation=45, labelrotation=45)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=200)
logger.info(f"Saved cluster vs DSSP overlap plot to {PLOT_PATH}")

# === Core-only plot (balanced core subset) ===
PLOT_PATH_CORE = os.path.join(
    ALGO_DIR,
    f"{FEATURES_DESC}_{CLUSTERING_ALGO}{'_' + str(DOWNSAMPLE_SIZE) if DOWNSAMPLE else ''}_pc20_core.png"
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

plt.figure(figsize=(12, 8))
ax = sns.heatmap(ct_w_core, cmap="viridis", xticklabels=xlabels_core, yticklabels=ylabels_core, annot=ct_counts_core, fmt="d")
ax.set_title(PLOT_TITLE_CORE)
ax.set_xlabel(PLOT_XLABEL)
ax.set_ylabel("Cluster (balanced-core only)")
ax.tick_params(axis='x', rotation=45, labelrotation=45)
plt.tight_layout()
plt.savefig(PLOT_PATH_CORE, dpi=200)
logger.info(f"Saved CORE-ONLY cluster vs DSSP overlap plot to {PLOT_PATH_CORE}")

# Persist minimal artifacts for downstream profiling
# 1) Cluster labels aligned to original row order in this run
clusters_path = os.path.join(ALGO_DIR, "clusters.parquet")
pd.DataFrame({"cluster": full_labels}).to_parquet(clusters_path, index=False)

# 2) Scaler so profiling can reproduce the exact transform
scaler_path = os.path.join(ALGO_DIR, "scaler.joblib")
dump(scaler, scaler_path)

# 3) Metadata for reproducibility
meta = {
    "timestamp": datetime.now().isoformat(),
    "data_version": DATA_VERSION,
    "features_desc": FEATURES_DESC,
    "algo": CLUSTERING_ALGO,
    "params": {
        "hdbscan": {
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
            "min_samples": HDBSCAN_MIN_SAMPLES
        },
        "agglomerative": {
            "distance_threshold": AGGLOMERATIVE_DISTANCE_THRESHOLD,
            "linkage": "ward"
        },
        "mfa": {  # <<< ADDED
            "q": MFA_Q,
            "bootstrap": {
                "method": "agglomerative",
                "distance_threshold": AGGLOMERATIVE_DISTANCE_THRESHOLD,
                "linkage": "ward"
            }
        },
        "proclus": {  # <<< ADDED
            "k": PROCLUS_K,
            "l": PROCLUS_L,
            "iters": PROCLUS_ITERS,
            "metric": "manhattan_projected",
            "l_auto_rule": "ceil(0.1*d), clamped to [2,d] when --proclus_l=0"
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
    "class_counts_core": dict(zip(*np.unique(y[core_idx], return_counts=True))),
    "class_counts_full": dict(zip(*np.unique(y, return_counts=True))),
    "plots": {
        "full": PLOT_PATH,
        "core": PLOT_PATH_CORE
    },
    "artifacts": {
        "clusters_parquet": clusters_path,
        "scaler_joblib": scaler_path
    },
    "metrics": {
        "external": {"AMI_core": ami_core},
        "internal": {"DBCV_full": dbcv_full}
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

with open(os.path.join(ALGO_DIR, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, default=_to_py)

logger.info(f"Saved clustering artifacts to {ALGO_DIR}")
