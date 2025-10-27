#!/usr/bin/env python3
# cluster.py

"""
Cluster DSSP features using specified algorithm and persist results.

Usage (MFA subspace clustering only):

python scripts/cluster.py \
    --algo mfa \
    --features_desc sincosphi_sincospsi_tco_hbondflags \
    --data_version v5
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, adjusted_mutual_info_score, silhouette_score
from sklearn.decomposition import FactorAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import logging

import argparse, os, json
from datetime import datetime
# from joblib import dump

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# argparse (MFA only)
parser = argparse.ArgumentParser(description="Cluster DSSP features and persist results.")
parser.add_argument("--algo", choices=["mfa"], required=True,
                    help="Clustering algorithm to use (only 'mfa').")
parser.add_argument("--features_desc", required=True,
                    help="Short descriptor for features used (goes into output paths).")
parser.add_argument("--data_version", required=True,
                    help="Data version tag, e.g. v5.")
parser.add_argument("--downsample", action="store_true",
                    help="Optional: enable downsampling for quick checks.")
parser.add_argument("--downsample_size", type=int, default=100_000,
                    help="Downsample size if --downsample is set.")
parser.add_argument("--agg_distance_threshold", type=float, default=50.0,
                    help="Agglomerative distance_threshold used for MFA bootstrap.")
parser.add_argument("--mfa_q", type=int, default=8,
                    help="MFA latent dimensionality per component (q).")
parser.add_argument("--fa_tol", type=float, default=1e-3,
                    help="Tolerance for FactorAnalysis convergence.")
args = parser.parse_args()

DATA_VERSION = args.data_version
FEATURES_DESC = args.features_desc
CLUSTERING_ALGO = args.algo
DOWNSAMPLE = args.downsample
DOWNSAMPLE_SIZE = args.downsample_size
AGGLOMERATIVE_DISTANCE_THRESHOLD = args.agg_distance_threshold
MFA_Q = args.mfa_q
FA_TOL = args.fa_tol
CLASS_CAP = 6_000

# derive IO paths and per-algo subdir
BASE_DIR = f"/home/ubuntu/p2nd/data/output/pc20_{DATA_VERSION}"
ALGO_DIR = os.path.join(BASE_DIR, FEATURES_DESC, CLUSTERING_ALGO)
os.makedirs(ALGO_DIR, exist_ok=True)

TRANSFORMED_PATH_X = f"/home/ubuntu/p2nd/data/output/pc20_{DATA_VERSION}/dssp_dataset_transformed_X.parquet"
TRANSFORMED_PATH_Y = f"/home/ubuntu/p2nd/data/output/pc20_{DATA_VERSION}/dssp_dataset_transformed_Y.parquet"

algo_name_for_title = "Mixture of Factor Analyzers"
PLOT_TITLE = f"Cluster - DSSP Overlap : {algo_name_for_title} : features={FEATURES_DESC} : data=pc20_{DATA_VERSION}"
PLOT_PATH = os.path.join(
    ALGO_DIR,
    f"{FEATURES_DESC}_{CLUSTERING_ALGO}{'_' + str(DOWNSAMPLE_SIZE) if DOWNSAMPLE else ''}_pc20.png"
)

PLOT_XLABEL = "DSSP label"
PLOT_YLABEL = "Cluster (balanced-core + medoid assignment)"
PLOT_COL_ORDER = ["C", "B", "E", "G", "H", "I", "P", "S", "T"]

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
mpl.rcParams["mathtext.fontset"] = "stixsans"

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
# ----------------------------------------
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
    invPsi = 1.0 / np.clip(psi, 1e-7, None)
    Lt_invPsi = (L.T * invPsi)
    A = np.eye(L.shape[1]) + Lt_invPsi @ L
    signA, logdetA = np.linalg.slogdet(A)
    if signA <= 0:
        logdetA = np.log(np.linalg.det(A + 1e-6*np.eye(A.shape[0])))
    logdetSigma = np.sum(np.log(np.clip(psi, 1e-12, None))) + logdetA
    invPsi_Xc = Xc * invPsi
    Lt_invPsi_Xc_T = Lt_invPsi @ Xc.T
    sol = np.linalg.solve(A, Lt_invPsi_Xc_T)
    quad = np.einsum("ij,ij->i", Xc, invPsi_Xc) - np.einsum("ij,ij->j", Lt_invPsi_Xc_T, sol)
    return -0.5 * (d * np.log(2.0*np.pi) + logdetSigma + quad)

# -----------------------
# MFA with agglom bootstrap
# -----------------------
logger.info(f"Clustering algorithm: MFA (bootstrap via Agglomerative distance_threshold={AGGLOMERATIVE_DISTANCE_THRESHOLD}, q={MFA_Q}, tol={FA_TOL})")  # <<< CHANGED (log tol)
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
        fa = FactorAnalysis(
            n_components=q_eff,
            random_state=42,
            max_iter=2000,
            tol=FA_TOL                                           # <<< CHANGED (use CLI tol)
        )
        fa.fit(Xk_c)
        L_k = fa.components_.T
        psi_k = np.maximum(fa.noise_variance_, 1e-6)
    pi_k = nk / float(n_core)
    comps.append({"mu": mu_k, "L": L_k, "psi": psi_k, "pi": max(pi_k, 1e-12)})

# One-shot E-step style assignment by max posterior (log pi + log N_MFA)
log_probs = np.zeros((n_core, len(comps)))
for j, c in enumerate(comps):
    log_probs[:, j] = np.log(c["pi"]) + mfa_logpdf_batch(Xs_core, c["mu"], c["L"], c["psi"])
core_labels = log_probs.argmax(axis=1)

# Map cluster labels to 0..K-1 for clean indexing
uniq = np.unique(core_labels)
label_to_compact = {lab:i for i, lab in enumerate(uniq)}
core_labels_compact = np.array([label_to_compact[lab] for lab in core_labels])
K = len(uniq)
logger.info(f"Found {K} clusters in the core dataset.")

# ===== Metrics (AMI_core + Silhouette_core) =====
try:
    logger.info("Computing AMI_core...")
    AMI_core = float(adjusted_mutual_info_score(y[core_idx], core_labels_compact))
except Exception as e:
    logger.warning(f"AMI_core computation failed: {e}")
    AMI_core = None

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
    radii.append(np.percentile(Dk, 95))

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
        assigned_rest[i] = -1

# Stitch full labels
full_labels = np.full(N, -1)
full_labels[core_idx] = core_labels_compact
full_labels[rest_idx] = assigned_rest

# inverse-frequency weights per DSSP class
w_map = {c: 1.0/count_map[c] for c in classes}
w_map_core = {c: 1.0/CLASS_CAP for c in classes}
w = np.array([w_map[lab] for lab in y])
w_core = np.array([w_map_core[lab] for lab in y])

df = pd.DataFrame({"cluster": full_labels, "dssp": y, "w": w})

# weighted crosstab (row-normalized)
ct_w = df.pivot_table(index="cluster", columns="dssp", values="w", aggfunc="sum", fill_value=0.0)
ct_w = ct_w.div(ct_w.sum(axis=1), axis=0)
ct_w = ct_w.reindex(ct_w.idxmax(axis=1).sort_values().index)
desired_cols = [c for c in PLOT_COL_ORDER if c in ct_w.columns]
ct_w = ct_w.reindex(columns=desired_cols)

# absolute counts for annotations (full set)
ct_counts = df.pivot_table(index="cluster", columns="dssp", values="w", aggfunc="count", fill_value=0)
ct_counts = ct_counts.loc[ct_w.index, ct_w.columns].astype(int).to_numpy()

label_map = {
    "B": "β-bridge", "C": "coil/other", "E": "β-strand", "G": "3₁₀ helix",
    "H": "α-helix", "I": "π-helix", "P": "PPII helix", "S": "bend", "T": "turn",
}

# DSSP labels with totals (full set)
totals_per_dssp_full = df["dssp"].value_counts()
xlabels = [f"{c}:{label_map.get(c)} (n={int(totals_per_dssp_full.get(c, 0))})" for c in ct_w.columns]

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
PLOT_PATH_CORE = os.path.join(
    ALGO_DIR,
    f"{FEATURES_DESC}_{CLUSTERING_ALGO}{'_' + str(DOWNSAMPLE_SIZE) if DOWNSAMPLE else ''}_pc20_core.png"
)
PLOT_TITLE_CORE = PLOT_TITLE + " — CORE ONLY"

core_df = pd.DataFrame({
    "cluster": core_labels_compact,
    "dssp":    y[core_idx],
    "w":       w_core[core_idx]
})

ct_w_core = core_df.pivot_table(index="cluster", columns="dssp", values="w", aggfunc="sum", fill_value=0.0)
ct_w_core = ct_w_core.div(ct_w_core.sum(axis=1), axis=0)
ct_w_core = ct_w_core.reindex(ct_w_core.idxmax(axis=1).sort_values().index)
desired_cols = [c for c in PLOT_COL_ORDER if c in ct_w_core.columns]
ct_w_core = ct_w_core.reindex(columns=desired_cols)

ct_counts_core = core_df.pivot_table(index="cluster", columns="dssp", values="w", aggfunc="count", fill_value=0)
ct_counts_core = ct_counts_core.loc[ct_w_core.index, ct_w_core.columns].astype(int).to_numpy()

totals_per_dssp_core = core_df["dssp"].value_counts()
xlabels_core = [f"{c}:{label_map.get(c)} (n={int(totals_per_dssp_core.get(c, 0))})" for c in ct_w_core.columns]

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

# Persist minimal artifacts for downstream profiling
clusters_path = os.path.join(ALGO_DIR, "clusters.parquet")
pd.DataFrame({"cluster": full_labels}).to_parquet(clusters_path, index=False)

scaler_path = os.path.join(ALGO_DIR, "scaler.joblib")
# dump(scaler, scaler_path)

logger.info(f"Scores — AMI_core={AMI_core}, Silhouette_core={Silhouette_core}: ARGS: {args}")

meta = {
    "timestamp": datetime.now().isoformat(),
    "data_version": DATA_VERSION,
    "features_desc": FEATURES_DESC,
    "algo": CLUSTERING_ALGO,
    "params": {
        "agglomerative": {
            "distance_threshold": AGGLOMERATIVE_DISTANCE_THRESHOLD,
            "linkage": "ward"
        },
        "mfa": {
            "q": MFA_Q,
            "tol": FA_TOL,                                            # <<< ADDED
            "bootstrap": {
                "method": "agglomerative",
                "distance_threshold": AGGLOMERATIVE_DISTANCE_THRESHOLD,
                "linkage": "ward"
            }
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
        "Silhouette_core": Silhouette_core,
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
