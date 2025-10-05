import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt

CLUSTERING_THRESHOLD = 40.0
CLASS_CAP = 5000
TRANSFORMED_PATH_X = "/home/ubuntu/p2nd/data/output/pc20_v1/dssp_dataset_transformed_X.parquet"
TRANSFORMED_PATH_Y = "/home/ubuntu/p2nd/data/output/pc20_v1/dssp_dataset_transformed_Y.parquet"
PLOT_PATH = "/home/ubuntu/p2nd/data/output/pc20_v1/clustered_heatmap.png"
PLOT_TITLE = "Balanced cluster ↔ DSSP overlap (inverse-frequency weighted) pc20_v1"
PLOT_XLABEL = "DSSP label"
PLOT_YLABEL = "Cluster (balanced-core + medoid assignment)"

features = pd.read_parquet(TRANSFORMED_PATH_X).to_numpy()
labels = pd.read_parquet(TRANSFORMED_PATH_Y)["DSSP_label"].tolist()

rng = np.random.default_rng(42)

X = features
y = np.array(labels)
N = X.shape[0]
print(f"Clustering: N={N:,}, features={X.shape[1]}")

# sub sample to make it faster for a quick check
if N > 100_000:
    sel = rng.choice(N, size=100_000, replace=False)
    X = X[sel]
    y = y[sel]
    N = X.shape[0]
    print(f"Clustering: downsampled to N={N:,}, features={X.shape[1]}")

classes, counts = np.unique(y, return_counts=True)
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

print(f"Clustering: balanced core size={len(core_idx):,}, rest size={len(rest_idx):,}")

# Scale using ONLY the balanced core
scaler   = StandardScaler().fit(X[core_idx])
Xs_core  = scaler.transform(X[core_idx])
Xs_rest  = scaler.transform(X[rest_idx])
Xs_all   = scaler.transform(X)  # for later use

agg = AgglomerativeClustering(
    n_clusters=None,           # let threshold decide
    distance_threshold=CLUSTERING_THRESHOLD,    # tune this!
    linkage='ward'
)
core_labels = agg.fit_predict(Xs_core)

print(f"Clustering: core clustered to K={len(np.unique(core_labels))} clusters")

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
w = np.array([w_map[lab] for lab in y])

df = pd.DataFrame({"cluster": full_labels, "dssp": y, "w": w})
df = df[df["cluster"] != -1]  # optional: drop noise for the heatmap

# weighted crosstab (row-normalized)
ct_w = df.pivot_table(index="cluster", columns="dssp", values="w", aggfunc="sum", fill_value=0.0)
ct_w = ct_w.div(ct_w.sum(axis=1), axis=0)

# sort rows by their dominant DSSP column for readability
ct_w = ct_w.reindex(ct_w.idxmax(axis=1).sort_values().index)

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

# Option A: pass mapped labels to seaborn
xlabels = [label_map.get(c, c) for c in ct_w.columns]

plt.figure(figsize=(10, 6))
ax = sns.heatmap(ct_w, cmap="viridis", xticklabels=xlabels)
ax.set_title(PLOT_TITLE)
ax.set_xlabel(PLOT_XLABEL)
ax.set_ylabel(PLOT_YLABEL)
ax.tick_params(axis='x', rotation=45, labelrotation=45)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
