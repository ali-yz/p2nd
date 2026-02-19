import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTDIR = Path("mock_plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

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

# Use exactly 9 DSSP classes (same as your label_map)
DSSP_CLASSES = list(label_map.keys())  # ["B","C","E","G","H","I","P","S","T"]
N_CLUSTERS = 9
CLUSTERS = list(range(N_CLUSTERS))

# Make results reproducible
RNG = np.random.default_rng(42)


# ---------------------------
# Helper functions
# ---------------------------
def make_df_perfect_diagonal(n_per_cluster: int = 200) -> pd.DataFrame:
    """
    Each cluster k maps perfectly to one DSSP class DSSP_CLASSES[k],
    producing a near-perfect diagonal crosstab.
    """
    rows = []
    for k in CLUSTERS:
        dssp = DSSP_CLASSES[k]
        rows.extend([(k, dssp)] * n_per_cluster)
    df = pd.DataFrame(rows, columns=["cluster", "dssp"])
    return df


def make_df_random(n_total: int = 1800) -> pd.DataFrame:
    """
    Random assignment of cluster and DSSP.
    """
    clusters = RNG.integers(0, N_CLUSTERS, size=n_total)
    dssp = RNG.choice(DSSP_CLASSES, size=n_total, replace=True)
    df = pd.DataFrame({"cluster": clusters, "dssp": dssp})
    return df


def plot_cluster_vs_dssp_heatmap(df: pd.DataFrame, plot_title: str, plot_path: Path) -> None:
    # Crosstab of cluster vs DSSP (counts)
    ct_w = pd.crosstab(df["cluster"], df["dssp"]).reindex(index=CLUSTERS, columns=DSSP_CLASSES, fill_value=0)

    # Labels
    xlabels = [label_map.get(c, c) for c in ct_w.columns]
    cluster_sizes = df.groupby("cluster").size().reindex(CLUSTERS, fill_value=0)
    ylabels = [f"Cluster {k}" for k in ct_w.index]

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        ct_w,
        cmap="viridis",
        xticklabels=xlabels,
        yticklabels=ylabels,
        fmt="d",
        cbar=True
    )
    ax.set_title(plot_title)
    ax.tick_params(axis="x", rotation=45, labelrotation=45)
    plt.tight_layout()

    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved plot to {plot_path.resolve()}")


# ---------------------------
# Generate mocked datasets + plots
# ---------------------------
df_diag = make_df_perfect_diagonal(n_per_cluster=200)
df_rand = make_df_random(n_total=9 * 200)  # same total size for easy comparison

plot_cluster_vs_dssp_heatmap(
    df_diag,
    plot_title="Too Perfect!",
    plot_path=OUTDIR / "cluster_vs_dssp_perfect_diagonal.png",
)

plot_cluster_vs_dssp_heatmap(
    df_rand,
    plot_title="Random Clusters",
    plot_path=OUTDIR / "cluster_vs_dssp_random.png",
)

print(f"Done. Plots saved in: {OUTDIR.resolve()}")
