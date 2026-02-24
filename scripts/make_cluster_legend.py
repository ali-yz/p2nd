"""
Generate a minimal colour legend for multi-cluster PyMOL renders.

Takes exact cluster→RGB mappings as arguments so colours match the render.

Usage:
    python scripts/make_cluster_legend.py \
        17:0.76,0.04,0.04  25:0.04,0.76,0.04  13:0.04,0.04,0.76 \
        -o data/output/validations/legend_17_25_13.png
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def parse_entry(s: str) -> tuple[int, tuple[float, float, float]]:
    """Parse 'CLUSTER:R,G,B' → (cluster_id, (r, g, b))."""
    cid_str, rgb_str = s.split(":")
    r, g, b = (float(v) for v in rgb_str.split(","))
    return int(cid_str), (r, g, b)


def main():
    p = argparse.ArgumentParser(description="Render a cluster colour legend")
    p.add_argument("entries", nargs="+", metavar="CID:R,G,B",
                   help="Cluster ID and RGB, e.g. 17:0.76,0.04,0.04")
    p.add_argument("-o", "--output", default="cluster_legend.png", help="Output PNG path")
    args = p.parse_args()

    items = [parse_entry(e) for e in args.entries]
    patches = [
        Patch(facecolor=rgb, edgecolor="black", label=f"Cluster {cid}")
        for cid, rgb in items
    ]

    fig, ax = plt.subplots(figsize=(2.5, 0.4 * len(items)))
    ax.axis("off")
    ax.legend(handles=patches, loc="center", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight", transparent=True)
    print(f"Saved legend to {args.output}")


if __name__ == "__main__":
    main()
