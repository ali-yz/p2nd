"""
Validate cluster assignments by rendering a PyMOL image of a protein chain
with multiple clusters highlighted in distinct colours.

All residues are shown in a light colour; residues belonging to each chosen
cluster are painted in a unique, saturated colour.

Reads directly from the parquet files (meta + Y + clusters) — no
pre-compiled CSV needed.

Usage:
    .venv_pymol/bin/python scripts/validate_multicluster_pymol.py \
        --pdb_id 8hui --clusters 3 10 15 --version v6

    .venv_pymol/bin/python scripts/validate_multicluster_pymol.py \
        --pdb_id 2pne --clusters 1 2 5 8 --version v5 --chain A
"""

import argparse
import colorsys
import os
import sys

import pandas as pd
import requests

# ── PyMOL headless init ──────────────────────────────────────────────
import pymol

pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()
from pymol import cmd

# ── defaults ─────────────────────────────────────────────────────────
DATA_DIR = "/home/ubuntu/p2nd/data/output"
OUT_DIR = "/home/ubuntu/p2nd/data/output/validations"

LIGHT_COLOR = "lightblue"

# version → cluster sub-directory name
VERSION_CLUSTER_DIR = {
    "v5": "sincosphi_sincospsi_tco_hbondflags",
    "v6": "sincosphi_sincospsi_sincosalpha_hbondflags",
    "v7": "sincosphi_sincospsi_hbondflags",
}


def generate_colors(n: int) -> list[tuple[float, float, float]]:
    """Generate *n* visually distinct saturated colours (RGB 0-1)."""
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hls_to_rgb(hue, 0.4, 0.9)
        colors.append((r, g, b))
    return colors


def parse_args():
    p = argparse.ArgumentParser(description="Render multi-cluster validation image with PyMOL")
    p.add_argument("--pdb_id", required=True, help="PDB identifier (e.g. 8hui, 2pne)")
    p.add_argument("--clusters", required=True, type=int, nargs="+", help="Cluster IDs to highlight")
    p.add_argument(
        "--version",
        required=True,
        choices=list(VERSION_CLUSTER_DIR.keys()),
        help="Dataset version (v5, v6, v7)",
    )
    p.add_argument("--chain", default=None, help="Chain to render (default: first chain for this PDB)")
    p.add_argument("--out_dir", default=OUT_DIR, help="Output directory")
    return p.parse_args()


def load_merged_df(version: str) -> pd.DataFrame:
    """Read meta + Y + clusters parquets and merge them on index."""
    base = os.path.join(DATA_DIR, f"pc20_{version}")
    cluster_dir = VERSION_CLUSTER_DIR[version]

    meta_path = os.path.join(base, "dssp_dataset_transformed_meta.parquet")
    y_path = os.path.join(base, "dssp_dataset_transformed_Y.parquet")
    cluster_path = os.path.join(base, cluster_dir, "agglomerative", "clusters.parquet")

    for p in (meta_path, y_path, cluster_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: file not found: {p}")

    print(f"Loading {version} parquets ...")
    meta = pd.read_parquet(meta_path)
    y = pd.read_parquet(y_path)
    clusters = pd.read_parquet(cluster_path)

    df = meta.merge(y, left_index=True, right_index=True).merge(
        clusters, left_index=True, right_index=True
    )
    print(f"  merged shape: {df.shape}")
    return df


def download_pdb(pdb_id: str, out_dir: str) -> str:
    pdb_path = os.path.join(out_dir, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        print(f"Downloading {url} ...")
        r = requests.get(url)
        r.raise_for_status()
        with open(pdb_path, "w") as f:
            f.write(r.text)
        print(f"Saved to {pdb_path}")
    else:
        print(f"Using cached {pdb_path}")
    return pdb_path


def main():
    args = parse_args()
    pdb_id = args.pdb_id.lower()
    cluster_ids = args.clusters
    version = args.version
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── load & merge parquets ────────────────────────────────────────
    df = load_merged_df(version)

    # filter to pdb
    df = df[df["pdb_id"] == pdb_id]
    if df.empty:
        sys.exit(f"ERROR: no rows for pdb_id={pdb_id} in {version}")

    # pick chain
    chain = args.chain or df["Chain"].iloc[0]
    df_chain = df[df["Chain"] == chain]
    if df_chain.empty:
        sys.exit(f"ERROR: no rows for chain={chain} in {version}")

    # ── resolve residues per cluster ─────────────────────────────────
    available = sorted(df_chain["cluster"].unique().tolist())
    cluster_residue_map: dict[int, list[int]] = {}
    for cid in cluster_ids:
        residues = df_chain[df_chain["cluster"] == cid]["RESIDUE"].astype(int).tolist()
        if not residues:
            print(
                f"WARNING: cluster {cid} has no residues for {pdb_id} chain {chain} "
                f"(available: {available}), skipping"
            )
            continue
        cluster_residue_map[cid] = residues

    if not cluster_residue_map:
        sys.exit(f"ERROR: none of the requested clusters have residues for {pdb_id} chain {chain}")

    total_residues = len(df_chain)
    for cid, residues in cluster_residue_map.items():
        print(f"  cluster {cid}: {len(residues)}/{total_residues} residues")

    # ── generate distinct colours ────────────────────────────────────
    colors = generate_colors(len(cluster_residue_map))
    cluster_color_map: dict[int, tuple[str, tuple[float, float, float]]] = {}
    for (cid, _residues), rgb in zip(cluster_residue_map.items(), colors):
        color_name = f"clr_{cid}"
        cmd.set_color(color_name, list(rgb))
        cluster_color_map[cid] = (color_name, rgb)

    # ── download & load PDB ──────────────────────────────────────────
    pdb_path = download_pdb(pdb_id, out_dir)
    cmd.load(pdb_path, pdb_id)
    cmd.remove(f"not chain {chain}")

    # ── representation ───────────────────────────────────────────────
    cmd.hide("everything")
    cmd.show("cartoon", "polymer")

    # ── cartoon settings ─────────────────────────────────────────────
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_flat_sheets", 1)
    cmd.set("cartoon_transparency", 0)
    cmd.set("cartoon_oval_length", 1.2)
    cmd.set("cartoon_oval_width", 0.3)
    cmd.set("cartoon_rect_length", 1.5)
    cmd.set("cartoon_rect_width", 0.3)

    # ── colour: light base, distinct colour per cluster ──────────────
    cmd.color(LIGHT_COLOR, f"chain {chain}")

    for cid, residues in cluster_residue_map.items():
        color_name, rgb = cluster_color_map[cid]
        resi_str = "+".join(str(r) for r in residues)
        sel_name = f"cluster_{cid}"
        cmd.select(sel_name, f"chain {chain} and resi {resi_str}")
        cmd.color(color_name, sel_name)
        r, g, b = rgb
        print(f"  cluster {cid} → RGB({r:.2f}, {g:.2f}, {b:.2f})")

    cmd.deselect()

    # ── render settings ──────────────────────────────────────────────
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("ray_opaque_background", 0)
    cmd.set("antialias", 2)
    cmd.set("specular", "off")
    cmd.set("ambient", 0.5)
    cmd.bg_color("white")
    cmd.set("ray_trace_gain", 0.1)
    cmd.set("ray_trace_disco_factor", 1)
    cmd.set("two_sided_lighting", "on")

    # ── orient & render ──────────────────────────────────────────────
    cmd.orient()
    cmd.zoom(f"chain {chain}", 10)

    clusters_tag = "_".join(str(c) for c in cluster_residue_map)
    png_path = os.path.join(out_dir, f"{pdb_id}_chain{chain}_clusters_{clusters_tag}_{version}.png")
    cmd.ray(1600, 1200)
    cmd.png(png_path, dpi=300)
    print(f"Saved render to {png_path}")

    cmd.quit()


if __name__ == "__main__":
    main()
