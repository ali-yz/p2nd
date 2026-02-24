"""
Quick test: fetch 8hui chain A from RCSB and render with PyMOL
using the preferred cartoon style.

Run with: /home/ubuntu/p2nd/.venv_pymol/bin/python scripts/pymol_test_8hui.py
"""

import os
import requests
import pymol

pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()
from pymol import cmd

# ── paths ──────────────────────────────────────────────────────────
PDB_ID = "8hui"
CHAIN = "A"
OUT_DIR = "/home/ubuntu/p2nd/data/output/validations"
PDB_PATH = os.path.join(OUT_DIR, f"{PDB_ID}.pdb")
PNG_PATH = os.path.join(OUT_DIR, f"{PDB_ID}_chain{CHAIN}_cartoon.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ── download PDB if not cached ─────────────────────────────────────
if not os.path.exists(PDB_PATH):
    url = f"https://files.rcsb.org/download/{PDB_ID.upper()}.pdb"
    print(f"Downloading {url} ...")
    r = requests.get(url)
    r.raise_for_status()
    with open(PDB_PATH, "w") as f:
        f.write(r.text)
    print(f"Saved to {PDB_PATH}")
else:
    print(f"Using cached {PDB_PATH}")

# ── load & select chain A ──────────────────────────────────────────
cmd.load(PDB_PATH, PDB_ID)
cmd.remove(f"not chain {CHAIN}")

# ── representation ─────────────────────────────────────────────────
cmd.show("cartoon")
cmd.hide("everything", "not polymer")

# ── cartoon settings ───────────────────────────────────────────────
cmd.set("cartoon_fancy_helices", 1)
cmd.set("cartoon_smooth_loops", 1)
cmd.set("cartoon_flat_sheets", 1)
cmd.set("cartoon_transparency", 0)
cmd.set("cartoon_oval_length", 1.2)
cmd.set("cartoon_oval_width", 0.3)
cmd.set("cartoon_rect_length", 1.5)
cmd.set("cartoon_rect_width", 0.3)

# ── surface / texture settings ─────────────────────────────────────
cmd.set("ray_trace_mode", 1)
cmd.set("ray_shadows", 0)
cmd.set("ray_opaque_background", 0)
cmd.set("antialias", 2)
cmd.set("specular", "off")
cmd.set("ambient", 0.5)

# ── colour ─────────────────────────────────────────────────────────
cmd.color("teal", f"chain {CHAIN}")

# ── background & rendering ─────────────────────────────────────────
cmd.bg_color("white")
cmd.set("ray_trace_gain", 0.1)
cmd.set("ray_trace_disco_factor", 1)
cmd.set("two_sided_lighting", "on")

# ── orient & rotate to match reference view ────────────────────────
cmd.orient()
cmd.rotate("x", -70)
cmd.rotate("y", -20)
cmd.rotate("z", -15)
cmd.zoom(f"chain {CHAIN}", 5)

# ── render ─────────────────────────────────────────────────────────
cmd.ray(1600, 1200)
cmd.png(PNG_PATH, dpi=300)
print(f"Saved render to {PNG_PATH}")

cmd.quit()
