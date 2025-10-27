#!/usr/bin/env python3
"""
Build a polished Parquet dataset by merging legacy .dssp residue data with
secondary structure labels computed from the corresponding mmCIF coordinates.

Requires:
  pip install gemmi pandas pyarrow

Usage:
  python aggregate_all.py \
      --dssp-dir data/downloads/pc20_legacy \
      --mmcif-dir data/downloads/pc20_mmcif \
      --out parquet/dssp_dataset.parquet
"""
import argparse
import re
import math
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
from tqdm.auto import tqdm


_DSSP_ALLOWED = set(list("HETSBIGP"))
# 3-letter -> 1-letter (covers all standard residues + common alternates)
_AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
    'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
    'THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    # Common alt/modified fallbacks (map to parent or X)
    'MSE':'M','SEC':'U','PYL':'O','ASX':'B','GLX':'Z','XLE':'J','UNK':'X'
}

# -------- helpers --------

RESNUM_RE = re.compile(r"^\s*(\d+)([A-Za-z]?)\s*$")  # capture resseq and optional icode

def parse_resnum(token: str) -> Tuple[Optional[int], str]:
    """
    Parse a residue number token that may contain an insertion code, e.g. '89A'.
    Returns (resseq:int or None, icode:str) with icode possibly ''.
    """
    m = RESNUM_RE.match(token)
    if not m:
        return None, ""
    return int(m.group(1)), m.group(2) or ""

def _to_float_or_none(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def _parse_hbond_pair(tok: str) -> Tuple[Optional[int], Optional[float]]:
    # tokens like "3,-0.6" or "0,0.0" or "65,-0.0" (tolerate spaces)
    if not tok or "," not in tok:
        return None, None
    i_str, e_str = tok.split(",", 1)
    i_str = i_str.strip()
    e_str = e_str.strip()
    try:
        i_val = int(i_str)
    except Exception:
        i_val = None
    try:
        e_val = float(e_str)
    except Exception:
        e_val = None
    return i_val, e_val

# Float and H-bond regexes (handle missing spaces before minus signs)
_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_HBOND_RE = re.compile(r"(-?\d+)\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))")

def parse_legacy_dssp_lines(lines: List[str], pdb_id: str) -> pd.DataFrame:
    """
    Parser for legacy DSSP. Robust to missing whitespace before negative numbers.

      IDX RESNUM CHAIN AA  [STRUCTURE ... may contain spaces ...]  BP1  BP2  ACC
           NH-O1     OHN1     NH-O2     OHN2    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA
    """
    # Find header line
    start_idx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("#  RESIDUE"):
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError("DSSP file missing header line starting with '#  RESIDUE'.")

    recs = []
    for ln in lines[start_idx:]:
        s = ln.rstrip("\n")
        if not s.strip() or s.lstrip().startswith("#"):
            continue

        # 1) First four columns with a single regex; capture the rest as 'rest'
        m = re.match(r"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$", s)
        if not m:
            continue
        dssp_index, resnum_token, chain_id, aa, rest = m.groups()

        # 2) Last 8 floats anywhere in the line: take the last 8 matches and their spans in 'rest'
        float_matches = list(_FLOAT_RE.finditer(rest))
        if len(float_matches) < 8:
            continue
        last8 = float_matches[-8:]
        tail_vals = [rest[m_.start():m_.end()] for m_ in last8]
        # Map to named columns: TCO, KAPPA, ALPHA, PHI, PSI, X-CA, Y-CA, Z-CA
        tco, kappa, alpha, phi, psi, x_ca, y_ca, z_ca = [_to_float_or_none(v) for v in tail_vals]
        tail_start = min(m_.start() for m_ in last8)
        head = rest[:tail_start].rstrip()

        # 3) From 'head', take the last 4 H-bond pairs
        hb_matches = list(_HBOND_RE.finditer(head))
        if len(hb_matches) < 4:
            continue
        last4_hb = hb_matches[-4:]
        hb1_m, hb2_m, hb3_m, hb4_m = last4_hb
        nho1_i, nho1_e = _parse_hbond_pair(hb1_m.group(0))
        ohn1_i, ohn1_e = _parse_hbond_pair(hb2_m.group(0))
        nho2_i, nho2_e = _parse_hbond_pair(hb3_m.group(0))
        ohn2_i, ohn2_e = _parse_hbond_pair(hb4_m.group(0))
        hb_block_start = last4_hb[0].start()

        # 4) Tokens immediately before the HB block are: ... BP1 BP2 ACC
        before_hb = head[:hb_block_start].strip()
        btoks = re.split(r"\s+", before_hb) if before_hb else []
        if len(btoks) < 3:
            continue
        bp1, bp2, acc = btoks[-3], btoks[-2], btoks[-1]
        structure_tokens = btoks[:-3]
        structure = " ".join(structure_tokens) if structure_tokens else ""

        # Residue number + optional insertion code
        resseq, icode = parse_resnum(resnum_token)

        recs.append({
            "pdb_id": pdb_id.lower(),
            "Chain": chain_id,
            "RESIDUE": resseq,
            "icode": icode,
            "AA": aa,
            "STRUCTURE": structure,
            "BP1": bp1,
            "BP2": bp2,
            "ACC": acc,
            "N-H-->O_1_i": nho1_i,
            "N-H-->O_1_E": nho1_e,
            "O-->H-N_1_i": ohn1_i,
            "O-->H-N_1_E": ohn1_e,
            "N-H-->O_2_i": nho2_i,
            "N-H-->O_2_E": nho2_e,
            "O-->H-N_2_i": ohn2_i,
            "O-->H-N_2_E": ohn2_e,
            "TCO": tco,
            "KAPPA": kappa,
            "ALPHA": alpha,
            "PHI": phi,
            "PSI": psi,
            "X-CA": x_ca,
            "Y-CA": y_ca,
            "Z-CA": z_ca,
        })

    df = pd.DataFrame.from_records(recs)

    cols = [
        "RESIDUE", "AA", "STRUCTURE", "BP1", "BP2", "ACC",
        "N-H-->O_1_i", "N-H-->O_1_E", "O-->H-N_1_i", "O-->H-N_1_E",
        "N-H-->O_2_i", "N-H-->O_2_E", "O-->H-N_2_i", "O-->H-N_2_E",
        "TCO", "KAPPA", "ALPHA", "PHI", "PSI", "X-CA", "Y-CA", "Z-CA",
        "pdb_id", "Chain", "icode"
    ]
    return df[[c for c in cols if c in df.columns]]

def _aa1_from_3(res3: str) -> str:
    if not res3:
        return "X"
    res3 = res3.upper()
    return _AA3_TO_1.get(res3, "X")

def dssp_like_from_mmcif(mmcif_path: Path) -> pd.DataFrame:
    """
    Parse the `_dssp_struct_summary` loop from a DSSP-annotated mmCIF using plain text.
    Returns columns: pdb_id, Chain, RESIDUE, icode, AA_from_mmcif, DSSP_label, X-CA, Y-CA, Z-CA
    """
    text = Path(mmcif_path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # 1) Find the start of the desired loop_ section
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == "loop_":
            j = i + 1
            found = False
            while j < len(lines) and lines[j].strip().startswith("_"):
                if lines[j].strip().startswith("_dssp_struct_summary."):
                    found = True
                    break
                j += 1
            if found:
                start = i
                break
    if start is None:
        raise RuntimeError(f"_dssp_struct_summary loop not found in {mmcif_path}")

    # 2) Collect all tag lines for this loop
    tag_lines = []
    cur = start + 1
    while cur < len(lines):
        s = lines[cur].strip()
        if s.startswith("_dssp_struct_summary."):
            tag_lines.append(s)
            cur += 1
            continue
        break
    if not tag_lines:
        raise RuntimeError(f"No _dssp_struct_summary.* tags found in loop for {mmcif_path}")

    # 3) Collect data rows until the loop ends
    data_rows = []
    while cur < len(lines):
        s = lines[cur].rstrip()
        st = s.strip()
        if not st:
            break
        if st.startswith("#") or st == "loop_" or (st.startswith("_") and not st.startswith("_dssp_struct_summary.")):
            break
        data_rows.append(s)
        cur += 1

    # 4) Build tag -> index map and locate columns we need
    tags = [t.split()[0] for t in tag_lines]
    tag_to_idx = {t: i for i, t in enumerate(tags)}

    def need(tagname):
        if tagname not in tag_to_idx:
            raise KeyError(f"Missing tag {tagname} in {mmcif_path}")
        return tag_to_idx[tagname]

    i_entry = need("_dssp_struct_summary.entry_id")
    i_asym  = need("_dssp_struct_summary.label_asym_id")
    i_seq   = need("_dssp_struct_summary.label_seq_id")
    i_comp  = need("_dssp_struct_summary.label_comp_id")
    i_ss    = need("_dssp_struct_summary.secondary_structure")
    i_x     = need("_dssp_struct_summary.x_ca")
    i_y     = need("_dssp_struct_summary.y_ca")
    i_z     = need("_dssp_struct_summary.z_ca")

    parsed = []
    for row in data_rows:
        toks = re.split(r"\s+", row.strip())
        req = max(i_entry, i_asym, i_seq, i_comp, i_ss, i_x, i_y, i_z)
        if len(toks) <= req:
            continue

        try:
            entry_id = toks[i_entry]
            chain    = toks[i_asym]
            resseq   = int(toks[i_seq])
            comp3    = toks[i_comp]
            ss_raw   = toks[i_ss]
            x_ca     = float(toks[i_x])
            y_ca     = float(toks[i_y])
            z_ca     = float(toks[i_z])
        except Exception:
            continue

        aa1 = _aa1_from_3(comp3)
        ss = (ss_raw or "").strip()
        dssp_label = ss if (len(ss) == 1 and ss in _DSSP_ALLOWED) else "C"

        pdb_id = (entry_id or Path(mmcif_path).stem).lower()

        parsed.append({
            "pdb_id": pdb_id,
            "Chain": chain,
            "RESIDUE": resseq,
            "icode": "",
            "AA_from_mmcif": aa1,
            "DSSP_label": dssp_label,
            "X-CA": x_ca,
            "Y-CA": y_ca,
            "Z-CA": z_ca,
        })

    return pd.DataFrame(parsed)

def merge_leg_dssp_with_mmcif(leg_df: pd.DataFrame, mm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge by matching Cα coordinates within a tolerance, per (pdb_id, Chain).
    Warn if multiple hits occur within the tolerance; pick the nearest.
    """
    eps = 0.01
    e2  = eps*eps

    # quick checks
    for cols, df, name in [({"pdb_id","Chain","X-CA","Y-CA","Z-CA"}, leg_df, "legacy"),
                           ({"pdb_id","Chain","X-CA","Y-CA","Z-CA","AA_from_mmcif","DSSP_label"}, mm_df, "mmcif")]:
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} dataframe missing columns: {missing}")

    # Pre-join on rounded coords to 1 decimal, then refine
    def _round3(df):
        out = df.copy()
        out["_xr"] = out["X-CA"].round(1)
        out["_yr"] = out["Y-CA"].round(1)
        out["_zr"] = out["Z-CA"].round(1)
        return out

    leg_r = _round3(leg_df)
    mm_r  = _round3(mm_df)

    key = ["pdb_id","Chain","_xr","_yr","_zr"]
    merged = pd.merge(
        leg_r,
        mm_r[key + ["AA_from_mmcif","DSSP_label","RESIDUE","icode"]].rename(
            columns={"RESIDUE":"RESIDUE_mm","icode":"icode_mm"}
        ),
        on=key, how="left"
    )

    # rows needing refinement: no hit or duplicate merges on the rounded key
    duplicate_mask = merged.duplicated(subset=list(leg_df.columns), keep=False)
    nohit_mask     = merged["DSSP_label"].isna()
    need_refine    = merged[duplicate_mask | nohit_mask].copy()
    keep_ok        = merged[~(duplicate_mask | nohit_mask)].copy()

    # group mm_r for fast lookup
    idx_cols = ["pdb_id","Chain"]
    mm_groups = {k: g for k, g in mm_r.groupby(idx_cols, sort=False)}

    refined_rows: List[dict] = []
    warn_count = 0
    warn_cap = 10

    def _d2(ax, ay, az, bx, by, bz):
        dx = ax - bx; dy = ay - by; dz = az - bz
        return dx*dx + dy*dy + dz*dz

    for (pdb, ch), grp in need_refine.groupby(idx_cols, sort=False):
        mmg = mm_groups.get((pdb, ch))
        if mmg is None or mmg.empty:
            refined_rows.extend(
                grp.assign(_ambiguous=True, _reason="no_mm_group").to_dict("records")
            )
            continue

        mm_points = mmg[["X-CA","Y-CA","Z-CA","AA_from_mmcif","DSSP_label","RESIDUE","icode"]].reset_index(drop=True)

        for _, row in grp.iterrows():
            ax, ay, az = float(row["X-CA"]), float(row["Y-CA"]), float(row["Z-CA"])
            candidates = []
            mind2 = float("inf")
            minj  = None
            for j, mmrow in mm_points.iterrows():
                d2 = _d2(ax, ay, az, float(mmrow["X-CA"]), float(mmrow["Y-CA"]), float(mmrow["Z-CA"]))
                if d2 <= e2:
                    candidates.append((j, d2))
                if d2 < mind2:
                    mind2, minj = d2, j

            if len(candidates) == 1:
                j = candidates[0][0]
                m = mm_points.loc[j]
                out = row.copy()
                out["AA_from_mmcif"] = m["AA_from_mmcif"]
                out["DSSP_label"]    = m["DSSP_label"]
                out["RESIDUE_mm"]    = m["RESIDUE"]
                out["icode_mm"]      = m["icode"]
                out["_ambiguous"]    = False
                out["_reason"]       = "distance_ok"
                refined_rows.append(out.to_dict())
            elif len(candidates) > 1:
                j = min(candidates, key=lambda t: t[1])[0]
                m = mm_points.loc[j]
                out = row.copy()
                out["AA_from_mmcif"] = m["AA_from_mmcif"]
                out["DSSP_label"]    = m["DSSP_label"]
                out["RESIDUE_mm"]    = m["RESIDUE"]
                out["icode_mm"]      = m["icode"]
                out["_ambiguous"]    = True
                out["_reason"]       = f"multi_hits_within_{eps}A"
                refined_rows.append(out.to_dict())
                if warn_count < warn_cap:
                    print(f"[WARN] {pdb} {ch}: multiple mmCIF hits within {eps} Å for CA ({ax:.3f},{ay:.3f},{az:.3f}); taking nearest and marking ambiguous.")
                    warn_count += 1
            else:
                out = row.copy()
                out["_ambiguous"] = True
                out["_reason"]    = f"no_hit_within_{eps}A"
                refined_rows.append(out.to_dict())
                if warn_count < warn_cap:
                    print(f"[WARN] {pdb} {ch}: no mmCIF CA within {eps} Å for ({ax:.3f},{ay:.3f},{az:.3f}).")
                    warn_count += 1

    refined = pd.DataFrame(refined_rows)
    if not refined.empty:
        # Match the column layout of the first merge so concat is stable
        refined = refined.reindex(columns=merged.columns)
        merged2 = pd.concat([keep_ok, refined], ignore_index=True)
    else:
        merged2 = keep_ok.copy()

    # Optional check: AA mismatch
    if "AA" in merged2.columns and "AA_from_mmcif" in merged2.columns:
        mism = merged2["AA"].notna() & merged2["AA_from_mmcif"].notna() & (merged2["AA"] != merged2["AA_from_mmcif"])
        if mism.any():
            n = int(mism.sum())
            print(f"[WARN] {n} AA mismatches between legacy and mmCIF.")

    return merged2


# -------- main pipeline --------

def main():
    print("Starting aggregation of DSSP data...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dssp-dir", required=True, help="Directory with legacy .dssp files")
    ap.add_argument("--mmcif-dir", required=True, help="Directory with matching .dssp/.mmcif files")
    ap.add_argument("--out", required=True, help="Output Parquet file path")
    ap.add_argument("--suffix", default=".dssp", help="DSSP file suffix (default: .dssp)")
    args = ap.parse_args()

    dssp_dir = Path(args.dssp_dir)
    mmcif_dir = Path(args.mmcif_dir)
    out_path = Path(args.out)
    print(f"Args: {args}")
    all_rows = []

    dssp_files = sorted(dssp_dir.glob(f"*{args.suffix}"))
    print(f"Found {len(dssp_files)} DSSP files in {dssp_dir} with suffix {args.suffix}")
    if not dssp_files:
        raise SystemExit(f"No DSSP files found in {dssp_dir} with suffix {args.suffix}")

    for dssp_file in tqdm(dssp_files, desc="Processing DSSP files", unit="file"):
        pdb_id = dssp_file.stem.lower()

        # Find corresponding mmCIF (look for real mmCIFs too)
        mmcif_path = None
        for ext in [".mmcif", ".cif", ".dssp"]:
            cand = mmcif_dir / f"{pdb_id}{ext}"
            if cand.exists():
                mmcif_path = cand
                break
        if mmcif_path is None:
            print(f"[WARN] No mmCIF for {pdb_id}, skipping.")
            continue

        # Parse legacy DSSP file
        with open(dssp_file, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        leg_df = parse_legacy_dssp_lines(lines, pdb_id=pdb_id)

        if leg_df.empty:
            print(f"[WARN] No residues parsed from {dssp_file}, skipping.")
            continue

        # Compute DSSP-like labels from mmCIF (includes CA coordinates)
        mm_df = dssp_like_from_mmcif(mmcif_path)

        if mm_df.empty:
            print(f"[WARN] No protein residues found in {mmcif_path}, skipping merge.")
            continue

        merged = merge_leg_dssp_with_mmcif(leg_df, mm_df)

        # Keep only polished columns; keep AA_from_mmcif at the end for QC
        polished_cols = [
            "RESIDUE", "AA", "STRUCTURE", "BP1", "BP2", "ACC",
            "N-H-->O_1_i", "N-H-->O_1_E", "O-->H-N_1_i", "O-->H-N_1_E",
            "N-H-->O_2_i", "N-H-->O_2_E", "O-->H-N_2_i", "O-->H-N_2_E",
            "TCO", "KAPPA", "ALPHA", "PHI", "PSI", "X-CA", "Y-CA", "Z-CA",
            "pdb_id", "Chain", "DSSP_label"
        ]
        optional = ["AA_from_mmcif", "icode"]
        out_cols = [c for c in polished_cols if c in merged.columns] + [c for c in optional if c in merged.columns]
        merged = merged[out_cols].copy()

        all_rows.append(merged)

    if not all_rows:
        raise SystemExit("Nothing to write: no merged rows produced.")

    final_df = pd.concat(all_rows, ignore_index=True)

    # Write CSV for easy inspection/debugging head 1000 rows
    csv_path = out_path.with_suffix(".csv")
    final_df.head(1000).to_csv(csv_path, index=False)
    print(f"Wrote head 1000 rows to {csv_path} for inspection.")

    # Write parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(final_df):,} rows to {out_path}")


if __name__ == "__main__":
    print("Running aggregate_all.py...")
    main()
