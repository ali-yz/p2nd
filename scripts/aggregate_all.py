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
import random
import argparse
import re
from pathlib import Path
from typing import Tuple, Optional, List
import re
import pandas as pd
from pathlib import Path


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
    # tokens like "3,-0.6" or "0,0.0" or "65,-0.0"
    if not tok or "," not in tok:
        return None, None
    i_str, e_str = tok.split(",", 1)
    try:
        i_val = int(i_str)
    except Exception:
        i_val = None
    try:
        e_val = float(e_str)
    except Exception:
        e_val = None
    return i_val, e_val

def parse_legacy_dssp_lines(lines: List[str], pdb_id: str) -> pd.DataFrame:
    """
    Parser for legacy DSSP (like the sample you pasted). Layout:

      IDX RESNUM CHAIN AA  [STRUCTURE ... may contain spaces ...]  BP1  BP2  ACC
           NH-O1     OHN1     NH-O2     OHN2    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA
    Where the 4 hydrogen-bond columns are single tokens formatted "offset,energy".

    We detect columns by:
      - taking first 4 tokens (IDX, RESNUM, CHAIN, AA),
      - taking the last 8 tokens as floats (Z-CA..TCO in reverse),
      - taking the 4 tokens before those as H-bond pairs,
      - just before those: ACC, BP2, BP1 (kept as strings),
      - everything between AA and BP1 joined as STRUCTURE.
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
        s = ln.rstrip()
        if not s or s.startswith("#"):
            continue
        # ignore non-data footer lines that may look like comments
        toks = re.split(r"\s+", s.strip())
        #if len(toks) < 20:
        #    continue  # too short to be a residue row

        # 1) First four tokens are stable:
        dssp_index = toks[0]                  # unused, but could be kept if you want
        resnum_token = toks[1]
        chain_id = toks[2]
        aa = toks[3]

        # 2) From the end, pull trailing numeric floats (8 of them):
        #if len(toks) < 4 + 1 + 3 + 4 + 8:
        #    # Needs at least: 4 (head) + STRUCT (>=1) + 3 (BP1,BP2,ACC) + 4 (hbonds) + 8 (floats)
        #    # Skip if not enough tokens
        #    continue

        # Last 8 numeric columns
        z_ca = _to_float_or_none(toks[-1])
        y_ca = _to_float_or_none(toks[-2])
        x_ca = _to_float_or_none(toks[-3])
        psi  = _to_float_or_none(toks[-4])
        phi  = _to_float_or_none(toks[-5])
        alpha= _to_float_or_none(toks[-6])
        kappa= _to_float_or_none(toks[-7])
        tco  = _to_float_or_none(toks[-8])

        # 3) Four H-bond tokens before those 8:
        hb4 = toks[-9]   # OHN2
        hb3 = toks[-10]  # NHO2
        hb2 = toks[-11]  # OHN1
        hb1 = toks[-12]  # NHO1

        nho1_i, nho1_e = _parse_hbond_pair(hb1)
        ohn1_i, ohn1_e = _parse_hbond_pair(hb2)
        nho2_i, nho2_e = _parse_hbond_pair(hb3)
        ohn2_i, ohn2_e = _parse_hbond_pair(hb4)

        # 4) Three tokens before those are ACC, BP2, BP1 (order in the file: BP1 BP2 ACC)
        acc  = toks[-13]
        bp2  = toks[-14]
        bp1  = toks[-15]

        # 5) Everything between AA and BP1 is the STRUCTURE field (can contain spaces)
        structure_tokens = toks[4:-15]
        structure = " ".join(structure_tokens) if structure_tokens else ""

        # Parse residue number and optional insertion code (e.g., '89A')
        # In your file, RESNUM (2nd col) is plain integer; still be safe:
        resseq = None
        icode = ""
        m = re.match(r"^\s*(-?\d+)([A-Za-z]?)\s*$", resnum_token)
        if m:
            resseq = int(m.group(1))
            icode  = m.group(2) or ""

        recs.append({
            "pdb_id": pdb_id.lower(),
            "Chain": chain_id,
            "RESIDUE": resseq,
            "icode": icode,
            "AA": aa,
            "STRUCTURE": structure,   # free-form legacy structure field
            "BP1": bp1,               # keep as string (can be like '0A')
            "BP2": bp2,               # keep as string
            "ACC": acc,               # keep as string (numeric-looking)
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

    # Order columns like your header, plus id/meta at the end.
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
    Returns columns: pdb_id, Chain, RESIDUE, icode, AA_from_mmcif, DSSP_label
    """
    text = Path(mmcif_path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # 1) Find the start of the desired loop_ section
    start = None
    for i, ln in enumerate(lines):
        if ln.strip() == "loop_":
            # Check if the next lines are _dssp_struct_summary.* tags
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
        break  # first non-tag after loop_ header
    if not tag_lines:
        raise RuntimeError(f"No _dssp_struct_summary.* tags found in loop for {mmcif_path}")

    # 3) Collect data rows until the loop ends (next loop_ / next category / '#'/blank)
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

    # 4) Build tag -> index map and locate only columns we need
    tags = [t.split()[0] for t in tag_lines]  # robust: ignore inline comments
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

    # 5) Tokenize rows: mmCIF loop rows are whitespace-separated; values don’t have spaces here
    # (multi-line fields are delimited with ';' blocks, which do not occur in this section)
    parsed = []
    for row in data_rows:
        toks = re.split(r"\s+", row.strip())
        # Skip short/bad rows
        if len(toks) < len(tags):
            # Some writers may compress trailing '.' columns; guard by only using required indices
            # We still require at least up to max(index we need)+1 tokens
            req = max(i_entry, i_asym, i_seq, i_comp, i_ss)
            if len(toks) <= req:
                continue

        try:
            entry_id = toks[i_entry]
            chain    = toks[i_asym]
            resseq   = int(toks[i_seq])
            comp3    = toks[i_comp]
            ss_raw   = toks[i_ss]
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
            "DSSP_label": dssp_label
        })

    return pd.DataFrame(parsed)

def merge_leg_dssp_with_mmcif(leg_df: pd.DataFrame, mm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge on pdb_id, Chain, RESIDUE, icode.
    Prefer AA from legacy (since that’s what you showed), but keep mmCIF AA to debug mismatches.
    """
    # temp write the mm_df to disk for debugging
    rnd = random.randint(1, 100)
    mm_df.to_csv(f"mmcif_dssp_debug{rnd}.csv", index=False)
    leg_df.to_csv(f"legacy_dssp_debug{rnd}.csv", index=False)

    key = ["pdb_id", "Chain", "RESIDUE"]
    merged = pd.merge(
        leg_df, mm_df[key + ["AA_from_mmcif", "DSSP_label"]],
        on=key, how="left"
    )
    return merged


# -------- main pipeline --------

def main():
    print("Starting aggregation of DSSP data...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dssp-dir", required=True, help="Directory with legacy .dssp files")
    ap.add_argument("--mmcif-dir", required=True, help="Directory with matching .dssp files")
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

    for dssp_file in dssp_files:
        pdb_id = dssp_file.stem.lower()

        # Find corresponding mmCIF
        mmcif_path = None
        for ext in [".dssp",]:
            cand = mmcif_dir / f"{pdb_id}{ext}"
            print(f"Checking for mmCIF candidate: {cand}")
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

        # Compute DSSP-like labels from mmCIF
        mm_df = dssp_like_from_mmcif(mmcif_path)

        if mm_df.empty:
            print(f"[WARN] No protein residues found in {mmcif_path}, skipping merge.")
            continue

        merged = merge_leg_dssp_with_mmcif(leg_df, mm_df)

        # Keep only your polished columns up-front; keep AA_from_mmcif at the end for QC
        # Exact names to match your example:
        polished_cols = [
            "RESIDUE", "AA", "STRUCTURE_legacy", "BP1", "BP2", "ACC",
            "N-H-->O_1_i", "N-H-->O_1_E", "O-->H-N_1_i", "O-->H-N_1_E",
            "N-H-->O_2_i", "N-H-->O_2_E", "O-->H-N_2_i", "O-->H-N_2_E",
            "TCO", "KAPPA", "ALPHA", "PHI", "PSI", "X-CA", "Y-CA", "Z-CA",
            "pdb_id", "Chain", "DSSP_label"
        ]
        # Add AA_from_mmcif for debugging mismatches, if present
        optional = ["AA_from_mmcif", "icode"]
        out_cols = [c for c in polished_cols if c in merged.columns] + [c for c in optional if c in merged.columns]
        merged = merged[out_cols].copy()

        all_rows.append(merged)

    if not all_rows:
        raise SystemExit("Nothing to write: no merged rows produced.")

    final_df = pd.concat(all_rows, ignore_index=True)

    # Write CSV for easy inspection/debugging
    csv_path = out_path.with_suffix(".csv")
    final_df.to_csv(csv_path, index=False)
    print(f"Wrote intermediate CSV with {len(final_df):,} rows to {csv_path}")

    # Write parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(final_df):,} rows to {out_path}")


if __name__ == "__main__":
    print("Running aggregate_all.py...")
    main()
