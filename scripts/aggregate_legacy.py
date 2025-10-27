#!/usr/bin/env python3
"""
Build a polished Parquet dataset **only from legacy .dssp files**.

Secondary-structure label is taken as the first character of the legacy
STRUCTURE field (fallback to 'C' if missing or unknown).

Requires:
  pip install pandas pyarrow

Usage:
  python aggregate_all.py \
      --dssp-dir data/downloads/pc20_legacy \
      --out parquet/dssp_dataset.parquet
"""
import argparse
import re
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
from tqdm.auto import tqdm

_DSSP_ALLOWED = set(list("HETSBIGP"))  # allowed single-letter DSSP codes

# -------- helpers --------

RESNUM_RE = re.compile(r"^\s*(\d+)([A-Za-z]?)\s*$")  # capture resseq and optional icode
_FLOAT_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_HBOND_RE = re.compile(r"(-?\d+)\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))")

def parse_resnum(token: str) -> Tuple[Optional[int], str]:
    """Parse a residue number token that may contain an insertion code, e.g. '89A'."""
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

def label_from_legacy_structure(structure: Optional[str]) -> str:
    """First non-space char of STRUCTURE if in allowed set, else 'C'."""
    if not structure:
        return "C"
    s = structure.strip()
    if not s:
        return "C"
    ch = s[0]
    return ch if ch in _DSSP_ALLOWED else "C"

def parse_legacy_dssp_lines(lines: List[str], pdb_id: str) -> pd.DataFrame:
    """
    Parser for legacy DSSP (robust to missing whitespace before negative numbers).

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

    # Derive DSSP_label from STRUCTURE’s first character
    if not df.empty:
        df["DSSP_label"] = df["STRUCTURE"].map(label_from_legacy_structure)

    # Safe subset ordering
    cols = [
        "RESIDUE", "AA", "STRUCTURE", "BP1", "BP2", "ACC",
        "N-H-->O_1_i", "N-H-->O_1_E", "O-->H-N_1_i", "O-->H-N_1_E",
        "N-H-->O_2_i", "N-H-->O_2_E", "O-->H-N_2_i", "O-->H-N_2_E",
        "TCO", "KAPPA", "ALPHA", "PHI", "PSI", "X-CA", "Y-CA", "Z-CA",
        "pdb_id", "Chain", "icode", "DSSP_label"
    ]
    return df[[c for c in cols if c in df.columns]]

# -------- main pipeline --------

def main():
    print("Starting aggregation of DSSP data (legacy-only)...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dssp-dir", required=True, help="Directory with legacy .dssp files")
    # kept for backward compatibility; ignored
    ap.add_argument("--mmcif-dir", required=False, help="(ignored) Directory with mmCIF files")
    ap.add_argument("--out", required=True, help="Output Parquet file path")
    ap.add_argument("--suffix", default=".dssp", help="DSSP file suffix (default: .dssp)")
    args = ap.parse_args()

    dssp_dir = Path(args.dssp_dir)
    out_path = Path(args.out)
    print(f"Args: {args}")
    all_rows = []

    dssp_files = sorted(dssp_dir.glob(f"*{args.suffix}"))
    print(f"Found {len(dssp_files)} DSSP files in {dssp_dir} with suffix {args.suffix}")
    if not dssp_files:
        raise SystemExit(f"No DSSP files found in {dssp_dir} with suffix {args.suffix}")

    for dssp_file in tqdm(dssp_files, desc="Processing DSSP files", unit="file"):
        pdb_id = dssp_file.stem.lower()

        with open(dssp_file, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        leg_df = parse_legacy_dssp_lines(lines, pdb_id=pdb_id)

        if leg_df.empty:
            print(f"[WARN] No residues parsed from {dssp_file}, skipping.")
            continue

        all_rows.append(leg_df)

    if not all_rows:
        raise SystemExit("Nothing to write: no rows produced from legacy files.")

    final_df = pd.concat(all_rows, ignore_index=True)

    # Write head 1000 rows to CSV for easy inspection/debugging
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
