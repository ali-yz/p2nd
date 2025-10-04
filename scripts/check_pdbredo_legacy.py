#!/usr/bin/env python3
"""
check_pdbredo_legacy.py

Usage:
    python3 check_pdbredo_legacy.py input.csv [output.csv] [-j N]

Reads a whitespace-delimited CSV with a header containing a "PDBchain" column
(e.g. "5D8VA", "3NIRA"), checks the PDB-REDO DSSP legacy endpoint
https://pdb-redo.eu/dssp/db/<pdbid>/legacy using curl, and writes a CSV with
http_code and legacy_exists columns.

Requires: Python 3.6+, curl installed.
"""
import sys
import subprocess
import concurrent.futures
import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Check PDB-REDO DSSP legacy availability with curl")
    p.add_argument("infile", help="Input CSV (whitespace-delimited) with a PDBchain column")
    p.add_argument("outfile", nargs="?", default="legacy_check_results.csv",
                   help="Output CSV (default: legacy_check_results.csv)")
    p.add_argument("-j", "--jobs", type=int, default=8, help="Parallel curl jobs (default: 8)")
    return p.parse_args()

def normalize_pdbid(pdbchain):
    """Extract 4-char pdb id (lowercase) from PDBchain like '5D8VA' or '1ABC_A'."""
    if pdbchain is None:
        return None
    s = str(pdbchain).strip()
    if len(s) >= 4:
        return s[:4].lower()
    return None

def check_legacy_with_curl(pdbid, timeout=20):
    """
    Return a tuple (pdbid, http_code_str, exists_bool, err_msg)
    Uses: curl -s -o /dev/null -w "%{http_code}" -I -L <url>
    """
    url = f"https://pdb-redo.eu/dssp/db/{pdbid}/legacy"
    cmd = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-I", "-L", url]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        code = proc.stdout.strip()
        # Sometimes curl returns empty stdout on failure; fall back to returncode
        if code == "":
            # use returncode: 0 usually means success, but no header, mark as '000'
            code = str(proc.returncode).zfill(3)
        exists = (code == "200")
        return pdbid, code, exists, ""
    except subprocess.TimeoutExpired:
        return pdbid, "TMO", False, "timeout"
    except Exception as e:
        return pdbid, "ERR", False, str(e)

def main():
    args = parse_args()
    infile = Path(args.infile)
    outfile = Path(args.outfile)
    if not infile.exists():
        print(f"ERROR: input file not found: {infile}", file=sys.stderr)
        sys.exit(2)

    # Read input using whitespace delimiter (handles your pasted example)
    try:
        df = pd.read_csv(infile, sep=r'\s+', engine='python', dtype=str)
    except Exception as e:
        print("Failed to parse input file with whitespace delimiter. Error:", e, file=sys.stderr)
        print("Try providing a proper CSV with a PDBchain column.", file=sys.stderr)
        sys.exit(3)

    if 'PDBchain' not in df.columns and 'PDB_chain' not in df.columns:
        # try case-insensitive match
        cols = {c.lower(): c for c in df.columns}
        if 'pdbchain' in cols:
            df.rename(columns={cols['pdbchain']: 'PDBchain'}, inplace=True)
        else:
            print("ERROR: no 'PDBchain' column found in input. Columns:", df.columns.tolist(), file=sys.stderr)
            sys.exit(4)

    # normalize and deduplicate pdb ids
    df['pdbid'] = df['PDBchain'].apply(normalize_pdbid)
    # warn about rows where pdbid could not be extracted
    missing = df['pdbid'].isna().sum()
    if missing:
        print(f"Warning: {missing} rows have no valid pdbid extracted from PDBchain column.")

    unique_pdbids = sorted(df['pdbid'].dropna().unique())
    print(f"Found {len(unique_pdbids)} unique PDB IDs to check.")

    # parallel check
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = { ex.submit(check_legacy_with_curl, pdbid): pdbid for pdbid in unique_pdbids }
        for fut in concurrent.futures.as_completed(futures):
            pdbid = futures[fut]
            try:
                pid, code, exists, err = fut.result()
                results[pid] = {"http_code": code, "legacy_exists": exists, "error": err}
            except Exception as e:
                results[pdbid] = {"http_code": "ERR", "legacy_exists": False, "error": str(e)}

    # map back to dataframe
    df['http_code'] = df['pdbid'].map(lambda x: results.get(x, {}).get('http_code') if x else None)
    df['legacy_exists'] = df['pdbid'].map(lambda x: results.get(x, {}).get('legacy_exists') if x else False)
    df['legacy_error'] = df['pdbid'].map(lambda x: results.get(x, {}).get('error') if x else None)

    # Save results
    df.to_csv(outfile, index=False)
    print(f"Wrote results to {outfile}")

    # summary
    total = len(df)
    exists_count = df['legacy_exists'].sum()
    unique_exists = sum(1 for v in results.values() if v['legacy_exists'])
    print(f"Summary: rows={total}, rows-with-legacy=True={exists_count}, unique-pdbs-with-legacy={unique_exists}")

    # also print list of pdbs with legacy available
    if unique_exists:
        available = [pid for pid, rv in results.items() if rv['legacy_exists']]
        print("PDBs with legacy available:", ", ".join(sorted(available)))

if __name__ == "__main__":
    main()
