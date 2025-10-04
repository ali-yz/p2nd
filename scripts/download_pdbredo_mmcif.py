#!/usr/bin/env python3
"""
download_pdbredo_mmcif.py

Parallel download of pdb-redo mmcif DSSP files with tqdm.

Usage:
    python download_pdbredo_mmcif.py input.txt --outdir mmcif_dssp --jobs 6

Input: whitespace-delimited file with a "PDBchain" column (like your sample).
"""
from __future__ import print_function
import argparse
import os
import sys
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import queue
import re
import urllib.parse

def parse_args():
    p = argparse.ArgumentParser(description="Parallel download of pdb-redo mmcif DSSP files with tqdm")
    p.add_argument("input", help="Input whitespace-delimited CSV with PDBchain column")
    p.add_argument("--outdir", default="mmcif_dssp", help="Output directory (default: mmcif_dssp)")
    p.add_argument("--jobs", "-j", type=int, default=8, help="Parallel download workers (default 8)")
    p.add_argument("--log", default="download_results.csv", help="CSV log file (default download_results.csv)")
    p.add_argument("--timeout", type=int, default=60, help="Per-request timeout seconds (default 60)")
    return p.parse_args()

def normalize_pdbid(pdbchain):
    if pd.isna(pdbchain):
        return None
    s = str(pdbchain).strip()
    if len(s) >= 4:
        return s[:4].lower()
    return None

def requests_session_with_retries(total_retries=5, backoff=0.5):
    s = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(['HEAD', 'GET', 'OPTIONS'])
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def filename_from_content_disposition(cd):
    """
    Parse Content-Disposition header for filename if present.
    Returns filename (str) or None.
    Handles filename* (encoded) and filename.
    """
    if not cd:
        return None
    # Try filename*= (e.g. UTF-8''name%20with%20spaces)
    m = re.search(r"filename\*\s*=\s*([^;]+)", cd, flags=re.IGNORECASE)
    if m:
        val = m.group(1).strip().strip('"\'')
        # If value contains encoding//lang''encoded_name
        if "''" in val:
            parts = val.split("''", 1)
            encoded = parts[1]
            try:
                return urllib.parse.unquote(encoded)
            except Exception:
                return encoded
        else:
            try:
                return urllib.parse.unquote(val)
            except Exception:
                return val

    # Fallback to filename="..."
    m2 = re.search(r'filename\s*=\s*"([^"]+)"', cd, flags=re.IGNORECASE)
    if m2:
        return m2.group(1)
    m3 = re.search(r"filename\s*=\s*([^;]+)", cd, flags=re.IGNORECASE)
    if m3:
        return m3.group(1).strip().strip('"\'')
    return None

def download_single(pdbid, outdir, timeout, pos_queue):
    url = f"https://pdb-redo.eu/dssp/db/{pdbid}/mmcif"
    result = {"pdbid": pdbid, "url": url, "http_code": None, "filepath": None, "filesize": 0, "error": None}
    session = requests_session_with_retries()
    pos = None
    try:
        # HEAD to discover headers & existence
        head = session.head(url, allow_redirects=True, timeout=min(10, timeout))
        code = head.status_code
        result["http_code"] = str(code)
        if code != 200:
            result["error"] = f"HTTP {code}"
            return result

        # determine filename
        cd = head.headers.get("content-disposition")
        fname = filename_from_content_disposition(cd)
        if not fname:
            fname = f"{pdbid}.dssp"
        outpath = outdir / fname

        # get a tqdm position slot (1..jobs)
        try:
            pos = pos_queue.get(timeout=5)
        except Exception:
            pos = None

        # stream download with a per-file tqdm bar
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            total = r.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            tmp_path = outdir / (fname + ".part")
            chunk_size = 8192

            perfile_tq = tqdm(total=total, unit="B", unit_scale=True, desc=pdbid,
                              position=(pos if pos is not None else None), leave=False, ascii=True)
            bytes_written = 0
            with open(tmp_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
                        bytes_written += len(chunk)
                        perfile_tq.update(len(chunk))
            perfile_tq.close()

            os.replace(tmp_path, outpath)
            result["filepath"] = str(outpath.resolve())
            result["filesize"] = bytes_written
            result["http_code"] = str(r.status_code)
            return result

    except requests.exceptions.RequestException as e:
        result["error"] = f"requests error: {e}"
        return result
    except Exception as e:
        result["error"] = str(e)
        return result
    finally:
        if pos is not None:
            try:
                pos_queue.put(pos)
            except Exception:
                pass

def main():
    args = parse_args()
    infile = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not infile.exists():
        print("Input file not found:", infile, file=sys.stderr)
        sys.exit(2)

    # read whitespace-delimited input
    try:
        df = pd.read_csv(infile, sep=r'\s+', engine='python', dtype=str)
    except Exception as e:
        print("Failed to parse input file. Ensure it's whitespace-delimited and has a PDBchain column.", e, file=sys.stderr)
        sys.exit(3)

    if 'PDBchain' not in df.columns:
        cols = {c.lower(): c for c in df.columns}
        if 'pdbchain' in cols:
            df.rename(columns={cols['pdbchain']: 'PDBchain'}, inplace=True)
        else:
            print("ERROR: No PDBchain column found. Columns are:", df.columns.tolist(), file=sys.stderr)
            sys.exit(4)

    df['pdbid'] = df['PDBchain'].apply(normalize_pdbid)
    pdbids = sorted(df['pdbid'].dropna().unique())
    if not pdbids:
        print("No PDB ids found in input.", file=sys.stderr)
        sys.exit(5)

    print(f"Found {len(pdbids)} unique PDB ids. Downloading with {args.jobs} workers...")

    # allocate positions for per-file bars (position 0 reserved for main bar)
    pos_queue = queue.Queue()
    for i in range(1, max(1, args.jobs) + 1):
        pos_queue.put(i)

    results = []
    main_bar = tqdm(total=len(pdbids), desc="files", position=0, leave=True, ascii=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = { ex.submit(download_single, pid, outdir, args.timeout, pos_queue): pid for pid in pdbids }
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"pdbid": pid, "url": f"https://pdb-redo.eu/dssp/db/{pid}/mmcif", "http_code": None, "filepath": None, "filesize": 0, "error": str(e)}
            results.append(res)
            main_bar.update(1)
    main_bar.close()

    outlog = Path(args.log)
    pd.DataFrame(results).sort_values("pdbid").to_csv(outlog, index=False)
    print(f"Wrote log to {outlog}")
    succ = [r for r in results if r.get("filepath")]
    failed = [r for r in results if not r.get("filepath")]
    print(f"Success: {len(succ)} downloaded. Failed: {len(failed)}.")
    if failed:
        print("Failed entries (pdbid, error):")
        for r in failed:
            print(" ", r.get("pdbid"), r.get("error"))

if __name__ == "__main__":
    main()
