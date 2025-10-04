# A Fundamental Redo of Protein Structure Clustering:

* New and bigger dataset based on:
  * PISCES 20% Sim 2.5A Res Only XRay (Size=6495 Chains / 1528095 residue)
  * PISCES 40% Sim 2.5A Res Only XRay (Size=19054 Chains / 4755357 residue)
  * From Previous Small Size of = 3812 Chains / 22301 residue
* PISCES Link: 
  * https://dunbrack.fccc.edu/pisces/download/cullpdb_pc20.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d2025_08_11_chains6495
  * https://dunbrack.fccc.edu/pisces/download/cullpdb_pc20.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d2025_08_11_chains6495.fasta
  * https://dunbrack.fccc.edu/pisces/download/cullpdb_pc40.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d2025_08_11_chains19053
  * https://dunbrack.fccc.edu/pisces/download/cullpdb_pc40.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d2025_08_11_chains19053.fasta
* Start with H-Bond Categorized Pattern + Dihedral + LDDT


## Scripts
1. Check DSSP Legacy Availability
```bash
python scripts/check_pdbredo_legacy.py data/metadata/cullpdb_pc40.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d2025_08_11_chains19053
```
Reads a PISCES metadata file (whitespace-delimited with PDBchain column).

For each PDB ID, queries PDB-REDO DSSP service:
https://pdb-redo.eu/dssp/db/<pdbid>/legacy

Outputs a CSV with columns: http_code, legacy_exists, legacy_error.

2. Download DSSP Legacy Files
```bash
python scripts/download_pdbredo_legacy.py data/metadata/cullpdb_pc40.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d2025_08_11_chains19053 \
  --outdir data/downloads/pc40_legacy --jobs 8
```

Downloads available legacy DSSP files in parallel (configurable with --jobs).
Shows progress with tqdm (per-file bars + global counter).
Saves a log to download_results.csv with details:
pdbid, url, http_code, filepath, filesize, error.


3. Download DSSP mmCIF Files (For better DSSP Labels)
```bash
python scripts/download_pdbredo_mmcif.py data/metadata/cullpdb_pc40.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d2025_08_11_chains19053 \
  --outdir data/downloads/pc40_mmcif --jobs 8
``` 
4. Aggregate All Data into a Single Parquet File
```bash
python scripts/aggregate_all.py \
   --dssp-dir /home/ubuntu/p2nd/data/downloads/agg_test_legacy \
   --mmcif-dir /home/ubuntu/p2nd/data/downloads/agg_test_mmcif \
   --out /home/ubuntu/p2nd/data/output/agg_test/dssp_dataset.parquet
```
