import pandas as pd
import numpy as np

PISCES_REF = "/home/ubuntu/p2nd/data/output/pc20_base_fixed/dssp_dataset_pisces_filtered.parquet"
TRANSFORMED_PATH_X = "/home/ubuntu/p2nd/data/output/pc20_v5/dssp_dataset_transformed_X.parquet"
TRANSFORMED_PATH_Y = "/home/ubuntu/p2nd/data/output/pc20_v5/dssp_dataset_transformed_Y.parquet"
TRANSFORMED_PATH_META = "/home/ubuntu/p2nd/data/output/pc20_v5/dssp_dataset_transformed_meta.parquet"

_FULL = [
    "RESIDUE", "AA", "STRUCTURE_legacy", "BP1", "BP2", "ACC",
    "N-H-->O_1_i", "N-H-->O_1_E", "O-->H-N_1_i", "O-->H-N_1_E",
    "N-H-->O_2_i", "N-H-->O_2_E", "O-->H-N_2_i", "O-->H-N_2_E",
    "TCO", "KAPPA", "ALPHA", "PHI", "PSI", "X-CA", "Y-CA", "Z-CA",
    "pdb_id", "Chain", "DSSP_label"
]

COLUMNS_TO_META = ["pdb_id", "Chain", "RESIDUE", "AA"]
COLUMNS_TO_KEEP = ["PHI", "PSI", "TCO"]   # in X (PHI/PSI -> sin/cos; TCO raw)
ANGLE_COLUMNS = ["PHI", "PSI"]            # sin/cos only for these
Y_COLUMN = "DSSP_label"

# Hydrogen bond columns: (offset, energy) pairs we should scan
HBOND_PAIRS = [
    ("N-H-->O_1_i", "N-H-->O_1_E"),
    ("N-H-->O_2_i", "N-H-->O_2_E"),
    ("O-->H-N_1_i", "O-->H-N_1_E"),
    ("O-->H-N_2_i", "O-->H-N_2_E"),
]

HBOND_OFFSETS_TO_FLAG = (3, 4, 5, 6)
HBOND_ENERGY_THRESHOLD = -0.5  # kcal/mol

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Safely coerce possibly-string columns to numeric."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_hbond_flags(df: pd.DataFrame,
                        pairs=HBOND_PAIRS,
                        offsets=HBOND_OFFSETS_TO_FLAG,
                        energy_thresh=HBOND_ENERGY_THRESHOLD) -> pd.DataFrame:
    """
    Build binary columns:
      is_hbond_i-+{3,4,5,6} = 1 if any H-bond entry has |offset| == n and energy < threshold
      is_hbond_far          = 1 if any H-bond entry has |offset| >= 7 and energy < threshold
    Notes:
      - Offset==0 or NaN => treated as 'no bond'
      - Combines donor/acceptor and best/second best
    """
    # Ensure numeric types
    offset_cols = [p[0] for p in pairs]
    energy_cols = [p[1] for p in pairs]
    df = _coerce_numeric(df, offset_cols + energy_cols)

    # Prepare per-pair valid-bond masks (energy < thresh and non-zero offset)
    per_pair = []
    for i_col, e_col in pairs:
        # Conditions: valid energy, valid non-zero offset, passes threshold
        # Using fillna to simplify vectorized comparisons
        e = df[e_col].fillna(np.inf)  # inf will fail (>= thresh)
        o = df[i_col].fillna(0).astype(float)
        valid = (o != 0) & (e < energy_thresh)
        per_pair.append((o, valid))

    # Helper to OR across pairs for a given offset predicate
    def any_pair(predicate):
        mask = pd.Series(False, index=df.index)
        for o, valid in per_pair:
            mask = mask | (valid & predicate(o))
        return mask

    out = pd.DataFrame(index=df.index)

    # Specific ±n flags
    for n in offsets:
        colname = f"is_hbond_i-+{n}"
        out[colname] = any_pair(lambda o: np.abs(o) == n).astype(np.uint8)

    # Far flag: |offset| >= 7 (i.e., not caught by 3..6)
    out["is_hbond_far"] = any_pair(lambda o: np.abs(o) >= 7).astype(np.uint8)

    return out

def drop_tco_zero_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where TCO is exactly zero as a way to clean low quality/missing or intrinsically disordered residues."""
    if "TCO" not in df.columns:
        print("TCO column not found in dataframe; skipping drop_tco_zero_rows.")
        return df
    
    initial_count = len(df)
    df_cleaned = df[df["TCO"] != 0.0].copy()
    final_count = len(df_cleaned)
    print(f"Dropped {initial_count - final_count} rows with TCO == 0.0; remaining rows: {final_count}")
    return df_cleaned

def transform_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Transforming: initial rows={len(df):,}, unique pdb_ids={df['pdb_id'].nunique():,}")

    # Separate features, target, and metadata
    meta = df[COLUMNS_TO_META].copy()
    y = df[[Y_COLUMN]].copy()

    # Base X (angles + TCO)
    X_blocks = []
    for col in COLUMNS_TO_KEEP:
        if col in ANGLE_COLUMNS:
            rad = np.deg2rad(df[col])
            X_blocks.append(np.sin(rad).rename(f"{col}_sin").to_frame())
            X_blocks.append(np.cos(rad).rename(f"{col}_cos").to_frame())
        else:
            X_blocks.append(df[[col]])  # keep raw (e.g., TCO)

    # Add hydrogen bond pattern flags
    hb_flags = compute_hbond_flags(df)
    X_blocks.append(hb_flags)

    # Concatenate all features
    X = pd.concat(X_blocks, axis=1)

    print(
        f"Transforming: final X rows={len(X):,}, columns={X.shape[1]}, "
        f"final y rows={len(y):,}, columns={y.shape[1]}, "
        f"final meta rows={len(meta):,}, columns={meta.shape[1]}"
    )
    return X, y, meta

if __name__ == "__main__":
    df = pd.read_parquet(PISCES_REF)
    print(f"Read {len(df):,} rows from {PISCES_REF}")

    # Clean data by dropping rows with TCO == 0.0
    df = drop_tco_zero_rows(df)

    X, y, meta = transform_dataset(df)

    # write the head of the original file as csv for debugging
    original_head_path = PISCES_REF.replace(".parquet", "_head.csv")
    df.head(100).to_csv(original_head_path, index=False)
    print(f"Wrote head 100 rows of original data to {original_head_path}")
    
    # write the head of files as csv for debugging
    X_head_path = TRANSFORMED_PATH_X.replace(".parquet", ".csv")
    y_head_path = TRANSFORMED_PATH_Y.replace(".parquet", ".csv")
    meta_head_path = TRANSFORMED_PATH_META.replace(".parquet", ".csv")

    X.head(100).to_csv(X_head_path, index=False)
    y.head(100).to_csv(y_head_path, index=False)
    meta.head(100).to_csv(meta_head_path, index=False)
    print(f"Wrote head 100 rows of X to {X_head_path}, y to {y_head_path}, meta to {meta_head_path}")

    X.to_parquet(TRANSFORMED_PATH_X, index=False)
    y.to_parquet(TRANSFORMED_PATH_Y, index=False)
    meta.to_parquet(TRANSFORMED_PATH_META, index=False)
    print(f"Saved transformed X to {TRANSFORMED_PATH_X} and y to {TRANSFORMED_PATH_Y} and meta to {TRANSFORMED_PATH_META}")
