import pandas as pd

PISCES_REF = "/home/ubuntu/p2nd/data/output/pc20_base_fixed/dssp_dataset_pisces_filtered.parquet"
TRANSFORMED_PATH_X = "/home/ubuntu/p2nd/data/output/pc20_v2/dssp_dataset_transformed_X.parquet"
TRANSFORMED_PATH_Y = "/home/ubuntu/p2nd/data/output/pc20_v2/dssp_dataset_transformed_Y.parquet"
TRANSFORMED_PATH_META = "/home/ubuntu/p2nd/data/output/pc20_v2/dssp_dataset_transformed_meta.parquet"

_FULL = [
    "RESIDUE", "AA", "STRUCTURE_legacy", "BP1", "BP2", "ACC",
    "N-H-->O_1_i", "N-H-->O_1_E", "O-->H-N_1_i", "O-->H-N_1_E",
    "N-H-->O_2_i", "N-H-->O_2_E", "O-->H-N_2_i", "O-->H-N_2_E",
    "TCO", "KAPPA", "ALPHA", "PHI", "PSI", "X-CA", "Y-CA", "Z-CA",
    "pdb_id", "Chain", "DSSP_label"
]

COLUMNS_TO_META = ["pdb_id", "Chain", "RESIDUE", "AA"]
COLUMNS_TO_KEEP = ["KAPPA", "ALPHA", "TCO"]
Y_COLUMN = "DSSP_label"

def transform_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Transforming: initial rows={len(df):,}, unique pdb_ids={df['pdb_id'].nunique():,}")
    
    # Separate features and target and metadata
    meta = df[COLUMNS_TO_META]
    X = df[COLUMNS_TO_KEEP]
    y = df[[Y_COLUMN]]
    
    print(f"Transforming: final X rows={len(X):,}, columns={X.shape[1]}, final y rows={len(y):,}, columns={y.shape[1]}, final meta rows={len(meta):,}, columns={meta.shape[1]}")
    
    return X, y, meta
if __name__ == "__main__":

    df = pd.read_parquet(PISCES_REF)

    print(f"Read {len(df):,} rows from {PISCES_REF}")

    X, y, meta = transform_dataset(df)

    X.to_parquet(TRANSFORMED_PATH_X, index=False)
    y.to_parquet(TRANSFORMED_PATH_Y, index=False)
    meta.to_parquet(TRANSFORMED_PATH_META, index=False)
    print(f"Saved transformed X to {TRANSFORMED_PATH_X} and y to {TRANSFORMED_PATH_Y} and meta to {TRANSFORMED_PATH_META}")
