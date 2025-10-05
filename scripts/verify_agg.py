import pandas as pd

if __name__ == "__main__":

    result_path = "/home/ubuntu/p2nd/data/output/pc20_v1/dssp_dataset.parquet"
    df = pd.read_parquet(result_path)
    print(f"Read {len(df):,} rows from {result_path}")
    print(df.head())
    print(df.columns)
    print("####")
    print(df["DSSP_label"].value_counts())
    print("####")
    print(df["AA_from_mmcif"].value_counts())
    print("####")
    print(df["AA"].value_counts())