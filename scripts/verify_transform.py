import pandas as pd

if __name__ == "__main__":

    x_path = "/home/ubuntu/p2nd/data/output/pc20_v1/dssp_dataset_transformed_X.parquet"
    y_path = "/home/ubuntu/p2nd/data/output/pc20_v1/dssp_dataset_transformed_Y.parquet"

    df = pd.read_parquet(x_path)
    print(f"Read {len(df):,} rows from {x_path}")
    print(df.head())

    print("####")
    df_y = pd.read_parquet(y_path)
    print(f"Read {len(df_y):,} rows from {y_path}")
    print(df_y.head())

    print("####")
    print(df.shape, df_y.shape)

    print("####")
    print(df_y.value_counts())