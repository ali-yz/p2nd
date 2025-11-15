"""Check Result Cluster

Example usage:
python scripts/check_result_cluster.py --cluster 6 --data_version v5 --features_desc sincosphi_sincospsi_tco_hbondflags --algo hdbscan
"""

import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Cluster profiling tool")

    parser.add_argument("--cluster", type=int, required=True,
                        help="Cluster number to filter (e.g., 6)")
    parser.add_argument("--data_version", type=str, default="v5",
                        help="Data version (default: v5)")
    parser.add_argument("--features_desc", type=str, default="sincosphi_sincospsi_tco_hbondflags",
                        help="Features description")
    parser.add_argument("--algo", type=str, default="hdbscan",
                        help="Clustering algorithm")

    args = parser.parse_args()

    cluster_num = args.cluster
    data_version = args.data_version
    features_desc = args.features_desc
    algo = args.algo

    CLUSTER_PARQUET_PATH = f"data/output/pc20_{data_version}/{features_desc}/{algo}/clusters.parquet"
    PROFILE_OUTPUT_DIR = f"data/output/pc20_{data_version}/{features_desc}/{algo}/profile_base/"
    BASEDATA_PATH = "/home/ubuntu/p2nd/data/output/pc20_base_fixed/dssp_dataset_pisces_filtered.parquet"
    TRANSFORMED_X_DATA_PATH = f"/home/ubuntu/p2nd/data/output/pc20_{data_version}/dssp_dataset_transformed_X.parquet"
    TRANSFORMED_Y_DATA_PATH = f"/home/ubuntu/p2nd/data/output/pc20_{data_version}/dssp_dataset_transformed_Y.parquet"
    PLOT_TITLE_PREFIX = f"pc20_{data_version} | {features_desc} | {algo}"

    # Ensure output directory exists
    os.makedirs(PROFILE_OUTPUT_DIR, exist_ok=True)

    # Load data
    cluster_label = pd.read_parquet(CLUSTER_PARQUET_PATH)
    base_df = pd.read_parquet(BASEDATA_PATH)
    transformed_X = pd.read_parquet(TRANSFORMED_X_DATA_PATH)
    transformed_Y = pd.read_parquet(TRANSFORMED_Y_DATA_PATH)

    print(cluster_label.shape)
    print(base_df.shape)
    print(transformed_X.shape)
    print(transformed_Y.shape)

    # Merge dataframes with prefixes
    merged_df = pd.concat(
        [
            base_df.add_prefix("base_"),
            transformed_X.add_prefix("transformed_X_"),
            transformed_Y.add_prefix("transformed_Y_"),
            cluster_label.add_prefix("result_"),
        ],
        axis=1,
    )

    print(merged_df.shape)
    print(merged_df.columns)
    print(merged_df.head(2))

    print("***********")
    print(merged_df[merged_df["result_cluster"] == cluster_num].head(10))

    # Save filtered cluster rows
    output_path = os.path.join(PROFILE_OUTPUT_DIR, f"cluster_{cluster_num}_profile.csv")
    merged_df[merged_df["result_cluster"] == cluster_num].to_csv(output_path, index=False)

    print(f"Saved cluster {cluster_num} profile to: {output_path}")


if __name__ == "__main__":
    main()
