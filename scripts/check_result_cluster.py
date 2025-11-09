import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm


data_version = "v5"
features_desc = "sincosphi_sincospsi_tco_hbondflags"
algo = "hdbscan"
CLUSTER_PARQUET_PATH = f"data/output/pc20_{data_version}/{features_desc}/{algo}/clusters.parquet"
PROFILE_OUTPUT_DIR = f"data/output/pc20_{data_version}/{features_desc}/{algo}/profile_base/"
BASEDATA_PATH = "/home/ubuntu/p2nd/data/output/pc20_base_fixed/dssp_dataset_pisces_filtered.parquet"
TRANSFORMED_X_DATA_PATH = f"/home/ubuntu/p2nd/data/output/pc20_{data_version}/dssp_dataset_transformed_X.parquet"
TRANSFORMED_Y_DATA_PATH = f"/home/ubuntu/p2nd/data/output/pc20_{data_version}/dssp_dataset_transformed_Y.parquet"
PLOT_TITLE_PREFIX = f"pc20_{data_version} | {features_desc} | {algo}"

# make sure output directory exists
os.makedirs(PROFILE_OUTPUT_DIR, exist_ok=True)

## SET FONTS
preferred = ["Roboto", "Roboto Regular", "Roboto Condensed", "Roboto Slab", "Roboto Mono"]
available = {f.name for f in fm.fontManager.ttflist}
for fam in preferred:
    if fam in available:
        mpl.rcParams["font.family"] = fam
        break
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"]  = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["mathtext.fontset"] = "stixsans"  # STIX sans pairs reasonably with Roboto
## END SET FONTS

cluster_label = pd.read_parquet(CLUSTER_PARQUET_PATH)
base_df = pd.read_parquet(BASEDATA_PATH)
transformed_X = pd.read_parquet(TRANSFORMED_X_DATA_PATH)
transformed_Y = pd.read_parquet(TRANSFORMED_Y_DATA_PATH)

print(cluster_label.shape)
print(base_df.shape)
print(transformed_X.shape)
print(transformed_Y.shape)

# merge dataframes since they have same number of rows, add the name of the file as column prefix
merged_df = pd.concat([base_df.add_prefix('base_'),
                       transformed_X.add_prefix('transformed_X_'),
                       transformed_Y.add_prefix('transformed_Y_'),
                       cluster_label.add_prefix('result_')], axis=1)

print(merged_df.shape)
print(merged_df.columns)
print(merged_df.head(2))

print("***********")
# print head of merged_df with result_cluster==6
print(merged_df[merged_df['result_cluster'] == 6].head(10))

# save to csv file
merged_df[merged_df['result_cluster'] == 6].to_csv(os.path.join(PROFILE_OUTPUT_DIR, "cluster_6_profile.csv"), index=False)