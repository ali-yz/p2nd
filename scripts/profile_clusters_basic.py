"""Profile Clusters Basic
Example usage:
python scripts/profile_clusters_basic.py --data-version v5 --desc sincosphi_sincospsi_tco_hbondflags --algo hdbscan

"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data-version", default="v5")
parser.add_argument("--desc", default="sincosphi_sincospsi_tco_hbondflags")
parser.add_argument("--algo", default="hdbscan")
parser.add_argument("--only-combined-output", action='store_true', help="If set, only output combined profile figure")
args = parser.parse_args()

data_version = args.data_version
features_desc = args.desc
algo = args.algo
ONLY_COMBINED_OUTPUT = args.only_combined_output

CLUSTER_PARQUET_PATH = f"data/output/pc20_{data_version}/{features_desc}/{algo}/clusters.parquet"
PROFILE_OUTPUT_DIR = f"data/output/pc20_{data_version}/{features_desc}/{algo}/profile_base/"
BASEDATA_PATH = "/home/ubuntu/p2nd/data/output/pc20_base_fixed_dropped/dssp_dataset_pisces_filtered.parquet"
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

print(f"Loaded dataframes:")
print(f"  cluster_label: {cluster_label.shape}")
print(f"  base_df: {base_df.shape}")
print(f"  transformed_X: {transformed_X.shape}")
print(f"  transformed_Y: {transformed_Y.shape}")

# merge dataframes since they have same number of rows, add the name of the file as column prefix
merged_df = pd.concat([base_df.add_prefix('base_'),
                       transformed_X.add_prefix('transformed_X_'),
                       transformed_Y.add_prefix('transformed_Y_'),
                       cluster_label.add_prefix('result_')], axis=1)

print(f"  merged: {merged_df.shape}")
print(merged_df.columns)

# for each result_cluster plot base_PSI vs base_PHI
if not ONLY_COMBINED_OUTPUT:
    for cluster in merged_df['result_cluster'].unique():
        cluster_df = merged_df[merged_df['result_cluster'] == cluster]
        plt.figure(figsize=(7,7))
        plt.scatter(cluster_df['base_PHI'], cluster_df['base_PSI'], s=2, alpha=0.5)
        plt.xlim(-180, 180)
        plt.ylim(-180, 180)
        plt.xticks([-180, -90, 0, 90, 180])
        plt.yticks([-180, -90, 0, 90, 180])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linewidth=0.5, alpha=0.3)
        plt.xlabel('ϕ (phi) [degrees]')
        plt.ylabel('ψ (psi) [degrees]')
        plt.title(f'Ramachandran Plot - Cluster {cluster} Size {len(cluster_df)}\n{PLOT_TITLE_PREFIX}')
        plot_dir = os.path.join(PROFILE_OUTPUT_DIR, 'psi_vs_phi')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'{cluster}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot for cluster {cluster} to {plot_path}")

# for each result_cluster plot a five bar plot of percent with        'transformed_X_is_hbond_i-+3', 'transformed_X_is_hbond_i-+4',
#       'transformed_X_is_hbond_i-+5', 'transformed_X_is_hbond_i-+6',
#       'transformed_X_is_hbond_far'

if not ONLY_COMBINED_OUTPUT:
    for cluster in merged_df['result_cluster'].unique():
        cluster_df = merged_df[merged_df['result_cluster'] == cluster]
        hbond_cols = ['transformed_X_is_hbond_i-+3', 'transformed_X_is_hbond_i-+4',
                    'transformed_X_is_hbond_i-+5', 'transformed_X_is_hbond_i-+6',
                        'transformed_X_is_hbond_far']
        hbond_means = cluster_df[hbond_cols].mean() * 100  # percent
        plt.figure(figsize=(8,5))
        sns.barplot(x=hbond_means.index.str.replace('transformed_X_is_hbond_', ''),
                    y=hbond_means.values,
                    hue=hbond_means.index.str.replace('transformed_X_is_hbond_', ''),
                    legend=False, palette="viridis")
        plt.ylim(0, 100)
        plt.ylabel('Percentage (%)')
        plt.title(f'H-Bond Profile - Cluster {cluster} Size {len(cluster_df)}\n{PLOT_TITLE_PREFIX}')
        plot_dir = os.path.join(PROFILE_OUTPUT_DIR, 'hbond_profile')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'{cluster}_hbond_profile.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved H-bond profile plot for cluster {cluster} to {plot_path}")

# for each result_cluster plot histograms of base_ALPHAs
if not ONLY_COMBINED_OUTPUT:
    for cluster in merged_df['result_cluster'].unique():
        cluster_df = merged_df[merged_df['result_cluster'] == cluster]
        plt.figure(figsize=(8,5))
        sns.histplot(cluster_df['base_ALPHA'], bins=50, kde=True, color='skyblue')
        plt.xlabel('Alpha Angle [degrees]')
        plt.ylabel('Count')
        plt.title(f'Alpha Angle Distribution - Cluster {cluster} Size {len(cluster_df)}\n{PLOT_TITLE_PREFIX}')
        plot_dir = os.path.join(PROFILE_OUTPUT_DIR, 'alpha_histograms')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'{cluster}_alpha_histogram.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved Alpha angle histogram for cluster {cluster} to {plot_path}")

# for each result_cluster plot histograms of base_TCOs
if not ONLY_COMBINED_OUTPUT:
    for cluster in merged_df['result_cluster'].unique():
        cluster_df = merged_df[merged_df['result_cluster'] == cluster]
        plt.figure(figsize=(8,5))
        sns.histplot(cluster_df['base_TCO'], bins=50, kde=True, color='salmon')
        plt.xlabel('TCO Value')
        plt.ylabel('Count')
        plt.title(f'TCO Distribution - Cluster {cluster} Size {len(cluster_df)}\n{PLOT_TITLE_PREFIX}')
        plot_dir = os.path.join(PROFILE_OUTPUT_DIR, 'tco_histograms')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'{cluster}_tco_histogram.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved TCO histogram for cluster {cluster} to {plot_path}")

# combine all cluster profiles into a single figure
num_clusters = len(merged_df['result_cluster'].unique())
fig, axes = plt.subplots(num_clusters, 4, figsize=(14, 5 * num_clusters))
for i, cluster in enumerate(sorted(merged_df['result_cluster'].unique())):
    cluster_df = merged_df[merged_df['result_cluster'] == cluster]
    # save a 10000 sample of the cluster_df to csv for external analysis
    sample_csv_path = os.path.join(PROFILE_OUTPUT_DIR, f'cluster_{cluster}_sample.csv')
    cluster_df.sample(n=min(10000, len(cluster_df)), random_state=42).to_csv(sample_csv_path, index=False)
    print(f"Saved sample of cluster {cluster} to {sample_csv_path}")

    # Ramachandran plot
    axes[i, 0].scatter(cluster_df['base_PHI'], cluster_df['base_PSI'], s=2, alpha=0.5)
    axes[i, 0].set_xlim(-180, 180)
    axes[i, 0].set_ylim(-180, 180)
    axes[i, 0].set_xticks([-180, -90, 0, 90, 180])
    axes[i, 0].set_yticks([-180, -90, 0, 90, 180])
    axes[i, 0].set_aspect('equal', adjustable='box')
    axes[i, 0].grid(True, linewidth=0.5, alpha=0.3)
    axes[i, 0].set_xlabel('ϕ (phi) [degrees]')
    axes[i, 0].set_ylabel('ψ (psi) [degrees]')
    axes[i, 0].set_title(f'Ramachandran Plot - Cluster {cluster} Size {len(cluster_df)}')
    
    # H-bond profile
    hbond_cols = ['transformed_X_is_hbond_i-+3', 'transformed_X_is_hbond_i-+4',
                  'transformed_X_is_hbond_i-+5', 'transformed_X_is_hbond_i-+6',
                    'transformed_X_is_hbond_far']
    # check if there are any hbond columns in the dataframe
    if not all(col in cluster_df.columns for col in hbond_cols):
        print(f"Skipping H-bond profile for cluster {cluster} as not all hbond columns are present.")
        hbond_means = pd.Series([0]*5, index=hbond_cols)
    else:
        hbond_means = cluster_df[hbond_cols].mean() * 100  # percent
    sns.barplot(x=hbond_means.index.str.replace('transformed_X_is_hbond_', ''),
                y=hbond_means.values,
                hue=hbond_means.index.str.replace('transformed_X_is_hbond_', ''),
                legend=False, palette="viridis", ax=axes[i, 1])
    axes[i, 1].set_ylim(0, 100)
    axes[i, 1].set_ylabel('Percentage (%)')
    axes[i, 1].set_title(f'H-Bond Profile - Cluster {cluster} Size {len(cluster_df)}')

    # Alpha angle histogram
    sns.histplot(cluster_df['base_ALPHA'], bins=50, kde=True, color='skyblue', ax=axes[i, 2])
    axes[i, 2].set_xlabel('Alpha Angle [degrees]')
    axes[i, 2].set_ylabel('Count')
    axes[i, 2].set_title(f'Alpha Angle Distribution - Cluster {cluster} Size {len(cluster_df)}')

    # TCO histogram
    sns.histplot(cluster_df['base_TCO'], bins=50, kde=True, color='salmon', ax=axes[i, 3])
    axes[i, 3].set_xlabel('TCO Value')
    axes[i, 3].set_ylabel('Count')
    axes[i, 3].set_title(f'TCO Distribution - Cluster {cluster} Size {len(cluster_df)}')

plt.suptitle(f'Cluster Profiles\n{PLOT_TITLE_PREFIX}', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
combined_plot_path = os.path.join(PROFILE_OUTPUT_DIR, f'combined_cluster_profiles_{features_desc}_{algo}_{data_version}.png')
plt.savefig(combined_plot_path, dpi=300)
plt.close()
print(f"Saved combined cluster profiles to {combined_plot_path}")
