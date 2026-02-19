import pandas as pd


# v5 files
# data/output/pc20_v5/dssp_dataset_transformed_meta.parquet
# data/output/pc20_v5/dssp_dataset_transformed_Y.parquet
# data/output/pc20_v5/sincosphi_sincospsi_tco_hbondflags/agglomerative/clusters.parquet
v5_meta = pd.read_parquet("/home/ubuntu/p2nd/data/output/pc20_v5/dssp_dataset_transformed_meta.parquet")
v5_y = pd.read_parquet("/home/ubuntu/p2nd/data/output/pc20_v5/dssp_dataset_transformed_Y.parquet")
v5_clusters = pd.read_parquet("/home/ubuntu/p2nd/data/output/pc20_v5/sincosphi_sincospsi_tco_hbondflags/agglomerative/clusters.parquet")

print(v5_meta.shape)
print(v5_y.shape)
print(v5_clusters.shape)

# v6 files
# data/output/pc20_v6/dssp_dataset_transformed_meta.parquet
# data/output/pc20_v6/dssp_dataset_transformed_Y.parquet
# data/output/pc20_v6/sincosphi_sincospsi_sincosalpha_hbondflags/agglomerative/clusters.parquet
v6_meta = pd.read_parquet("/home/ubuntu/p2nd/data/output/pc20_v6/dssp_dataset_transformed_meta.parquet")
v6_y = pd.read_parquet("/home/ubuntu/p2nd/data/output/pc20_v6/dssp_dataset_transformed_Y.parquet")
v6_clusters = pd.read_parquet("/home/ubuntu/p2nd/data/output/pc20_v6/sincosphi_sincospsi_sincosalpha_hbondflags/agglomerative/clusters.parquet")

print(v6_meta.shape)
print(v6_y.shape)
print(v6_clusters.shape)

# v7 files
# data/output/pc20_v7/dssp_dataset_transformed_meta.parquet
# data/output/pc20_v7/dssp_dataset_transformed_Y.parquet
# data/output/pc20_v7/sincosphi_sincospsi_hbondflags/agglomerative/clusters.parquet
v7_meta = pd.read_parquet("/home/ubuntu/p2nd/data/output/pc20_v7/dssp_dataset_transformed_meta.parquet")
v7_y = pd.read_parquet("/home/ubuntu/p2nd/data/output/pc20_v7/dssp_dataset_transformed_Y.parquet")
v7_clusters = pd.read_parquet("/home/ubuntu/p2nd/data/output/pc20_v7/sincosphi_sincospsi_hbondflags/agglomerative/clusters.parquet")

print(v7_meta.shape)
print(v7_y.shape)
print(v7_clusters.shape)


# Merge dfs for each data
df_v5 = v5_meta.merge(v5_y, left_index=True, right_index=True).merge(v5_clusters, left_index=True, right_index=True)
df_v6 = v6_meta.merge(v6_y, left_index=True, right_index=True).merge(v6_clusters, left_index=True, right_index=True)
df_v7 = v7_meta.merge(v7_y, left_index=True, right_index=True).merge(v7_clusters, left_index=True, right_index=True)

print(df_v5.shape)
print(df_v6.shape)
print(df_v7.shape)

print("-"*50)

# Print only rows for v6 where col pdb_id is 2pne
print(df_v6[df_v6['pdb_id'] == '2pne'])

# Save it to file
df_v6[df_v6['pdb_id'] == '2pne'].to_csv("/home/ubuntu/p2nd/data/output/validations/2pne_v6_validation.csv", index=False)

# 8hui
print(df_v6[df_v6['pdb_id'] == '8hui'])
df_v6[df_v6['pdb_id'] == '8hui'].to_csv("/home/ubuntu/p2nd/data/output/validations/8hui_v6_validation.csv", index=False)