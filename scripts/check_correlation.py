import pandas as pd

TRANSFORMED_PATH_X = f"/home/ubuntu/p2nd/data/output/pc20_vfull/dssp_dataset_transformed_X.parquet"
TRANSFORMED_PATH_Y = f"/home/ubuntu/p2nd/data/output/pc20_vfull/dssp_dataset_transformed_Y.parquet"

X = pd.read_parquet(TRANSFORMED_PATH_X)
y = pd.read_parquet(TRANSFORMED_PATH_Y)

print("header of X:")
print(X.head())

print("header of y:")
print(y.head())

# how many rows have 360.0 in PHI PSI KAPP ALPHA each?
for col in ["PHI", "PSI", "KAPPA", "ALPHA"]:
    num_360 = (X[col] == 360.0).sum()
    print(f"Number of rows with {col} == 360.0: {num_360} / {len(X)} ({num_360 / len(X):.4%})")
