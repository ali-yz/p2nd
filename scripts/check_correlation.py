import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# mutual information
from sklearn.feature_selection import mutual_info_regression

TRANSFORMED_PATH_X = f"/home/ubuntu/p2nd/data/output/pc20_vfull/dssp_dataset_transformed_X.parquet"
TRANSFORMED_PATH_Y = f"/home/ubuntu/p2nd/data/output/pc20_vfull/dssp_dataset_transformed_Y.parquet"
OUTPUT_FIG_PATH = f"/home/ubuntu/p2nd/data/output/pc20_vfull/feature_correlation_matrix.png"

X = pd.read_parquet(TRANSFORMED_PATH_X)
y = pd.read_parquet(TRANSFORMED_PATH_Y)

print("header of X:")
print(X.head())

print("header of y:")
print(y.head())

# how many rows have 360.0 in PHI PSI KAPP ALPHA each?
#for col in ["PHI", "PSI", "KAPPA", "ALPHA"]:
#    num_360 = (X[col] == 360.0).sum()
#    print(f"Number of rows with {col} == 360.0: {num_360} / {len(X)} ({num_360 / len(X):.4%})")

# draw the correlation matrix
corr = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": .8})
plt.title("Correlation Matrix of Transformed Features")
plt.tight_layout()
plt.savefig(OUTPUT_FIG_PATH)
print(f"Correlation matrix saved to {OUTPUT_FIG_PATH}")

# compute mutual information between pairs of features
def compute_mutual_info(X: pd.DataFrame) -> pd.DataFrame:
    """Compute mutual information between all pairs of features in X."""
    features = X.columns
    n = len(features)
    mi_matrix = pd.DataFrame(np.zeros((n, n)), index=features, columns=features)

    for i in range(n):
        for j in range(i, n):
            print(f"Computing MI between {features[i]} and {features[j]}", end=" ")
            if i == j:
                mi = 1.0  # Mutual information with itself
            else:
                mi = mutual_info_regression(X[[features[i]]], X[features[j]], discrete_features=False)
                mi = mi[0]  # mutual_info_regression returns an array
                print(f"=> MI={mi:.4f}", end="\n")
            mi_matrix.iloc[i, j] = mi
            mi_matrix.iloc[j, i] = mi  # Symmetric matrix

    return mi_matrix

mi_matrix = compute_mutual_info(X)
OUTPUT_MI_FIG_PATH = f"/home/ubuntu/p2nd/data/output/pc20_vfull/feature_mutual_information_matrix.png"
plt.figure(figsize=(10, 8))
sns.heatmap(mi_matrix, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"shrink": .8})
plt.title("Mutual Information Matrix of Transformed Features")
plt.tight_layout()
plt.savefig(OUTPUT_MI_FIG_PATH)
print(f"Mutual information matrix saved to {OUTPUT_MI_FIG_PATH}")
