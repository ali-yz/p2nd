import pandas as pd

BASE_PATH = "/home/ubuntu/p2nd/data/output/pc20_base_fixed/dssp_dataset_pisces_filtered.parquet"

df = pd.read_parquet(BASE_PATH)

print("Header of the dataset:")
print(df.head())

# print value counts of DSSP_label
print("\nDSSP_label value counts:")
print(df['DSSP_label'].value_counts())

# filter and keep only rows with DSSP_label == 'H' (alpha helix)
helix_df = df[df['DSSP_label'] == 'H']

print(f"Total number of residues labeled as helix (H): {len(helix_df)}")

# filter rows with phi<0 and psi>+30
helix_df = helix_df[(helix_df['PHI'] < 0) & (helix_df['PSI'] > 30)]

print(f"Number of helix residues with phi<0 and psi>+30: {len(helix_df)}")

helix_df.to_csv("/home/ubuntu/p2nd/data/output/pc20_base_fixed/dssp_helix_issues.csv", index=False)
print("Saved helix residues to dssp_helix_issues.csv")

# filter and keep only rows with DSSP_label == 'E' (beta sheet)
sheet_df = df[df['DSSP_label'] == 'E']

print(f"Total number of residues labeled as sheet (E): {len(sheet_df)}")

# filter rows with phi<0 and psi<20
sheet_df = sheet_df[(sheet_df['PHI'] < 0) & (sheet_df['PSI'] < 20)]
print(f"Number of sheet residues with phi<0 and psi<20: {len(sheet_df)}")
sheet_df.to_csv("/home/ubuntu/p2nd/data/output/pc20_base_fixed/dssp_sheet_issues.csv", index=False)
print("Saved sheet residues to dssp_sheet_issues.csv")