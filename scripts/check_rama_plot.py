import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRANSFORMED_PATH_X = f"/home/ubuntu/p2nd/data/output/pc20_vfull/dssp_dataset_transformed_X.parquet"
TRANSFORMED_PATH_Y = f"/home/ubuntu/p2nd/data/output/pc20_vfull/dssp_dataset_transformed_Y.parquet"
PLOT_PATH = "/home/ubuntu/p2nd/data/output/pc20_vfull/ramachandran_plot.png"

X = pd.read_parquet(TRANSFORMED_PATH_X)
y = pd.read_parquet(TRANSFORMED_PATH_Y)

# --- 1) Wrap angles to [-180, 180) so values like 360 become 0, etc.
def wrap_deg(a):
    return ((a + 180.0) % 360.0) - 180.0

df = X[['PHI', 'PSI']].copy()
df['PHI'] = wrap_deg(df['PHI'].astype(float))
df['PSI'] = wrap_deg(df['PSI'].astype(float))

# Attach DSSP labels (assumes indices align; if not, merge on your key)
df['DSSP_label'] = y['DSSP_label'].values

print(f"Total residues for Ramachandran plot: {len(df)}")

# --- 2) Optional: map labels to human-friendly groups
# Keep all DSSP codes but also a coarse group for legend clarity
coarse_map = {
    "B": "β-bridge",
    "C": "coil/other",
    "E": "β-strand",
    "G": "3₁₀ helix",
    "H": "α-helix",
    "I": "π-helix",
    "P": "PPII helix",
    "S": "bend",
    "T": "turn",
}

df['Group'] = df['DSSP_label'].map(coarse_map).fillna('Other')

# --- 3) Plot
plt.figure(figsize=(7, 7))

# Distinct groups for legend; matplotlib will cycle colors automatically
for group, sub in df.groupby('Group'):
    plt.scatter(
        sub['PHI'], sub['PSI'],
        s=2, alpha=1.0, label=group, edgecolors='none'
    )

plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.xticks([-180, -90, 0, 90, 180])
plt.yticks([-180, -90, 0, 90, 180])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linewidth=0.5, alpha=0.3)

plt.xlabel('ϕ (phi) [degrees]')
plt.ylabel('ψ (psi) [degrees]')
plt.title('Ramachandran Plot')
plt.legend(markerscale=2, fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=200)

print(f"Ramachandran plot saved to: {PLOT_PATH}")

# save only alpha helix plot
alpha_df = df[df['Group'] == 'α-helix']

print(f"Total alpha helix residues for Ramachandran plot: {len(alpha_df)}")
plt.figure(figsize=(7, 7))
plt.scatter(
    alpha_df['PHI'], alpha_df['PSI'],
    s=2, alpha=1.0, label='Alpha helix', edgecolors='none'
)
plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.xticks([-180, -90, 0, 90, 180])
plt.yticks([-180, -90, 0, 90, 180])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linewidth=0.5, alpha=0.3)
plt.xlabel('ϕ (phi) [degrees]')
plt.ylabel('ψ (psi) [degrees]')
plt.title('Ramachandran Plot - Alpha Helices')
plt.legend(markerscale=2, fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
plt.tight_layout()
alpha_plot_path = PLOT_PATH.replace('.png', '_alpha_helices.png')
plt.savefig(alpha_plot_path, dpi=200)
print(f"Alpha helix Ramachandran plot saved to: {alpha_plot_path}")

# only beta strand plot
beta_df = df[df['Group'] == 'β-strand']
print(f"Total beta strand residues for Ramachandran plot: {len(beta_df)}")
plt.figure(figsize=(7, 7))
plt.scatter(
    beta_df['PHI'], beta_df['PSI'],
    s=2, alpha=1.0, label='Beta strand', edgecolors='none'
)
plt.xlim(-180, 180)
plt.ylim(-180, 180) 
plt.xticks([-180, -90, 0, 90, 180])
plt.yticks([-180, -90, 0, 90, 180])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linewidth=0.5, alpha=0.3)
plt.xlabel('ϕ (phi) [degrees]')
plt.ylabel('ψ (psi) [degrees]')
plt.title('Ramachandran Plot - Beta Strands')
plt.legend(markerscale=2, fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
plt.tight_layout()
beta_plot_path = PLOT_PATH.replace('.png', '_beta_strands.png')
plt.savefig(beta_plot_path, dpi=200)
print(f"Beta strand Ramachandran plot saved to: {beta_plot_path}")  

# PPII helix plot
ppii_df = df[df['Group'] == 'PPII helix']
print(f"Total PPII helix residues for Ramachandran plot: {len(ppii_df)}")
plt.figure(figsize=(7, 7))
plt.scatter(
    ppii_df['PHI'], ppii_df['PSI'],
    s=2, alpha=1.0, label='PPII helix', edgecolors='none'
)
plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.xticks([-180, -90, 0, 90, 180])
plt.yticks([-180, -90, 0, 90, 180])
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linewidth=0.5, alpha=0.3)
plt.xlabel('ϕ (phi) [degrees]')
plt.ylabel('ψ (psi) [degrees]')
plt.title('Ramachandran Plot - PPII Helices')
plt.legend(markerscale=2, fontsize=8, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
plt.tight_layout()
ppii_plot_path = PLOT_PATH.replace('.png', '_ppii_helices.png')
plt.savefig(ppii_plot_path, dpi=200)
print(f"PPII helix Ramachandran plot saved to: {ppii_plot_path}")
