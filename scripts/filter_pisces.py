import pandas as pd

PISCES_REF = "/home/ubuntu/p2nd/data/metadata/cullpdb_pc20.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d2025_08_11_chains6495"
# TODO: remove some (5?) residues from both terminal

def get_pisces_pdb_chains(pisces_ref_path: str) -> tuple[set[str], set[str]]:
    pisces_df = pd.read_csv(pisces_ref_path, sep=r'\s+')
    id_chains = pisces_df["PDBchain"].tolist()
    id_chains_tuples = []

    discarded = 0
    for id_chain in id_chains:
        if len(id_chain) == 5:
            id_chains_tuples.append((id_chain[:4].lower(), id_chain[4].upper()))
        else:
            discarded += 1
    
    print(f"Discarded {discarded} invalid PDBchain entries")

    return id_chains_tuples

def filter_to_keep_pisces(aggregated_df, id_chains_to_keep) -> pd.DataFrame:
    print(f"Filtering: initial rows={len(aggregated_df):,}, unique pdb_ids={aggregated_df['pdb_id'].nunique():,}")
    filtered_df = aggregated_df[aggregated_df[["pdb_id", "Chain"]].apply(tuple, axis=1).isin(id_chains_to_keep)]
    print(f"Filtering: after filter pisces chain ids rows={len(filtered_df):,}, unique pdb_ids={filtered_df['pdb_id'].nunique():,}")
    return filtered_df

if __name__ == "__main__":

    result_path = "/home/ubuntu/p2nd/data/output/pc20_base/dssp_dataset.parquet"
    df = pd.read_parquet(result_path)

    print(f"Read {len(df):,} rows from {result_path}")

    id_chains_to_keep = get_pisces_pdb_chains(PISCES_REF)
    filtered_df = filter_to_keep_pisces(df, id_chains_to_keep)

    output_path = "/home/ubuntu/p2nd/data/output/pc20_base/dssp_dataset_pisces_filtered.parquet"
    filtered_df.to_parquet(output_path, index=False)