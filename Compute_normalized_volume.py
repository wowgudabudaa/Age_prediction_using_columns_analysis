## This is to compute the normalized regional brain volume

# Read the volume data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from scipy.stats import zscore

# === Load raw regional volume data ===
vol_path = "../AD_Decode_Regional_Stats/AD_Decode_studywide_stats_for_volume.txt"
df_vol_raw = pd.read_csv(vol_path, sep="\t").iloc[1:]  # Remove header row used as data
df_vol_raw = df_vol_raw[df_vol_raw["ROI"] != "0"].reset_index(drop=True)

# === Extract subject columns and transpose for per-subject format ===
subject_cols = [col for col in df_vol_raw.columns if col.startswith("S")]
df_vol_transposed = df_vol_raw[subject_cols].transpose()
df_vol_transposed.columns = [f"Index_{i+1}" for i in range(df_vol_transposed.shape[1])]
df_vol_transposed.index.name = "subject_id"
df_vol_transposed = df_vol_transposed.astype(float)

# === Clean and standardize subject IDs to 5-digit format ===
cleaned_vol = {}
for subj in df_vol_transposed.index:
    match = re.search(r"S(\d{5})", subj)
    if match:
        subj_id = match.group(1)  # Extract e.g. "02842"
        cleaned_vol[subj_id] = df_vol_transposed.loc[subj]

# === Reconstruct volume DataFrame with subject ID index ===
df_vol_clean = pd.DataFrame.from_dict(cleaned_vol, orient="index")
df_vol_clean.index.name = "subject_id"

# Print the column names for reference
print("Volume DataFrame columns:")
print(df_vol_clean.columns.tolist())
# Read the index look up file
index_lookup_path = "../IITmean_RPI_lookup.xlsx"
index_lookup_df = pd.read_excel(index_lookup_path)
# The index_lookup_df contains the region names and the 'Index_i' in df_vol_clean means the i-th region in the index_lookup_df
# Create a mapping from index to region name
index_to_region = {f"Index_{i+1}": row['Structure'] for i, row in index_lookup_df.iterrows()}
# Map the column names in df_vol_clean to region names
df_vol_clean.columns = [index_to_region.get(col, col) for col in df_vol_clean.columns]
# Print the updated column names
print("Updated Volume DataFrame columns:")
print(df_vol_clean.columns.tolist())

# === Normalize volumes using z-score normalization ===
# Read whole brain volume data
whole_brain_vol_path = "./whole_brain_volume.csv"
df_whole_brain_vol = pd.read_csv(whole_brain_vol_path)
df_whole_brain_vol.set_index("subject", inplace=True)
# Ensure the index format matches
# The format of subject IDs in df_whole_brain_vol is e.g. 'J02842', so cut the 'J' or 'S' or 'T' prefix from df_vol_clean
df_whole_brain_vol.index = [str(subj_id).lstrip("JST") for subj_id in df_whole_brain_vol.index]
# Clean and standardize subject IDs to 5-digit format
df_vol_clean.index = df_vol_clean.index.str.zfill(5)
# Merge with regional volume data
df_merged = df_vol_clean.merge(df_whole_brain_vol[["nonzero_voxels"]], left_index=True, right_index=True, how="inner")
df_merged.rename(columns={"nonzero_voxels": "whole_brain_volume"}, inplace=True)
# Normalize each regional volume by whole brain volume and apply z-score normalization
df_normalized = df_merged.drop(columns=["whole_brain_volume"]).div(df_merged["whole_brain_volume"], axis=0)
# df_normalized = df_normalized.apply(zscore, axis=0)
# Save the normalized volume data
output_path = "./normalized_regional_volumes.csv"
df_normalized.to_csv(output_path)
print(f"Normalized regional volumes saved to {output_path}")
