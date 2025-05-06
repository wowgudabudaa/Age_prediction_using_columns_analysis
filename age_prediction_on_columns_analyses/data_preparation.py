#%% md
# # Prepare data for age prediction using GNN
#%%
import pandas as pd

# Path to your Excel file
xlsx_path = '/mnt/newStor/paros//paros_WORK/hanwen/ad_decode_test/code/AD_DECODE_data3.xlsx'

# Subject IDs you're interested in
subject_ids = ['2110', '2363', '2373', '2386', '2390', '2231', '2402', '2410', '2421',
               '1912', '2424', '2446', '2451', '2485', '2473', '2491', '2506', '2524',
               '2535', '2654', '2666', '2670', '2690', '2695', '2715', '2720', '2737',
               '2753', '2227', '2765', '2771', '2781', '2802', '2804', '2813', '2817',
               '2840', '2877', '2898', '2926', '2938', '2939', '2954', '2967', '2987',
               '3010', '2320', '3017', '3028', '3033', '3034', '3045', '3069', '3225',
               '3265', '3293', '3308', '3343', '3350', '3378', '3391', '3847', '3866',
               '3889', '3890', '3896', '0775', '4493', '1412', '4526', '1470', '1619']

# Load Excel
df = pd.read_excel(xlsx_path)
df = df.dropna(subset=['MRI_Exam'])
# Ensure MRI_Exam is string and strip leading/trailing spaces
df['MRI_Exam'] = df['MRI_Exam'].astype(str).str.strip()

# Also normalize subject_ids
subject_ids = [sid.strip().lstrip('0') if sid != '0775' else '0775' for sid in subject_ids]

# Fix: pad IDs if needed (assumes IDs are supposed to be 4-digit strings)
df['MRI_Exam'] = df['MRI_Exam'].astype(float).astype(int).astype(str).str.zfill(4)
subject_ids = [sid.zfill(4) for sid in subject_ids]

# Filter and reorder
df_filtered = df[df['MRI_Exam'].isin(subject_ids)]
df_filtered = df_filtered.set_index('MRI_Exam').loc[subject_ids].reset_index()

# Save
df_filtered[['MRI_Exam']].to_csv("subject_ids.csv", index=False, header=False)
df_filtered[['age']].to_csv("ages.csv", index=False, header=False)

#%% md
# ## For graph
# 
# ${hemi}_${region}_columns_file.csv: num_of_columns_in_${hemi}_${region} * 21
# ${hemi}_${region}_coordinates_file: 4 * num_of_columns, [x1, x2, x3, ..., xn; y1, y2, y3, ..., yn; z1, z2, z3, ..., zn; 1, 1, 1, ..., 1]
# 
# 
#%%
REGION_NAMES = ['bankssts',
'caudalanteriorcingulate',
'caudalmiddlefrontal',
'cuneus',
'entorhinal',
'fusiform',
'inferiorparietal',
'inferiortemporal',
'isthmuscingulate',
'lateraloccipital',
'lateralorbitofrontal',
'lingual',
'medialorbitofrontal',
'middletemporal',
'parahippocampal',
'paracentral',
'parsopercularis',
'parsorbitalis',
'parstriangularis',
'pericalcarine',
'postcentral',
'posteriorcingulate',
'precentral',
'precuneus',
'rostralanteriorcingulate',
'rostralmiddlefrontal',
'superiorfrontal',
'superiorparietal',
'superiortemporal',
'supramarginal',
'frontalpole',
'temporalpole',
'transversetemporal',
'insula'
]
#%%
import os
import pandas as pd
import numpy as np
from glob import glob

# Path
file_dir = '/mnt/newStor/paros//paros_WORK/hanwen/ad_decode_test/output/'

# Constants
HEMIS = ['lh', 'rh']
REGION_NAMES = [
    'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus',
    'entorhinal', 'fusiform', 'inferiorparietal', 'inferiortemporal',
    'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual',
    'medialorbitofrontal', 'middletemporal', 'parahippocampal', 'paracentral',
    'parsopercularis', 'parsorbitalis', 'parstriangularis', 'pericalcarine',
    'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
    'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal',
    'superiorparietal', 'superiortemporal', 'supramarginal', 'frontalpole',
    'temporalpole', 'transversetemporal', 'insula'
]

# Output DataFrames
column_features = pd.DataFrame()
column_coords = pd.DataFrame()

# Process each region and hemisphere
for subj_id in subject_ids:
    for hemi in HEMIS:
        for region in REGION_NAMES:
            prefix = f"S0{subj_id}_{hemi}_{region}"

            # # Features
            # column_file = os.path.join(file_dir,f"S0{subj_id}", f"columns_1mm", f"{prefix}_cols_md.csv")
            # if not os.path.exists(column_file):
            #     print(f"File not found: {column_file}")
            #     continue
            # features_df = pd.read_csv(column_file, header=None)
            # num_columns = features_df.shape[0]

            # Coordinates
            coord_file = os.path.join(file_dir,f"S0{subj_id}", f"label_coord_1mm", f"{hemi}_{region}.csv")
            if not os.path.exists(coord_file):
                print(f"File not found: {coord_file}")
                continue
            coords = pd.read_csv(coord_file, header=None)
            x_coords, y_coords, z_coords = coords.iloc[0], coords.iloc[1], coords.iloc[2]
            num_points = len(x_coords)

            # Generate column_ids
            column_ids = [f"{prefix}_p{i}" for i in range(num_points)]

            # # Create feature DataFrame
            # feature_rows = features_df.iloc[:, :21].copy()
            # feature_rows.columns = [f'md_{i+1}' for i in range(21)]
            # feature_rows.insert(0, 'column_id', column_ids)
            # feature_rows.insert(0, 'brain_id', subj_id)
            # column_features = pd.concat([column_features, feature_rows], ignore_index=True)

            # Create coordinate DataFrame
            coords_matrix = pd.DataFrame({
                'x': x_coords.values[:num_points],
                'y': y_coords.values[:num_points],
                'z': z_coords.values[:num_points],
                'column_id': column_ids,
                'brain_id': [subj_id] * num_points
            })
            column_coords = pd.concat([column_coords, coords_matrix], ignore_index=True)


# Convert to DataFrames
#column_features_df = pd.DataFrame(column_features)
column_coords_df = pd.DataFrame(column_coords)

# Load brain metadata
brain_metadata_df = df_filtered[['MRI_Exam', 'age', 'sex', 'genotype', 'BMI']].copy()
brain_metadata_df.columns = brain_metadata_df.columns.str.lower()

# Save output
#column_features_df.to_csv("column_features.csv", index=False)
column_coords_df.to_csv("column_coords.csv", index=False)
brain_metadata_df.to_csv("brain_metadata.csv", index=False)

print("Files saved: column_features.csv, column_coords.csv, brain_metadata.csv")
