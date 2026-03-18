import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

# Load merged dataset
file_path = "/mnt/data/Merged_Data_with_APOE4.csv"
merged_df = pd.read_csv(file_path)

# Generate fake SHAP embeddings for 68 regions
np.random.seed(42)
shap_columns = [f"region_{i+1}" for i in range(68)]
shap_data = pd.DataFrame(np.random.randn(len(merged_df), 68), columns=shap_columns)
shap_data.index = merged_df["MRI_Exam"]

# Combine with metadata (Risk, APOE4, and Sex)
meta_data = merged_df.set_index("MRI_Exam")[["Risk", "APOE4", "Sex"]]

# Standardize SHAP data for clustering
scaler = StandardScaler()
shap_scaled = scaler.fit_transform(shap_data)
shap_scaled_df = pd.DataFrame(shap_scaled, index=shap_data.index, columns=shap_columns)

# Assign colors to Risk, APOE4, and Sex
risk_colors = {
    'No risk': 'blue',
    'Familial': 'green',
    'MCI': 'orange',
    'AD': 'red'
}
sex_colors = {
    'Male': 'purple',
    'Female': 'yellow'
}
apoe4_colors = {
    0: 'black',  # Non-carrier
    1: 'cyan'    # Carrier
}

# Map metadata to colors
row_colors = {
    'Risk': meta_data["Risk"].map(risk_colors),
    'Sex': meta_data["Sex"].map(sex_colors),
    'APOE4': meta_data["APOE4"].map(apoe4_colors)
}

# Create clustered heatmap
sns.set(style="white")
cg = sns.clustermap(
    shap_scaled_df,
    method="ward",
    metric="euclidean",
    row_colors=pd.concat([row_colors['Risk'], row_colors['Sex'], row_colors['APOE4']], axis=1),
    cmap="RdBu_r",
    figsize=(14, 10),
    col_cluster=True,
    row_cluster=True,
    xticklabels=False,
    yticklabels=False
)

# Add legend for Risk, Sex, and APOE4
handles = [
    mpatches.Patch(color=color, label=label) for label, color in risk_colors.items()
]
handles += [
    mpatches.Patch(color=color, label=label) for label, color in sex_colors.items()
]
handles += [
    mpatches.Patch(color=color, label="APOE4 Carrier" if k == 1 else "Non-carrier") for k, color in apoe4_colors.items()
]
plt.legend(handles=handles, title="Risk, Sex, and APOE4", bbox_to_anchor=(1.05, 1), loc='upper left')

# Save figure
output_path = "/mnt/data/SHAP_Clustermap_with_APOE4_Sex_ADDECODE.png"
plt.savefig(output_path, bbox_inches='tight')

# Export clustering results
clustered_subjects = pd.DataFrame({
    'Subject': shap_scaled_df.index,
    'Cluster': cg.dendrogram_row.reordered_ind
})

clustered_subjects.to_csv("/mnt/data/Clustered_Subjects.csv", index=False)

output_path, "/mnt/data/Clustered_Subjects.csv"
