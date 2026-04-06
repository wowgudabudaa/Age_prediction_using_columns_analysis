import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import os
import warnings
import numpy as np
from scipy.stats import pearsonr


# Define config parameters
analyse_cognitive = True
analyse_biological = True
analyse_volumes = True
patterns_dir = 'extracted_patterns/md_graph_patterns_summary.csv'
save_dir = 'patterns_analysis_results_md'
metadata_path = "biological_metrics_MCIandAD/merged_results_with_metadata.csv"
vol_path = "../../normalized_regional_volumes.csv"

# Load gcn patterns
patterns_summary = pd.read_csv(patterns_dir, index_col=0)  # (N, 256) with subject IDs as index

# Load cognitive scores and biological metrics
df = pd.read_csv(metadata_path, header=0) # (N, metrics)
df = df[df['subject_id'].isin(patterns_summary.index)]  # Keep only subjects with patterns
df = df.set_index('subject_id').loc[patterns_summary.index]  # Align with patterns order

# === Load raw regional volume data ===
vol_df = pd.read_csv(vol_path)
print(vol_df.head())
print(vol_df['subject_id'].dtype)
print(patterns_summary.index.dtype)

region_columns = [c for c in vol_df.columns if c != 'subject_id']

for col in region_columns:
    vol_df[col] = pd.to_numeric(vol_df[col], errors='coerce')

# Align with patterns_summary
vol_df = vol_df[vol_df['subject_id'].isin(patterns_summary.index)]
vol_df = vol_df.set_index('subject_id').loc[patterns_summary.index]
print("Volume data shape:", vol_df.shape)


def safe_pearsonr(x, y):
    """safely compute Pearson correlation, returning 0.0 if either input is constant or if any numerical issues arise."""
    # check if either x or y is constant (std = 0), which would cause pearsonr to fail
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    # safely compute the correlation, catching potential warnings (but we've already checked for constants)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        r, _ = pearsonr(x, y)
        # if the result is NaN due to numerical issues, return 0.0
        if np.isnan(r):
            return 0.0
        return r


def evaluate_regression(X, y, n_permutations=1000, random_state=42):
    """
    X: features (n_samples, n_features) - z-scored
    y: target variable (n_samples,)
    n_permutations: number of permutations for the test
    random_state: seed for reproducibility
    return: (mean Pearson r, MAE, RMSE, R², permutation p-value)
    """
    print("regression evaluation...")
    
    # check if y is constant
    if np.std(y) == 0:
        print("Warning: target variable y is constant. All metrics are meaningless.")
        return (0.0, 0.0, 0.0, 0.0, 1.0)
    
    model = Lasso(alpha=0.1, max_iter=10000)
    model.fit(X, y)
    print("regression fitted...")
    y_pred = model.predict(X)
    # print(y_pred.shape, y.shape)

    # Use the safe Pearson correlation function
    r_orig = safe_pearsonr(y, y_pred)
    print(f"Pearson r: {r_orig:.4f}")
    mae = mean_absolute_error(y, y_pred)
    print(f"MAE: {mae:.4f}")
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f"RMSE: {rmse:.4f}")
    r2 = r2_score(y, y_pred)
    print(f"R²: {r2:.4f}")

    # ----- Permutation test -----
    np.random.seed(random_state)
    perm_r = []
    n = len(y)
    
    for i in range(n_permutations):
        # permute the target variable
        y_perm = np.random.permutation(y)
        # fit the model on the original features and permuted target
        model_perm = Lasso(alpha=0.1, max_iter=10000)
        model_perm.fit(X, y_perm)
        y_pred_perm = model_perm.predict(X)
        # compute the Pearson correlation for the permuted data using the safe function
        r_perm = safe_pearsonr(y_perm, y_pred_perm)
        perm_r.append(r_perm)
    
    # compute the p-value as the proportion of permuted correlations that are greater than or equal to the original correlation
    # Adding 1 to both numerator and denominator to avoid p-value of 0 and correct for small sample bias
    p_perm = (np.sum(np.array(perm_r) >= r_orig) + 1) / (n_permutations + 1)
    print(f"Permutation p-value (for correlation > random): {p_perm:.4f}")

    return (r_orig, mae, rmse, r2, p_perm)


os.makedirs(save_dir, exist_ok=True)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(patterns_summary)  # shape (n_subjects, feature_dim)
print(X_normalized.shape)

cognitive_start = df.columns.get_loc('MOCA_TOTAL')
cognitive_end = df.columns.get_loc('Delayed_paraphrase') + 1
cognitive_columns = df.columns[cognitive_start:cognitive_end]

cognition_composite_vars = [
    "Memory_Composite",
    "Executive_Function_Composite",
    "Processing_Speed_Composite",
    "Language_Composite",
    "Visuospatial_Composite",
    "Global_Cognition_Composite"
]
cognitive_composite_columns = [col for col in cognition_composite_vars if col in df.columns]

biological_start = df.columns.get_loc('Systolic')
biological_end = df.columns.get_loc('BMI') + 1
biological_columns = df.columns[biological_start:biological_end]

metrics = {"Cognitive": cognitive_columns,
           "Cognitive_Composite": cognitive_composite_columns,
           "Biological": biological_columns}

if analyse_cognitive and analyse_biological:
    # Evaluate regression performance for each metric type
    for metric_type, columns in metrics.items():
        print(f"\nEvaluating regression performance for {metric_type} metrics:")
        r_list = []
        p_list = []
        mae_list = []
        rmse_list = []
        r2_list = []

        # Evaluate regression performance for cognitive columns
        for name, y_true in df[columns].items():
            # print(name, "regression...")
            # print(y_true)
            r, mae, rmse, r2, p = evaluate_regression(X_normalized, y_true.values)
            print("Results for", name)
            print(r, mae, rmse, r2, p)
            r_list.append(r)
            p_list.append(p)
            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)
            print(name, "done")

        # Save results to DataFrame
        results_df = pd.DataFrame({
            'Metric': columns,
            'Pearson_r': r_list,
            'p_value': p_list,
            'MAE': mae_list,
            'RMSE': rmse_list,
            'R²': r2_list
        })

        # Multiple comparisons correction
        reject, pvals_corrected, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
        results_df['p_value_corrected'] = pvals_corrected
        results_df['Significant'] = reject
        results_df.to_csv(f'{save_dir}/regression_results_{metric_type.lower().replace(" ", "_")}.csv', index=False)
        print("Results saved for", metric_type)

if analyse_volumes:
    # Evaluate regression performance for regional volumes
    print("\nEvaluating regression performance for regional volumes:")
    r_list = []
    p_list = []
    mae_list = []
    rmse_list = []
    r2_list = []

    for name, y_true in vol_df.items():
        r, mae, rmse, r2, p = evaluate_regression(X_normalized, y_true.values)
        print("Results for", name)
        print(r, mae, rmse, r2, p)
        r_list.append(r)
        p_list.append(p)
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)
        print(name, "done")

    # Save results to DataFrame
    results_df = pd.DataFrame({
        'Region': vol_df.columns,
        'Pearson_r': r_list,
        'p_value': p_list,
        'MAE': mae_list,
        'RMSE': rmse_list,
        'R²': r2_list
    })

    # Multiple comparisons correction
    reject, pvals_corrected, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['p_value_corrected'] = pvals_corrected
    results_df['Significant'] = reject
    results_df.to_csv(f'{save_dir}/regression_results_regional_volumes.csv', index=False)
    print("Results saved for regional volumes")
