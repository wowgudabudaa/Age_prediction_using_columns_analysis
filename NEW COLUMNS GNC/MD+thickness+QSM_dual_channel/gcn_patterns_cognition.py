import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import os

# Load gcn patterns
patterns_dir = 'extracted_patterns/md_graph_patterns_summary.csv'
patterns_summary = pd.read_csv(patterns_dir, index_col=0)  # (N, 256) with subject IDs as index

# Load cognitive scores and biological metrics
df = pd.read_csv("biological_metrics_MCIandAD/merged_results_with_metadata.csv", header=0) # (N, metrics)
df = df[df['subject_id'].isin(patterns_summary.index)]  # Keep only subjects with patterns
df = df.set_index('subject_id').loc[patterns_summary.index]  # Align with patterns order


def evaluate_regression(X, y):
    """
    X: features (n_samples, n_features) - z-scored
    y: target variable (n_samples,)
    return: mean Pearson r, MAE, RMSE, R², p-value for slope significance
    """
    print("regression evaluation...")
    model = Lasso(alpha=0.1, max_iter=10000)
    model.fit(X, y)
    print("regression fitted...")
    y_pred = model.predict(X)
    print(y_pred.shape, y.shape)

    r, _ = pearsonr(y, y_pred)
    print(r)
    mae = mean_absolute_error(y, y_pred)
    print(mae)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(rmse)
    r2 = r2_score(y, y_pred)
    print(r2)
    ymodel = sm.OLS(y, sm.add_constant(y_pred)).fit()
    print(ymodel.summary())
    p = ymodel.pvalues[1]  # p-value for slope
    print(p)

    return (r, mae, rmse, r2, p)


save_dir = 'patterns_analysis_results'
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
        r, mae, rmse, r2, p = evaluate_regression(X_normalized, y_true)
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
