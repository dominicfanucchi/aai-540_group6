# custom_drift.py
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def main():
    # Input and output directories provided by SageMaker Processing job
    input_dir = "/opt/ml/processing/input"
    output_dir = "/opt/ml/processing/output"
    
    # List all CSV files in the input directory
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]
    if not input_files:
        raise ValueError("No CSV files found in input directory: " + input_dir)
    
    # Read and combine the captured data
    df_current = pd.concat([pd.read_csv(file) for file in input_files], ignore_index=True)
    print(f"Loaded captured input data shape: {df_current.shape}")
    
    # Load baseline input data from where its saved
    baseline_path = os.path.join("/opt/ml/processing/baseline", "baseline_input.csv")
    if not os.path.exists(baseline_path):
        raise ValueError("Baseline input file not found at: " + baseline_path)
    df_baseline = pd.read_csv(baseline_path)
    print(f"Loaded baseline input data shape: {df_baseline.shape}")
    
    # Compare distribution of a key factor
    feature = "svd_0"
    if feature not in df_current.columns or feature not in df_baseline.columns:
        raise ValueError(f"Feature {feature} not found in both datasets.")
    
    # Compute the Kolmogorov–Smirnov statistic between the two distributions
    stat, p_value = ks_2samp(df_baseline[feature].dropna(), df_current[feature].dropna())
    print(f"KS statistic for {feature}: {stat}, p-value: {p_value}")
    
    result = {
        "drift_metrics": {
            feature: {
                "ks_statistic": stat,
                "p_value": p_value
            }
        }
    }
    
    # Write the result to output directory as JSON
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "drift_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(result, f)
    print("Drift evaluation complete. Results written to:", output_path)

if __name__ == "__main__":
    main()