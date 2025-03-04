# custom_baseline.py
import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def main():
    # SageMaker Processing job directories
    input_dir = "/opt/ml/processing/input"
    output_dir = "/opt/ml/processing/output"
    
    # All CSV files in the input directory
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]
    if not input_files:
        raise ValueError("No CSV files found in input directory: " + input_dir)
    
    # Read and concatenate all CSV files into one DataFrame
    df = pd.concat([pd.read_csv(file) for file in input_files], ignore_index=True)
    print(f"Loaded input data shape: {df.shape}")
    
    # Extract feature columns beginning with svd from preprocessing
    svd_cols = [col for col in df.columns if col.startswith("svd_")]
    if not svd_cols:
        raise ValueError("No SVD columns found in the input data.")
    X = df[svd_cols].values
    print(f"Extracted feature matrix shape: {X.shape}")
    
    # Define the number of clusters matching training amount
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=39)
    cluster_labels = kmeans.fit_predict(X)
    
    # Compute the silhouette score if there is more than one unique cluster
    if len(np.unique(cluster_labels)) > 1:
        score = silhouette_score(X, cluster_labels)
    else:
        score = 0.0
    print(f"Silhouette Score: {score}")
    
    # Prepare the result in JSON format
    result = {
        "regression_metrics": {
            "silhouette_score": {
                "value": score,
                "standard_deviation": None # No need for variability
            }
        }
    }
    
    # Write the result JSON to the output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "evaluation.json")
    with open(output_path, "w") as f:
        json.dump(result, f)
    print("Evaluation complete. Results written to:", output_path)

if __name__ == "__main__":
    main()