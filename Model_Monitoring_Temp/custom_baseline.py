import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import argparse

# NO way around it, need to drop versions after reading way too problematic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True, help="Path to baseline data (folder containing parquet files)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for baseline metrics")
    parser.add_argument("--n_clusters", type=int, default=5, help="Number of clusters")
    args = parser.parse_args()
    
    # Read baseline dataset from the folder.
    # This will read all parquet files in the directory.
    df = pd.read_parquet(args.input_data)
    
    # Drop the problematic column, if it exists
    if 'versions' in df.columns:
        print("Dropping 'versions' column due to unsupported data types.")
        df = df.drop(columns=['versions'])
    
    # Assume the last 10 columns are the TF-IDF features (adjust if necessary)
    feature_columns = df.columns[-10:]
    X = df[feature_columns].values
    
    # Impute missing values (silhouette score cannot handle NaNs)
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    
    # Run KMeans clustering on the baseline dataset
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Compute the silhouette score
    silhouette = silhouette_score(X, clusters)
    
    baseline_metrics = {
        "silhouette_metrics": {
            "mean": silhouette
        }
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "baseline_metrics.json")
    with open(output_file, "w") as f:
        json.dump(baseline_metrics, f)
    print("Baseline metrics written to", output_file)

if __name__ == "__main__":
    main()

