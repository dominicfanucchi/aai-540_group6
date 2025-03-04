#!/usr/bin/env python3
import boto3
import sagemaker
import pandas as pd
import numpy as np
import joblib
import os
import tarfile
import awswrangler as wr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def main():
    # Configuration parameters
    bucket_name = "arxiv-project-bucket"
    role = "arn:aws:iam::221082214706:role/MYLabRole"
    region = "us-east-1"
    
    # Setting up our SageMaker session
    sess = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    print("Using bucket:", bucket_name)
    
    # S3 path for processed training data in csv format
    s3_train_path = f"s3://{bucket_name}/processed_csv/train/"
    df_train = wr.s3.read_csv(path=s3_train_path)
    print("Loaded processed training data shape:", df_train.shape)
    
    # Identify SVD feature columns
    svd_cols = [col for col in df_train.columns if col.startswith("svd_")]
    if not svd_cols:
        raise ValueError("No columns starting with 'svd_' found in the training data.")
    
    # Extract feature matrix
    X = df_train[svd_cols].values
    print("Feature matrix shape (X):", X.shape)
    
    # Train KMeans clustering model
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=39)
    clusters = kmeans.fit_predict(X)
    print("KMeans clustering complete.")
    
    # Compute silhouette score on a random sample to save computation time
    sample_size = 200000
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
        clusters_sample = kmeans.predict(X_sample)
    else:
        X_sample = X
        clusters_sample = clusters

    score = silhouette_score(X_sample, clusters_sample)
    print("Silhouette Score (on sampled data):", score)
    
    # Append cluster labels to the DataFrame for debugging
    df_train["cluster"] = kmeans.labels_
    print("Sample of training data with cluster assignments:")
    print(df_train.head())
    
    # Save the trained model to disk using joblib
    model_filename = "kmeans_arxiv_model.joblib"
    joblib.dump(kmeans, model_filename)
    print("Model saved locally as:", model_filename)
    
    # Archive the model into a tar.gz file for our sklearn container
    archive_filename = "model.tar.gz"
    with tarfile.open(archive_filename, "w:gz") as tar:
        tar.add(model_filename)
    print("Model archived as:", archive_filename)
    
    # Upload the archived model to S3 in models folder
    sess.upload_data(archive_filename, bucket=bucket_name, key_prefix="models")
    model_s3_path = f"s3://{bucket_name}/models/{archive_filename}"
    print("Model uploaded to:", model_s3_path)

if __name__ == "__main__":
    main()