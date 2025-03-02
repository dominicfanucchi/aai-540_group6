# train_model.py
#!pip install faiss-cpu, boto3

import os
import zipfile
import boto3
import torch
import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score

# AWS Configuration (placeholder)
S3_BUCKET = "your-s3-bucket-name"
S3_RESULTS_PATH = "results/"
S3_MODELS_PATH = "models/"

# Paths
ZIP_PATH = "/content/embedding.zip"
EXTRACT_PATH = "/content/extracted_embeddings"
OUTPUT_CSV_PATH = "/content/results/clustered_results.csv"
MODEL_PATH = "/content/models/kmeans_model.faiss"
METRICS_PATH = "/content/results/cluster_metrics.csv"
PREV_RESULTS_PATH = "/content/results/previous_clustered_results.csv"

# Hyperparameters
NUM_CLUSTERS = 5
BATCH_SIZE = 512
PCA_COMPONENTS = 2  
FAISS_ITERATIONS = 10  

# ===========================
# Extract ZIP File If Needed
# ===========================
if os.path.exists(ZIP_PATH) and not os.path.exists(EXTRACT_PATH):
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

# ===========================
# Load Embeddings 
# ===========================

dfs = []
embedding_files = [f for f in os.listdir(EXTRACT_PATH) if f.endswith(".csv")]

if not embedding_files:
    raise ValueError(f"No CSV files found in extracted path: {EXTRACT_PATH}")

def parse_embedding(embedding):
    """Convert string embeddings to lists of floats."""
    try:
        return list(map(float, embedding.strip("[]").split(","))) if isinstance(embedding, str) else np.zeros(128).tolist()
    except:
        return np.zeros(128).tolist()

embedding_list = []

for file in embedding_files:
    file_path = os.path.join(EXTRACT_PATH, file)
    try:
        for chunk in pd.read_csv(file_path, chunksize=5000, encoding="utf-8"):
            chunk["abstract_embedding"] = chunk["abstract_embedding"].apply(parse_embedding)
            embedding_list.append(np.vstack(chunk["abstract_embedding"].values))
            dfs.append(chunk)  
    except Exception:
        pass

if not dfs:
    raise ValueError("No valid CSV files loaded.")

df = pd.concat(dfs, ignore_index=True)

embeddings_np = np.vstack(embedding_list)
np.save("large_embeddings.npy", embeddings_np)
X_memmap = np.load("large_embeddings.npy", mmap_mode="r")

# ===========================
# K-means Clutering 
# ===========================
kmeans = faiss.Kmeans(d=X_memmap.shape[1], k=NUM_CLUSTERS, niter=FAISS_ITERATIONS)
kmeans.train(X_memmap)

df["cluster"] = kmeans.index.search(X_memmap, 1)[1].flatten()

# ===========================
# Apply PCA After Clustering
# ===========================
pca = PCA(n_components=PCA_COMPONENTS)
X_pca = pca.fit_transform(X_memmap)

df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]

df.drop(columns=["abstract_embedding"], inplace=True)

# ===========================
# Compute Metrics
# ===========================
metrics_results = []

for (year, month), sub_df in df.groupby(["year", "month"]):
    if len(sub_df) < NUM_CLUSTERS:
        continue

    filtered_pca = X_pca[df.index.isin(sub_df.index)]
    inertia = np.sum(kmeans.centroids)
    davies_bouldin = davies_bouldin_score(filtered_pca, sub_df["cluster"])

    metrics_results.append({"year": year, "month": month, "inertia": inertia, "davies_bouldin_score": davies_bouldin})

metrics_df = pd.DataFrame(metrics_results)

# Ensure directories exist before saving
os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

df.to_csv(OUTPUT_CSV_PATH, index=False)
metrics_df.to_csv(METRICS_PATH, index=False)
faiss.write_index(kmeans.index, MODEL_PATH)

# ===========================
# Upload Results to AWS S3
# ===========================
s3 = boto3.client("s3")

def upload_to_s3(local_path, bucket, s3_path):
    """Uploads a file to AWS S3."""
    s3.upload_file(local_path, bucket, s3_path)

upload_to_s3(OUTPUT_CSV_PATH, S3_BUCKET, f"{S3_RESULTS_PATH}clustered_results.csv")
upload_to_s3(METRICS_PATH, S3_BUCKET, f"{S3_RESULTS_PATH}cluster_metrics.csv")
upload_to_s3(MODEL_PATH, S3_BUCKET, f"{S3_MODELS_PATH}kmeans_model.faiss")

