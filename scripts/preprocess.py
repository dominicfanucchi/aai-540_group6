'''
Python script for sagemaker processing job. This script reads the raw JSON data from the input directory (which will be provided via ProcessingInput), performs data cleaning and text preprocessing (stopword removal, punctuation removal, lemmatization), generates TF‑IDF features, applies dimensionality reduction (via TruncatedSVD), splits the data by year into test, train, and validation sets, and then writes each split as a CSV file to the output directory. (In a pipeline job, the ProcessingOutput from the job will then be automatically uploaded to S3.)
'''

import os
import io
import re
import time
import boto3
import nltk
import warnings
import pandas as pd
import seaborn as sns
import numpy as np
import awswrangler as wr
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sagemaker.feature_store.feature_group import FeatureGroup

# Suppress warnings and set plot style
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# Setup input/output directories for SageMaker job
# These directories are provided by SageMaker when the job is run
input_dir = "/opt/ml/processing/input"
output_dir = "/opt/ml/processing/output"

# For convenience, create an output subdirectory for the CSV splits.
splits_output_dir = os.path.join(output_dir, "csv_splits")
os.makedirs(splits_output_dir, exist_ok=True)

# Setup configuration parameters unique to each user
bucket_name = "arxiv-project-bucket" 
data_key = "arxiv-metadata-oai-snapshot.json"
role = "arn:aws:iam::221082214706:role/MYLabRole"
region = "us-east-1"

print("Preprocessing job started at", datetime.utcnow().isoformat())

# Download all necessary nltk resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load Raw Data
raw_file_path = os.path.join(input_dir, data_key)
print("Reading raw data from:", raw_file_path)
df = pd.read_json(raw_file_path, lines=True)
print("Raw data shape:", df.shape)

# Data Cleaning replacing any string of "None" with NaN to identify and drop missing values
df = df.replace("None", np.nan)
nan_counts = df.isna().sum()
print("NaN counts per column before drop:")
print(nan_counts)
print("Total NaNs in dataset:", nan_counts.sum())
df.dropna(inplace=True)

# Drop columns that are too sparse or not needed
columns_to_drop = ['version', 'versions', 'authors_parsed', 'license', 'report-no', 'journal-ref', 'comments', 'doi']
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"Dropped column: {col}")
print("Data cleaning complete.")

# Text Preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def remove_stop_words(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

def remove_punct_numbers(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def lemmatize_text(text):
    tokens = text.split()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

df['title'] = df['title'].fillna("")
df['abstract'] = df['abstract'].fillna("")
df['text'] = df['title'] + " " + df['abstract']
df['text'] = df['text'].str.lower().apply(remove_stop_words)
df['text'] = df['text'].apply(remove_punct_numbers)
df['text'] = df['text'].apply(lemmatize_text)
print("Text preprocessing complete.")

# Convert update_time to datetime format
df["update_date"] = pd.to_datetime(df["update_date"], format="%Y-%m-%d", errors="coerce")
df = df.sort_values(by='update_date')
print("Data sorted by update_date.")
df['year'] = df['update_date'].dt.year
print("Year counts:")
print(df['year'].value_counts().sort_index())

max_date = df['update_date'].max()
print("Max date in dataset:", max_date)
eight_years_ago = max_date - pd.DateOffset(years=8)
df_recent = df[df['update_date'] >= eight_years_ago]
print("Data shape after filtering for the most recent 8 years:", df_recent.shape)

# Feature Store Preparation
# Setting up a copy for feature store ingestion
df_feature_group = df_recent.copy()
for col in df_feature_group.columns:
    if df_feature_group[col].dtype == "object":
        df_feature_group[col] = df_feature_group[col].astype("string")
if "update_date" in df_feature_group.columns:
    df_feature_group["update_date"] = df_feature_group["update_date"].astype("string")
df_feature_group["EventTime"] = datetime.utcnow().isoformat()
feature_group_name = "arxiv-feature-group-" + datetime.now().strftime("%Y%m%d%H%M%S")
print("Feature Group Name:", feature_group_name)
# Create the feature group in Feature Store
try:
    feature_group = FeatureGroup(
        name=feature_group_name,
        sagemaker_session=sagemaker.Session(boto_session=boto3.Session(region_name=region))
    )
    feature_group.load_feature_definitions(data_frame=df_feature_group)
    feature_group.create(
        s3_uri=f"s3://{bucket_name}/feature_store",
        record_identifier_name="id",
        event_time_feature_name="EventTime",
        role_arn=role,
        enable_online_store=True
    )
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for feature group creation...")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    if status != "Created":
        raise RuntimeError(f"Failed to create feature group {feature_group_name}")
    print(f"Feature Group {feature_group_name} successfully created.")
except Exception as e:
    print("Feature store creation skipped or failed:", e)

# TF-IDF and Dimensionality Reduction 
tfidf_vectorizer = TfidfVectorizer(max_features=300, min_df=2, max_df=0.8)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_recent['text'])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

svd = TruncatedSVD(n_components=50, random_state=39)
X_reduced = svd.fit_transform(tfidf_matrix)
print("Reduced features shape:", X_reduced.shape)

df_reduced = pd.DataFrame(X_reduced, columns=[f'svd_{i}' for i in range(50)])
df_recent = df_recent.reset_index(drop=True)
df_final = pd.concat([df_recent, df_reduced], axis=1)
print("Final data shape with reduced features:", df_final.shape)

# Data splitting into train, test, and validation sets
df_final['year'] = df_final['update_date'].dt.year
unique_years = sorted(df_final['year'].unique())
print("Unique years in final data:", unique_years)
if len(unique_years) >= 8:
    recent_8_years = unique_years[-8:]
else:
    recent_8_years = unique_years
print("Recent 8 years for splitting:", recent_8_years)

# Split into Test (oldest 4), Train (next 2), and Validation (most recent 2)
if len(recent_8_years) == 8:
    train_years = recent_8_years[:4]
    test_years = recent_8_years[4:6]
    val_years = recent_8_years[6:]
else:
    n = len(recent_8_years)
    train_years = recent_8_years[:n//2]
    test_years = recent_8_years[n//2:n//2+1]
    val_years = recent_8_years[n//2+1:]
    
print("Train years:", train_years)
print("Test years:", test_years)
print("Validation years:", val_years)

dfs_test = {yr: df_final[df_final['year'] == yr] for yr in test_years}
dfs_train = {yr: df_final[df_final['year'] == yr] for yr in train_years}
dfs_val = {yr: df_final[df_final['year'] == yr] for yr in val_years}

for yr, d in dfs_test.items():
    print(f"Test set for {yr}: {d.shape}")
for yr, d in dfs_train.items():
    print(f"Training set for {yr}: {d.shape}")
for yr, d in dfs_val.items():
    print(f"Validation set for {yr}: {d.shape}")

# Save all data splits as csv files
def save_csv_locally_and_log(df, filename):
    local_path = os.path.join(splits_output_dir, filename)
    df.to_csv(local_path, index=False)
    print(f"Saved {filename} to {local_path}")
    return local_path

# Save Test, Train, and Validation splits at one file per year
for yr, d in dfs_test.items():
    fname = f"arxiv_test_{yr}.csv"
    save_csv_locally_and_log(d, fname)
for yr, d in dfs_train.items():
    fname = f"arxiv_train_{yr}.csv"
    save_csv_locally_and_log(d, fname)
for yr, d in dfs_val.items():
    fname = f"arxiv_val_{yr}.csv"
    save_csv_locally_and_log(d, fname)

print("Preprocessing complete. All processed CSV files are saved in", splits_output_dir)