"""
AWS CI/CD Pipeline Tools for ArXiv Recommendation System

This module contains utility functions for:
- Reading the ArXiv JSON metadata from S3
- Preprocessing and basic cleaning of the data
- Setting up feature groups in SageMaker Feature Store

Author: Your Name
Date: [Today's Date]
"""

import json
import boto3
import awswrangler as wr
import pandas as pd
import time
from time import strftime, gmtime
from sagemaker.feature_store.feature_group import FeatureGroup

def get_arxiv_data(bucket_name, data_key):
    """
    Reads the ArXiv JSON metadata file from S3 and returns a DataFrame
    With assumption that the JSON file is newline-delimited JSON (NDJSON).
    """
    s3_path = f"s3://{bucket_name}/{data_key}"
    try:
        df = wr.s3.read_json(path=s3_path, lines=True)
    except Exception as e:
        # Fallback using boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket_name, Key=data_key)
        data = obj["Body"].read().decode("utf-8")
        records = [json.loads(line) for line in data.splitlines() if line.strip()]
        df = pd.DataFrame(records)
    return df

def preprocess_arxiv_data(df):
    """
    Basic preprocessing for ArXiv data:
    - Fill missing values
    - Standardize text fields
    - Create additional features if necessary
    - Can add more if required
    """
    # Fill missing abstracts with empty string
    df["abstract"] = df["abstract"].fillna("")
    # Convert 'categories' to a list (assumes space-separated categories)
    if df["categories"].dtype == object:
        df["categories"] = df["categories"].apply(lambda x: x.split() if isinstance(x, str) else [])
    return df

def setup_feature_groups(pq_data, feature_store_session, event_time_feature_name="EventTime"):
    """
    Set up feature groups in SageMaker Feature Store for different data splits.
    Uses the 'id' field as the record identifier.
    """
    stamp_mark = strftime("%d-%H-%M-%S", gmtime())
    current_time_sec = int(round(time.time()))
    f_groups = []
    for key in pq_data.keys():
        df = pq_data[key].reset_index(drop=True)
        feature_group_name = f"arxiv-{key}-feature-group-{stamp_mark}"
        feature_group = FeatureGroup(
            name=feature_group_name, sagemaker_session=feature_store_session
        )
        # Ensure object dtypes are converted to string
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("string")
        # Append event time feature
        df[event_time_feature_name] = current_time_sec
        feature_group.load_feature_definitions(data_frame=df)
        f_groups.append(feature_group)
    return f_groups

def wait_for_feature_group_creation_complete(feature_group):
    """
    Wait until the feature group is successfully created
    """
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print(f"Waiting for Feature Group {feature_group.name} creation...")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    if status != "Created":
        raise RuntimeError(f"Failed to create feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully created.")
