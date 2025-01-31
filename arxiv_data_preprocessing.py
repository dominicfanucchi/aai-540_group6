#!/usr/bin/env python
# coding: utf-8

'''
FILE STRUCTURE GENERAL
arxiv_project/
  ├─ requirements.txt
  ├─ scripts/
  │   ├─ data_preprocessing.py #Clean, transform JSON -> CSV
  │   ├─ embedding_script.py #Generate embeddings via Transformers
  │   ├─ train_preparation.py #Convert embeddings to numeric CSV
  │   ├─ cluster_evaluation.py #Evaluate silhouette, Davies-Bouldin
  │   ├─ pipeline_definition.py #SageMaker Pipeline definition
  └─ buildspec.yml #For building up
'''

import subprocess
import sys

# ------------------------------------------------------
# Dynamically install needed libraries
# ------------------------------------------------------
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "transformers==4.26.1",
    "torch==1.13.1",
    "scikit-learn",
    "boto3",
    "sagemaker"
])

# ------------------------------------------------------
# Import them plus standard libraries
# ------------------------------------------------------
import argparse
import pandas as pd
import os
from datetime import datetime
import json

# ------------------------------------------------------
# Parse function
# ------------------------------------------------------
def parse_record(record_dict):
    """
    Convert a single JSON record (dict) into a standardized row
    for our DataFrame.
    """
    arxiv_id = record_dict.get("id", "")
    abstract = record_dict.get("abstract", "")
    update_date = record_dict.get("update_date", "")
    
    # Try to parse update_date as YYYY-MM-DD
    year, month = None, None
    if update_date:
        try:
            dt = datetime.strptime(update_date, "%Y-%m-%d")
            year = dt.year
            month = dt.month
        except ValueError:
            pass

    authors = record_dict.get("authors", "")
    abstract_clean = abstract.replace("\n", " ").strip()

    return {
        "id": arxiv_id,
        "abstract": abstract_clean,
        "update_date": update_date,
        "year": year,
        "month": month,
        "authors": authors
    }

# ------------------------------------------------------
# Stream JSON, chunk into CSV
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input",
                        help="Path to the folder containing the large JSON file.")
    parser.add_argument("--output-data", type=str, default="/opt/ml/processing/output",
                        help="Path to store the output CSV chunks.")
    parser.add_argument("--chunk-size", type=int, default=200_000,
                        help="Number of lines/records to accumulate before writing a partial CSV.")
    args = parser.parse_args()

    input_path = args.input_data
    output_path = args.output_data
    chunk_size = args.chunk_size

    # Suppose our large JSON file is named 'arxiv-metadata-oai-snapshot.json' in the input path
    json_file = os.path.join(input_path, "arxiv-metadata-oai-snapshot.json")

    os.makedirs(output_path, exist_ok=True)

    # We'll store records in a buffer until we reach chunk_size, then flush to CSV
    records_buffer = []
    file_counter = 0
    total_records = 0

    print(f"Reading file: {json_file} in streaming mode with chunk_size={chunk_size}")
    
    # Example for a JSON-LINES file:
    with open(json_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record_dict = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line {line_num}: {e}")
                continue
            
            parsed = parse_record(record_dict)
            records_buffer.append(parsed)
            total_records += 1
            
            # Once we reach chunk_size, flush to a partial CSV
            if len(records_buffer) >= chunk_size:
                df_chunk = pd.DataFrame(records_buffer)
                partial_csv = os.path.join(output_path, f"arxiv_preprocessed_part{file_counter}.csv")
                df_chunk.to_csv(partial_csv, index=False)
                print(f"Wrote chunk {file_counter} with {len(records_buffer)} records to {partial_csv}")
                records_buffer.clear()
                file_counter += 1

    # Write any leftover records
    if records_buffer:
        df_chunk = pd.DataFrame(records_buffer)
        partial_csv = os.path.join(output_path, f"arxiv_preprocessed_part{file_counter}.csv")
        df_chunk.to_csv(partial_csv, index=False)
        print(f"Wrote final chunk {file_counter} with {len(records_buffer)} records to {partial_csv}")

    print(f"Finished processing. Total records: {total_records}")

if __name__ == "__main__":
    main()
