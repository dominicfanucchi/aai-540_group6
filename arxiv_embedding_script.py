#!/usr/bin/env python
# coding: utf-8
# embedding_script.py
import argparse
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


def get_sentence_embedding(text, tokenizer, model, device='cpu'):
    # General embedding function
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean Pool
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-csv-dir", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--print-interval", type=int, default=10000, 
                        help="Print progress every N rows processed")
    args = parser.parse_args()
    
    # SEtup and Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    model.eval()
    
    # All CSV chunk files in the input dir
    files_in_input = os.listdir(args.input_csv_dir)
    csv_files = [f for f in files_in_input if f.endswith(".csv")]
    print("CSV chunk files:", csv_files)

    # Loop over each chunk
    for csv_file in csv_files:
        input_path = os.path.join(args.input_csv_dir, csv_file)
        df = pd.read_csv(input_path) 

        embeddings_list = []
        row_count = len(df)

        # Print once at the start
        print(f"Start embedding {csv_file} (rows: {row_count})")

        for i, abstract in enumerate(df["abstract"]):
            # Handle non-string abstracts gracefully
            if not isinstance(abstract, str):
                embeddings_list.append("")
            else:
                emb = get_sentence_embedding(abstract, tokenizer, model, device=device)
                emb_str = ",".join(str(x) for x in emb)
                embeddings_list.append(emb_str)

            # Print progress every 'print_interval' rows
            if (i + 1) % args.print_interval == 0:
                print(f"...{csv_file}: processed {i + 1} / {row_count} rows")

        # Add a new column for embeddings
        df["abstract_embedding"] = embeddings_list

        # Write output CSV (e.g., "part0_embedded.csv")
        output_file_name = csv_file.replace(".csv", "_embedded.csv")
        output_path = os.path.join(args.output_csv_dir, output_file_name)
        df.to_csv(output_path, index=False)
        print(f"Finished {csv_file}, wrote embedded file: {output_path}\n")