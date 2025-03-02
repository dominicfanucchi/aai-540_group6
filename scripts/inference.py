import json
import joblib
import numpy as np
import os
import pandas as pd

# Loads in serialized model from disk 
# Different setup, changed kmeans_arxiv_model.joblib to reduced sample
# May need to directly call model.tar.gz as model dir

def model_fn(model_dir):
    # Construct the full path to the model file
    model_path = os.path.join(model_dir, "kmeans_arxiv_model_reduced_sample.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = joblib.load(model_path)
    return model

# Converts incoming request data into a NumPy array or dataframe from content type 

def input_fn(request_body, request_content_type):
    """Deserialize the input data."""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data)
    elif request_content_type == "text/csv":
        import io
        return pd.read_csv(io.StringIO(request_body)).values
    else:
        raise ValueError("Unsupported content type: " + request_content_type)

# Runs model predictions

def predict_fn(input_data, model):
    """Run model prediction (cluster assignment)."""
    predictions = model.predict(input_data)
    return predictions.tolist()

# Formats prediction into json or csv

def output_fn(prediction, response_content_type):
    """Serialize the prediction output."""
    if response_content_type == "application/json":
        return json.dumps({"prediction": prediction})
    elif response_content_type == "text/csv":
        return ",".join(str(x) for x in prediction)
    else:
        raise ValueError("Unsupported response content type: " + response_content_type)