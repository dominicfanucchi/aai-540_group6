import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, Response
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "kmeans_arxiv_model.joblib")
    logger.info("Loading model from %s", model_path)
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise e
    return model

def input_fn(request_body, content_type):
    logger.info("Parsing input. Content-Type: %s", content_type)
    if content_type == "application/json":
        try:
            # Expecting a JSON-encoded list like [0.1, 0.2, ..., 0.5]
            data = json.loads(request_body)
            logger.info("Parsed JSON data: %s", data)
            return np.array(data).reshape(1, -1)
        except Exception as e:
            logger.error("Error parsing JSON input: %s", e)
            raise ValueError("Error parsing JSON input: " + str(e))
    elif content_type == "text/csv":
        try:
            data = [float(x) for x in request_body.strip().split(",")]
            logger.info("Parsed CSV data: %s", data)
            return np.array(data).reshape(1, -1)
        except Exception as e:
            logger.error("Error parsing CSV input: %s", e)
            raise ValueError("Error parsing CSV input: " + str(e))
    else:
        logger.error("Unsupported content type: %s", content_type)
        raise ValueError("Unsupported content type: " + content_type)

def predict_fn(input_data, model):
    logger.info("Making prediction on input: %s", input_data)
    try:
        prediction = model.predict(input_data)
        logger.info("Prediction result: %s", prediction)
    except Exception as e:
        logger.error("Error during prediction: %s", e)
        raise e
    return prediction

def output_fn(prediction, accept):
    logger.info("Formatting output. Accept: %s", accept)
    if accept == "application/json":
        try:
            output = json.dumps({"prediction": prediction.tolist()})
            logger.info("JSON output: %s", output)
            return output
        except Exception as e:
            logger.error("Error formatting JSON output: %s", e)
            raise e
    elif accept == "text/csv":
        try:
            output = ",".join(map(str, prediction.tolist())) + "\n"
            logger.info("CSV output: %s", output)
            return output
        except Exception as e:
            logger.error("Error formatting CSV output: %s", e)
            raise e
    else:
        # Default to JSON if unspecified
        try:
            output = json.dumps({"prediction": prediction.tolist()})
            logger.info("Defaulting to JSON output: %s", output)
            return output
        except Exception as e:
            logger.error("Error formatting default JSON output: %s", e)
            raise e

@app.route("/ping", methods=["GET"])
def ping():
    logger.info("Received /ping request")
    return Response("pong", status=200)

@app.route("/invocations", methods=["POST"])
def invocations():
    logger.info("Received /invocations request")
    try:
        data = request.data.decode("utf-8")
        logger.info("Raw request data: %s", data)
        input_data = input_fn(data, request.content_type)
        logger.info("Parsed input data: %s", input_data)
        model = model_fn(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
        prediction = predict_fn(input_data, model)
        best_mimetype = request.accept_mimetypes.best_match(["application/json", "text/csv"])
        response = output_fn(prediction, best_mimetype)
        logger.info("Returning response with mimetype %s: %s", best_mimetype, response)
        return Response(response, mimetype=best_mimetype)
    except Exception as e:
        logger.error("Error in /invocations: %s", e)
        return Response("Error: " + str(e), status=500)
