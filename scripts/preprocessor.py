import json

def preprocess_handler(inference_record):
    """
    Preprocess the inference record for model monitoring.
    Flattens the prediction and ground truth data from the inference response.
    """
    flattened_data = {}

    # Process inference output if available
    if hasattr(inference_record, 'endpoint_output'):
        output_data_json = json.loads(inference_record.endpoint_output.data.rstrip("\n"))
        for i, pred in enumerate(output_data_json.get("prediction", [])):
            flattened_data[f"endpointOutput_prediction{i}"] = pred

    # Process ground truth data if available
    if hasattr(inference_record, 'ground_truth'):
        ground_truth_data_json = json.loads(inference_record.ground_truth.data.rstrip("\n"))
        for i, gt in enumerate(ground_truth_data_json):
            flattened_data[f"groundTruthData_{i}"] = gt

    return flattened_data
