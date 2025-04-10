import requests
import json
from datetime import datetime

def save_model_to_api(file_path, model_type, last_trained_date, best_params):
    """
    Uploads a model file to the API and inserts metadata into the database.

    Args:
        file_path (str): Path to the model file to be uploaded.
        model_type (str): Type of the model (e.g., 'LSTM').
        last_trained_date (str): Last trained date in 'YYYY-MM-DD' format.
        best_params (dict): Best parameters of the model.

    Returns:
        dict: Response from the API.
    """
    url = "https://termpro-api-production.up.railway.app/Model/"
    headers = {"Content-Type": "application/json"}
    
    # Read the file as binary
    with open(file_path, "rb") as file:
        blob_data = file.read()
    
    # Prepare the payload
    payload = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "type": model_type,
        "last_trained_date": last_trained_date,
        "lstm_model": blob_data.hex(),  # Convert binary to hex string
        "best_params": json.dumps(best_params)
    }
    
    # Send POST request
    response = requests.post(url, headers=headers, json=payload)
    
    # Check response status
    if response.status_code == 200:
        print("Model saved successfully!")
    else:
        print(f"Failed to save model. Status code: {response.status_code}, Response: {response.text}")
    
    return response.json()

# Example usage
if __name__ == "__main__":
    file_path = "path_to_your_model_file.h5"
    model_type = "LSTM"
    last_trained_date = "2025-04-01"
    best_params = {"learning_rate": 0.001, "epochs": 50}
    
    save_model_to_api(file_path, model_type, last_trained_date, best_params)