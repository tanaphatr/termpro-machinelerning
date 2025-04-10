import requests
import json
from datetime import datetime

def get_latest_models_by_date():
    """
    Retrieves models from the API based on the latest saved date for each type.

    Returns:
        dict: A dictionary containing models grouped by type for the latest date.
    """
    url = "https://termpro-api-production.up.railway.app/Model/"
    headers = {"Content-Type": "application/json"}
    
    # Send GET request to fetch all models
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to retrieve models. Status code: {response.status_code}, Response: {response.text}")
        return None

    # Parse the response JSON
    models = response.json()
    
    if not models:  # Check if the models list is empty
        print("No models found.")
        return None

    # Group models by type and find the latest date
    grouped_models = {}
    latest_date = None

    for model in models:
        model_date = datetime.strptime(model["date"], "%Y-%m-%d")
        model_type = model["type"]

        # Update the latest date
        if latest_date is None or model_date > latest_date:
            latest_date = model_date

        # Group models by type
        if model_type not in grouped_models:
            grouped_models[model_type] = []
        grouped_models[model_type].append(model)

    # Ensure latest_date is not None before proceeding
    if latest_date is None:
        print("No valid dates found in models.")
        return None

    # Filter models to include only those from the latest date
    latest_date_str = latest_date.strftime("%Y-%m-%d")
    latest_models = {
        model_type: [
            model for model in models if model["date"] == latest_date_str
        ]
        for model_type, models in grouped_models.items()
    }

    print(f"Latest models retrieved for date: {latest_date_str}")
    return latest_models

# Example usage
if __name__ == "__main__":
    latest_models = get_latest_models_by_date()
    if latest_models:
        print(json.dumps(latest_models, indent=4))