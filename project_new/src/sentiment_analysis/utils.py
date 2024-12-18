import json

def load_json(file_path: str):
    """
    Load JSON data from a file.

    Parameters:
    file_path (str): Path to the JSON file.

    Returns:
    dict: Parsed JSON data.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data: dict, file_path: str):
    """
    Save data to a JSON file.

    Parameters:
    data (dict): Data to be saved.
    file_path (str): Path to the output JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
