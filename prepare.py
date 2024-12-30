import os
import json
import pandas as pd
from occupation_ratio import calculate_occupation_area_shapely
from datetime import datetime

# Function to extract vehicle information from the JSON data
def extract_vehicle_info(json_data):
    trucks = 0
    cars = 0
    buses = 0
    two_wheelers = 0
    total_count = len(json_data['predictions'])

    for pred in json_data['predictions']:
        if pred['class'] == 'car':
            cars += 1
        elif pred['class'] == 'bus':
            buses += 1
        elif pred['class'] == 'truck':
            trucks += 1
        elif pred['class'] == 'two-wheeler':
            two_wheelers += 1

    return {
        'total_count': total_count,
        'trucks': trucks,
        'cars': cars,
        'buses': buses,
        'two_wheelers': two_wheelers
    }

# Path to the directory containing the JSON files
json_directory = "data/3796/json"

# ROI area for occupation ratio calculation
roi_area = 992766.0

# List to store the features for the dataset
data = []

# Iterate over each JSON file in the directory
for filename in os.listdir(json_directory):
    if filename.endswith(".json"):
        file_path = os.path.join(json_directory, filename)

        # Read the JSON file
        with open(file_path, 'r') as f:
            predictions_json = json.load(f)

        # Extract the timestamp from the filename
        timestamp_str = filename.split('_')[1] + ' ' + filename.split('_')[2].replace('.json', '')
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d %H%M%S")

        # Extract vehicle info
        vehicle_info = extract_vehicle_info(predictions_json)

        # Calculate occupation ratio
        occupation_ratio = calculate_occupation_area_shapely(predictions_json, roi_area)

        # Append the features to the dataset list
        data.append({
            'Timestamp': timestamp,
            'Total Vehicles': vehicle_info['total_count'],
            'Number of Cars': vehicle_info['cars'],
            'Number of Trucks': vehicle_info['trucks'],
            'Number of Two Wheelers': vehicle_info['two_wheelers'],
            'Number of Buses': vehicle_info['buses'],
            'Occupation Ratio': occupation_ratio
        })

# Convert the list of data to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('3796_vehicle_data.csv', index=False)

print("Data extraction and CSV file generation complete.")
