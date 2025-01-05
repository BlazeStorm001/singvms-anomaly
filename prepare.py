import os
import json
import pandas as pd
from occupation_ratio import calculate_occupation_area_shapely, get_poly_area
from datetime import datetime
import argparse

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

def process_json_file(file_path, roi_area):
    with open(file_path, 'r') as f:
        predictions_json = json.load(f)

    filename = os.path.basename(file_path)
    timestamp_str = filename.split('_')[1] + ' ' + filename.split('_')[2].replace('.json', '')
    timestamp = datetime.strptime(timestamp_str, "%Y%m%d %H%M%S")

    vehicle_info = extract_vehicle_info(predictions_json)
    occupation_ratio = calculate_occupation_area_shapely(predictions_json, roi_area)

    return {
        'Timestamp': timestamp,
        'Total Vehicles': vehicle_info['total_count'],
        'Number of Cars': vehicle_info['cars'],
        'Number of Trucks': vehicle_info['trucks'],
        'Number of Two Wheelers': vehicle_info['two_wheelers'],
        'Number of Buses': vehicle_info['buses'],
        'Occupation Ratio': occupation_ratio
    }

def process_directory(json_directory, roi_area):
    data = []
    for filename in os.listdir(json_directory):
        if filename.endswith(".json"):
            print("Processing:", filename)
            file_path = os.path.join(json_directory, filename)
            data.append(process_json_file(file_path, roi_area))
    return data

def save_to_csv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="Process JSON files and generate a CSV report.")
    parser.add_argument("--json_directory", type=str, required=True, help="Directory containing JSON files.")
    parser.add_argument("--points_json", type=str, required=True, help="Path to the JSON file containing polygon points.")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file path.")
    args = parser.parse_args()
    # Load polygon points
    with open(args.points_json, 'r') as file:
        points = json.load(file)
    json_directory = args.json_directory
    roi_area = get_poly_area(points)
    output_file = args.output_file

    data = process_directory(json_directory, roi_area)
    save_to_csv(data, output_file)
    print(f"Data extraction and CSV file generation complete. File saved to: {output_file}")

if __name__ == "__main__":
    main()
