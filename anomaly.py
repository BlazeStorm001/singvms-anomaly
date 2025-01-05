import pandas as pd
import numpy as np
from scipy.stats import zscore, norm
import matplotlib.pyplot as plt
import argparse

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Time_Window'] = df['Timestamp'].dt.floor('15T')
    return df

def aggregate_data(df):
    return df.groupby('Time_Window').agg({
        'Total Vehicles': 'mean',
        'Occupation Ratio': 'mean'
    }).reset_index()

def calculate_z_scores(agg_df):
    agg_df['Total_Vehicles_Z'] = zscore(agg_df['Total Vehicles'])
    agg_df['Occupation_Ratio_Z'] = zscore(agg_df['Occupation Ratio'])
    return agg_df

def detect_anomalies(agg_df, z_threshold):
    agg_df['High_Traffic_Anomaly'] = (
        (agg_df['Total_Vehicles_Z'] > z_threshold) &
        (agg_df['Occupation_Ratio_Z'] > z_threshold)
    )
    return agg_df

def visualize_anomalies(agg_df, anomalies):
    plt.figure(figsize=(10, 6))
    plt.scatter(agg_df['Total Vehicles'], agg_df['Occupation Ratio'], c='blue', label='Normal Traffic')
    plt.scatter(anomalies['Total Vehicles'], anomalies['Occupation Ratio'], c='red', label='High Traffic Anomaly')
    plt.xlabel('Total Vehicles')
    plt.ylabel('Occupation Ratio')
    plt.title('Traffic Anomalies Detected (Right-Tailed Test)')
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Detect high traffic anomalies based on Z-scores.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file containing traffic data.")
    parser.add_argument("--significance_level", type=float, required=True, help="Significance level (e.g., 0.01 for 1%).")
    args = parser.parse_args()

    file_path = args.file_path
    significance_level = args.significance_level

    # Calculate Z-threshold for right-tailed test
    z_threshold = norm.ppf(1 - significance_level)
    print(f"Calculated Z-threshold for significance level {significance_level}: {z_threshold:.4f}")

    df = load_data(file_path)
    df = preprocess_data(df)
    agg_df = aggregate_data(df)
    agg_df = calculate_z_scores(agg_df)
    agg_df = detect_anomalies(agg_df, z_threshold)
    anomalies = agg_df[agg_df['High_Traffic_Anomaly']]

    print("High Traffic Anomalies:")
    print(anomalies.head())
    visualize_anomalies(agg_df, anomalies)

if __name__ == "__main__":
    main()
