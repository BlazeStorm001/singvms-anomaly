import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('3796_vehicle_data.csv')

# Convert 'Timestamp' to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Create a time window for clustering (group by 15-minute intervals)
df['Time_Window'] = df['Timestamp'].dt.floor('15T')

# Aggregate data by time window (sum the vehicle counts, average occupation ratio)
agg_df = df.groupby('Time_Window').agg({
    'Total Vehicles': 'mean',
    'Occupation Ratio': 'mean'
}).reset_index()

# Calculate Z-scores for 'Total Vehicles' and 'Occupation Ratio'
agg_df['Total_Vehicles_Z'] = zscore(agg_df['Total Vehicles'])
agg_df['Occupation_Ratio_Z'] = zscore(agg_df['Occupation Ratio'])

# Define the significance level (0.1%)
z_threshold = 3.291  # Corresponding to 99.9% confidence interval

# Detect anomalies based on Z-scores
agg_df['High_Traffic_Anomaly'] = (
    (abs(agg_df['Total_Vehicles_Z']) > z_threshold) &
    (abs(agg_df['Occupation_Ratio_Z']) > z_threshold)
)

# Print the rows marked as high traffic anomalies
anomalies = agg_df[agg_df['High_Traffic_Anomaly']]
print("High Traffic Anomalies:")
print(anomalies)

# Visualize anomalies on a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(agg_df['Total Vehicles'], agg_df['Occupation Ratio'], c='blue', label='Normal Traffic')
plt.scatter(anomalies['Total Vehicles'], anomalies['Occupation Ratio'], c='red', label='High Traffic Anomaly')
plt.xlabel('Total Vehicles')
plt.ylabel('Occupation Ratio')
plt.title('Traffic Anomalies Detected (Z-Score Method, 0.1% Significance Level)')
plt.legend()
plt.show()
