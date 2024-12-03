import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = "datasets/unprocessed_dataset.xlsx"  # Replace with your Excel file path
df = pd.read_excel(file_path)

# Display dataset info
print("Dataset Info:")
print(df.info())
print("\nSample Data:")
print(df.head())

# Fill missing values
# Fill numeric columns with the median
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical columns with the mode
categorical_cols = df.select_dtypes(include=["object"]).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Feature Engineering
# Handle missing components in Start and End dates
df['Start Month'] = df['Start Month'].fillna(1)  # Default missing months to January
df['Start Day'] = df['Start Day'].fillna(1)      # Default missing days to the first day of the month
df['End Month'] = df['End Month'].fillna(1)      # Default missing months to January
df['End Day'] = df['End Day'].fillna(1)          # Default missing days to the first day of the month

# Convert Start and End dates to datetime
df['Start Date'] = pd.to_datetime(
    dict(year=df['Start Year'], month=df['Start Month'], day=df['Start Day']),
    errors='coerce'
)
df['End Date'] = pd.to_datetime(
    dict(year=df['End Year'], month=df['End Month'], day=df['End Day']),
    errors='coerce'
)

# Calculate duration of disaster in days
df['Duration (Days)'] = (df['End Date'] - df['Start Date']).dt.days
df['Duration (Days)'] = df['Duration (Days)'].fillna(0).clip(lower=0)

# Add seasonal feature
df['Season'] = df['Start Date'].dt.month % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Fall

# Encode categorical features
label_encoders = {}
for col in ['Disaster Type', 'Disaster Subtype', 'Region', 'Country']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for future use

# Normalize numeric features
scaler = StandardScaler()
scaled_cols = ['Magnitude', 'Latitude', 'Longitude', 'Duration (Days)', 'Total Deaths', 'Total Affected']
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Ensure datasets folder exists in the root directory
os.makedirs("datasets", exist_ok=True)

# Save the preprocessed dataset as CSV
output_path = os.path.join("datasets", "processed_dataset.csv")
df.to_csv(output_path, index=False)

print(f"\nData preprocessing complete! Preprocessed dataset saved to {output_path}")
