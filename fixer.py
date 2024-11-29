# import pandas as pd
# import torch
# from sklearn.preprocessing import MinMaxScaler
# import joblib

# # Load the dataset (ensure the correct file path is used)
# file_path = "realdata.xlsx"  # Adjust if the file is elsewhere
# data = pd.read_excel(file_path)  # Correct method for reading Excel files

# # Handle missing day/month with 1 for constructing dates
# data['Start_Month_Filled'] = data['Start Month'].fillna(1).astype(int)
# data['Start_Day_Filled'] = data['Start Day'].fillna(1).astype(int)
# data['End_Month_Filled'] = data['End Month'].fillna(1).astype(int)
# data['End_Day_Filled'] = data['End Day'].fillna(1).astype(int)

# # Create Start_Date and End_Date using filled columns
# data['Start_Date'] = pd.to_datetime(
#     data[['Start Year', 'Start_Month_Filled', 'Start_Day_Filled']].rename(columns={
#         'Start Year': 'year', 'Start_Month_Filled': 'month', 'Start_Day_Filled': 'day'
#     }), errors='coerce'
# )
# data['End_Date'] = pd.to_datetime(
#     data[['End Year', 'End_Month_Filled', 'End_Day_Filled']].rename(columns={
#         'End Year': 'year', 'End_Month_Filled': 'month', 'End_Day_Filled': 'day'
#     }), errors='coerce'
# )

# # Calculate the duration in days
# data['Duration_Days'] = (data['End_Date'] - data['Start_Date']).dt.days

# # Select and rename relevant columns
# filtered_data = data[[
#     'Duration_Days',
#     'Total Damage (\'000 US$)',
#     'Total Deaths',
#     'Total Affected',
#     'Disaster Type'
# ]].dropna()  # Dropping rows with missing values in selected columns

# filtered_data.columns = ['Duration_Days', 'Economic_Loss_USD', 'Deaths', 'Total_Affected', 'Disaster_Type']

# # Create a binary label (1 for "Secondary Disaster", 0 otherwise)
# filtered_data['label'] = (filtered_data['Disaster_Type'] == 'Secondary Disaster').astype(int)

# # Define features and ensure all columns exist in the dataset
# features = ["Duration_Days", "Economic_Loss_USD", "Deaths", "Total_Affected"]
# data = filtered_data.dropna(subset=features)  # Ensure no missing values

# # Extract features (X) and target (y)
# X = data[features]
# y = data['label']

# # Normalize the features
# scaler = MinMaxScaler()
# X_normalized = scaler.fit_transform(X)

# # Convert to PyTorch tensors
# X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
# y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

# # Save the scaler for future use
# joblib.dump(scaler, "scaler.pkl")

# # Export the cleaned data to CSV
# data.to_csv("cleaned_disaster_data.csv", index=False)

# print(f"Data tensor shape: {X_tensor.shape}, Label tensor shape: {y_tensor.shape}")
# print("Cleaned data has been saved as 'cleaned_disaster_data.csv'.")
