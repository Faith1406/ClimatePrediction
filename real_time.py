import torch
import torch.nn as nn
import pandas as pd
import joblib  # For loading the preprocessor
import numpy as np
from openvino.runtime import Core  # Updated import for OpenVINO

# Define the model class (ensure this matches the model definition used during training)
class DisasterPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(DisasterPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Load the preprocessor
preprocessor = joblib.load('preprocessor.pkl')

# Define input size based on preprocessed sample input
input_size = preprocessor.transform(pd.DataFrame([{
    'Duration_Days': 0,
    'Economic_Loss_USD': 0,
    'Deaths': 0,
    'Total_Affected': 0,
    'Primary_Disaster_Type': 'Flood',
    'Region': 'North',
    'Disaster_Frequency': 0
}])).shape[1]

# Load the OpenVINO model using the Core API
ie = Core()

# Load the network from IR files (XML and BIN)
compiled_model = ie.compile_model(model="secondary_disaster_model.xml", device_name="CPU")

# Retrieve input and output information
input_blob = compiled_model.inputs[0]
output_blob = compiled_model.outputs[0]

# Prediction function using OpenVINO
def predict_secondary_disaster(input_df):
    # Preprocess input data
    input_data = preprocessor.transform(input_df)
    input_data = np.array(input_data, dtype=np.float32)

    # Perform inference
    result = compiled_model([input_data])
    predicted_label = result[output_blob]

    # Apply threshold for binary classification
    return "YES" if predicted_label[0] > 0.5 else "NO"

# Function to take user input and make a prediction
def get_user_input():
    print("Enter details for secondary disaster prediction:")

    # Get user inputs for the prediction
    duration = float(input("Duration of the disaster (in days): "))
    economic_loss = float(input("Estimated economic loss (in INR): "))
    deaths = int(input("Number of deaths reported: "))
    affected = int(input("Total number of affected people: "))

    # Adjusted to India-specific disaster types
    disaster_type = input("Type of primary disaster (e.g., Flood, Earthquake, Cyclone, Landslide): ")

    # Adjusted to Indian regions
    region = input("Region (e.g., North, South, East, West, Central, Northeast): ")

    disaster_frequency = int(input("Disaster frequency in the region (on a scale of 1-10): "))

    # Return the inputs in dictionary form for easier DataFrame creation
    return {
        'Duration_Days': duration,
        'Economic_Loss_USD': economic_loss,  # You may consider converting INR to USD if necessary
        'Deaths': deaths,
        'Total_Affected': affected,
        'Primary_Disaster_Type': disaster_type,
        'Region': region,
        'Disaster_Frequency': disaster_frequency
    }

# Main function to execute the prediction
def main():
    # Get user input
    user_inputs = get_user_input()

    # Create DataFrame from user input
    input_df = pd.DataFrame([user_inputs])

    # Make the prediction using the input data
    result = predict_secondary_disaster(input_df)

    print(f"Secondary disaster prediction: {result}")

if __name__ == "__main__":
    main()
