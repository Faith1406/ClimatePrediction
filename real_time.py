import pandas as pd
import numpy as np
import joblib
from openvino.runtime import Core  # Ensure OpenVINO is installed and set up correctly

# Load the preprocessor (adjusted path)
preprocessor = joblib.load('processed_file/preprocessor.pkl')  # Path to preprocessor.pkl

# Load the OpenVINO model using the Core API (adjusted paths for XML and BIN files)
ie = Core()

# Load the network from IR files (XML and BIN) stored in the 'open_model/' folder
compiled_model = ie.compile_model(model="open_model/secondary_disaster_model.xml", device_name="CPU")

# Retrieve input and output information
input_blob = compiled_model.inputs[0]
output_blob = compiled_model.outputs[0]

def predict_secondary_disaster(input_df):
    # Preprocess input data
    input_data = preprocessor.transform(input_df)
    input_data = np.array(input_data, dtype=np.float32)

    # Perform inference
    result = compiled_model([input_data])
    predicted_label = result[output_blob]

    # Apply threshold for binary classification
    return "YES" if predicted_label[0] > 0.5 else "NO"

def validate_numeric_input(value, field_name, min_value=None, max_value=None):
    try:
        value = float(value)
        if min_value is not None and value < min_value:
            return f"Error: {field_name} should be at least {min_value}."
        if max_value is not None and value > max_value:
            return f"Error: {field_name} should not exceed {max_value}."
        return None  # No error
    except ValueError:
        return f"Error: {field_name} should be a valid number."

def validate_text_input(value, field_name, valid_values=None):
    if not value or (valid_values and value not in valid_values):
        return f"Error: {field_name} should be a valid option."
    return None  # No error

# Define the run_prediction function that Flask will call
def run_prediction(duration, economic_loss, deaths, affected, disaster_type, region, disaster_frequency):
    # Validate inputs
    error = validate_numeric_input(duration, 'Duration of the disaster', min_value=1)
    if error:
        return error
    error = validate_numeric_input(economic_loss, 'Estimated economic loss', min_value=0)
    if error:
        return error
    error = validate_numeric_input(deaths, 'Number of deaths', min_value=0)
    if error:
        return error
    error = validate_numeric_input(affected, 'Total affected people', min_value=0)
    if error:
        return error
    error = validate_text_input(disaster_type, 'Type of primary disaster', valid_values=['Flood', 'Earthquake', 'Cyclone', 'Landslide', 'Tsunami'])
    if error:
        return error
    error = validate_text_input(region, 'Region', valid_values=['North', 'South', 'East', 'West', 'Central', 'Northeast'])
    if error:
        return error
    error = validate_numeric_input(disaster_frequency, 'Disaster frequency', min_value=1, max_value=10)
    if error:
        return error

    # Create input dictionary
    user_input = {
        'Duration_Days': duration,
        'Economic_Loss_USD': economic_loss,
        'Deaths': deaths,
        'Total_Affected': affected,
        'Primary_Disaster_Type': disaster_type,
        'Region': region,
        'Disaster_Frequency': disaster_frequency
    }

    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_input])

    # Get prediction result
    result = predict_secondary_disaster(input_df)

    return result
