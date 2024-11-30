import torch
import torch.onnx
import joblib
import os
from aimodel import DisasterPredictionModel

# Load the preprocessor
preprocessor = joblib.load('processed_file/preprocessor.pkl')  # Adjusted path

# Assuming the preprocessed data gives 15 features, so we set input_size=15
model = DisasterPredictionModel(input_size=15)  # Adjust input size

# Load the trained model weights
model.load_state_dict(torch.load("models/secondary_disaster_model.pth"))  # Adjusted path
model.eval()

# Create a dummy input for the model's forward pass (15 features)
dummy_input = torch.randn(1, 15)  # Ensure it has 15 features

# Ensure the open_model/ directory exists
output_dir = 'open_model'
os.makedirs(output_dir, exist_ok=True)

# Define the ONNX export path
onnx_model_path = os.path.join(output_dir, "secondary_disaster_model.onnx")

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, onnx_model_path)

print(f"Model exported to ONNX format and saved at: {onnx_model_path}")
