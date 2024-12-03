import torch
import torch.onnx
import joblib
import os
from aimodel import DisasterPredictionModel  # Ensure this is the correct model class

# Load the preprocessor (make sure it's correct)
preprocessor = joblib.load('processed_file/preprocessor.pkl')  # Adjust path if needed

# Initialize the model with the correct input size (assuming 15 features in the input)
model = DisasterPredictionModel(input_size=15)  # Ensure this is the correct input size

# Load the trained model weights (ensure the model path is correct)
model_weights_path = "models/disaster_prediction_model.pth"

# Make sure that the model path exists
if os.path.exists(model_weights_path):
    model.load_state_dict(torch.load(model_weights_path))  # Load the saved weights into the model
else:
    raise FileNotFoundError(f"Model weights file not found at: {model_weights_path}")

model.eval()  # Set the model to evaluation mode

# Create a dummy input tensor with 15 features for the forward pass (make sure it's the correct size)
dummy_input = torch.randn(1, 15)  # Assuming the model expects 15 features

# Ensure the output directory exists for saving the ONNX model
output_dir = 'open_model'
os.makedirs(output_dir, exist_ok=True)

# Define the ONNX model output path
onnx_model_path = os.path.join(output_dir, "secondary_disaster_model.onnx")

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)  # You can change opset_version if needed

print(f"Model successfully exported to ONNX format and saved at: {onnx_model_path}")
