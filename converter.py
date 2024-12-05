import torch
import torch.nn as nn
import joblib
import numpy as np
from aimodel import DisasterPredictionModelWithLSTM  # Assuming model class in separate file

# Paths
scaler_path = "models/scaler.pkl"
encoder_path = "models/le_type.pkl"
model_path = "models/disaster_prediction_model.pth"
onnx_path = "open_model/disaster_prediction_model.onnx"

# Load the scaler and encoder
scaler = joblib.load(scaler_path)
le_type = joblib.load(encoder_path)
num_classes = len(le_type.classes_)

# Define the model structure
input_size = scaler.mean_.shape[0]  # Number of input features
model = DisasterPredictionModelWithLSTM(input_size=input_size, num_classes=num_classes)

# Load the trained model weights
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Create a dummy input tensor (shape: batch_size x input_size)
batch_size = 1  # You can adjust this as needed
dummy_input = torch.randn(batch_size, input_size, dtype=torch.float32)

# Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,  # Store trained parameter weights
    opset_version=11,    # Use a compatible ONNX opset
    do_constant_folding=True,  # Optimize the model graph
    input_names=['input'],     # Name of the input layer
    output_names=['occur_output', 'type_output', 'intensity_output'],  # Names of the output layers
    dynamic_axes={'input': {0: 'batch_size'}, 'occur_output': {0: 'batch_size'},
                  'type_output': {0: 'batch_size'}, 'intensity_output': {0: 'batch_size'}}  # Allow dynamic batch sizes
)

print(f"Model successfully exported to {onnx_path}")
