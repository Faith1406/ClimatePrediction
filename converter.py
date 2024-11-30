import torch
import torch.onnx
import joblib
from aimodel import DisasterPredictionModel

# Load the preprocessor
preprocessor = joblib.load('preprocessor.pkl')

# Assuming the preprocessed data gives 15 features, so we set input_size=15
model = DisasterPredictionModel(input_size=15)  # Adjust input size

# Load the trained model weights
model.load_state_dict(torch.load("secondary_disaster_model.pth"))
model.eval()

# Create a dummy input for the model's forward pass (15 features)
dummy_input = torch.randn(1, 15)  # Ensure it has 15 features

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, "secondary_disaster_model.onnx")
