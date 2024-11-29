import torch
import torch.onnx
from aimodel import DisasterPredictor

# Initialize the model and load trained weights
model = DisasterPredictor(input_dim=4)  # Adjust input_dim as needed
model.load_state_dict(torch.load("disaster_model.pth"))
model.eval()

# Define the dummy input with the correct shape
dummy_input = torch.randn(1, 4)  # Batch size of 1, 4 features

# Export the PyTorch model to ONNX with dynamic axes and input/output names
torch.onnx.export(
    model,
    dummy_input,
    "disaster_model.onnx",
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    input_names=["input"],
    output_names=["output"]
)

print("Model successfully exported to disaster_model.onnx")
