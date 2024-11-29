import torch
import torch.nn as nn
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from openvino.runtime import Core

# Load dataset
data = pd.read_csv("cleaned_disaster_data.csv")

# Features and target selection
features = ["Duration_Days", "Economic_Loss_USD", "Deaths", "Total_Affected"]
target_condition = data["Disaster_Type"] == "Secondary Disaster"
data["label"] = target_condition.astype(int)
data = data.dropna(subset=features)

X = data[features].values
y = data["label"].values

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
class DisasterPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DisasterPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Initialize the model
input_dim = train_X.shape[1]
model = DisasterPredictor(input_dim=input_dim)

# Train the PyTorch model
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):  # Reduced epochs for quick demonstration
    model.train()
    optimizer.zero_grad()
    output = model(train_X)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), "disaster_model.pth")

# PyTorch Inference Benchmark
model.eval()
with torch.no_grad():
    start_time = time.time()
    pytorch_preds = model(test_X)
    pytorch_duration = time.time() - start_time

pytorch_preds = (pytorch_preds.numpy() > 0.5).astype(int)
pytorch_accuracy = (pytorch_preds == test_y.numpy()).mean()
print(f"PyTorch Inference Time: {pytorch_duration:.4f} seconds")
print(f"PyTorch Accuracy: {pytorch_accuracy:.4f}")

# OpenVINO Inference Benchmark
core = Core()
compiled_model = core.compile_model("disaster_model.xml", device_name="CPU")
infer_request = compiled_model.create_infer_request()

# Get the actual input and output names
input_name = compiled_model.inputs[0].get_any_name()
output_name = compiled_model.outputs[0].get_any_name()

start_time = time.time()
openvino_results = infer_request.infer({input_name: test_X.numpy()})
openvino_duration = time.time() - start_time

openvino_preds = (openvino_results[output_name] > 0.5).astype(int)
openvino_accuracy = (openvino_preds == test_y.numpy()).mean()

print(f"OpenVINO Inference Time: {openvino_duration:.4f} seconds")
print(f"OpenVINO Accuracy: {openvino_accuracy:.4f}")
