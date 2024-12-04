import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from sklearn.metrics import classification_report
import os

# Paths
dataset_path = "datasets/processed_dataset.csv"
scaler_path = "models/scaler.pkl"
encoder_path = "models/le_type.pkl"
model_path = "models/disaster_prediction_model.pth"

# Load the dataset
df = pd.read_csv(dataset_path)

# Define features and labels
features = ['Disaster Type', 'Magnitude', 'Latitude', 'Longitude', 'Duration (Days)', 'Total Deaths', 'Total Affected']
target_secondary_occur = 'Secondary Disaster Occurred'
target_secondary_type = 'Secondary Disaster Type'
target_secondary_intensity = 'Secondary Intensity'

# Generate labels (for demonstration; replace with actual labels)
np.random.seed(42)
df[target_secondary_occur] = np.random.choice([0, 1], size=len(df))
df[target_secondary_type] = np.random.choice(range(5), size=len(df))  # Assuming 5 disaster types
df[target_secondary_intensity] = np.random.uniform(0, 10, size=len(df))

# Encode features
scaler = StandardScaler()
X = scaler.fit_transform(df[features])  # Fit and transform features
joblib.dump(scaler, scaler_path)        # Save the fitted scaler

# Encode labels
le_type = LabelEncoder()
df[target_secondary_type] = le_type.fit_transform(df[target_secondary_type])
joblib.dump(le_type, encoder_path)      # Save the fitted label encoder

# Prepare data
y_occur = df[target_secondary_occur].values
y_type = df[target_secondary_type].values
y_intensity = df[target_secondary_intensity].values

# Train-test split
X_train, X_test, y_occur_train, y_occur_test, y_type_train, y_type_test, y_intensity_train, y_intensity_test = train_test_split(
    X, y_occur, y_type, y_intensity, test_size=0.2, random_state=42
)

# PyTorch Dataset class
class DisasterDataset(Dataset):
    def __init__(self, X, y_occur, y_type, y_intensity):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_occur = torch.tensor(y_occur, dtype=torch.float32)
        self.y_type = torch.tensor(y_type, dtype=torch.long)
        self.y_intensity = torch.tensor(y_intensity, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_occur[idx], self.y_type[idx], self.y_intensity[idx]

# Create datasets and dataloaders
train_dataset = DisasterDataset(X_train, y_occur_train, y_type_train, y_intensity_train)
test_dataset = DisasterDataset(X_test, y_occur_test, y_type_test, y_intensity_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model with LSTM
class DisasterPredictionModelWithLSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=64, num_layers=2):
        super(DisasterPredictionModelWithLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.occur_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.type_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )
        self.intensity_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Add a sequence dimension for LSTM
        x = x.unsqueeze(1)  # Add sequence length = 1
        lstm_out, _ = self.lstm(x)
        shared = self.shared_layer(lstm_out[:, -1, :])  # Use the last hidden state
        occur = self.occur_layer(shared)
        type_pred = self.type_layer(shared)
        intensity = self.intensity_layer(shared)
        return occur, type_pred, intensity

# Instantiate the model
input_size = X.shape[1]
num_classes = len(le_type.classes_)

# Create the model instance
model = DisasterPredictionModelWithLSTM(input_size, num_classes)

# Define loss functions and optimizer
criterion_occur = nn.BCELoss()
criterion_type = nn.CrossEntropyLoss()
criterion_intensity = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_occur_batch, y_type_batch, y_intensity_batch in train_loader:
        optimizer.zero_grad()

        # Forward pass
        occur_pred, type_pred, intensity_pred = model(X_batch)

        # Calculate losses
        loss_occur = criterion_occur(occur_pred.squeeze(), y_occur_batch)
        loss_type = criterion_type(type_pred, y_type_batch)
        loss_intensity = criterion_intensity(intensity_pred.squeeze(), y_intensity_batch)

        # Backward pass
        loss = loss_occur + loss_type + loss_intensity
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Save the model after training
torch.save(model.state_dict(), model_path)

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred_occur = []
    y_pred_type = []
    y_pred_intensity = []
    y_true_occur = []
    y_true_type = []
    y_true_intensity = []
    for X_batch, y_occur_batch, y_type_batch, y_intensity_batch in test_loader:
        occur_pred, type_pred, intensity_pred = model(X_batch)
        y_pred_occur.append(occur_pred.squeeze().numpy())
        y_pred_type.append(type_pred.argmax(dim=1).numpy())
        y_pred_intensity.append(intensity_pred.squeeze().numpy())
        y_true_occur.append(y_occur_batch.numpy())
        y_true_type.append(y_type_batch.numpy())
        y_true_intensity.append(y_intensity_batch.numpy())

# Flatten the lists
y_pred_occur = np.concatenate(y_pred_occur)
y_pred_type = np.concatenate(y_pred_type)
y_pred_intensity = np.concatenate(y_pred_intensity)
y_true_occur = np.concatenate(y_true_occur)
y_true_type = np.concatenate(y_true_type)
y_true_intensity = np.concatenate(y_true_intensity)

# Print classification report for type prediction
print("Classification Report (Type Prediction):")
print(classification_report(y_true_type, y_pred_type))

# Print mean squared error for intensity prediction
print(f"Mean Squared Error (Intensity Prediction): {np.mean((y_pred_intensity - y_true_intensity) ** 2)}")
