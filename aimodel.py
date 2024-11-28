import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("corrected_data.csv")

# Define the threshold value for disaster
threshold = 100

# Create a label column based on the max value across the 1st to 31st columns
data['label'] = (data[['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th',
                        '11th', '12th', '13th', '14th', '15th', '16th', '17th', '18th', '19th',
                        '20th', '21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th',
                        '29th', '30th', '31st']].max(axis=1) > threshold).astype(int)

# Select features and target
X = data[["month", "1st", "2nd"]].values
y = data["label"].values

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Split data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
class DisasterPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DisasterPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Initialize the model
model = DisasterPredictor(input_dim=3)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# List to store loss values for plotting
losses = []

# Train the model
for epoch in range(50):  # 50 epochs
    optimizer.zero_grad()
    output = model(train_X)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()

    # Store the loss for visualization
    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "disaster_model.pth")

# Save the scaler for later use
import joblib
joblib.dump(scaler, 'scaler.pkl')

# Plot the loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
