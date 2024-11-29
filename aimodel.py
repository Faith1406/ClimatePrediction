import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score

# Load the cleaned dataset
data = pd.read_csv("cleaned_disaster_data.csv")

# Check the dataset structure
print(data.head())

# Select relevant features and target
features = ["Duration_Days", "Economic_Loss_USD", "Deaths", "Total_Affected"]

# Assuming 'label' is the target variable
X = data[features].values
y = data["label"].values

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension to match output shape

# Split data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
class DisasterPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DisasterPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Sigmoid activation to get a probability between 0 and 1
        )

    def forward(self, x):
        return self.fc(x)

# Initialize the model
input_dim = train_X.shape[1]  # Based on selected features
model = DisasterPredictor(input_dim=input_dim)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
losses = []
for epoch in range(50):  # Adjust epochs if needed
    model.train()
    optimizer.zero_grad()
    output = model(train_X)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # Calculate accuracy for training
    predicted_train = (output > 0.5).float()  # Convert probabilities to class labels (0 or 1)
    train_accuracy = (predicted_train == train_y).float().mean().item()  # Compute accuracy

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.4f}")

# Save the model and scaler
torch.save(model.state_dict(), "disaster_model.pth")
joblib.dump(scaler, "scaler.pkl")

# Evaluate on the test set
model.eval()
with torch.no_grad():
    test_preds = model(test_X)
    test_loss = criterion(test_preds, test_y)
    predicted_test = (test_preds > 0.5).float()  # Convert probabilities to class labels
    test_accuracy = (predicted_test == test_y).float().mean().item()

    print(f"Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}")

    # If you want to calculate more metrics like precision, recall, and F1-score
    print("Test Accuracy:", test_accuracy)

    # Optionally: Calculate classification report (precision, recall, f1-score)
    from sklearn.metrics import classification_report
    print(classification_report(test_y.numpy(), predicted_test.numpy()))

# Visualize training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, 51), losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()
