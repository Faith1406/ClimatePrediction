import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib  # Used for saving preprocessor
import os  # For creating directories if they don't exist

# Define the model architecture
class DisasterPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(DisasterPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)

# Preprocess the data (One-Hot Encoding for categorical variables)
def preprocess_data(df):
    categorical_columns = ['Primary_Disaster_Type', 'Region']
    numerical_columns = ['Duration_Days', 'Economic_Loss_USD', 'Deaths', 'Total_Affected', 'Disaster_Frequency']

    # StandardScaler for numerical features and OneHotEncoder for categorical features
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_columns),
                      ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
                     ])

    X = preprocessor.fit_transform(df)  # Apply transformations
    y = df['label'].values  # Labels column
    return X, y, preprocessor

# Load the dataset (adjust path if necessary)
data = pd.read_csv("datasets/synthetic_disaster_data_with_type.csv")

# Preprocess the data
X, y, preprocessor = preprocess_data(data)

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
input_size = X.shape[1]  # This will be the number of features after preprocessing
model = DisasterPredictionModel(input_size=input_size)

# Set up loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate of 0.001

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    inputs = torch.tensor(X_train, dtype=torch.float32)  # Convert to tensor
    targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Convert to tensor and reshape

    # Forward pass
    outputs = model(inputs)  # Get predictions from the model
    loss = criterion(outputs, targets)  # Compute the loss

    # Backward pass
    optimizer.zero_grad()  # Zero the gradients before the backward pass
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the model parameters

    # Print loss for every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Define the directories where the model and preprocessor will be saved
model_dir = 'models'
preprocessor_dir = 'processed_file'

# Ensure the directories exist, create them if necessary
os.makedirs(model_dir, exist_ok=True)
os.makedirs(preprocessor_dir, exist_ok=True)

# Save the trained model's weights and preprocessor
model_save_path = os.path.join(model_dir, 'secondary_disaster_model.pth')
preprocessor_save_path = os.path.join(preprocessor_dir, 'preprocessor.pkl')

torch.save(model.state_dict(), model_save_path)  # Save model weights
joblib.dump(preprocessor, preprocessor_save_path)  # Save preprocessor

print(f"Training complete. Model saved to {model_save_path} and preprocessor saved to {preprocessor_save_path}.")
