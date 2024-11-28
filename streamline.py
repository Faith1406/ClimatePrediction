import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
from aimodel import DisasterPredictor

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

# Load the saved scaler and apply transformation
scaler = joblib.load('scaler.pkl')
X = scaler.transform(X)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Split data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
model = DisasterPredictor(input_dim=3)
model.load_state_dict(torch.load('disaster_model.pth'))
model.eval()  # Set the model to evaluation mode

# Evaluate the model accuracy
correct_predictions = 0
total_predictions = 0

accuracy_per_epoch = []

with torch.no_grad():  # We don't need gradients during evaluation
    # Get the model's predictions for the test set
    predictions = model(test_X)

    # Apply a threshold of 0.5 to classify as 0 or 1
    predicted_labels = (predictions > 0.5).float()

    # Calculate accuracy
    correct_predictions = (predicted_labels == test_y).float()
    accuracy = correct_predictions.mean()  # Mean of correct predictions gives accuracy

    print(f"Accuracy on test data: {accuracy.item() * 100:.2f}%")

    # For visualization, let's store the accuracy values over time
    accuracy_per_epoch.append(accuracy.item())

# Plot the accuracy on test data
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(accuracy_per_epoch) + 1), accuracy_per_epoch, label='Test Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
