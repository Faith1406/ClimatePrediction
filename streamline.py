# import torch
# import pandas as pd
# import joblib
# from sklearn.metrics import accuracy_score

# from aimodel import DisasterPredictor
# # Load the trained model with weights_only=True
# model = DisasterPredictor(input_dim=4)  # Ensure input_dim matches the training setup
# model.load_state_dict(torch.load("disaster_model.pth", weights_only=True))
# model.eval()

# # Load the scaler
# scaler = joblib.load("scaler.pkl")

# # Example input data for prediction (modify as needed)
# example_data = [
#     [30, 100000, 500, 10000],  # Example disaster 1
#     [10, 200000, 100, 5000],   # Example disaster 2
#     [365, 1000000, 50, 500000] # Example disaster 3
# ]

# # Corresponding actual labels for the above data (replace with real labels)
# actual_labels = [1, 0, 1]  # Example: 1 for Secondary Disaster, 0 otherwise

# # Convert example data to DataFrame
# example_df = pd.DataFrame(example_data, columns=["Duration_Days", "Economic_Loss_USD", "Deaths", "Total_Affected"])

# # Scale the input data after converting to NumPy array
# scaled_data = scaler.transform(example_df.to_numpy())

# # Convert scaled data to PyTorch tensor
# input_tensor = torch.tensor(scaled_data, dtype=torch.float32)

# # Make predictions
# with torch.no_grad():
#     predictions = model(input_tensor)

# # Convert predictions to binary output (0 or 1)
# predicted_labels = (predictions > 0.5).int().squeeze().numpy()

# # Calculate accuracy
# accuracy = accuracy_score(actual_labels, predicted_labels)

# # Print the results and accuracy
# for i, (pred, label) in enumerate(zip(predictions, predicted_labels)):
#     print(f"Disaster {i + 1}: Probability of 'Secondary Disaster': {pred.item():.4f}, Predicted Label: {label}")

# print(f"\nAccuracy: {accuracy * 100:.2f}%")
