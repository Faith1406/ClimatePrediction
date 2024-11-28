import pandas as pd

# Load the dataset with the correct delimiter
data = pd.read_csv("data.csv", delimiter=';', quotechar='"')

# Display the first few rows to confirm
print(data.head())
print(data.columns)

# Save the corrected dataset to a new CSV file
data.to_csv("corrected_data.csv", index=False)

print("Corrected dataset saved as 'corrected_data.csv'.")
