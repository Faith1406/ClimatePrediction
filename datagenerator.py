import numpy as np
import pandas as pd
import random

# Set a seed for reproducibility
random.seed(42)
np.random.seed(42)

# Function to generate synthetic disaster data with primary disaster type and region
def generate_synthetic_data(num_samples=100):
    disaster_types = ['Flood', 'Earthquake', 'Wildfire', 'Hurricane', 'Tornado']
    regions = ['Coastal', 'Urban', 'Rural', 'Mountainous', 'Flood-Prone']

    data = {
        "Primary_Disaster_Type": [random.choice(disaster_types) for _ in range(num_samples)],
        "Region": [random.choice(regions) for _ in range(num_samples)],
        "Duration_Days": [random.randint(1, 30) for _ in range(num_samples)],  # Random duration between 1 and 30 days
        "Economic_Loss_USD": [random.randint(100000, 5000000) for _ in range(num_samples)],  # Random economic loss between 100k and 5M USD
        "Deaths": [random.randint(0, 1000) for _ in range(num_samples)],  # Random deaths between 0 and 1000
        "Total_Affected": [random.randint(100, 10000) for _ in range(num_samples)],  # Random affected population between 100 and 10,000
        "Disaster_Frequency": [random.randint(1, 10) for _ in range(num_samples)],  # Frequency of disasters in the region (higher = more likely secondary disaster)
        "label": [random.choice([0, 1]) for _ in range(num_samples)]  # Random label (0 or 1)
    }

    return pd.DataFrame(data)

# Generate synthetic data
synthetic_data = generate_synthetic_data(num_samples=50)

# Save to CSV for easy loading (optional)
synthetic_data.to_csv("synthetic_disaster_data_with_type.csv", index=False)

# Preview the generated data
print(synthetic_data.head())
