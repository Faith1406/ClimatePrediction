from flask import Flask, render_template, request
import torch
import joblib
import numpy as np
from aimodel import DisasterPredictionModel  # Assuming the model code is in a file named aimodel.py

# Create Flask app
app = Flask(__name__)

# Load the trained model and preprocessing tools
model_path = "models/disaster_prediction_model.pth"
scaler_path = "models/scaler.pkl"
le_type_path = "models/le_type.pkl"

# Load model parameters dynamically (stored in the model or config file)
input_size = 7  # Number of input features (static, adjust if needed)
num_classes = 5  # Number of disaster types (static, adjust if needed)

# Load model
model = DisasterPredictionModel(input_size=input_size, num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()  # Set model to evaluation mode

# Load scaler and label encoder
scaler = joblib.load(scaler_path)
le_type = joblib.load(le_type_path)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Extract input data from the form
            data = {
                "Disaster Type": int(request.form['disaster_type']),
                "Magnitude": float(request.form['magnitude']),
                "Latitude": float(request.form['latitude']),
                "Longitude": float(request.form['longitude']),
                "Duration (Days)": int(request.form['duration']),
                "Total Deaths": int(request.form['deaths']),
                "Total Affected": int(request.form['affected'])
            }

            # Ensure that all required fields are provided
            if any(v is None for v in data.values()):
                raise ValueError("All input fields must be filled out.")

            # Preprocess the input data
            input_data = np.array([[data[feature] for feature in data]])
            input_scaled = scaler.transform(input_data)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            # Get predictions from the model
            with torch.no_grad():
                occur_pred, type_pred, intensity_pred = model(input_tensor)

            # Decode predictions
            occur_prob = occur_pred.item()  # Probability of secondary disaster occurrence
            type_idx = torch.argmax(type_pred, dim=1).item()
            disaster_type = le_type.inverse_transform([type_idx])[0]
            intensity = intensity_pred.item()

            # Format the result for display
            result = {
                "Secondary Disaster Occurred": round(occur_prob, 4),
                "Secondary Disaster Type": disaster_type,
                "Secondary Disaster Intensity": round(intensity, 4)
            }

            return render_template('index.html', result=result)

        except ValueError as ve:
            error = str(ve)
            return render_template('index.html', error=error)
        except Exception as e:
            # Log the error for debugging
            print(f"Error occurred: {e}")
            error = "An unexpected error occurred. Please try again."
            return render_template('index.html', error=error)

    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
