from flask import Flask, render_template, request
import torch
import joblib
import numpy as np
from threading import Thread
import logging
from aimodel import DisasterPredictionModelWithLSTM

# Initialize Flask app
app = Flask(__name__)

# Initialize the model and tools
model = None
scaler = None
le_type = None

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load model and tools when the app starts
def load_model_and_tools():
    global model, scaler, le_type

    model_path = "models/disaster_prediction_model.pth"
    scaler_path = "models/scaler.pkl"
    le_type_path = "models/le_type.pkl"

    input_size = 7  # Adjust as needed
    num_classes = 5  # Adjust as needed

    try:
        # Load the model
        model = DisasterPredictionModelWithLSTM(input_size=input_size, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()  # Set model to evaluation mode
        logging.debug("Model loaded successfully.")

        # Load scaler and label encoder
        scaler = joblib.load(scaler_path)
        le_type = joblib.load(le_type_path)
        logging.debug("Scaler and label encoder loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model and tools: {e}")

# Call the load_model_and_tools function in a separate thread to avoid blocking the app startup
Thread(target=load_model_and_tools).start()

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

            logging.debug(f"Input data: {data}")

            # Preprocess the input data
            input_data = np.array([[data[feature] for feature in data]]).reshape(1, -1)
            logging.debug(f"Input data array: {input_data}")

            input_scaled = scaler.transform(input_data)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            # Get predictions from the model
            with torch.no_grad():
                occur_pred, type_pred, intensity_pred = model(input_tensor)

            # Decode predictions
            occur_prob = occur_pred.item()
            type_idx = torch.argmax(type_pred, dim=1).item()
            disaster_type = le_type.inverse_transform([type_idx])[0]
            intensity = intensity_pred.item()

            # Prepare the result to display
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
            logging.error(f"Error occurred: {e}")
            error = "An unexpected error occurred. Please try again."
            return render_template('index.html', error=error)

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
