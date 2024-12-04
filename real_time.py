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
        model.eval()
        logging.debug(f"Model loaded from {model_path}")

        # Load scaler and label encoder
        scaler = joblib.load(scaler_path)
        le_type = joblib.load(le_type_path)

        logging.debug("Scaler and Label Encoder loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model or tools: {e}")

# Prediction function
def make_prediction(input_data):
    try:
        # Preprocess input data
        input_data = np.array(input_data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        with torch.no_grad():
            prediction = model(torch.tensor(input_data_scaled, dtype=torch.float32))
            predicted_class = torch.argmax(prediction, dim=1).item()

        secondary_disaster_occurred = "Yes" if predicted_class == 1 else "No"
        secondary_disaster_type = le_type.inverse_transform([predicted_class])[0]
        secondary_disaster_intensity = prediction[0, predicted_class].item()

        result = {
            "Secondary Disaster Occurred": secondary_disaster_occurred,
            "Secondary Disaster Type": secondary_disaster_type,
            "Secondary Disaster Intensity": secondary_disaster_intensity
        }
        return result
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return None

# Route for the homepage
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            # Get form data
            disaster_type = int(request.form["disaster_type"])
            magnitude = float(request.form["magnitude"])
            latitude = float(request.form["latitude"])
            longitude = float(request.form["longitude"])
            duration = int(request.form["duration"])
            deaths = int(request.form["deaths"])
            affected = int(request.form["affected"])

            # Prepare input data for the model
            input_data = [disaster_type, magnitude, latitude, longitude, duration, deaths, affected]

            # Make prediction
            result = make_prediction(input_data)

            if not result:
                error = "There was an error processing your request. Please try again later."

        except Exception as e:
            error = f"Error processing form data: {e}"

    return render_template("index.html", result=result, error=error)

# Run the app
if __name__ == "__main__":
    load_model_and_tools()  # Load the model and tools before running the app
    app.run(debug=True)
