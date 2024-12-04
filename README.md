
# NovAI

## Overview

This project's aim is to predict secondary disasters based on primary disaster features, such as disaster type, magnitude, duration, and other related parameters. The model predicts three key outcomes:

1. **Secondary Disaster Occurrence**: Whether a secondary disaster occurred.
2. **Secondary Disaster Type**: The type of secondary disaster that occurred.
3. **Secondary Disaster Intensity**: The intensity of the secondary disaster.

The system uses a machine learning model trained on historical disaster data and predicts these outcomes for new disaster events.

---

## Project Structure

- `datasets/`: Folder containing the processed datasets.
- `models/`: Folder containing the trained models, scalers, and encoders.
- `scripts/`: Python scripts for model training, evaluation, and inference.
- `flask_app/`: Flask-based web application for serving the model.
- `README.md`: This file.

---

## Technologies Used

- **Python**: Core language for development.
- **PyTorch**: Deep learning framework for model training and inference.
- **Scikit-learn**: For data preprocessing and machine learning utilities.
- **Flask**: Lightweight web framework for serving the model.
- **Pandas and NumPy**: For data manipulation and numerical computations.
- **Joblib**: For saving and loading models and preprocessors.
- **ONNX**: For model export and integration.

---

## Features

- **Data Preprocessing**: The data is cleaned, transformed, and scaled to be fed into the machine learning model.
- **Model Training**: The model is trained using PyTorch and consists of three distinct layers for predicting the occurrence, type, and intensity of secondary disasters.
- **Flask Web Interface**: A web interface for predicting outcomes based on user input.
- **Model Export**: The model is exported to the ONNX format for easy deployment on different platforms.

---

## Model

The model is built using a multi-task learning approach to predict three outcomes simultaneously. The architecture consists of:

- **Shared Layer**: A fully connected layer to process common features.
- **Secondary Disaster Occurrence**: A binary classification output for whether a secondary disaster occurred.
- **Secondary Disaster Type**: A multi-class classification output for the type of secondary disaster.
- **Secondary Intensity**: A regression output predicting the intensity of the secondary disaster.

The model is trained using the following loss functions:

- **Binary Cross-Entropy** for secondary disaster occurrence.
- **Cross-Entropy Loss** for secondary disaster type.
- **Mean Squared Error** for secondary disaster intensity.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/disaster-prediction.git
cd disaster-prediction
```

### Dataset

Ensure the processed dataset `processed_dataset.csv` is available in the `datasets/` folder. You can use the provided dataset or add your own.

---

### Flask Application

To run the Flask web app, use the following command:

```bash
python flask_app/app.py
```
This will start a local web server, and the application will be available at http://127.0.0.1:5000.

---

### Training the Model

To train the model:

```bash
python scripts/train.py
```

This script will train the model using the dataset and save the trained model to the models/ folder.

---

### Evaluation

To evaluate the model on the test set:

```bash
python scripts/evaluate.py
```

This will print out classification reports and the Mean Squared Error for intensity predictions.

---

### Web Interface

The Flask application allows you to input disaster data and predict the likelihood of secondary disasters, their type, and intensity.

---

#### Input Fields:
- **Disaster Type**: The type of primary disaster.
- **Magnitude**: The magnitude of the primary disaster.
- **Latitude**: The geographical latitude of the disaster location.
- **Longitude**: The geographical longitude of the disaster location.
- **Duration (Days)**: The duration of the primary disaster in days.
- **Total Deaths**: The total number of deaths caused by the primary disaster.
- **Total Affected**: The total number of people affected by the primary disaster.


---

### Model Export to ONNX

To export the trained model to the ONNX format for deployment:

```bash
python scripts/export_onnx.py
```

This will export the model to the open_model/ folder in the ONNX format

---

### Acknowledgments

- Inspired by real-world disaster prediction research and machine learning applications.  
- **PyTorch** and **Scikit-learn** for their powerful tools in building and training models.  
- **Flask** for easy deployment of the web application.

---