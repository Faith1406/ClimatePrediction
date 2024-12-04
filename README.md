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
- **LSTM (Long Short-Term Memory)**: A type of recurrent neural network used for sequential data modeling.
- **Intel AI Analytics Toolkit**: Optimized libraries to accelerate PyTorch performance.
- **Scikit-learn**: For data preprocessing and machine learning utilities.
- **Flask**: Lightweight web framework for serving the model.
- **Pandas and NumPy**: For data manipulation and numerical computations.
- **Joblib**: For saving and loading models and preprocessors.
- **ONNX**: For model export and integration.

---

## Features

- **Data Preprocessing**: The data is cleaned, transformed, and scaled to be fed into the machine learning model.
- **LSTM-Based Model**: The core architecture leverages LSTM layers to capture sequential dependencies in disaster data, particularly for predicting secondary disaster outcomes.
- **Intel AI Toolkit Optimization**: The model is optimized for better performance using Intel’s AI Toolkit for PyTorch.
- **Flask Web Interface**: A web interface for predicting outcomes based on user input.
- **Model Export**: The model is exported to the ONNX format for easy deployment on different platforms.

---

# How to Test the Disaster Prediction Model

To test the disaster prediction model, we provide a `real_time.py` script. This script processes real-time input data, utilizes the trained LSTM model, and outputs predictions for secondary disaster occurrence, type, and intensity.

---

## Prerequisites
Before running the script, ensure the following:
1. The trained model (`disaster_prediction_model.pth`) is available in the `models` directory.
2. The necessary preprocessing tools (`scaler.pkl` and `le_type.pkl`) are also in the `models` directory.
3. The required Python dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
---

## Model

The model uses a Long Short-Term Memory (LSTM) architecture to effectively handle sequential features in disaster data. The architecture includes:  

- **Shared LSTM Layer**: Captures sequential dependencies and extracts features relevant to all three outputs.  
- **Secondary Disaster Occurrence**: A binary classification output for whether a secondary disaster occurred.  
- **Secondary Disaster Type**: A multi-class classification output for the type of secondary disaster.  
- **Secondary Intensity**: A regression output predicting the intensity of the secondary disaster.

The model is trained using the following loss functions:

- **Binary Cross-Entropy** for secondary disaster occurrence.
- **Cross-Entropy Loss** for secondary disaster type.
- **Mean Squared Error** for secondary disaster intensity.

---

# Why Use LSTM for Disaster Prediction?

In this project, we utilize a Long Short-Term Memory (LSTM) model to predict secondary disasters, including their occurrence, type, and intensity. Below, we explain why LSTMs were chosen for this project and how they outperform other models like feedforward neural networks (FNN) or simpler machine learning algorithms.

---

## 1. Sequential Data Handling
Disaster data often involves temporal or sequential relationships. For example:
- Changes in magnitudes, locations, or duration might indicate patterns leading to secondary disasters.
- Historical disaster data may contain trends that are important for predictions.

LSTMs excel in capturing these **time-dependent relationships** because they are designed to handle sequential data through memory cells and gates, allowing them to retain relevant past information while ignoring irrelevant data.

---

## 2. Feature Interactions Over Time
Unlike traditional models like FNNs, which process data independently for each instance, LSTMs can learn **dynamic interactions between features over time**, such as:
- How disaster magnitude correlates with affected populations over time.
- How geospatial features (latitude and longitude) interact with other variables like duration and total deaths.

These interactions are crucial in predicting secondary disasters more accurately.

---

## 3. Advantages Over Feedforward Neural Networks
| **Criteria**               | **LSTM**                                | **FNN**                                |
|-----------------------------|-----------------------------------------|----------------------------------------|
| **Captures Temporal Patterns** | ✅ Yes                                 | ❌ No                                  |
| **Handles Sequential Data**     | ✅ Yes                                 | ❌ No                                  |
| **Dynamic Feature Interactions**| ✅ Excellent                          | ❌ Limited                             |
| **Memory Mechanism**        | ✅ Retains important information        | ❌ No memory of previous states        |

While FNNs are simpler and faster, they are insufficient for datasets where the sequence of events or time-dependent features play a critical role in predictions.

---

## 4. Robustness in Modeling Complex Patterns
Disaster data is inherently complex. LSTMs handle this by:
- Storing long-term dependencies for recurring patterns (e.g., seasonal disaster trends).
- Forgetting irrelevant details through their gating mechanisms.
- Adapting to irregularities in the data without relying solely on feature engineering.

This makes LSTMs a more robust choice for handling real-world disaster data compared to simpler models like decision trees, support vector machines (SVMs), or even FNNs.

---

## 5. Considerations
### Why Not Other Models?
- **Decision Trees/Random Forests**: These models are not suited for capturing sequential or temporal dependencies in the data.
- **Support Vector Machines (SVMs)**: SVMs are limited to static data and can’t efficiently handle large-scale or sequential datasets.
- **Transformers**: While powerful, transformers are computationally expensive and might be overkill for smaller disaster datasets. LSTMs strike the right balance between performance and computational efficiency.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Faith1014/NovaAI.git
cd NovAI
```

### Dataset

Ensure the processed dataset `processed_dataset.csv` is available in the `datasets/` folder. You can use the provided dataset or add your own.

---

### Flask Application

To run the Flask web app, use the following command:

```bash
python real_time.py
```
This will start a local web server, and the application will be available at http://127.0.0.1:5000.

---

### Training the Model

To train the model with the LSTM architecture:

```bash
python aimodel.py
```

This script will train the model using the dataset and save the trained model to the models/ folder.
The training process is optimized using Intel’s AI Toolkit if installed.

---

### Evaluation

To evaluate the model on the test set:
It is included in the aimodel script

```bash
python aimodel.py
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
python converter.py
```

This will export the model to the open_model/ folder in the ONNX format

---

### Acknowledgments

- Inspired by real-world disaster prediction research and machine learning applications.
- **PyTorch**, **LSTM**, **Intel AI Toolkit**, and **Scikit-learn** for their powerful tools in building and training models.  
- **Flask** for easy deployment of the web application.

---
