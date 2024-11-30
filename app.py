from flask import Flask, render_template, request
import real_time  # Import your real-time module

# Initialize Flask app
app = Flask(__name__)

# Flask route to display the input form
@app.route('/', methods=['GET', 'POST'])
def home():
    error_message = None
    if request.method == 'POST':
        # Get user inputs from the form
        duration = request.form.get('duration')
        economic_loss = request.form.get('economic_loss')
        deaths = request.form.get('deaths')
        affected = request.form.get('affected')
        disaster_type = request.form.get('disaster_type')
        region = request.form.get('region')
        disaster_frequency = request.form.get('disaster_frequency')

        # Validate inputs
        error_message = validate_inputs(duration, economic_loss, deaths, affected, disaster_type, region, disaster_frequency)

        # If there is no error, perform prediction
        if not error_message:
            # Convert the validated input values to appropriate data types
            duration = float(duration)
            economic_loss = float(economic_loss)
            deaths = int(deaths)
            affected = int(affected)
            disaster_frequency = int(disaster_frequency)

            # Run the prediction using the run_prediction function from real_time.py
            result = real_time.run_prediction(duration, economic_loss, deaths, affected, disaster_type, region, disaster_frequency)
            return render_template('result.html', prediction=result)

    # Render the form with the error message (if any)
    return render_template('index.html', error_message=error_message)

def validate_inputs(duration, economic_loss, deaths, affected, disaster_type, region, disaster_frequency):
    # Validate all the fields and return appropriate error messages if they are invalid

    # Duration validation
    if not duration or not duration.isdigit() or float(duration) < 1:
        return "Error: Duration should be a number greater than or equal to 1."

    # Economic loss validation
    if not economic_loss or not economic_loss.isdigit() or float(economic_loss) < 0:
        return "Error: Economic loss should be a number greater than or equal to 0."

    # Deaths validation
    if not deaths or not deaths.isdigit() or int(deaths) < 0:
        return "Error: Deaths should be a number greater than or equal to 0."

    # Affected validation
    if not affected or not affected.isdigit() or int(affected) < 0:
        return "Error: Affected people should be a number greater than or equal to 0."

    # Disaster type validation
    valid_disasters = ['Flood', 'Earthquake', 'Cyclone', 'Landslide', 'Tsunami']
    if disaster_type not in valid_disasters:
        return f"Error: Disaster type should be one of the following: {', '.join(valid_disasters)}."

    # Region validation
    valid_regions = ['North', 'South', 'East', 'West', 'Central', 'Northeast']
    if region not in valid_regions:
        return f"Error: Region should be one of the following: {', '.join(valid_regions)}."

    # Disaster frequency validation
    if not disaster_frequency or not disaster_frequency.isdigit() or not (1 <= int(disaster_frequency) <= 10):
        return "Error: Disaster frequency should be a number between 1 and 10."

    return None  # No error

@app.route('/about')
def about():
    return render_template('about.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
