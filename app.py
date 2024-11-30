from flask import Flask, render_template, request
import real_time  # Assuming this is your module for predictions

# Initialize Flask app
app = Flask(__name__)

# Home route to handle form input and display the form
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
        error_message = validate_inputs(
            duration, economic_loss, deaths, affected,
            disaster_type, region, disaster_frequency
        )

        # If validation passes, run prediction
        if not error_message:
            duration = float(duration)
            economic_loss = float(economic_loss)
            deaths = int(deaths)
            affected = int(affected)
            disaster_frequency = int(disaster_frequency)

            # Perform prediction
            result = real_time.run_prediction(
                duration, economic_loss, deaths, affected,
                disaster_type, region, disaster_frequency
            )
            return render_template('result.html', prediction=result)

    return render_template('index.html', error_message=error_message)

# Validation function for user input
def validate_inputs(duration, economic_loss, deaths, affected, disaster_type, region, disaster_frequency):
    if not duration or not duration.isdigit() or float(duration) < 1:
        return "Error: Duration should be a number greater than or equal to 1."
    if not economic_loss or not economic_loss.isdigit() or float(economic_loss) < 0:
        return "Error: Economic loss should be a number greater than or equal to 0."
    if not deaths or not deaths.isdigit() or int(deaths) < 0:
        return "Error: Deaths should be a number greater than or equal to 0."
    if not affected or not affected.isdigit() or int(affected) < 0:
        return "Error: Affected people should be a number greater than or equal to 0."

    valid_disasters = ['Flood', 'Earthquake', 'Cyclone', 'Landslide', 'Tsunami']
    if disaster_type not in valid_disasters:
        return f"Error: Disaster type must be one of: {', '.join(valid_disasters)}."

    valid_regions = ['North', 'South', 'East', 'West', 'Central', 'Northeast']
    if region not in valid_regions:
        return f"Error: Region must be one of: {', '.join(valid_regions)}."

    if not disaster_frequency or not disaster_frequency.isdigit() or not (1 <= int(disaster_frequency) <= 10):
        return "Error: Disaster frequency must be between 1 and 10."

    return None

# About route
@app.route('/about')
def about():
    return render_template('about.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
