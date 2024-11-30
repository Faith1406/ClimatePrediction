from flask import Flask, render_template, request
import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENWEATHER_API_KEY')
if not api_key:
    raise ValueError("API Key is missing. Please set the OPENWEATHER_API_KEY environment variable.")

BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    weather_data = None
    error_message = None

    if request.method == 'POST':
        city = request.form['city']

        if not city:
            error_message = "City name cannot be empty."
        else:
            params = {'q': city, 'appid': api_key, 'units': 'metric'}
            try:
                response = requests.get(BASE_URL, params=params)
                response.raise_for_status()
                weather_data = response.json()

                if weather_data['cod'] != 200:
                    error_message = weather_data.get('message', 'City not found.')
            except requests.exceptions.RequestException as e:
                error_message = f"Error: {str(e)}"

    return render_template('index.html', weather_data=weather_data, error_message=error_message)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
