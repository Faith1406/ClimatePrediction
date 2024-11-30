import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENWEATHER_API_KEY')
if not api_key:
    raise ValueError("API Key is missing. Please set the OPENWEATHER_API_KEY environment variable.")

BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"


def get_weather(city):
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric',
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        if data['cod'] != 200:
            print(f"Error: {data['message']}")
            return None

        return data

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {params['q']} not found")
        return None


def convert_timestamp(timestamp):
    hours, remainder = divmod(timestamp, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def temperature_emoji(temp):
    if temp > 30:
        return "🔥 Hot 🔥"
    elif 20 <= temp <= 30:
        return "🌞 Warm 🌞"
    elif 10 <= temp < 20:
        return "🌤️ Mild 🌤️"
    elif 0 <= temp < 10:
        return "❄️ Chilly ❄️"
    else:
        return "☃️ Freezing ☃️"

def humidity_emoji(humidity):
    if humidity > 80:
        return "💧 Very Humid 💧"
    elif 60 <= humidity <= 80:
        return "🌫️ Humid 🌫️"
    elif 40 <= humidity < 60:
        return "💦 Moderate 💦"
    else:
        return "🌬️ Dry 🌬️"


def wind_emoji(wind_speed):
    if wind_speed > 15:
        return "🌪️ Strong Wind 🌪️"
    elif 5 <= wind_speed <= 15:
        return "🌬️ Moderate Wind 🌬️"
    else:
        return "🌫️ Calm 🌫️"


def weather_emoji(description):
    description = description.lower()
    if "clear" in description:
        return "☀️"
    elif "cloud" in description:
        return "☁️"
    elif "rain" in description:
        return "🌧️"
    elif "snow" in description:
        return "❄️"
    elif "storm" in description:
        return "🌩️"
    else:
        return "🌤️"


def weather_alerts(temp, humidity, wind_speed):
    alerts = []


    if temp > 30:
        alerts.append("🔥 Alert: It's very hot! Stay hydrated and avoid direct sunlight.")
    elif temp < 0:
        alerts.append("❄️ Alert: Freezing temperatures! Bundle up and stay warm.")


    if humidity > 80:
        alerts.append("💧 Alert: High humidity! Expect sticky and uncomfortable conditions.")
    elif humidity < 30:
        alerts.append("🌵 Alert: Low humidity! Dry conditions, stay hydrated and protect your skin.")


    if wind_speed > 15:
        alerts.append("🌬️ Alert: Strong winds! Be cautious of flying debris and high winds.")

    return alerts


def display_weather(data):

    city_name = data['name']
    temp = data['main']['temp']
    humidity = data['main']['humidity']
    pressure = data['main']['pressure']
    wind_speed = data['wind']['speed']
    description = data['weather'][0]['description']
    sunrise = convert_timestamp(data['sys']['sunrise'])
    sunset = convert_timestamp(data['sys']['sunset'])


    print("\nWeather Information:")
    print("----------------------")
    print(f"City: {city_name}")
    print(f"Temperature: {temp}°C {temperature_emoji(temp)}")
    print(f"Humidity: {humidity}% {humidity_emoji(humidity)}")
    print(f"Pressure: {pressure} hPa")
    print(f"Wind Speed: {wind_speed} m/s {wind_emoji(wind_speed)}")
    print(f"Weather: {description.capitalize()} {weather_emoji(description)}")
    print(f"Sunrise: {sunrise}")
    print(f"Sunset: {sunset}")
    print("----------------------\n")


    alerts = weather_alerts(temp, humidity, wind_speed)

    if alerts:
        print("\nWeather Alerts:")
        for alert in alerts:
            print(alert)
        print("----------------------\n")

def main():
    city = input("Enter a city name: ").strip()

    if not city:
        print("Error: City name cannot be empty.")
        return


    weather_data = get_weather(city)

    if weather_data:
        display_weather(weather_data)

if __name__ == "__main__":
    main()
