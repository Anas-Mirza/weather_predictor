import os

class Config:
    """Configuration class for the Flask application"""
    # Base directory of the project
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Path to the weather data CSV
    CSV_PATH = os.path.join(BASE_DIR, 'weather_data.csv')
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    DEBUG = True 
