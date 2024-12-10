import pandas as pd
from flask import current_app, jsonify, request, render_template
from models.weather_predictor import WeatherPredictor

# Create a single global predictor instance
predictor = WeatherPredictor(current_app.config['CSV_PATH'])

def init_routes(app):
    """Initialize all routes for the application"""
    
    @app.route('/')
    def index():
        """Render the main page with prediction options"""
        return render_template('index.html')

    @app.route('/predict_direct', methods=['POST'])
    def predict_direct():
        """Predict weather for a specific date"""
        try:
            date_str = request.form.get('date')
            prediction_date = pd.to_datetime(date_str)
            
            direct_prediction = predictor.predict_direct(prediction_date)
            
            return jsonify({
                'date': date_str,
                'prediction': {
                    'rain': round(direct_prediction['Rain'], 2),
                    'temp_max': round(direct_prediction['Temp Max'], 2),
                    'temp_min': round(direct_prediction['Temp Min'], 2)
                }
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/predict_weekly', methods=['POST'])
    def predict_weekly():
        """Predict weather based on past week's average"""
        try:
            date_str = request.form.get('date')
            prediction_date = pd.to_datetime(date_str)
            
            weekly_prediction = predictor.predict_weekly_avg(
                prediction_date, 
                predictor.data
            )
            
            return jsonify({
                'date': date_str,
                'prediction': {
                    'rain': round(weekly_prediction['Rain'], 2),
                    'temp_max': round(weekly_prediction['Temp Max'], 2),
                    'temp_min': round(weekly_prediction['Temp Min'], 2)
                }
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Custom 404 error handler"""
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def server_error(error):
        """Custom 500 error handler"""
        return jsonify({'error': 'Internal server error'}), 500

# In __init__.py, you would then call:
# init_routes(app)