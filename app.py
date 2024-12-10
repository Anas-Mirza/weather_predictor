from weather_predictor import create_app, global_predictor

# Create Flask app
app = create_app()

def main():
    # Optional: additional setup before running
    global_predictor.train_direct_model()
    global_predictor.train_weekly_avg_model()

    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()"" 
