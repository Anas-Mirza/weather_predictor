import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

class WeatherPredictor:
    def __init__(self, csv_path):
        # Load and preprocess data
        self.data = pd.read_csv(csv_path)
        
        # Convert Date column to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Filter data from 2018 to 2023
        self.data = self.data[(self.data['Date'].dt.year >= 2018) & 
                               (self.data['Date'].dt.year <= 2023)]
        
        # Create features from date
        self.data['DayOfYear'] = self.data['Date'].dt.dayofyear
        self.data['Year'] = self.data['Date'].dt.year
        
        # Prepare two prediction models
        self.direct_model = None
        self.weekly_avg_model = None
        
        # Scalers
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
    
    def prepare_direct_data(self):
        # Prepare data for direct date prediction
        features = ['DayOfYear', 'Year']
        targets = ['Rain', 'Temp Max', 'Temp Min']
        
        X = self.data[features]
        y = self.data[targets]
        
        # Scale features and targets
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def prepare_weekly_avg_data(self):
        # Prepare data with weekly averages
        def get_weekly_averages(df):
            df_sorted = df.sort_values('Date')
            weekly_avgs = df_sorted.groupby(df_sorted['Date'].dt.to_period('W')).agg({
                'Rain': 'mean',
                'Temp Max': 'mean', 
                'Temp Min': 'mean',
                'DayOfYear': 'first',
                'Year': 'first'
            }).reset_index()
            weekly_avgs['DayOfYear'] = weekly_avgs['Date'].dt.start_time.dt.dayofyear
            return weekly_avgs
        
        weekly_data = get_weekly_averages(self.data)
        
        features = ['DayOfYear', 'Year']
        targets = ['Rain', 'Temp Max', 'Temp Min']
        
        X = weekly_data[features]
        y = weekly_data[targets]
        
        # Scale features and targets
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3)  # 3 output targets
        ])
        
        model.compile(optimizer='adam', 
                      loss='mean_squared_error', 
                      metrics=['mae'])
        return model
    
    def train_direct_model(self):
        X_train, X_test, y_train, y_test = self.prepare_direct_data()
        
        self.direct_model = self.build_model(X_train.shape[1])
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        self.direct_model.fit(
            X_train, y_train, 
            validation_split=0.2, 
            epochs=100, 
            batch_size=32, 
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        loss, mae = self.direct_model.evaluate(X_test, y_test)
        print("Direct Model - Test Loss:", loss)
        print("Direct Model - Test MAE:", mae)
    
    def train_weekly_avg_model(self):
        X_train, X_test, y_train, y_test = self.prepare_weekly_avg_data()
        
        self.weekly_avg_model = self.build_model(X_train.shape[1])
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        self.weekly_avg_model.fit(
            X_train, y_train, 
            validation_split=0.2, 
            epochs=100, 
            batch_size=32, 
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        loss, mae = self.weekly_avg_model.evaluate(X_test, y_test)
        print("Weekly Average Model - Test Loss:", loss)
        print("Weekly Average Model - Test MAE:", mae)
    
    def predict_direct(self, date):
        if self.direct_model is None:
            raise ValueError("Direct model not trained. Call train_direct_model() first.")
        
        # Prepare input features
        day_of_year = date.timetuple().tm_yday
        year = date.year
        
        # Scale input
        input_features = self.feature_scaler.transform([[day_of_year, year]])
        
        # Predict and inverse transform
        prediction_scaled = self.direct_model.predict(input_features)
        prediction = self.target_scaler.inverse_transform(prediction_scaled)
        
        return {
            'Rain': prediction[0][0],
            'Temp Max': prediction[0][1],
            'Temp Min': prediction[0][2]
        }
    
    def predict_weekly_avg(self, date, historical_data):
        if self.weekly_avg_model is None:
            raise ValueError("Weekly average model not trained. Call train_weekly_avg_model() first.")
        
        # Calculate weekly average from historical data
        weekly_avg_data = historical_data[
            (historical_data['Date'] >= date - timedelta(days=7)) & 
            (historical_data['Date'] < date)
        ]
        
        week_avg = {
            'Rain': weekly_avg_data['Rain'].mean(),
            'Temp Max': weekly_avg_data['Temp Max'].mean(),
            'Temp Min': weekly_avg_data['Temp Min'].mean()
        }
        
        # Prepare input features
        day_of_year = date.timetuple().tm_yday
        year = date.year
        
        # Scale input
        input_features = self.feature_scaler.transform([[day_of_year, year]])
        
        # Predict and inverse transform
        prediction_scaled = self.weekly_avg_model.predict(input_features)
        prediction = self.target_scaler.inverse_transform(prediction_scaled)
        
        return {
            'Rain': prediction[0][0],
            'Temp Max': prediction[0][1],
            'Temp Min': prediction[0][2]
        }

# Example usage
def main():
    # Replace with your actual CSV path
    predictor = WeatherPredictor('weather_data.csv')
    
    # Train models
    predictor.train_direct_model()
    predictor.train_weekly_avg_model()
    
    # Predict for a specific date in 2024
    prediction_date = pd.to_datetime('2024-03-15')
    
    # Direct prediction
    direct_prediction = predictor.predict_direct(prediction_date)
    print("Direct Prediction for", prediction_date)
    print(direct_prediction)
    
    # Weekly average prediction (requires historical data)
    weekly_prediction = predictor.predict_weekly_avg(
        prediction_date, 
        predictor.data
    )
    print("\nWeekly Average Prediction for", prediction_date)
    print(weekly_prediction)

if __name__ == '__main__':
    main()