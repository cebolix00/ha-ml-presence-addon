import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import datetime
import requests
import json
import time
import os

class PresenceDetector:
    def __init__(self):
        self.data = pd.DataFrame()
        self.labels = pd.Series()
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.is_trained = False

    def collect_data(self, state, attributes):
        time = datetime.datetime.now()
        new_data = pd.DataFrame({
            'hour': [time.hour],
            'minute': [time.minute],
            'day_of_week': [time.weekday()],
            'humidity': [float(attributes.get('humidity', 0))],
            'illuminance': [attributes.get('illuminance', 0)],
            'door_state': [attributes.get('door_state', False)],
            'motion_intensity': [float(attributes.get('motion_intensity', 0))],
        })
        return new_data

    def update_data(self, new_data, is_valid):
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        self.labels = pd.concat([self.labels, pd.Series([is_valid])], ignore_index=True)
        
        if len(self.data) > 10000:  # Limit the amount of stored data
            self.data = self.data.iloc[-10000:]
            self.labels = self.labels.iloc[-10000:]

    def train_model(self):
        if len(self.data) < 100:
            return False
        
        X = self.scaler.fit_transform(self.data)
        y = self.labels
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        score = self.model.score(X_test, y_test)
        print(f"Model trained with accuracy: {score}")
        
        return True

    def predict(self, features):
        if not self.is_trained:
            return 0.5
        features = self.scaler.transform(features)
        return self.model.predict_proba(features)[0][1]

    def save_model(self, path='/config/ml_presence_model.joblib'):
        if self.is_trained:
            joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
            print(f"Model saved to {path}")

    def load_model(self, path='/config/ml_presence_model.joblib'):
        try:
            loaded = joblib.load(path)
            self.model = loaded['model']
            self.scaler = loaded['scaler']
            self.is_trained = True
            print(f"Model loaded from {path}")
            return True
        except:
            print(f"Failed to load model from {path}")
            return False

detector = PresenceDetector()

def get_ha_data(entity_id, api_url, token):
    url = f"{api_url}/states/{entity_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data for {entity_id}: {e}")
        return None

def set_ha_state(entity_id, state, api_url, token):
    url = f"{api_url}/states/{entity_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = {
        "state": state,
        "attributes": {"source": "ml_presence_detector"}
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Error setting state for {entity_id}: {e}")
        return False

def main():
    # Load configuration
    config_path = '/data/options.json'
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON from configuration file: {config_path}")
        return

    sensor_entity = config.get('sensor_entity', 'binary_sensor.presence_sensor_fp2_2e03')
    light_entity = config.get('light_entity', 'light.luz_casa_de_banho_light')
    api_url = config.get('api_url', 'http://supervisor/core/api')
    token = config.get('long_lived_token')

    if not token:
        print("Long-lived token not provided in configuration")
        return

    detector.load_model()  # Try to load a previously saved model

    while True:
        try:
            sensor_data = get_ha_data(sensor_entity, api_url, token)
            if sensor_data:
                features = detector.collect_data(sensor_data['state'], sensor_data['attributes'])
                probability = detector.predict(features)
                
                is_present = probability > 0.6  # You can adjust this threshold
                
                detector.update_data(features, is_present)
                
                if len(detector.data) % 100 == 0:
                    detector.train_model()
                    detector.save_model()
                
                set_ha_state(light_entity, 'on' if is_present else 'off', api_url, token)
            
            time.sleep(10)  # Check every 10 seconds
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)  # Wait a minute before trying again if there's an error

if __name__ == "__main__":
    main()
