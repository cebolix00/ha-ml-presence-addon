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

   detector = PresenceDetector()

   def get_ha_data(entity_id):
       url = f"http://supervisor/core/api/states/{entity_id}"
       headers = {
           "Authorization": f"Bearer {os.environ.get('SUPERVISOR_TOKEN')}",
           "Content-Type": "application/json",
       }
       response = requests.get(url, headers=headers)
       return response.json()

   def set_ha_state(entity_id, state):
       url = f"http://supervisor/core/api/states/{entity_id}"
       headers = {
           "Authorization": f"Bearer {os.environ.get('SUPERVISOR_TOKEN')}",
           "Content-Type": "application/json",
       }
       data = {
           "state": state,
           "attributes": {"source": "ml_presence_detector"}
       }
       response = requests.post(url, headers=headers, json=data)
       return response.status_code == 201

   def main():
       options = json.loads(os.environ.get('OPTIONS', '{}'))
       sensor_entity = options.get('sensor_entity', 'binary_sensor.presence_sensor_fp2_2e03')
       light_entity = options.get('light_entity', 'light.luz_casa_de_banho_light')

       while True:
           sensor_data = get_ha_data(sensor_entity)
           features = detector.collect_data(sensor_data['state'], sensor_data['attributes'])
           probability = detector.predict(features)
           
           is_present = probability > 0.6  # You can adjust this threshold
           
           detector.update_data(features, is_present)
           
           if len(detector.data) % 100 == 0:
               detector.train_model()
           
           set_ha_state(light_entity, 'on' if is_present else 'off')
           
           time.sleep(10)  # Check every 10 seconds

   if __name__ == "__main__":
       main()
