import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Obyekt yaratish
class AnomalyDetector:
    def __init__(self, data):
        self.data = data

    # O'zgaruvchilarni olib, tayyorlash
    def prepare_data(self):
        self.data.dropna(inplace=True)
        self.data['target'] = 0
        self.data.loc[self.data['value'] > 2, 'target'] = 1

    # O'zgaruvchilarni olib, tayyorlash
    def scale_data(self):
        scaler = StandardScaler()
        self.data[['value']] = scaler.fit_transform(self.data[['value']])

    # O'zgaruvchilarni olib, tayyorlash
    def split_data(self):
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # O'zgaruvchilarni olib, tayyorlash
    def train_model(self):
        self.model = IsolationForest(n_estimators=100, random_state=42)
        self.model.fit(self.X_train)

    # O'zgaruvchilarni olib, tayyorlash
    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    # O'zgaruvchilarni olib, tayyorlash
    def evaluate(self):
        self.accuracy = accuracy_score(self.y_test, self.y_pred)

    # O'zgaruvchilarni olib, tayyorlash
    def detect_anomalies(self):
        self.anomalies = self.model.predict(self.X_test)
        self.data.loc[self.data['target'] == 1, 'anomaly'] = self.anomalies

# Ma'lumotlar yaratish
data = {
    'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'target': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

# O'zgaruvchilarni olib, tayyorlash
detector = AnomalyDetector(pd.DataFrame(data))
detector.prepare_data()
detector.scale_data()
detector.split_data()
detector.train_model()
detector.predict()
detector.evaluate()
detector.detect_anomalies()

# Natija chiqarish
print(detector.data)
