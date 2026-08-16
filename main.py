"""
MediMind AI: Multi-Modal Disease Prediction Engine
Author: Saboor Ahmad
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class MediMindEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = [
            "Age", "Systolic_BP", "Cholesterol", "Fasting_Blood_Sugar",
            "Max_Heart_Rate", "BMI", "Kidney_Function_GFR"
        ]
        self._initialize_dataset()

    def _initialize_dataset(self):
        np.random.seed(42)
        samples = 1000

        age = np.random.randint(25, 80, samples)
        systolic_bp = np.random.randint(90, 180, samples)
        cholesterol = np.random.randint(150, 320, samples)
        fasting_bs = np.random.randint(70, 200, samples)
        max_hr = np.random.randint(100, 200, samples)
        bmi = np.random.uniform(18.5, 40.0, samples)
        gfr = np.random.randint(45, 120, samples)

        risk_score = (
            (systolic_bp > 140).astype(int) * 2 +
            (cholesterol > 240).astype(int) * 2 +
            (fasting_bs > 126).astype(int) * 2.5 +
            (bmi > 30).astype(int) * 1.5
        )
        target = (risk_score >= 4).astype(int)

        data = np.column_stack([age, systolic_bp, cholesterol, fasting_bs, max_hr, bmi, gfr])
        self.df = pd.DataFrame(data, columns=self.feature_names)
        self.df["Target_Risk"] = target

    def train(self):
        X = self.df[self.feature_names]
        y = self.df["Target_Risk"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, self.model.predict(X_test_scaled))
        print(f"[SYSTEM LOG] MediMind Model Trained Successfully. Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    engine = MediMindEngine()
    engine.train()
