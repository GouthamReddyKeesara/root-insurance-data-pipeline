# root_insurance_pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Simulate data ingestion
def load_data():
    data = pd.DataFrame({
        'age': np.random.randint(18, 60, 1000),
        'claim_amount': np.random.randint(100, 10000, 1000),
        'policy_type': np.random.choice(['basic', 'premium'], 1000),
        'fraud': np.random.choice([0, 1], 1000)
    })
    return data

# Simulate model training
def train_model(data):
    X = data[['age', 'claim_amount']]
    y = data['fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Main pipeline
def main():
    data = load_data()
    model = train_model(data)
    print(f"Model accuracy: {model.score(data[['age', 'claim_amount']], data['fraud']) * 100:.2f}%")

if __name__ == '__main__':
    main()
