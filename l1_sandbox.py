import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Patient 3 has a biologically impossible Hemoglobin of 150.0
data = {
    'hemoglobin': [14.2, 15.1, 150.0],
    'glucose': [90, 105, 95],
    'risk_label': [0, 1, 0]  # 0 = Normal, 1 = High Risk
}

df = pd.DataFrame(data)
X = df[['hemoglobin', 'glucose']]
y = df['risk_label']

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# A new patient arrives with an impossible value
new_patient = pd.DataFrame({
    'hemoglobin': [180.0],
    'glucose': [92]
})

print('Prediction:', model.predict(new_patient))