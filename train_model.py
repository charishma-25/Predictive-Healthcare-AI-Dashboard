import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. Load and clean data
df = pd.read_csv('diabetes.csv')
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# 2. Separate features
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# 5. Save the Model and the Scaler together
joblib.dump({'model': model, 'scaler': scaler}, 'diabetes_model.pkl')
print("Model trained and saved successfully as 'diabetes_model.pkl'")