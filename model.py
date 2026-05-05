# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("car_data.csv")

# Convert categorical → numeric
df['Fuel'] = df['Fuel'].map({'Petrol':0, 'Diesel':1})
df['Transmission'] = df['Transmission'].map({'Manual':0, 'Automatic':1})
df['Car_Name'] = df['Car_Name'].astype('category').cat.codes

# Features & Target
X = df[['Year', 'Kms_Driven', 'Fuel', 'Transmission', 'Car_Name']]
y = df['Selling_Price']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")