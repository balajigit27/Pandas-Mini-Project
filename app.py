# app.py

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load data
df = pd.read_csv("car_data.csv")

# Title
st.title("🚗 Car Price Prediction Dashboard")

# ---------------------------
# Data Visualization
# ---------------------------

st.subheader("Car Count by Company")

car_count = df["Car_Name"].value_counts()

fig, ax = plt.subplots()
car_count.plot(kind='bar', ax=ax)
st.pyplot(fig)

# ---------------------------
# User Input
# ---------------------------

st.subheader("Enter Car Details")

year = st.slider("Year", 2010, 2025, 2020)
kms = st.number_input("Kms Driven", 1000, 100000, 20000)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
company = st.selectbox("Car Company", df["Car_Name"].unique())

# Encode input
fuel_val = 0 if fuel == "Petrol" else 1
trans_val = 0 if transmission == "Manual" else 1
company_val = df["Car_Name"].astype('category').cat.codes[df["Car_Name"] == company].iloc[0]

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Price"):
    input_data = pd.DataFrame([[year, kms, fuel_val, trans_val, company_val]],
                             columns=['Year', 'Kms_Driven', 'Fuel', 'Transmission', 'Car_Name'])

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Price: ₹ {int(prediction[0])}")