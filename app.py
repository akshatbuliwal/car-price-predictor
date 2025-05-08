import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Load the trained model and necessary encoders
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
ohe = pickle.load(open('OneHotEncoder.pkl', 'rb'))  # Load the OneHotEncoder used during training
scaler = pickle.load(open('StandardScaler.pkl', 'rb'))  # Load the scaler used during training

# Load the cleaned data
car = pd.read_csv('Cleaned_Car_data.csv')

# Streamlit title and description
st.title('Car Price Predictor')

# Dynamic dropdowns for user input
companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()

# User selects the company
company = st.selectbox('Select Car Company', ['Select Company'] + companies)

# Filter car models based on the selected company
if company != 'Select Company':
    car_models = sorted(car[car['company'] == company]['name'].unique())

car_model = st.selectbox('Select Car Model', car_models)

# User selects the year
year = st.selectbox('Select Year of Purchase', years)

# User selects the fuel type
fuel_type = st.selectbox('Select Fuel Type', fuel_types)

# User inputs the kilometers driven
kms_driven = st.number_input('Enter Distance Driven (in km)', min_value=0)

# Button to trigger the prediction
if st.button('Predict Price'):
    # Ensure user has selected valid options
    if company == 'Select Company' or not car_model or not fuel_type:
        st.error("Please select valid options for all fields.")
    else:
        # Prepare input data for prediction
        input_data = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        # OneHotEncode the categorical columns
        input_encoded = ohe.transform(input_data[['name', 'company', 'fuel_type']]).toarray()

        # Scale the numerical columns
        input_scaled = scaler.transform(input_data[['year', 'kms_driven']])

        # Concatenate the encoded and scaled columns
        final_input = np.concatenate([input_encoded, input_scaled], axis=1)

        # Make the prediction
        prediction = model.predict(final_input)[0]

        # Display the predicted price
        st.success(f"Estimated Car Price: ₹ {np.round(prediction, 2)}")
