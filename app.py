import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('LinearRegressionModel.pk1', 'rb'))

# Create the Streamlit app
st.title("Car Price Prediction")

# Get the user input
name = st.text_input("Car Name", "Maruti Suzuki Swift")
company = st.selectbox("Company", ['Maruti', 'Hyundai', 'Honda', 'Tata', 'Mahindra', 'Toyota'])
year = st.number_input("Year", min_value=1990, max_value=2023, value=2019, step=1)
kms_driven = st.number_input("Kilometers Driven", min_value=0, value=100, step=1)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel'])

# Create a dataframe with the user input
user_input = pd.DataFrame({
    'name': [name],
    'company': [company],
    'year': [year],
    'kms_driven': [kms_driven],
    'fuel_type': [fuel_type]
})

# Make the prediction
if st.button("Predict Price"):
    prediction = model.predict(user_input)[0]
    st.success(f"The predicted price of the car is: {int(prediction)} INR")

# Add some additional information
st.write("---")
st.write("About the Model")
st.write("This model was trained on a dataset of used cars from Quikr. The model uses a Linear Regression algorithm to predict the price of a car based on its features.")
st.write("The model has an R-squared score of around 0.8 on the test set, which means it can explain about 80% of the variance in the target variable (car price).")