import streamlit as st
import pickle
st.title(":blue[Mileage(MPG) ML Project]")
st.header('We are trying to predict the value of mileage of a car', divider='rainbow')

displacement = st.number_input('Displacement',value=300, placeholder="Enter a value for displacement")
horsepower = st.number_input('Horsepower',value=130, placeholder="Enter a value for horsepower")
weight = st.number_input('Weight',value=3000, placeholder="Enter a value for weight")
acceleration = st.number_input('Acceleration',value=12, placeholder="Enter a value for acceleration")

# Predicting the value of mpg
loaded_model = pickle.load(open('mpg_regression.sav', 'rb'))
prediction = loaded_model.predict([[displacement, horsepower, weight, acceleration]])
st.subheader(f'Predicted Mileage for the above parameter is:{prediction}', divider='red')
st.write(prediction)

