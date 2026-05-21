import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle

# setting the page
st.set_page_config(page_title="Life Expectancy Prediction", page_icon=":guardsman:", layout="wide")

#background color
st.markdown("""
    <style>
    .stApp {
        background-color: #c4c3d0;
    }
    </style>
""", unsafe_allow_html=True)

#Load the dataset
df = pd.read_csv('Life Expectancy Data.csv') 
df.columns = df.columns.str.strip()
x = df.drop('Life expectancy', axis=1)    
y = df['Life expectancy'] 

# Load the trained model and scaler
model = joblib.load('life_expectancy_model.pkl')
scaler = joblib.load('scaler.pkl')

# load training columns
model_columns = pickle.load(open('model_columns.pkl', 'rb'))
feature_names = model_columns

# load dropdown options
Country = pickle.load(open('countries.pkl', 'rb'))
status = pickle.load(open('status.pkl', 'rb'))


# Streamlit app
st.title('Life Expectancy Prediction')
st.write('Enter the values for the following features to predict life expectancy:')

# Create input fields for each feature

col1, col2, col3, col4 = st.columns(4)

with col1:
  
  country = st.selectbox("Country", Country)

  year = st.number_input("Year", min_value=1800, max_value=2100, value=2000)

  status = st.selectbox("Status", status)

  Adult_mortality = st.number_input("Adult Mortality", value=0.0)

  infant_deaths = st.number_input("infant deaths", value=0.0)

with col2:
  
  Alcohol = st.number_input("Alcohol", value=0.0)

  percentage_expenditure = st.number_input("percentage expenditure", value=0.0)

  Hepatitis_B = st.number_input("Hepatitis B", value=0.0)

  Measles = st.number_input("Measles", value=0.0)

  BMI = st.number_input("BMI", value=0.0)

  under_five_deaths = st.number_input("under-five deaths", value=0.0)
with col3:
   
  Polio = st.number_input("Polio", value=0.0)

  Total_expenditure = st.number_input("Total expenditure", value=0.0)

  Diphtheria = st.number_input("Diphtheria", value=0.0)

  HIV_AIDS = st.number_input("HIV/AIDS", value=0.0)

  GDP = st.number_input("GDP", value=0.0)

  Population = st.number_input("Population", value=0.0)

with col4:
  
  thinness_1_19_years = st.number_input("thinness 1-19 years", value=0.0)

  thinness_5_9_years = st.number_input("thinness 5-9 years", value=0.0)

  Income_composition_of_resources = st.number_input("Income composition of resources", value=0.0)

  schooling = st.number_input("Schooling", value=0.0)





# Predict button
if st.button('Predict'):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame({
        'Country': [country],
        'Year': [year],
        'Status': [status],
        'Adult Mortality': [Adult_mortality],
        'infant deaths': [infant_deaths],
        'Alcohol': [Alcohol],
        'percentage expenditure': [percentage_expenditure],
        'Hepatitis B': [Hepatitis_B],
        'Measles': [Measles],
        'BMI': [BMI],
        'under-five deaths': [under_five_deaths],
        'Polio': [Polio],
        'Total expenditure': [Total_expenditure],
        'Diphtheria': [Diphtheria],
        'HIV/AIDS': [HIV_AIDS],
        'GDP': [GDP],
        'Population': [Population],
        'thinness  1-19 years': [thinness_1_19_years],
        'thinness 5-9 years': [thinness_5_9_years],
        'Income composition of resources': [Income_composition_of_resources],
        'Schooling': [schooling]
     })

    #Apply get_dummies to the input data
    input_data = pd.get_dummies(input_df, columns=['Country', 'Status'])

    # Match training column
    input_data = input_data.reindex(columns=feature_names, fill_value=0)
    
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    # Display the predicted life expectancy
    st.success(f'Predicted Life Expectancy: {prediction[0]:.2f} years')


