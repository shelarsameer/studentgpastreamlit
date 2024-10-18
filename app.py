import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('student_gpa_model.joblib')
scaler = joblib.load('student_gpa_scaler.joblib')

# Define the feature names
feature_names = [
    'Age', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport',
    'Extracurricular', 'Sports', 'Music', 'Volunteering',
    'Gender_0', 'Gender_1',
    'Ethnicity_0', 'Ethnicity_1', 'Ethnicity_2', 'Ethnicity_3',
    'ParentalEducation_0', 'ParentalEducation_1', 'ParentalEducation_2',
    'ParentalEducation_3', 'ParentalEducation_4'
]

st.title('Student GPA Predictor')

# Create input fields for user
age = st.slider('Age', 15, 18, 16)
study_time = st.slider('Study Time Weekly (hours)', 0.0, 20.0, 10.0)
absences = st.slider('Number of Absences', 0, 29, 5)
tutoring = st.checkbox('Tutoring')
parental_support = st.slider('Parental Support (0-4)', 0, 4, 2)
extracurricular = st.checkbox('Extracurricular Activities')
sports = st.checkbox('Sports')
music = st.checkbox('Music')
volunteering = st.checkbox('Volunteering')

gender = st.selectbox('Gender', ['Male', 'Female'])
ethnicity = st.selectbox('Ethnicity', ['White', 'Black', 'Hispanic', 'Asian'])
parental_education = st.selectbox('Parental Education', 
                                  ['Some High School', 'High School', 'Some College', 'College', 'Graduate School'])

if st.button('Predict GPA'):
    # Prepare input data
    input_data = [age, study_time, absences, int(tutoring), parental_support,
                  int(extracurricular), int(sports), int(music), int(volunteering)]
    
    # One-hot encode categorical variables
    gender_encoded = [1, 0] if gender == 'Male' else [0, 1]
    ethnicity_encoded = [0, 0, 0, 0]
    ethnicity_encoded[['White', 'Black', 'Hispanic', 'Asian'].index(ethnicity)] = 1
    parental_education_encoded = [0, 0, 0, 0, 0]
    parental_education_encoded[['Some High School', 'High School', 'Some College', 'College', 'Graduate School'].index(parental_education)] = 1
    
    input_data.extend(gender_encoded)
    input_data.extend(ethnicity_encoded)
    input_data.extend(parental_education_encoded)
    
    # Scale the input data
    input_scaled = scaler.transform([input_data])
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    st.success(f'Predicted GPA: {prediction:.2f}')

st.sidebar.header('About')
st.sidebar.info('This app predicts a student\'s GPA based on various factors. '
                'Enter the student\'s information and click "Predict GPA" to see the result.')