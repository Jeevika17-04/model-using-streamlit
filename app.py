



import streamlit as st
import pickle
import numpy as np

# Load the saved Linear Regression model
with open('MLOPS.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict EMISSION using the loaded model
def predict_heart_failure(Age, RestingBP, Cholesterol, RestingECG, ExerciseAngina, Oldpeak):
    # Create a numpy array of input features
    row = np.array([Age, RestingBP, Cholesterol, RestingECG, ExerciseAngina, Oldpeak])
    # Reshape the array (not necessary in this case)
    # X = row.reshape(1,-1) 
    # Predict the heart failure
    prediction = model.predict(row.reshape(1, -1))
    return prediction[0]

# Streamlit UI
st.title('Heart Failure Prediction')
st.write("""## Input Features
Enter the values for the input features to predict PRICE.
""")

# Input fields for user 
Age = st.number_input("Enter the person's age: ")
RestingBP = st.number_input("Enter the person's Resting BP")
Cholesterol = st.number_input("Enter the Person's Cholesterol:")
RestingECG = st.number_input("Enter the Person's Resting ECG:") 
ExerciseAngina = st.number_input("Enter the Person's Exercise Angina:", 0, 3)
Oldpeak = st.number_input("Enter the Person's Old Peak:")

# Prediction button
if st.button('Predict'):
    # Predict heart failure
    prediction = predict_heart_failure(Age, RestingBP, Cholesterol, RestingECG, ExerciseAngina, Oldpeak)
    st.write(f"Possibilities of Heart Failure: {prediction}")