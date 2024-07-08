import streamlit as st
import numpy as np
import joblib
import requests

# URL to the model file on GitHub
url = 'https://github.com/sudbrl/defaultprediction/raw/main/model.pkl'

# Path to save the downloaded file
model_file_path = 'model.pkl'

# Function to download the model file
def download_model(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print("Model file downloaded successfully.")
    else:
        raise Exception("Failed to download the model file.")

# Streamlit app interface
def main():
    st.title('Loan Approval Prediction')
    st.write('Enter details to predict loan approval:')

    # Input fields for user input
    applicant_income = st.number_input('Applicant Income')
    coapplicant_income = st.number_input('Coapplicant Income')
    loan_amount = st.number_input('Loan Amount')
    loan_amount_term = st.number_input('Loan Amount Term')
    credit_history = st.selectbox('Credit History', [0.0, 1.0])
    gender = st.radio('Gender', ['Male', 'Female'])  # Assuming gender as categorical
    education = st.radio('Education', ['Graduate', 'Not Graduate'])  # Assuming education as categorical
    property_area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])  # Assuming property area as categorical

    # Convert categorical inputs to numerical representation
    gender_encoded = 1 if gender == 'Male' else 0
    education_encoded = 1 if education == 'Graduate' else 0
    property_rural = 1 if property_area == 'Rural' else 0
    property_semiurban = 1 if property_area == 'Semiurban' else 0
    property_urban = 1 if property_area == 'Urban' else 0

    # Prepare input data as a NumPy array with 12 features
    # Assuming you know the exact order and number of features expected
    input_data = np.array([[applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history,
                            gender_encoded, education_encoded, property_rural, property_semiurban, property_urban,
                            0, 0]])  # Add placeholders for additional features if needed

    # Download and load the model if not already loaded
    if 'model' not in st.session_state:
        download_model(url, model_file_path)
        st.session_state.model = joblib.load(model_file_path)
        st.success("Model loaded successfully!")

    # Predict loan approval on user input
    if st.button('Predict'):
        # Make prediction
        prediction = st.session_state.model.predict(input_data)
        if prediction == 1:
            st.success('Loan Approved!')
        else:
            st.error('Loan Not Approved')

# Run the Streamlit app
if __name__ == '__main__':
    main()
