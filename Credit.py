import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import streamlit as st
import requests
from io import BytesIO

# URL for the loan prediction CSV file on GitHub
url = 'https://raw.githubusercontent.com/sudbrl/defaultprediction/main/loan_prediction.csv'

# Download the file using requests
response = requests.get(url)
if response.status_code == 200:
    data = pd.read_csv(BytesIO(response.content))
else:
    st.error(f"Failed to retrieve data from {url}")

# Drop the 'Loan_ID' column if it exists
if 'Loan_ID' in data.columns:
    data = data.drop('Loan_ID', axis=1)

# Handle missing values
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median(), inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Married'] = label_encoder.fit_transform(data['Married'])
data['Education'] = label_encoder.fit_transform(data['Education'])
data['Self_Employed'] = label_encoder.fit_transform(data['Self_Employed'])
data['Property_Area'] = label_encoder.fit_transform(data['Property_Area'])
data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)
data['Loan_Status'] = label_encoder.fit_transform(data['Loan_Status'])

# Split the dataset into features and target variable
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Apply SMOTE
smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

# Train the Random Forest Classifier with class weights
classifier = RandomForestClassifier(n_estimators=100, random_state=0, class_weight='balanced')
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# Function to predict loan status
def predict_loan_status(input_data):
    prediction = classifier.predict(input_data)
    return 'Approved' if prediction[0] == 1 else 'Rejected'

# Streamlit code for GUI
st.title("Loan Prediction App")

# Input fields for the features
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Married', ['Yes', 'No'])
dependents = st.selectbox('Dependents', [0, 1, 2, 3])
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.number_input('Applicant Income', min_value=0)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
loan_amount = st.number_input('Loan Amount', min_value=0)
loan_amount_term = st.number_input('Loan Amount Term', min_value=0)
credit_history = st.selectbox('Credit History', [0, 1])
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

# Preprocess the input data
gender = 1 if gender == 'Male' else 0
married = 1 if married == 'Yes' else 0
education = 1 if education == 'Graduate' else 0
self_employed = 1 if self_employed == 'Yes' else 0
property_area = 2 if property_area == 'Urban' else 1 if property_area == 'Semiurban' else 0

input_data = pd.DataFrame({
    'Gender': [gender],
    'Married': [married],
    'Dependents': [dependents],
    'Education': [education],
    'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [property_area]
})

# Predict button
if st.button('Predict'):
    prediction = predict_loan_status(input_data)
    st.write(f'Loan Status: {prediction}')

# Display model evaluation metrics
st.header('Model Evaluation Metrics')

# Display classification report
st.subheader('Classification Report')
st.dataframe(class_report_df)

# Display precision, recall, f1-score for each class
st.subheader('Precision, Recall, F1-Score for Each Class')
metrics_dict = {
    'Precision': [class_report_df.loc['0', 'precision'], class_report_df.loc['1', 'precision']],
    'Recall': [class_report_df.loc['0', 'recall'], class_report_df.loc['1', 'recall']],
    'F1-Score': [class_report_df.loc['0', 'f1-score'], class_report_df.loc['1', 'f1-score']]
}
metrics_df = pd.DataFrame(metrics_dict, index=['Rejected', 'Approved'])
st.write(metrics_df)
