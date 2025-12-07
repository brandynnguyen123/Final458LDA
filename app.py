import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. Load the pre-trained model ---
# It's good practice to wrap model loading in a try-except block.
try:
    with open('my_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'my_model.pkl' not found. Make sure the model file is in the same directory.")
    st.stop() # Stop the app if model isn't found
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 2. Define the expected feature columns from training ---
# These are the columns from the X_train dataframe after one-hot encoding.
# This list is crucial for ensuring that user inputs are transformed into
# the exact same format as the training data.
expected_columns = [
    'Granted_Loan_Amount',
    'FICO_score',
    'Monthly_Gross_Income',
    'Monthly_Housing_Payment',
    'Ever_Bankrupt_or_Foreclose',
    'Reason_credit_card_refinancing',
    'Reason_debt_conslidation',
    'Reason_home_improvement',
    'Reason_major_purchase',
    'Reason_other',
    'Employment_Status_part_time',
    'Employment_Status_unemployed',
    'Employment_Sector_Unknown',
    'Employment_Sector_communication_services',
    'Employment_Sector_consumer_discretionary',
    'Employment_Sector_consumer_staples',
    'Employment_Sector_energy',
    'Employment_Sector_financials',
    'Employment_Sector_health_care',
    'Employment_Sector_industrials',
    'Employment_Sector_information_technology',
    'Employment_Sector_materials',
    'Employment_Sector_real_estate',
    'Employment_Sector_utilities',
    'Lender_B',
    'Lender_C'
]

# --- 3. Streamlit Application UI ---
st.title('Loan Approval Prediction App')
st.markdown("Enter customer details to predict loan approval status.")

# Input widgets for numerical features
st.sidebar.header('Customer Financial Details')
granted_loan_amount = st.sidebar.number_input('Granted Loan Amount', min_value=5000.0, max_value=2500000.0, value=50000.0, step=1000.0)
fico_score = st.sidebar.number_input('FICO Score', min_value=300.0, max_value=850.0, value=650.0, step=1.0)
monthly_gross_income = st.sidebar.number_input('Monthly Gross Income', min_value=0.0, max_value=20000.0, value=5000.0, step=100.0)
monthly_housing_payment = st.sidebar.number_input('Monthly Housing Payment', min_value=300.0, max_value=50000.0, value=1500.0, step=50.0)
ever_bankrupt_foreclose = st.sidebar.radio('Ever Bankrupt or Foreclosed?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Input widgets for categorical features
st.sidebar.header('Customer Demographics & Loan Purpose')
reason_options = [
    'debt_conslidation', 'credit_card_refinancing', 'home_improvement',
    'cover_an_unexpected_cost', 'major_purchase', 'other'
]
reason = st.sidebar.selectbox('Reason for Loan', reason_options)

employment_status_options = [
    'full_time', 'part_time', 'unemployed'
]
employment_status = st.sidebar.selectbox('Employment Status', employment_status_options)

employment_sector_options = [
    'information_technology', 'energy', 'financials', 'health_care',
    'consumer_discretionary', 'industrials', 'real_estate',
    'consumer_staples', 'utilities', 'communication_services', 'materials', 'Unknown'
]
employment_sector = st.sidebar.selectbox('Employment Sector', employment_sector_options)

lender_options = [
    'A', 'B', 'C'
]
lender = st.sidebar.selectbox('Lender', lender_options)


# --- 4. Preprocess user input ---
def preprocess_input(input_data):
    # Create a DataFrame from the input data
    data = {
        'Granted_Loan_Amount': [input_data['Granted_Loan_Amount']],
        'FICO_score': [input_data['FICO_score']],
        'Monthly_Gross_Income': [input_data['Monthly_Gross_Income']],
        'Monthly_Housing_Payment': [input_data['Monthly_Housing_Payment']],
        'Ever_Bankrupt_or_Foreclose': [input_data['Ever_Bankrupt_or_Foreclose']],
        'Reason': [input_data['Reason']],
        'Employment_Status': [input_data['Employment_Status']],
        'Employment_Sector': [input_data['Employment_Sector']],
        'Lender': [input_data['Lender']]
    }
    input_df = pd.DataFrame(data)

    # Apply one-hot encoding for categorical features
    categorical_cols = ['Reason', 'Employment_Status', 'Employment_Sector', 'Lender']
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Align columns with the model's expected features (important for consistency)
    # Initialize a DataFrame with all expected columns and fill with zeros
    final_input = pd.DataFrame(0, index=[0], columns=expected_columns)

    # Populate the final_input DataFrame with actual values from user input
    for col in input_df_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_df_encoded[col]

    # Ensure all expected columns are present, and in the correct order
    # This handles cases where a category might not be present in the user input
    # but was present in the training data.
    final_input = final_input[expected_columns]

    return final_input

# --- 5. Make prediction ---
if st.button('Predict Loan Approval'):
    user_input = {
        'Granted_Loan_Amount': granted_loan_amount,
        'FICO_score': fico_score,
        'Monthly_Gross_Income': monthly_gross_income,
        'Monthly_Housing_Payment': monthly_housing_payment,
        'Ever_Bankrupt_or_Foreclose': ever_bankrupt_foreclose,
        'Reason': reason,
        'Employment_Status': employment_status,
        'Employment_Sector': employment_sector,
        'Lender': lender
    }

    processed_input = preprocess_input(user_input)
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)[:, 1]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.success(f"The model predicts loan **Approval** for this customer.")
        st.write(f"Probability of Approval: {prediction_proba[0]:.2f}")
    else:
        st.error(f"The model predicts loan **Denial** for this customer.")
        st.write(f"Probability of Approval: {prediction_proba[0]:.2f}")

    st.markdown("--- Request Details ---")
    st.write("**Input Features:**")
    st.dataframe(pd.DataFrame([user_input]))
    st.write("**Processed Input for Model:**")
    st.dataframe(processed_input)
