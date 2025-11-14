import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title='👔 Employee Attrition Prediction', layout='wide')
st.title('👔 Employee Attrition Prediction')
st.write("Enter employee details to predict the likelihood of attrition.")

MODEL_FILE = 'attrition_model.pkl'
FEATURE_FILE = 'feature_columns.json'

# Load model and feature list
@st.cache_resource
def load_model_and_features():
    model = None
    feature_cols = None

    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        st.error(f"Model file '{MODEL_FILE}' not found.")

    if os.path.exists(FEATURE_FILE):
        with open(FEATURE_FILE, 'r') as f:
            feature_cols = json.load(f)
    else:
        st.error(f"Feature file '{FEATURE_FILE}' not found.")

    return model, feature_cols

model, feature_cols = load_model_and_features()

# --- User Inputs in Sidebar ---
st.sidebar.header('Employee Details')

# Numerical Inputs
age = st.sidebar.number_input('Age', min_value=18, max_value=60, value=36)
daily_rate = st.sidebar.number_input('Daily Rate', min_value=102, max_value=1499, value=802)
distance_from_home = st.sidebar.number_input('Distance From Home (km)', min_value=1, max_value=29, value=9)
education = st.sidebar.selectbox('Education', [1, 2, 3, 4, 5], index=2)
env_satisfaction = st.sidebar.selectbox('Environment Satisfaction', [1, 2, 3, 4], index=2)
hourly_rate = st.sidebar.number_input('Hourly Rate', min_value=30, max_value=100, value=66)
job_involvement = st.sidebar.selectbox('Job Involvement', [1, 2, 3, 4], index=2)
job_level = st.sidebar.selectbox('Job Level', [1, 2, 3, 4, 5], index=1)
job_satisfaction = st.sidebar.selectbox('Job Satisfaction', [1, 2, 3, 4], index=2)
monthly_income = st.sidebar.number_input('Monthly Income', min_value=1009, max_value=19999, value=6503)
monthly_rate = st.sidebar.number_input('Monthly Rate', min_value=2094, max_value=26999, value=14313)
num_companies_worked = st.sidebar.number_input('Num Companies Worked', min_value=0, max_value=9, value=2)
percent_salary_hike = st.sidebar.number_input('Percent Salary Hike', min_value=11, max_value=25, value=15)
performance_rating = st.sidebar.selectbox('Performance Rating', [1, 2, 3, 4], index=2)
rel_satisfaction = st.sidebar.selectbox('Relationship Satisfaction', [1, 2, 3, 4], index=2)
stock_option_level = st.sidebar.selectbox('Stock Option Level', [0, 1, 2, 3], index=1)
total_working_years = st.sidebar.number_input('Total Working Years', min_value=0, max_value=40, value=11)
training_times_last_year = st.sidebar.number_input('Training Times Last Year', min_value=0, max_value=6, value=3)
work_life_balance = st.sidebar.selectbox('Work Life Balance', [1, 2, 3, 4], index=2)
years_at_company = st.sidebar.number_input('Years At Company', min_value=0, max_value=40, value=7)
years_in_current_role = st.sidebar.number_input('Years In Current Role', min_value=0, max_value=18, value=4)
years_since_last_promotion = st.sidebar.number_input('Years Since Last Promotion', min_value=0, max_value=15, value=2)
years_with_curr_manager = st.sidebar.number_input('Years With Current Manager', min_value=0, max_value=17, value=4)

# Categorical Inputs
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
over_time = st.sidebar.selectbox('OverTime', ['Yes', 'No'])
business_travel = st.sidebar.selectbox('Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
department = st.sidebar.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
education_field = st.sidebar.selectbox('Education Field', ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
job_role = st.sidebar.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])


# --- Preprocessing Function ---
def preprocess_input(data, feature_cols_list):
    # Create a dataframe from the input dictionary
    input_df = pd.DataFrame([data])

    # Label encode binary features
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
    input_df['OverTime'] = input_df['OverTime'].map({'Yes': 1, 'No': 0})

    # One-hot encode categorical features
    categorical_cols_for_app = ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 'Sales', 'Research & Development', 'Human Resources', 'Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources', 'Single', 'Married', 'Divorced']

    # Use pd.Categorical to ensure all categories are present
    for col, options in [
        ('BusinessTravel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']),
        ('Department', ['Sales', 'Research & Development', 'Human Resources']),
        ('EducationField', ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources']),
        ('JobRole', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources']),
        ('MaritalStatus', ['Single', 'Married', 'Divorced'])
    ]:
        input_df[col] = pd.Categorical(input_df[col], categories=options)

    input_df_encoded = pd.get_dummies(
        input_df,
        columns=['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus'],
        prefix=['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus'],
        dtype=int
    )

    # --- Critical Step: Align columns with the model ---
    # Create an empty DataFrame with the model's feature columns
    final_input_df = pd.DataFrame(columns=feature_cols_list)
    # Fill with a single row of zeros
    final_input_df.loc[0] = 0

    # Get common columns
    common_cols = final_input_df.columns.intersection(input_df_encoded.columns)

    # Copy data from encoded input to the final DataFrame
    final_input_df[common_cols] = input_df_encoded[common_cols].values

    # Ensure all columns are of the correct type (int64)
    final_input_df = final_input_df.astype('int64')

    return final_input_df[feature_cols_list] # Ensure column order


# --- Prediction ---
if st.sidebar.button('🚀 Predict Attrition'):
    if model is not None and feature_cols is not None:
        # Collect all inputs into a dictionary
        raw_data = {
            'Age': age, 'DailyRate': daily_rate, 'DistanceFromHome': distance_from_home,
            'Education': education, 'EnvironmentSatisfaction': env_satisfaction,
            'Gender': gender, 'HourlyRate': hourly_rate, 'JobInvolvement': job_involvement,
            'JobLevel': job_level, 'JobSatisfaction': job_satisfaction,
            'MonthlyIncome': monthly_income, 'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies_worked, 'OverTime': over_time,
            'PercentSalaryHike': percent_salary_hike, 'PerformanceRating': performance_rating,
            'RelationshipSatisfaction': rel_satisfaction, 'StockOptionLevel': stock_option_level,
            'TotalWorkingYears': total_working_years, 'TrainingTimesLastYear': training_times_last_year,
            'WorkLifeBalance': work_life_balance, 'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_current_role,
            'YearsSinceLastPromotion': years_since_last_promotion,
            'YearsWithCurrManager': years_with_curr_manager,
            'BusinessTravel': business_travel, 'Department': department,
            'EducationField': education_field, 'JobRole': job_role,
            'MaritalStatus': marital_status
        }

        # Preprocess the data
        processed_data = preprocess_input(raw_data, feature_cols)

        # Make prediction
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]

        st.subheader('Prediction Result')
        if prediction == 1:
            st.error(f'**Prediction: Employee Will Leave (Attrition: Yes)**')
            st.write(f"Confidence Score: {prediction_proba[1]:.2f}")
        else:
            st.success(f'**Prediction: Employee Will Stay (Attrition: No)**')
            st.write(f"Confidence Score: {prediction_proba[0]:.2f}")

        st.subheader("Input Data Preview")
        st.dataframe(pd.DataFrame([raw_data]))

        st.subheader("Processed Feature Preview (for Model)")
        st.dataframe(processed_data)

    else:
        st.error("Model or feature list not loaded. Please check file paths.")

else:
    st.info("Click the 'Predict Attrition' button in the sidebar to get a prediction.")
