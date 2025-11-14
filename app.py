import streamlit as st
import pandas as pd
import joblib, json, os

st.set_page_config(page_title='🌸 Iris Flower Prediction', layout='centered')
st.title('🌸 Iris Flower Prediction App')

MODEL_FILE = 'iris_model.pkl'
FEATURE_FILE = 'feature_columns.json'

# Load model
model, feature_cols = None, None
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    st.success('✅ Model loaded successfully')
else:
    st.error('❌ Model file not found')

# Load feature list
if os.path.exists(FEATURE_FILE):
    with open(FEATURE_FILE, 'r') as f:
        feature_cols = json.load(f)
else:
    st.error('❌ Feature list not found')

st.header('Enter Flower Measurements:')
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input('Sepal Length (cm)', value=5.1)
    sepal_width = st.number_input('Sepal Width (cm)', value=3.5)
with col2:
    petal_length = st.number_input('Petal Length (cm)', value=1.4)
    petal_width = st.number_input('Petal Width (cm)', value=0.2)

input_df = pd.DataFrame([{
    'sepal length (cm)': sepal_length,
    'sepal width (cm)': sepal_width,
    'petal length (cm)': petal_length,
    'petal width (cm)': petal_width
}])

st.subheader('Input Preview')
st.dataframe(input_df)

if st.button('🌼 Predict'):
    if model is None or feature_cols is None:
        st.error('Model or feature list not loaded.')
    else:
        X_pred = input_df[feature_cols]
        pred = model.predict(X_pred)[0]
        flower_names = ['Setosa', 'Versicolor', 'Virginica']
        st.success(f'🌸 Predicted flower type: {flower_names[pred]}')
