# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
from preprocessing_utils import encode_features

# Set page configuration and theme
st.set_page_config(
    page_title="UTI Diagnosis App",
    page_icon=":test_tube:"
)

# Load the pre-trained machine learning model
model_path = r'C:\Users\Administrator\Downloads\best_model.joblib'
model = joblib.load(model_path, mmap_mode=None)

# Sidebar information
st.sidebar.title("Classification of Urinary Tract Infection (UTI) Diagnosis: Reducing Misdiagnosis Using Machine Learning")

with st.sidebar.expander("See information", expanded=True):
    st.write("A urinary tract infection (UTI) is an infection in any part of your urinary system: your kidneys, ureters, bladder, and urethra. Most UTIs are caused by bacteria. Escherichia coli (E. coli) bacteria are the most common cause of UTIs, but other types of bacteria can also cause them.")

st.sidebar.header("About the application")
with st.sidebar.expander("See information", expanded=True):
    st.write("A user-friendly web application, integrating the classification model for UTI diagnosis. The application will provide healthcare providers with a convenient interface to input urinalysis test results and receive the diagnosis of UTI. It will offer an intuitive user experience and facilitate seamless integration of the classification model into clinical practice.")

st.sidebar.header("About the model being used")
with st.sidebar.expander("See information", expanded=True):
    st.write("**Random Forest**")

# Input options
option_gender = ["MALE", "FEMALE"]
option_au = ["NONE SEEN", "RARE", "FEW", "OCCASIONAL", "MODERATE", "LOADED", "PLENTY"]
option_color = [
    "LIGHT YELLOW", "STRAW", "AMBER", "BROWN", "DARK YELLOW", "YELLOW", "REDDISH YELLOW", "REDDISH", "LIGHT RED", "RED",
]
option_transparency = ["CLEAR", "SLIGHTLY HAZY", "HAZY", "CLOUDY", "TURBID"]
option_pg = ["NEGATIVE", "TRACE", "1+", "2+", "3+", "4+"]
option_wbc = ["0-0", "0-1", "0-2", "1-2", "0-3", "0-4", "1-3", "2-3", "1-4", "2-4", "1-5", "2-5", "3-4", "1-6", "3-5", "2-6", "3-6", "2-7", "4-5", "3-7", "4-6", "4-7", "5-6", "5-7", "4-8", "3-10", "5-8", "6-8", "4-10", "5-10", "7-8", "7-9", "7-10", "8-10", "8-11", "6-14", "8-12", "9-11", "9-12", "7-15", "10-12", "9-15", "11-13", "10-15", "11-14", "10-16", "11-15", "12-14", "12-15", "10-18", "13-15", "12-17", "14-16", "15-17", "15-18", "16-18", "15-20", "15-21", "17-20", "15-22", "18-20", "18-21", "18-22", "20-22", "15-28", "18-25", "20-25", "22-24", "23-25", "25-30", "25-32", "28-30", "30-32", "28-35", "30-35", "34-36", "36-38", "35-40", "38-40", "45-50", ">50", "48-55", "50-55", "48-62", "55-58", "70-75", "79-85", "85-87", ">100", "LOADED", "TNTC"]

# Main page content
st.subheader("Enter the necessary information of the patient :test_tube:", divider='grey')
st.markdown("**Please fill in all fields.**")

# Create two columns
col1, col2 = st.columns(2)

# Place input fields in the first column
with col1:
    Age = st.number_input("Age:", step=1, format="%d", min_value=0, value=None)
    Gender = st.selectbox("Gender:", [None] + option_gender, format_func=lambda x: '' if x is None else x)
    Color = st.selectbox("Color:", [None] + option_color, format_func=lambda x: '' if x is None else x)
    Transparency = st.selectbox("Transparency:", [None] + option_transparency, format_func=lambda x: '' if x is None else x)
    Glucose = st.selectbox("Glucose:", [None] + option_pg, format_func=lambda x: '' if x is None else x)
    Protein = st.selectbox("Protein:", [None] + option_pg, format_func=lambda x: '' if x is None else x)
    pH = st.number_input("pH:", value=None)

# Place input fields in the second column
with col2:
    Specific_Gravity = st.number_input("Specific Gravity:", value=None)
    WBC = st.selectbox("White Blood Cells (WBC):", [None] + option_wbc, format_func=lambda x: '' if x is None else x)
    RBC = st.selectbox("Red Blood Cells (RBC):", [None] + option_wbc, format_func=lambda x: '' if x is None else x)
    Epithelial_Cells = st.selectbox("Epithelial Cells:", [None] + option_au, format_func=lambda x: '' if x is None else x)
    Mucous_Threads = st.selectbox("Mucous Threads:", [None] + option_au, format_func=lambda x: '' if x is None else x)
    Amorphous_Urates = st.selectbox("Amorphous Urates:", [None] + option_au, format_func=lambda x: '' if x is None else x)
    Bacteria = st.selectbox("Bacteria:", [None] + option_au, format_func=lambda x: '' if x is None else x)

st.markdown("**Make sure to double-check your inputs.**")

# Add a "Predict" button
predict_button = st.button("Diagnose")

# Process the input fields and display the diagnosis based on the button click
if predict_button:
    # Create a Pandas DataFrame from the input data
    user_input = {
        "Age": Age,
        "Gender": Gender,
        "Color": Color,
        "Transparency": Transparency,
        "Glucose": Glucose,
        "Protein": Protein,
        "pH": pH,
        "Specific Gravity": Specific_Gravity,
        "WBC": WBC,
        "RBC": RBC,
        "Epithelial_Cells": Epithelial_Cells,
        "Mucous_Threads": Mucous_Threads,
        "Amorphous_Urates": Amorphous_Urates,
        "Bacteria": Bacteria
    }

    user_input_df = pd.DataFrame([user_input])

    # Display the input data in a table
    st.subheader("Input Data :pencil:", divider='grey')
    st.dataframe(user_input_df)

    # Export the DataFrame to a CSV file
    csv_file_path = "urine_test_results.csv"
    user_input_df.to_csv(csv_file_path, encoding="utf-8",)

    # Trigger the download of the CSV file
    st.download_button("Download CSV File",csv_file_path, mime = "text/csv", file_name="urine_test_results.csv")

    # Encode the categorical inputs using the 'encode_features' function
    encoded_user_input = encode_features(
        user_input_df,
        ordinal_features=["Transparency", "Epithelial_Cells", "Mucous_Threads", "Amorphous_Urates", "Bacteria",
                          "Color", "Protein", "Glucose", "WBC", "RBC"],
        nominal_features=["Gender"]
    )

    # Make a prediction using the pre-trained model
    pred = model.predict(encoded_user_input)

    # Check if WBC is LOADED or TNTC
    if WBC == "LOADED" or WBC == "TNTC":
        diagnosis = "POSITIVE"
    else:
        diagnosis = "POSITIVE" if pred else "NEGATIVE"

    # Display the diagnosis with conditional formatting
    st.subheader("Diagnosis :stethoscope:", divider='grey')
    if diagnosis == "NEGATIVE":
        st.success(f"The patient is {diagnosis}.")
    else:
        st.error(f"The patient is {diagnosis}. Advice to see a doctor for medication.")
