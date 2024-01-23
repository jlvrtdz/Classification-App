# Import necessary libraries

import streamlit as st
import pandas as pd
import numpy as np
import os
import openpyxl


#from captum.attr import IntegratedGradients
#import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from prediction_utils import load_scaler, predict_diagnosis_MLP, predict_diagnosis_LGBM
from model import MLP, LGBM


# Set page configuration and theme
st.set_page_config(
    page_title="UTI Diagnosis App",
    page_icon=":test_tube:",
    layout="centered",
    initial_sidebar_state="expanded"
    
)

# Load LGBM model



# Load the pre-trained scaler
scaler_path = r"C:\Users\Administrator\Downloads\train_scaler.joblib"
scaler = load_scaler(scaler_path)

# Sidebar information
st.sidebar.title("UTI Diagnostic System")

with st.sidebar.expander("See information", expanded=True):
    st.write("Urinary Tract Infections (UTIs) have a significant global impact, affecting millions and posing challenges to healthcare systems. The World Health Organization identifies UTIs as among the most common bacterial infections worldwide.")

st.sidebar.header("About the application")
with st.sidebar.expander("See information", expanded=True):
    st.write("Empowering healthcare providers with a streamlined UTI diagnosis process. This user-friendly web application integrates an advanced ensemble model that combines the strengths of LGBM (Light Gradient Boosting Machine) and MLP (Multi-Layer Perceptron) for accurate UTI predictions. Input urinalysis test results effortlessly and receive instant diagnoses, enhancing clinical decision-making. Downloadable results provide convenient record-keeping. Improve patient care with our intuitive and efficient UTI Diagnosis Applicaton.")

# Input options
option_gender = ["MALE", "FEMALE"]
option_au = ["NONE SEEN", "RARE", "FEW", "OCCASIONAL", "MODERATE", "LOADED", "PLENTY"]
option_color = [
    "LIGHT YELLOW", "STRAW", "AMBER", "BROWN", "DARK YELLOW", "YELLOW", "REDDISH YELLOW", "REDDISH", "LIGHT RED", "RED",
]
option_transparency = ["CLEAR", "SLIGHTLY HAZY", "HAZY", "CLOUDY", "TURBID"]
option_pg = ["NEGATIVE", "TRACE", "1+", "2+", "3+", "4+"]
option_wbc = ["0-0", "0-1", "0-2", "1-2", "0-3", "0-4", "1-3", "2-3", "1-4", "2-4", "1-5", "2-5", "3-4", "1-6", "3-5", "2-6", "3-6", "2-7", "4-5", "3-7", "4-6", "4-7", "5-6", "5-7", "4-8", "3-10", "5-8", "6-8", "4-10", "5-10", "7-8", "7-9", "7-10", "8-10", "8-11", "6-14", "8-12", "9-11", "9-12", "7-15", "10-12", "9-15", "11-13", "10-15", "11-14", "10-16", "11-15", "12-14", "12-15", "10-18", "13-15", "12-17", "14-16", "15-17", "15-18", "16-18", "15-20", "15-21", "17-20", "15-22", "18-20", "18-21", "18-22", "20-22", "15-28", "18-25", "20-25", "22-24", "23-25", "25-30", "25-32", "28-30", "30-32", "28-35", "30-35", "34-36", "36-38", "35-40", "38-40", "45-50", ">50", "48-55", "50-55", "48-62", "55-58", "70-75", "79-85", "85-87", ">100", "LOADED", "TNTC"]
option_rbc = ["0-0", "0-1", "0-2", "1-2", "0-3", "0-4", "1-3", "2-3", "1-4", "2-4", "1-5", "2-5", "3-4", "1-6", "3-5", "2-6", "3-6", "2-7", "4-5", "3-7", "4-6", "4-7", "5-6", "5-7", "4-8", "3-10", "5-8", "6-8", "4-10", "5-10", "7-8", "7-9", "7-10", "8-10", "8-11", "6-14", "8-12", "9-11", "9-12", "7-15", "10-12", "9-15", "11-13", "10-15", "11-14", "10-16", "11-15", "12-14", "12-15", "10-18", "13-15", "12-17", "14-16", "15-17", "15-18", "16-18", "15-20", "15-21", "17-20", "15-22", "18-20", "18-21", "18-22", "20-22", "15-28", "18-25", "20-25", "22-24", "23-25", "25-30", "25-32", "28-30", "30-32", "28-35", "30-35", "34-36", "36-38", "35-40", "38-40", "45-50", ">50", "48-55", "50-55", "48-62", "55-58", "70-75", "79-85", "85-87", ">100", "LOADED", "TNTC"]
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
    RBC = st.selectbox("Red Blood Cells (RBC):", [None] + option_rbc, format_func=lambda x: '' if x is None else x)
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
        "WBC": str(WBC),
        "RBC": str(RBC),
        "Epithelial Cells": Epithelial_Cells,
        "Mucous Threads": Mucous_Threads,
        "Amorphous Urates": Amorphous_Urates,
        "Bacteria": Bacteria
    }
    

    user_input_df = pd.DataFrame([user_input])

    # Create a new DataFrame for Excel
    excel_output_df = user_input_df.copy()

    # Check if any of the required fields are empty
    if not all(user_input_df.notnull().values.flatten()):
        st.error("Please fill in all fields before diagnosing.")
        st.stop()

    # Display the input data in a table
    st.subheader("Input Data :pencil:", divider='grey')
    st.dataframe(user_input_df)


    # Make prediction using MLP and LGBM
    pred_MLP = predict_diagnosis_MLP(user_input_df, scaler, model=MLP)
    pred_LGBM = predict_diagnosis_LGBM(user_input_df, scaler, model=LGBM)

    # Assuming pred_MLP is a PyTorch tensor
    normalized_probs = F.softmax(pred_MLP, dim=1)

    
    numpy_array = normalized_probs.detach().numpy()

    pred_ensemble = (numpy_array + pred_LGBM)/2

    
    predicted_class = np.argmax(pred_ensemble, axis=1)

    if WBC == "LOADED" or WBC == "TNTC" or Bacteria == "PLENTY" or Bacteria == "LOADED":
        diagnosis = "POSITIVE"
    else:
        # Existing logic for diagnosis based on model predictions
        if np.any(predicted_class > 0):
            diagnosis = "POSITIVE"
        else:
            diagnosis = "NEGATIVE"
    print(numpy_array)
    print(pred_ensemble)
    print(predicted_class)

    # Create a new DataFrame with diagnosis result
    diagnosis_df = pd.DataFrame({"Diagnosis": [diagnosis]})

    # Create a new DataFrame for Excel
    excel_output_df1 = pd.concat([excel_output_df, diagnosis_df], axis=1)

    # Excel File and set cell format to "Text"
    xlsx_file_name = "urine_test_results.xlsx"
    excel_file_path = os.path.join(os.path.expanduser("~"), "Downloads", xlsx_file_name)
    excel_output_df1.to_excel(excel_file_path, index=False, engine='openpyxl')

    # Open the Excel file and set the format of all cells to "Text"
    wb = openpyxl.load_workbook(excel_file_path)
    ws = wb.active
    for row in ws.iter_rows():
        for cell in row:
            cell.number_format = openpyxl.styles.numbers.FORMAT_TEXT

    # Save the modified Excel file
    wb.save(excel_file_path)

    # Trigger the download of the Excel file
    st.download_button("Download File", excel_file_path, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_button", file_name=xlsx_file_name)

    # Display the diagnosis with conditional formatting
    st.subheader("Diagnosis :stethoscope:", divider='grey')
    if diagnosis == "NEGATIVE":
        st.success(f"The patient is {diagnosis}.")
    else:
        st.error(f"**The patient is {diagnosis}. Advice to see a doctor for medication.**")

    
    # Integrated Gradients Analysis
    #ig = IntegratedGradients(best_model)
    #input_tensor = user_input_df.drop('Gender', axis=1).values.astype('float32')

    # Convert NumPy array to torch.Tensor
    #input_tensor_tensor = torch.tensor(input_tensor)

    # Integrated Gradients Analysis
    #attribution_map = ig.attribute(input_tensor_tensor, target=predicted_class)



    #fig = go.Figure()

    #fig.add_trace(go.Bar(
        #x=attribution_map.sum(dim=0).numpy(),
        #y=user_input_df.columns[:-1],  # Exclude the last column (FEMALE)
        #orientation='h',
        #marker=dict(color='rgba(73, 131, 255, 0.6)', line=dict(color='rgba(73, 131, 255, 1.0)', width=1)),
    #))

    #fig.update_layout(
        
        #xaxis=dict(title='Attribution'),
        #height=550,  # Adjust the height as needed
        #width=700  # Adjust the width as needed
        
    #)


    #st.plotly_chart(fig)
    #"""
