import streamlit as st
import numpy as np
import pandas as pd
from src.Iris_classification.pipeline.prediction import PredictionPipeline
from sklearn.preprocessing import StandardScaler

st.title("Iris Flower Classification")

# Define the input form for prediction
with st.form("prediction_form"):
    st.header("Enter Flower Measurements:")
    
    # Input fields for numerical data (Iris dataset features)
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

    # Submit button
    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            # Prepare the input data as a 2D numpy array
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

            # Apply the same scaler as during training
            scaler = StandardScaler()
            input_data = scaler.fit_transform(input_data)  # Replace with `scaler.transform` for pre-trained scaler
            
            # convert input_data to data frame with column names
            input_data = pd.DataFrame(input_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

            # Create an instance of the PredictionPipeline and make predictions
            obj = PredictionPipeline()
            prediction = obj.predict(input_data)

            # Map numeric predictions to flower names
            flower_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
            predicted_flower = flower_names[prediction[0]]

            # Show the result
            st.success(f"Predicted class of flower: {predicted_flower}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
