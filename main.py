import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

st.title('Model Evaluation App')

selected_model_name = st.selectbox('Choose a model', ['Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier', 'Deep Learning Model'])
loaded_data = None

if st.button("Load Model"):
    st.write(f"Loading {selected_model_name} model...")
    if selected_model_name == "Deep Learning Model":
        global loaded_model
        st.text(open('Deep Learning Model.txt').read())
        st.image('Deep Learning Model.png', caption='PNG Image', use_column_width=True)
    else:
        st.text(open(f'{selected_model_name}.txt').read())
        st.image(f'{selected_model_name}.png', caption='PNG Image', use_column_width=True)
        with open(f'{selected_model_name}.pkl', 'rb') as file:
            loaded_model = pickle.load(file)

    st.success(f"{selected_model_name} model successfully loaded!")

with st.form("model_evaluation_form"):
    feature_inputs = [None,None,None,None,None,None]
    X = ["apache_4a_hospital_death_prob", "apache_4a_icu_death_prob", "d1_lactate_max", "d1_lactate_min", "gcs_motor_apache", "gcs_eyes_apache"]
    feature_inputs[0] = st.slider(f'Slide to set {X[0]}', min_value=0.0, max_value=100.0, value=0.0)
    feature_inputs[1] = st.slider(f'Slide to set {X[1]}', min_value=0.0, max_value=100.0, value=0.0)
    feature_inputs[2] = st.slider(f'Slide to set {X[2]}', min_value=0.0, max_value=100.0, value=0.0)
    feature_inputs[3] = st.slider(f'Slide to set {X[3]}', min_value=0.0, max_value=100.0, value=0.0)
    feature_inputs[4] = st.slider(f'Slide to set {X[4]}', min_value=0.0, max_value=100.0, value=0.0)
    feature_inputs[5] = st.slider(f'Slide to set {X[5]}', min_value=0.0, max_value=100.0, value=0.0)
    evaluate_button = st.form_submit_button("Evaluate Model")

if evaluate_button:
    print(type(feature_inputs))
    if selected_model_name == "Deep Learning Model":
        loaded_model = load_model('Deep Learning Model.h5')
        # Convert the feature_inputs list to a NumPy array
        feature_inputs_array = np.array([feature_inputs])
        prediction = loaded_model.predict(feature_inputs_array)
    else:
        loaded_model = joblib.load(f'{selected_model_name}.joblib')
        print(loaded_model,type(loaded_model),"@@@@@@@@@@@@@@@@@@")
        # Convert the feature_inputs list to a NumPy array
        feature_inputs_array = np.array([feature_inputs])
        
        prediction = loaded_model.predict(feature_inputs_array)

    st.write("Model Predictions:")
    if selected_model_name == "Deep Learning Model":
        st.write(f"Prediction: {(round(prediction[0][0],2))}")
    else:
        st.write(f"Prediction: {prediction}")
