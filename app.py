import streamlit as st
import tensorflow as tf
import numpy as np
 
# Load model
model = tf.keras.models.load_model("my_model.h5")
 
st.title("🚀 TensorFlow Model Deployment with Streamlit")
 
st.write("Enter input features:")
 
# Example: 4 input features
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)
 
if st.button("Predict"):
    # Prepare input
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(input_data)
    
    st.success(f"✅ Prediction: {prediction[0][0]:.4f}")
 
 