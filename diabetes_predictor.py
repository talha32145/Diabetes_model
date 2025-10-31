import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Diabetes Prediction App", page_icon="💉", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3, h4 {
    color: #00e6ac;
}
.divider {
    border: 1px solid #333;
    margin: 15px 0;
}
.pulse {
    margin: 0 auto;
    height: 120px;
    width: 120px;
    background: radial-gradient(circle, #00e6ac 30%, #008060 90%);
    border-radius: 50%;
    animation: pulse 1.5s infinite;
    box-shadow: 0 0 25px #00e6ac;
}
@keyframes pulse {
    0% {
        transform: scale(1);
        opacity: 0.9;
    }
    50% {
        transform: scale(1.15);
        opacity: 0.7;
    }
    100% {
        transform: scale(1);
        opacity: 0.9;
    }
}
.result-box {
    border-radius: 12px;
    padding: 15px;
    background: #1b1e24;
    box-shadow: 0 0 10px rgba(0, 230, 172, 0.2);
    margin-top: 20px;
}
.result-box p{
            color=green;
            }
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

model = load_model("diabetes_model.keras")
with open("scaler.pkl", "rb") as file:
    std = pickle.load(file)

st.markdown("<h1 style='text-align:center;'>💉 Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>_An AI-powered tool to predict diabetes using health data._</h4>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown("<div class='pulse'></div>", unsafe_allow_html=True)

st.subheader("🧬 Enter Your Medical Details:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    BloodPressure = st.number_input("Blood Pressure", min_value=40, max_value=200, value=80)

with col2:
    Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=20)

with col3:
    Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    BMI = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)

with col4:
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    Age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("🔍 Predict"):
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])

    input_scaled = std.transform(input_data)
    prediction = model.predict(input_scaled)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.subheader("📊 Prediction Result:")

    if prediction >= 0.5:
        st.markdown("""
        <div class='result-box'>
            <h3 style='color:#ff4d4d;'>⚠️ You might be <b>Diabetic</b>.</h3>
            <p style="color:white;">Please consult a doctor for further advice.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='result-box'>
            <h3 style='color:#00e6ac;'>🎉 You are <b>Non-Diabetic</b>!</h3>
            <p style="color:white;">Keep up your healthy lifestyle 💪</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div class='footer'>Built with ❤️ by <b>Talha</b> | Powered by TensorFlow & Streamlit</div>", unsafe_allow_html=True)
