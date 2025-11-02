#HEADER
import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder

# LOAD MODEL
with open('best_model1.pkl', 'rb') as file:
    model = pickle.load(file)

# PREPROCESSING FUNCTION
def preprocess_input(data):
    encoders = {col: LabelEncoder() for col in data.columns if data[col].dtype == 'object'}
    for col, le in encoders.items():
        data[col] = le.fit_transform(data[col])
    return data

# PAGE CONFIG
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 12px;
    }
    h1 {
        color: #d9534f;
        text-align: center;
        font-family: 'Arial Black', sans-serif;
    }
    .result-card {
        background-color: #fff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    footer {
        text-align: center;
        margin-top: 30px;
        font-size: 0.9em;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# TITLE + IMAGE
st.title("❤️ Heart Disease Prediction Dashboard ❤️")
st.image("Heart_Disease.jpg", caption="Heart Disease Awareness", use_container_width=True)

# TABS
tab1, tab2, tab3 = st.tabs(["📝 Input Data", "🔍 Prediction", "ℹ️ About"])

# TAB 1: INPUT
with tab1:
    st.header("Enter Patient Information")
    col_form, col_img = st.columns([2,1])

    with col_form:
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 0, 120, 50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal pain", "asymtomatic"])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300, 120)

        with col2:
            chol = st.number_input("Serum Cholesterol (mg/dl)", 0, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            restecg = st.selectbox("Resting ECG Results", ["normal", "ST-T Wave abnormal", "probable or definite left ventricular hypertrophy"])
            thalach = st.number_input("Maximum Heart Rate Achieved", 0, 300, 150)

        with col3:
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
            ca = st.selectbox("Number of Major Vessels", ["Number of major vessels: 0", "Number of major vessels: 1", "Number of major vessels: 2", "Number of major vessels: 3"])
            thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

        # Store input in session state for use in Prediction tab
        st.session_state["input_data"] = pd.DataFrame({
            'age':[age], 'sex':[sex], 'cp':[cp], 'trestbps':[trestbps],
            'chol':[chol], 'fbs':[fbs], 'restecg':[restecg], 'thalach':[thalach],
            'exang':[exang], 'oldpeak':[oldpeak], 'slope':[slope], 'ca':[ca], 'thal':[thal]
        })

    with col_img:
        st.image("Heart_Disease.jpg", caption="Stay Heart Healthy", use_container_width=True)

# TAB 2: PREDICTION
with tab2:
    st.header("Prediction Result")
    if "input_data" in st.session_state:
        input_data = st.session_state["input_data"]
        preprocessed_data = preprocess_input(input_data)

        if st.button("Run Prediction"):
            prediction = model.predict(preprocessed_data)
            prediction_proba = model.predict_proba(preprocessed_data)

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            if prediction[0] == 0:
                st.success("🎉 No Disease Detected! 🎉")
            else:
                st.error("⚠️ Disease Detected! ⚠️")

            st.subheader("Prediction Probability")
            st.write(f"✅ Probability of No Disease: {prediction_proba[0][0]:.2f}")
            st.write(f"⚠️ Probability of Disease: {prediction_proba[0][1]:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter patient information in the Input tab first.")

# TAB 3: ABOUT
with tab3:
    st.header("About this App")
    st.image("Heart_Disease.jpg", caption="Heart Disease Awareness", use_container_width=True)
    st.write("""
    This dashboard is part of a **Data Science Capstone Project**.  
    It predicts the likelihood of **heart disease** based on clinical parameters.  

    **Features:**
    - User-friendly input form  
    - Machine learning model prediction  
    - Probability scores for better interpretation  

    **Disclaimer:**  
    This tool is for educational purposes only and should not replace professional medical advice.
    """)

# FOOTER
st.markdown("<footer>Built by Adrian Taufan – Capstone Project</footer>", unsafe_allow_html=True)