import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# =========================
# APP CONFIG
# =========================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model1.pkl"
DATA_PATH = BASE_DIR / "Heart_clean.csv"
IMAGE_PATH = BASE_DIR / "Heart_Disease.jpg"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

CATEGORICAL_COLUMNS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]

NUMERIC_COLUMNS = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

# IMPORTANT:
# If your target column was encoded with sklearn LabelEncoder from
# ['Disease', 'No disease'], then the usual mapping is:
# 0 = Disease, 1 = No disease.
# If your notebook manually used the opposite mapping, swap these values.
TARGET_LABEL_MAP = {
    0: "Disease",
    1: "No disease",
    "0": "Disease",
    "1": "No disease",
    "Disease": "Disease",
    "No disease": "No disease",
}

# =========================
# LOAD DATA, ENCODERS, MODEL
# =========================
@st.cache_data
def load_training_data():
    """Load training data used to recreate the same category mappings."""
    df = pd.read_csv(DATA_PATH, index_col=0)
    return df


@st.cache_resource
def build_encoders(train_df):
    """Fit LabelEncoders using the full training dataset, not one user input row."""
    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoder.fit(train_df[col].astype(str))
        encoders[col] = encoder
    return encoders


@st.cache_resource
def load_model():
    """Load the trained pickle model."""
    with open(MODEL_PATH, "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model


try:
    train_df = load_training_data()
    encoders = build_encoders(train_df)
    model = load_model()
except FileNotFoundError as e:
    st.error(f"Required file is missing: {e.filename}")
    st.stop()
except Exception as e:
    st.error("The app could not load the model or training data.")
    st.exception(e)
    st.stop()


# =========================
# PREPROCESSING
# =========================
def preprocess_input(input_df):
    """
    Convert Streamlit user input into the same numeric format expected by the model.
    Important: use transform(), not fit_transform(), for user input.
    """
    processed_df = input_df.copy()

    # Keep columns in the exact order used during training
    processed_df = processed_df[FEATURE_COLUMNS]

    # Encode categorical columns using encoders fitted on Heart_clean.csv
    for col in CATEGORICAL_COLUMNS:
        processed_df[col] = processed_df[col].astype(str)

        # Friendly validation for unexpected category values
        unknown_values = set(processed_df[col]) - set(encoders[col].classes_)
        if unknown_values:
            raise ValueError(
                f"Unknown value in column '{col}': {unknown_values}. "
                f"Allowed values are: {list(encoders[col].classes_)}"
            )

        processed_df[col] = encoders[col].transform(processed_df[col])

    # Force all columns to numeric float values for scikit-learn
    for col in NUMERIC_COLUMNS:
        processed_df[col] = pd.to_numeric(processed_df[col], errors="raise")

    return processed_df.astype(float)


def get_model_classes(model_object):
    """Return classes from GridSearchCV or normal estimator."""
    if hasattr(model_object, "classes_"):
        return list(model_object.classes_)
    if hasattr(model_object, "best_estimator_") and hasattr(model_object.best_estimator_, "classes_"):
        return list(model_object.best_estimator_.classes_)
    return None


def prediction_to_text(prediction_value):
    """
    Convert prediction output to display text.
    Edit TARGET_LABEL_MAP above if your training notebook used a different target mapping.
    """
    if prediction_value in TARGET_LABEL_MAP:
        return TARGET_LABEL_MAP[prediction_value]

    text_value = str(prediction_value).strip()
    if text_value in TARGET_LABEL_MAP:
        return TARGET_LABEL_MAP[text_value]

    return text_value


def show_prediction_result(prediction, prediction_proba):
    classes = get_model_classes(model)
    pred_value = prediction[0]
    pred_text = prediction_to_text(pred_value)

    st.markdown("---")
    st.subheader("Prediction Result")

    if pred_text == "Disease":
        st.error("⚠️ Disease Detected")
    elif pred_text == "No disease":
        st.success("🎉 No Disease Detected")
    else:
        st.info(f"Prediction: {pred_text}")

    if prediction_proba is not None:
        st.subheader("Prediction Probability")

        if classes is not None:
            probability_df = pd.DataFrame({
                "Class": [str(c) for c in classes],
                "Probability": prediction_proba[0]
            })
            st.dataframe(probability_df, width="stretch", hide_index=True)
        else:
            st.write(prediction_proba[0])

    st.caption(
        "Educational disclaimer: This app is for a data science project only and "
        "must not replace professional medical advice."
    )


# =========================
# USER INTERFACE
# =========================
st.title("❤️ Heart Disease Prediction Dashboard")

if IMAGE_PATH.exists():
    st.image(str(IMAGE_PATH), caption="Heart Disease Awareness", width="stretch")
else:
    st.warning("Image file Heart_Disease.jpg was not found. The app will continue without it.")

st.markdown(
    "Enter patient information, then open the **Prediction** tab and click "
    "**Run Prediction**."
)

tab1, tab2, tab3 = st.tabs(["📝 Input Data", "🔍 Prediction", "ℹ️ About"])

# =========================
# TAB 1: INPUT DATA
# =========================
with tab1:
    st.header("Enter Patient Information")

    col_form, col_img = st.columns([2, 1])

    with col_form:
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox(
                "Chest Pain Type",
                ["typical angina", "atypical angina", "non-anginal pain", "asymtomatic"]
            )
            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120
            )

        with col2:
            chol = st.number_input(
                "Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200
            )
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            restecg = st.selectbox(
                "Resting ECG Results",
                [
                    "normal",
                    "ST-T Wave abnormal",
                    "probable or definite left ventricular hypertrophy",
                ]
            )
            thalach = st.number_input(
                "Maximum Heart Rate Achieved", min_value=0, max_value=300, value=150
            )

        with col3:
            exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
            oldpeak = st.number_input(
                "ST Depression Induced by Exercise",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
            )
            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                ["upsloping", "flat", "downsloping"]
            )
            ca = st.selectbox(
                "Number of Major Vessels",
                [
                    "Number of major vessels: 0",
                    "Number of major vessels: 1",
                    "Number of major vessels: 2",
                    "Number of major vessels: 3",
                ]
            )
            thal = st.selectbox(
                "Thalassemia",
                ["normal", "fixed defect", "reversable defect"]
            )

        input_data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "cp": [cp],
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs],
            "restecg": [restecg],
            "thalach": [thalach],
            "exang": [exang],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "thal": [thal],
        })

        st.session_state["input_data"] = input_data

        with st.expander("Preview input data"):
            st.dataframe(input_data, width="stretch", hide_index=True)

    with col_img:
        if IMAGE_PATH.exists():
            st.image(str(IMAGE_PATH), caption="Stay Heart Healthy", width="stretch")

# =========================
# TAB 2: PREDICTION
# =========================
with tab2:
    st.header("Prediction")

    if "input_data" not in st.session_state:
        st.warning("Please enter patient information in the Input Data tab first.")
    else:
        st.write("Current input:")
        st.dataframe(st.session_state["input_data"], width="stretch", hide_index=True)

        if st.button("Run Prediction", type="primary"):
            try:
                preprocessed_data = preprocess_input(st.session_state["input_data"])

                prediction = model.predict(preprocessed_data)

                prediction_proba = None
                if hasattr(model, "predict_proba"):
                    prediction_proba = model.predict_proba(preprocessed_data)

                show_prediction_result(prediction, prediction_proba)

                with st.expander("Debug: preprocessed numeric input"):
                    st.dataframe(preprocessed_data, width="stretch", hide_index=True)

            except Exception as e:
                st.error("Prediction failed. Please check that preprocessing matches the model training step.")
                st.exception(e)

# =========================
# TAB 3: ABOUT
# =========================
with tab3:
    st.header("About this App")

    if IMAGE_PATH.exists():
        st.image(str(IMAGE_PATH), caption="Heart Disease Awareness", width="stretch")

    st.write(
        """
        This dashboard is part of a **Data Science Capstone Project**. It predicts
        the likelihood of **heart disease** based on clinical parameters.

        **Features:**
        - User-friendly input form
        - Machine learning model prediction
        - Probability scores for interpretation
        - Consistent preprocessing using category mappings from the training dataset

        **Disclaimer:**
        This tool is for educational purposes only and should not replace professional medical advice.
        """
    )

st.markdown("---")
st.caption("Built by Adrian Taufan – Capstone Project")