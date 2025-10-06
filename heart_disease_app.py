import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Health Checker", page_icon="❤️", layout="centered")
st.title("❤️ Heart Health Checker")
st.write("A simple tool that estimates your chance of heart disease. This is NOT medical advice.")

# Load model and columns
@st.cache_resource
def load_model_and_columns():
    model = joblib.load("heart_model.pkl")
    cols = pd.read_csv("heart.csv").drop("target", axis=1).columns.tolist()
    return model, cols

model, columns = load_model_and_columns()

st.sidebar.header("Model Info")
st.sidebar.write("Model: Random Forest (pretrained)")
st.sidebar.write("Test accuracy: **87.5%**")

# User inputs (friendly)
def user_input():
    age = st.slider("Age", 29, 77, 50)
    sex = st.radio("Gender", ("Male", "Female"))
    sex = 1 if sex == "Male" else 0

    cp = st.selectbox("Chest pain level", ["0 - None", "1 - Mild", "2 - Moderate", "3 - Severe"])
    cp = int(cp.split(" ")[0])

    trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
    chol = st.slider("Cholesterol (mg/dL)", 100, 550, 240)
    fbs = st.radio("Fasting blood sugar > 120 mg/dL?", ("No", "Yes"))
    fbs = 1 if fbs == "Yes" else 0

    restecg = st.selectbox("ECG result", ["0 - Normal", "1 - Slight abnormality", "2 - Definite abnormality"])
    restecg = int(restecg.split(" ")[0])

    thalach = st.slider("Max heart rate achieved", 70, 210, 150)
    exang = st.radio("Chest pain during exercise?", ("No", "Yes"))
    exang = 1 if exang == "Yes" else 0

    oldpeak = st.slider("ST depression (exercise strain)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST segment", ["0 - Downsloping", "1 - Flat", "2 - Upsloping"])
    slope = int(slope.split(" ")[0])

    ca = st.selectbox("Number of major blood vessels (0-4)", [0,1,2,3,4])
    thal = st.selectbox("Thalassemia (0=normal,1=fixed,2=reversible)", ["0 - Normal", "1 - Fixed defect", "2 - Reversible defect"])
    thal = int(thal.split(" ")[0])

    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    return pd.DataFrame(data, index=[0])

user_df = user_input()

# Ensure correct column order
user_df = user_df[columns]

# Prediction and probability
pred = model.predict(user_df)[0]
proba = model.predict_proba(user_df)[0]
# find probability for class 1
if list(model.classes_).count(1):
    p1 = float(proba[list(model.classes_).index(1)])
else:
    p1 = float(proba[0])

st.markdown("---")
st.subheader("Result")
st.write(f"Estimated chance of heart disease: **{p1*100:.1f}%**")

if p1 >= 0.6:
    st.error("⚠️ This indicates a higher chance of heart disease. Please consult a doctor.")
elif p1 >= 0.3:
    st.warning("⚠️ Moderate risk. Consider seeing a healthcare provider for check-up.")
else:
    st.success("✅ Low risk based on the provided information.")

st.caption("This is an educational tool and not a substitute for professional medical advice.")