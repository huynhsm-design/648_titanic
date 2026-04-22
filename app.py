import os
import pickle
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict survival:")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.number_input("Age", 0.0, 100.0, 25.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 30.0)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# Define a custom threshold (you can adjust this as needed)
SURVIVAL_THRESHOLD = 0.5
st.write("Survival Probability Threshold:", SURVIVAL_THRESHOLD)

if st.button("Predict Survival"):

    input_df = pd.DataFrame([{
        'pclass': pclass,
        'sex': sex,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'fare': fare,
        'embarked': embarked
    }])

    # Get probability for the positive class (survived=1)
    prob = model.predict_proba(input_df)[0][1]

    # Apply the custom threshold for prediction
    pred = 1 if prob >= SURVIVAL_THRESHOLD else 0

    label = "Survived! 🎉" if pred == 1 else "Did Not Survive 😢"

    st.subheader(label)
    st.write(f"Survival Probability: {prob:.2f}")
