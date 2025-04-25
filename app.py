import streamlit as st
import pandas as pd
import joblib

# Load model and label encoder
model = joblib.load("model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# App Title
st.title("🎯 Career Prediction App")

# Main Header for user input section
st.header("📝 Enter Your Details")

# Input fields (on main page)
work_env = st.selectbox("Preferred Work Environment", ["Office", "Remote", "Hybrid"])
work_env_map = {"Office": 1, "Remote": 2, "Hybrid": 3}
work_env_val = work_env_map[work_env]

risk = st.number_input("Risk-Taking Ability (1 to 10)", min_value=1, max_value=10, step=1)
age = st.number_input("Age", min_value=10, max_value=25, step=1)
finance = st.number_input("Financial Stability (1 - low, 10 - high)", min_value=1, max_value=10, step=1)

subject = st.selectbox("Preferred Subjects", ["Science", "Commerce", "Arts"])
subject_map = {"Science": 1, "Commerce": 2, "Arts": 3}
subject_val = subject_map[subject]

siblings = st.number_input("Number of Siblings", min_value=0, max_value=10, step=1)

extra = st.selectbox("Participation in Extracurricular Activities", ["No", "Yes"])
extra_val = 1 if extra == "Yes" else 0

music = st.selectbox("Preferred Music Genre", ["Pop", "Classical", "Rock"])
music_map = {"Pop": 1, "Classical": 2, "Rock": 3}
music_val = music_map[music]

leadership = st.number_input("Leadership Experience (0 to 10)", min_value=0, max_value=10, step=1)

tech_input = st.selectbox(
    "Tech-Savviness",
    ["Not Comfortable", "Basic", "Intermediate", "Comfortable", "Advanced"]
)
tech_map = {
    "Not Comfortable": 1,
    "Basic": 3,
    "Intermediate": 5,
    "Comfortable": 7,
    "Advanced": 10
}
tech = tech_map[tech_input]

motivation = st.selectbox("Motivation for Career Choice", ["Passion", "Money", "Stability"])
motivation_map = {"Passion": 1, "Money": 2, "Stability": 3}
motivation_val = motivation_map[motivation]

cgpa = st.number_input("Academic Performance (0–100)", min_value=0, max_value=100, step=1)
water = st.number_input("Daily Water Intake (in Litres)", min_value=0.0, max_value=10.0, step=0.5)

# Feature Engineering
input_data = {
    "Preferred Work Environment": work_env_val,
    "Risk-Taking Ability": risk,
    "Age": age,
    "Financial Stability - self/family (1 is low income and 10 is high income)": finance,
    "Preferred Subjects in Highschool/College": subject_val,
    "Number of Siblings": siblings,
    "Participation in Extracurricular Activities": extra_val,
    "Preferred Music Genre": music_val,
    "Leadership Experience": leadership,
    "Tech-Savviness": tech,
    "Motivation for Career Choice": motivation_val,
    "risk_x_leadership": risk * leadership,
    "low_finance_high_risk": int(risk > 8 and finance < 5),
    "cgpa_bucket_Medium": int(60 < cgpa <= 80),
    "cgpa_bucket_High": int(cgpa > 80),
    "water_category_Medium": int(2 < water <= 4),
    "water_category_High": int(water > 4),
    "academic_risk_profile": cgpa * risk,
    "leadership_to_siblings_ratio": leadership / (siblings + 1),
    "tech_finance_combo": tech * finance,
    "persona_cluster": 1  # Placeholder if clustering logic is applied
}

input_df = pd.DataFrame([input_data])

# Predict button
if st.button("🔍 Predict Career"):
    prediction = model.predict(input_df)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🎉 Your Predicted Career Path: **{predicted_label}**")
