# Import required libraries
import streamlit as st
import pandas as pd
import joblib

# Load the trained machine learning model, scaler, and feature names
model = joblib.load("gaming_addiction_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# App title
st.title("Gaming Addiction Risk Prediction")

# Force left-to-right layout for better UI formatting
st.markdown("""
<style>
html, body, [class*="css"] {
    direction: ltr !important;
}
</style>
""", unsafe_allow_html=True)

# Short description of the application
st.write(
    "This app predicts gaming addiction risk level "
    "based on lifestyle and gaming behavior."
)

# User input: average daily gaming hours
daily_gaming_hours = st.number_input(
    "Daily gaming hours",
    0.0, 24.0, 3.0
)

# User input: monthly spending on games
monthly_game_spending_usd = st.number_input(
    "Monthly game spending (USD)",
    0.0, 10000.0, 50.0
)

# User input: sleep quality score from 1 to 5
sleep_quality = st.number_input(
    "Sleep quality (1 = Very Poor, 5 = Excellent)",
    1, 5, 3
)

# User input: average sleep hours
sleep_hours = st.number_input(
    "Sleep hours",
    0.0, 24.0, 7.0
)

# User input: social isolation score from 1 to 5
social_isolation_score = st.number_input(
    "Social isolation score (1 = Low Isolation, 5 = High Isolation)",
    1, 5, 3
)

# User input: whether the user experiences withdrawal symptoms
withdrawal_symptoms = st.selectbox(
    "Withdrawal symptoms (0 = No, 1 = Yes)",
    [0, 1]
)

# User input: whether the user lost interest in other activities
loss_of_other_interests = st.selectbox(
    "Loss of other interests (0 = No, 1 = Yes)",
    [0, 1]
)

# User input: whether the user continues gaming despite problems
continued_despite_problems = st.selectbox(
    "Continued despite problems (0 = No, 1 = Yes)",
    [0, 1]
)

# User input: GPA value
grades_gpa = st.number_input(
    "GPA",
    0.0, 5.0, 3.5
)

# User input: academic or work performance score
academic_work_performance = st.number_input(
    "Academic/work performance (1 = Poor, 5 = Excellent)",
    1, 5, 3
)

# User input: weekly exercise hours
exercise_hours_weekly = st.number_input(
    "Exercise hours weekly",
    0.0, 40.0, 3.0
)

# Store all user inputs inside a DataFrame
input_data = pd.DataFrame([{
    "daily_gaming_hours": daily_gaming_hours,
    "monthly_game_spending_usd": monthly_game_spending_usd,
    "sleep_quality": sleep_quality,
    "sleep_hours": sleep_hours,
    "social_isolation_score": social_isolation_score,
    "withdrawal_symptoms": withdrawal_symptoms,
    "loss_of_other_interests": loss_of_other_interests,
    "continued_despite_problems": continued_despite_problems,
    "grades_gpa": grades_gpa,
    "academic_work_performance": academic_work_performance,
    "exercise_hours_weekly": exercise_hours_weekly
}])

# Arrange columns in the same order used during model training
input_data = input_data[features]

# Scale the input data using the saved scaler
input_scaled = scaler.transform(input_data)

# Dictionary for converting prediction numbers into readable labels
risk_labels = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
    3: "Very High Risk"
}

# Predict button
if st.button("Predict"):

    # Generate prediction using the trained model
    prediction = model.predict(input_scaled)[0]

    # Display prediction result to the user
    st.success(
        f"Predicted Gaming Addiction Risk Level: "
        f"{prediction} - {risk_labels[prediction]}"
    )