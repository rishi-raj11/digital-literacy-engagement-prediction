import streamlit as st
import pickle
import pandas as pd

# ==========================
# LOAD MODEL
# ==========================

model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Digital Literacy Engagement Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Digital Literacy Engagement Predictor")

st.markdown(
    """
    Predict learner engagement using Machine Learning.
    """
)

# ==========================
# INPUTS
# ==========================

computer = st.slider(
    "Computer Knowledge",
    1,
    5,
    3
)

internet = st.slider(
    "Internet Usage",
    1,
    5,
    3
)

mobile = st.slider(
    "Mobile Literacy",
    1,
    5,
    3
)

modules = st.slider(
    "Modules Completed",
    1,
    5,
    3
)

learning_time = st.slider(
    "Time Spent Learning",
    1,
    5,
    3
)

quiz = st.slider(
    "Quiz Performance",
    1,
    5,
    3
)

sessions = st.slider(
    "Session Frequency",
    1,
    5,
    3
)

adaptability = st.slider(
    "Adaptability",
    1,
    5,
    3
)

feedback = st.slider(
    "Feedback Rating",
    1,
    5,
    3
)

skill = st.slider(
    "Skill Application",
    1,
    5,
    3
)

# ==========================
# SCALE CONVERSION
# ==========================

computer_score = computer * 10
internet_score = internet * 10
mobile_score = mobile * 10

modules_completed = 5 + (modules - 1) * 2.5

avg_time = 10 + (learning_time - 1) * 5

quiz_score = 60 + (quiz - 1) * 10

session_count = 10 + (sessions - 1) * 5

adaptability_score = 50 + (adaptability - 1) * 12.5

skill_score = 50 + (skill - 1) * 12.5

overall_score = (
    computer_score
    + internet_score
    + mobile_score
    + adaptability_score
    + skill_score
) / 5

# ==========================
# PREDICT BUTTON
# ==========================

if st.button("Predict Engagement"):

    data = [[
        computer_score,
        internet_score,
        mobile_score,
        modules_completed,
        avg_time,
        quiz_score,
        session_count,
        adaptability_score,
        feedback,
        skill_score,
        overall_score
    ]]

    prediction = model.predict(data)[0]

    probs = model.predict_proba(data)[0]

    engagement = encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = round(
        max(probs) * 100,
        2
    )

    st.success(
        f"Predicted Engagement: {engagement}"
    )

    st.info(
        f"Confidence: {confidence}%"
    )

    st.subheader(
        "Probability Breakdown"
    )

    prob_df = pd.DataFrame({
        "Category": encoder.classes_,
        "Probability (%)":
        [round(p * 100, 2) for p in probs]
    })

    st.dataframe(prob_df)

    st.bar_chart(
        prob_df.set_index("Category")
    )