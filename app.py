import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Load Model and Preprocessing Objects
# ==========================================

model = joblib.load("recruitment_model.pkl")
scaler = joblib.load("scaler.pkl")
threshold = joblib.load("threshold.pkl")
feature_names = joblib.load("feature_names.pkl")


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Smart Recruitment Assistant",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# Title
# ==========================================

st.title("Smart Recruitment Assistant")

st.write(
    "An AI-powered recruitment screening system "
    "that predicts whether a candidate should advance "
    "to the next stage of the hiring process."
)

st.divider()


# ==========================================
# Candidate Information
# ==========================================

st.header("Candidate Information")

col1, col2 = st.columns(2)


# ==========================================
# Column 1
# ==========================================

with col1:

    city_development_index = st.slider(
        "City Development Index",
        min_value=0.0,
        max_value=1.0,
        value=0.80,
        step=0.01
    )

    relevant_experience = st.selectbox(
        "Relevant Experience",
        [
            "No relevant experience",
            "Has relevant experience",
            "Unknown"
        ]
    )

    enrolled_university = st.selectbox(
        "Enrolled University",
        [
            "no_enrollment",
            "Part time course",
            "Full time course",
            "Unknown"
        ]
    )

    education_level = st.selectbox(
        "Education Level",
        [
            "Primary School",
            "High School",
            "Graduate",
            "Masters",
            "Phd",
            "Unknown"
        ]
    )

    company_size = st.selectbox(
        "Company Size",
        [
            "Unknown",
            "<10",
            "10-49",
            "50-99",
            "100-500",
            "500-999",
            "1000-4999",
            "5000-9999",
            "10000+"
        ]
    )

    training_hours = st.number_input(
        "Training Hours",
        min_value=0,
        max_value=500,
        value=30
    )


# ==========================================
# Column 2
# ==========================================

with col2:

    experience = st.selectbox(
        "Experience",
        [
            "Unknown",
            "<1",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            ">20"
        ]
    )

    last_new_job = st.selectbox(
        "Last New Job",
        [
            "Unknown",
            "never",
            "1",
            "2",
            "3",
            "4",
            ">4"
        ]
    )

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male",
            "Other",
            "Unknown"
        ]
    )

    major_discipline = st.selectbox(
        "Major Discipline",
        [
            "Business Degree",
            "Humanities",
            "No Major",
            "Other",
            "STEM",
            "Unknown"
        ]
    )

    company_type = st.selectbox(
        "Company Type",
        [
            "Funded Startup",
            "NGO",
            "Other",
            "Public Sector",
            "Pvt Ltd",
            "Unknown"
        ]
    )


st.divider()


# ==========================================
# Prediction
# ==========================================

if st.button("Predict Candidate", use_container_width=True):

    # --------------------------------------
    # Create empty dataframe
    # --------------------------------------

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_names
    )


    # --------------------------------------
    # Numerical Features
    # --------------------------------------

    input_data["city_development_index"] = city_development_index
    input_data["training_hours"] = training_hours


    # --------------------------------------
    # Experience
    # --------------------------------------

    if experience == "<1":
        experience_clean = 0

    elif experience == ">20":
        experience_clean = 21

    elif experience == "Unknown":
        experience_clean = -1

    else:
        experience_clean = int(experience)

    input_data["experience_clean"] = experience_clean


    # --------------------------------------
    # Last New Job
    # --------------------------------------

    if last_new_job == "never":
        last_new_job_clean = 0

    elif last_new_job == ">4":
        last_new_job_clean = 5

    elif last_new_job == "Unknown":
        last_new_job_clean = -1

    else:
        last_new_job_clean = int(last_new_job)

    input_data["last_new_job_clean"] = last_new_job_clean


    # --------------------------------------
    # Relevant Experience
    # --------------------------------------

    relevant_mapping = {
        "No relevant experience": 0,
        "Has relevant experience": 1,
        "Unknown": -1
    }

    input_data["relevent_experience"] = (
        relevant_mapping[relevant_experience]
    )


    # --------------------------------------
    # Enrolled University
    # --------------------------------------

    university_mapping = {
        "no_enrollment": 0,
        "Part time course": 1,
        "Full time course": 2,
        "Unknown": -1
    }

    input_data["enrolled_university"] = (
        university_mapping[enrolled_university]
    )


    # --------------------------------------
    # Education Level
    # --------------------------------------

    education_mapping = {
        "Primary School": 0,
        "High School": 1,
        "Graduate": 2,
        "Masters": 3,
        "Phd": 4,
        "Unknown": -1
    }

    input_data["education_level"] = (
        education_mapping[education_level]
    )


    # --------------------------------------
    # Company Size
    # --------------------------------------

    company_size_mapping = {
        "Unknown": 0,
        "<10": 1,
        "10-49": 2,
        "50-99": 3,
        "100-500": 4,
        "500-999": 5,
        "1000-4999": 6,
        "5000-9999": 7,
        "10000+": 8
    }

    input_data["company_size"] = (
        company_size_mapping[company_size]
    )


    # --------------------------------------
    # Gender One-Hot Encoding
    # --------------------------------------

    gender_column = f"gender_{gender}"

    if gender_column in input_data.columns:
        input_data[gender_column] = 1


    # --------------------------------------
    # Major Discipline One-Hot Encoding
    # --------------------------------------

    major_column = f"major_discipline_{major_discipline}"

    if major_column in input_data.columns:
        input_data[major_column] = 1


    # --------------------------------------
    # Company Type One-Hot Encoding
    # --------------------------------------

    company_column = f"company_type_{company_type}"

    if company_column in input_data.columns:
        input_data[company_column] = 1


    # --------------------------------------
    # Make sure feature order is correct
    # --------------------------------------

    input_data = input_data[feature_names]


    # --------------------------------------
    # Scaling
    # --------------------------------------

    input_scaled = scaler.transform(input_data)


    # --------------------------------------
    # Prediction Probability
    # --------------------------------------

    probability = model.predict_proba(
        input_scaled
    )[0][1]


    # --------------------------------------
    # Apply Threshold
    # --------------------------------------

    prediction = int(
        probability >= threshold
    )


    # ======================================
    # Display Result
    # ======================================

    st.divider()

    st.header("Prediction Result")


    if prediction == 1:

        st.success(
            "Recommended for the Next Stage"
        )

    else:

        st.warning(
            "Not Recommended for the Next Stage"
        )


    st.metric(
        "Candidate Probability",
        f"{probability * 100:.2f}%"
    )


    st.progress(float(probability))


    st.info(
        f"Decision Threshold: {threshold}"
    )