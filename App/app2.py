#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Load trained models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load trained models
rf = joblib.load("RandomForestClassifier.pkl")
gb = joblib.load("GradientBoostingClassifier.pkl")
ada = joblib.load("AdaBoostClassifier.pkl")
meta_model = joblib.load("stacking_model.pkl")  # Logistic Regression meta-model
scaler = joblib.load("scaler.pkl")  # Load the scaler used during training

def predict_attrition(input_data):
    """
    Process raw features through base models and get meta-features for stacking.
    Then predict attrition using the meta-model.
    """
    try:
        input_data = np.array(input_data).reshape(1, -1)  # Ensure correct shape
        
        # Apply feature scaling
        input_data = scaler.transform(input_data)

        # Get probabilities from base models
        rf_prob = rf.predict_proba(input_data)[:, 1]  # Probability of Attrition (Class 1)
        gb_prob = gb.predict_proba(input_data)[:, 1]
        ada_prob = ada.predict_proba(input_data)[:, 1]

        # Stack probabilities as meta-features
        meta_features = np.column_stack([rf_prob, gb_prob, ada_prob])

        # Final prediction using stacking model
        probability = meta_model.predict_proba(meta_features)[0][1] * 100  # Attrition probability
        
        # Adjusting threshold
        prediction = 1 if probability > 50 else 0  # Default threshold: 50%

        return prediction, probability

    except Exception as e:
        print(f"Model Prediction Error: {e}")
        return 0, 0  # Default values instead of None


# 🎯 *Streamlit UI*
st.title("🔍 HR Attrition Prediction & Analytics")
st.subheader("📊 Predict Employee Attrition Risk & Analyze Trends")

# Tabs for Prediction & Analytics
tab1, tab2 = st.tabs(["📈 Attrition Prediction", "📊 Analytics Dashboard"])

# 🎯 *Tab 1: Attrition Prediction*
with tab1:
    st.subheader("📋 Employee Details")

    # User Inputs
    age = st.number_input("📅 Age", min_value=18, max_value=60, step=1)
    years_at_company = st.number_input("🏢 Years at Company", min_value=0, max_value=40, step=1)
    monthly_income = st.number_input("💰 Monthly Income ($)", min_value=1000, max_value=50000, step=500)
    distance_from_home = st.number_input("📍 Distance from Home (miles)", min_value=1, max_value=50, step=1)
    number_of_promotions = st.number_input("🚀 Number of Promotions", min_value=0, max_value=10, step=1)

    # Dropdown Inputs
    gender = st.selectbox("🧑 Gender", ["Male", "Female", "Other"])
    job_role = st.selectbox("💼 Job Role", ["Finance", "Healthcare", "Technology", "Education", "Media"])
    work_life_balance = st.selectbox("⚖ Work-Life Balance", ["Poor", "Below Average", "Good", "Excellent"])
    job_satisfaction = st.selectbox("😊 Job Satisfaction", ["Very Low", "Low", "Medium", "High"])
    performance_rating = st.selectbox("📊 Performance Rating", ["Low", "Below Average", "Average", "High"])
    education_level = st.selectbox("🎓 Education Level", ["High School", "Associate Degree", "Bachelor’s", "Master’s", "PhD"])
    marital_status = st.selectbox("❤ Marital Status", ["Divorced", "Married", "Single"])
    job_level = st.selectbox("📈 Job Level", ["Entry", "Mid", "Senior"])
    company_size = st.selectbox("🏢 Company Size", ["Small", "Medium", "Large"])
    remote_work = st.selectbox("🏡 Remote Work", ["Yes", "No"])
    leadership_opportunities = st.selectbox("🎯 Leadership Opportunities", ["Yes", "No"])
    innovation_opportunities = st.selectbox("🚀 Innovation Opportunities", ["Yes", "No"])
    company_reputation = st.selectbox("🏆 Company Reputation", ["Very Poor", "Poor", "Good", "Excellent"])
    employee_recognition = st.selectbox("🌟 Employee Recognition", ["Very Low", "Low", "Medium", "High"])

    # *📌 Compute Derived Features (MCDM Features)*
    tenure_ratio = years_at_company / (years_at_company + 1)  # Prevent division by zero
    promotion_rate = number_of_promotions / (years_at_company + 1)  # Prevent division by zero
    satisfaction_index = (0.6 * (["Very Low", "Low", "Medium", "High"].index(job_satisfaction) + 1)) + \
                         (0.4 * (["Poor", "Below Average", "Good", "Excellent"].index(work_life_balance) + 1))

    # Convert categorical inputs to numeric
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    job_role_map = {"Finance": 0, "Healthcare": 1, "Technology": 2, "Education": 3, "Media": 4}
    performance_map = {"Low": 0, "Below Average": 1, "Average": 2, "High": 3}
    education_map = {"High School": 0, "Associate Degree": 1, "Bachelor’s": 2, "Master’s": 3, "PhD": 4}
    marital_map = {"Divorced": 0, "Married": 1, "Single": 2}
    job_level_map = {"Entry": 0, "Mid": 1, "Senior": 2}
    company_size_map = {"Small": 0, "Medium": 1, "Large": 2}
    yes_no_map = {"No": 0, "Yes": 1}
    reputation_map = {"Very Poor": 0, "Poor": 1, "Good": 2, "Excellent": 3}
    recognition_map = {"Very Low": 0, "Low": 1, "Medium": 2, "High": 3}

    input_data = [
        age,
        years_at_company,
        monthly_income,
        distance_from_home,
        number_of_promotions,
        gender_map[gender],
        job_role_map[job_role],
        performance_map[performance_rating],
        education_map[education_level],
        marital_map[marital_status],
        job_level_map[job_level],
        company_size_map[company_size],
        yes_no_map[remote_work],
        yes_no_map[leadership_opportunities],
        yes_no_map[innovation_opportunities],
        reputation_map[company_reputation],
        recognition_map[employee_recognition],
        tenure_ratio,
        promotion_rate,
        satisfaction_index
    ]

    # *🎯 Prediction Button*
    if st.button("🔮 Predict Attrition Risk"):
        prediction, probability = predict_attrition(input_data)  # Get prediction

        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.error(f"⚠ High Attrition Risk ({probability:.2f}%)")
            st.write("🚨 *HR Action Needed:* Consider salary hikes, promotions, or workload balance.")
        else:
            st.success(f"✅ Low Attrition Risk ({100 - probability:.2f}%)")
            st.write("😊 *Employee is likely to stay.* Keep engagement high.")

