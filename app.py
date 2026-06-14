import streamlit as st
import numpy as np
import joblib
import time
import os
from datetime import datetime
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="HealthAI Diagnostics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# 2. Session State Management (This controls the Login system)
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# 3. Load the ML model
@st.cache_resource # Caches the model so it doesn't reload on every click
def load_model():
    try:
        data = joblib.load('diabetes_model.pkl')
        return data['model'], data['scaler']
    except FileNotFoundError:
        return None, None

model, scaler = load_model()
if model is None:
    st.error("Model file not found. Please run train_model.py first!")
    st.stop()

# 4. Simulated LLM Generation Engine
def generate_llm_advice(prediction, glucose, bmi, age):
    advice = ""
    if prediction == 1:
        advice += f"### 🩺 **AI Health Agent Clinical Analysis**\n\n"
        advice += f"Based on the predictive model, the patient is classified as **High Risk** for metabolic escalation. "
        advice += f"Specifically, the current glucose level of **{glucose} mg/dL** is a primary physiological contributing factor.\n\n"
        if bmi > 30:
            advice += f"Furthermore, a BMI of **{bmi}** indicates clinical obesity, which compounds insulin resistance.\n"
        advice += "\n### 📋 **Targeted Action Plan**\n"
        advice += "* **Clinical Protocol:** Schedule a formal HbA1c screening with an endocrinologist within 7 days.\n"
        advice += "* **Nutritional Intervention:** Transition immediately to a low-glycemic index dietary framework.\n"
    else:
        advice += f"### 🩺 **AI Health Agent Clinical Analysis**\n\n"
        advice += f"The predictive model classifies this profile as **Low Risk**. Current baseline metrics indicate stable metabolic markers. "
        if glucose > 100:
            advice += f"However, notice that the glucose level of **{glucose} mg/dL** is slightly elevated.\n"
            advice += "\n### 📋 **Preventative Action Plan**\n"
            advice += "* **Routine Screenings:** Perform a fasting blood glucose check every 6 months to monitor trends.\n"
        else:
            advice += "\n\n### 📋 **Maintenance Strategy**\n"
            advice += "* **Optimal Health:** Metrics are excellently balanced. Maintain current habits. No clinical interventions are indicated."
    return advice

# ==========================================
# SCREEN 1: LOGIN PAGE
# ==========================================
if not st.session_state['logged_in']:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h2 style='text-align: center;'>Patient Portal Login</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Please enter your details to access the AI Diagnostics Dashboard.</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            user_name = st.text_input("Full Name")
            user_email = st.text_input("Email Address")
            user_phone = st.text_input("Phone Number (Optional)")
            
            submit_login = st.form_submit_button("Secure Login", use_container_width=True)
            
            if submit_login:
                if user_name and user_email:
                    # Save user details to session state so we can use them later
                    st.session_state['user_name'] = user_name
                    st.session_state['user_email'] = user_email
                    st.session_state['logged_in'] = True
                    st.rerun() # Instantly refreshes the page to show the dashboard
                else:
                    st.error("Please fill in both Name and Email to continue.")

# ==========================================
# SCREEN 2: MAIN DASHBOARD
# ==========================================
else:
    # Header Section with Logout Button
    head_col1, head_col2 = st.columns([4, 1])
    with head_col1:
        st.markdown(f"# 🩺 Welcome, {st.session_state['user_name']}")
        st.markdown("##### *Predictive Healthcare & AI Coaching Dashboard*")
    with head_col2:
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
            
    st.markdown("---")

    # Dashboard Layout
    main_col1, main_col2 = st.columns([1, 1.2], gap="large")

    with main_col1:
        st.subheader("📊 Patient Clinical Vitals")
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, value=0, step=1)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, value=120.0, step=1.0)
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, value=70.0, step=1.0)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0, step=1.0)
        with sub_col2:
            insulin = st.number_input("Insulin Level (μIU/mL)", min_value=0.0, value=79.0, step=1.0)
            bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, value=32.0, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, step=0.01)
            age = st.number_input("Age (Years)", min_value=1, value=33, step=1)

        st.write("")
        run_diagnostic = st.button("🚀 Run System Diagnostic", use_container_width=True)

    with main_col2:
        st.subheader("🖥️ Real-Time System Output")
        
        if run_diagnostic:
            features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]
            
            # --- SAVE RECORD TO DATABASE WITH PERSONAL INFO ---
            record = pd.DataFrame([{
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Patient_Name": st.session_state['user_name'],
                "Patient_Email": st.session_state['user_email'],
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "Blood_Pressure": blood_pressure,
                "Skin_Thickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DPF": dpf,
                "Age": age,
                "Diagnosis": "High Risk" if prediction == 1 else "Low Risk"
            }])
            
            db_file = "patient_database.csv"
            if not os.path.isfile(db_file):
                record.to_csv(db_file, index=False)
            else:
                record.to_csv(db_file, mode='a', header=False, index=False)
            # ------------------------------------------------

            st.markdown("#### **1. Predictive Analytics Engine (ML)**")
            if prediction == 1:
                st.error("### 🚨 High Risk Detected")
            else:
                st.success("### ✅ Low Risk Detected")
                
            st.markdown("---")
            st.markdown("#### **2. Custom Intervention Strategy (LLM)**")
            with st.spinner("AI Health Agent compiling custom recommendations..."):
                time.sleep(1.2)
                
            custom_advice = generate_llm_advice(prediction, glucose, bmi, age)
            st.info(custom_advice)
            st.success("💾 Patient record and diagnosis successfully saved to database.")
            
        else:
            st.info("💡 **System Ready:** Adjust the parameters on the left and click 'Run System Diagnostic'.")