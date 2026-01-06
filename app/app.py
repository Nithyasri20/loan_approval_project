import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Loan Approval System",
    layout="wide"
)

# ----------------------------------
# Load CSS
# ----------------------------------
with open("styles/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ----------------------------------
# Session State
# ----------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ----------------------------------
# NAVIGATION BAR
# ----------------------------------
st.markdown("""
<div class="navbar">
    <div class="navbar-title">Loan Approval</div>
    <div class="navbar-links">
        <a href="#" class="nav-link">Home</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------
# HOME PAGE
# ----------------------------------
if st.session_state.page == "home":
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <h1 class="hero-title">Prediction Of Modernized Loan Approval System Based On Machine Learning Approach</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("🚀 Apply for Loan", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()

# ----------------------------------
# FORM PAGE
# ----------------------------------
if st.session_state.page == "form":

    st.markdown("""
    <div class="main-card">
        <h2 class="form-title">📄 Loan Application Form</h2>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------------
    # Load Model Files
    # ----------------------------------
    with open("models/loan_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/features.pkl", "rb") as f:
        feature_order = pickle.load(f)

    # ----------------------------------
    # Form
    # ----------------------------------
    with st.form("loan_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            Applicant_Income = st.number_input("Applicant Income", 0)
            Employment_Status = st.selectbox("Employment Status", ["Salaried", "Self-employed"])
            Credit_Score = st.number_input("Credit Score", 300, 850)

        with c2:
            Coapplicant_Income = st.number_input("Coapplicant Income", 0)
            Age = st.number_input("Age", 18, 70)
            Loan_Amount = st.number_input("Loan Amount", 0)

        with c3:
            Loan_Term = st.number_input("Loan Term (months)", 12, 360, step=12)
            Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            Education_Level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

        submit = st.form_submit_button("📊 Predict Loan Status")

    # ----------------------------------
    # Prediction + Chart
    # ----------------------------------
    if submit:

        mappings = {
            "Employment_Status": {"Salaried": 1, "Self-employed": 0},
            "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0},
            "Education_Level": {"Graduate": 1, "Not Graduate": 0},
        }

        input_data = {
            "Applicant_Income": Applicant_Income,
            "Coapplicant_Income": Coapplicant_Income,
            "Employment_Status": mappings["Employment_Status"][Employment_Status],
            "Age": Age,
            "Credit_Score": Credit_Score,
            "Loan_Amount": Loan_Amount,
            "Loan_Term": Loan_Term,
            "Property_Area": mappings["Property_Area"][Property_Area],
            "Education_Level": mappings["Education_Level"][Education_Level],
        }

        df = pd.DataFrame([input_data])

        for col in feature_order:
            if col not in df:
                df[col] = 0

        df = df[feature_order]

        prediction = model.predict(scaler.transform(df))[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        # -----------------------------
        # Decision Zone Chart
        # -----------------------------
        score = round(((Credit_Score - 300) / 550) * 100, 2)

        base = pd.DataFrame({"Score": [0, 100], "Y": [1, 1]})
        fig = px.line(base, x="Score", y="Y", title="Loan Approval Decision Zones")

        fig.update_yaxes(visible=False)

        fig.add_vrect(0, 40, fillcolor="red", opacity=0.3, annotation_text="Rejected")
        fig.add_vrect(40, 60, fillcolor="yellow", opacity=0.3, annotation_text="Review")
        fig.add_vrect(60, 100, fillcolor="green", opacity=0.3, annotation_text="Approved")

        fig.add_vline(
            x=score,
            line_dash="dash",
            annotation_text=f"Your Score: {score}%",
            line_color="black"
        )

        st.plotly_chart(fig, use_container_width=True)
    
    # Back Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()