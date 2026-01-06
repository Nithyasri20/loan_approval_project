Loan Approval Prediction App

This is an interactive web application that predicts whether a loan will be approved based on applicant details. The app uses a Random Forest Classifier trained on real loan applicant data and is deployed using Streamlit for easy user interaction.

Project Overview

Many banks and financial institutions need to quickly assess whether a loan application is likely to be approved. This project uses machine learning to automate that decision-making process based on applicant features like income, age, credit score, employment, and more.

The model has been trained on historical loan data and can predict loan approval for new applicants.

Project Structure
Loan_ML_Project/
│
├── app/
│   └── app.py
├── styles/
│   └── style.css
├── models/
│   ├── loan_model.pkl
│   └── scaler.pkl
├── data/
│   └── loan_approval.csv
└── notebook/
    └── loan_approval.ipynb


Features

Interactive form to input applicant details

Supports both numerical and categorical features

Uses pre-trained Random Forest model for quick prediction

Displays results clearly:

✅ Loan Approved

❌ Loan Not Approved

Uses StandardScaler for consistent feature scaling

Lightweight and easy to deploy with Streamlit

Machine Learning Model

Model Type: Random Forest Classifier (Supervised Learning – Classification)

Target Variable: Loan_Approved (1 = Yes, 0 = No)

Features Used:

Applicant Income, Coapplicant Income, Age, Credit Score, Employment Status

Loan Amount, Loan Term, Loan Purpose, Property Area, Education Level

Gender, Employer Category, Dependents, Savings, Collateral Value, DTI Ratio

Preprocessing:

Categorical features encoded to numeric values

StandardScaler used to scale numeric features

Training: Done once in Jupyter Notebook, predictions handled in the Streamlit app

How to Run the App

Clone the repository:

git clone <repository_url>
cd Loan_ML_Project


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app/app.py


Use the app:

Enter applicant details in the form

Click Predict

View the loan approval result

Example Input
Feature	Example Value
Applicant Income	12000
Coapplicant Income	2000
Employment Status	Salaried
Age	35
Marital Status	Single
Dependents	1
Credit Score	650
Loan Amount	15000
Loan Term	60 months
Loan Purpose	Personal
Example Output
Prediction Result: ✅ Loan Approved


or

Prediction Result: ❌ Loan Not Approved
