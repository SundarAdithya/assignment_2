
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Page setup

st.set_page_config(page_title="SmartPremium Insurance Predictor", page_icon="🏥", layout="wide")


# Custom CSS

st.markdown("""
    <style>
    /* Apply to all alert messages */
    .stAlert p {
        color: black !important;
        font-weight: 500;
    }

    [data-theme="dark"] .stAlert p {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


# App Header

st.markdown('<h1 class="main-header">🏥 SmartPremium Insurance Predictor</h1>', unsafe_allow_html=True)


# Generate synthetic data & train model

@st.cache_data
def get_model():
    np.random.seed(42)
    n = 2000
    data = {
        'Age': np.random.randint(18, 80, n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Annual_Income': np.random.lognormal(10.5, 0.8, n),
        'Health_Score': np.clip(np.random.normal(70, 15, n), 1, 100),
        'Smoking_Status': np.random.choice(['Yes', 'No'], n, p=[0.25, 0.75]),
        'Previous_Claims': np.random.poisson(0.3, n),
        'Vehicle_Age': np.random.randint(0, 20, n),
        'Credit_Score': np.random.choice(range(300, 850), n),
        'Policy_Type': np.random.choice(['Basic', 'Comprehensive', 'Premium'], n),
        'Location': np.random.choice(['Urban', 'Suburban', 'Rural'], n),
    }
    
    data['Premium_Amount'] = (
        1000 + data['Age'] * 20 + (data['Annual_Income'] / 10000) * 15 +
        (data['Health_Score'] - 50) * 20 + (data['Smoking_Status'] == 'Yes') * 500 +
        data['Previous_Claims'] * 200 + (data['Vehicle_Age'] > 10) * 300 +
        np.random.normal(0, 200, n)
    )
    
    df = pd.DataFrame(data)
    X = df[['Age', 'Gender', 'Annual_Income', 'Health_Score', 'Smoking_Status', 
            'Previous_Claims', 'Vehicle_Age', 'Credit_Score', 'Policy_Type', 'Location']]
    y = df['Premium_Amount']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['Age', 'Annual_Income', 'Health_Score', 'Previous_Claims', 'Vehicle_Age', 'Credit_Score']),
        ('cat', OneHotEncoder(), ['Gender', 'Smoking_Status', 'Policy_Type', 'Location'])
    ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X, y)
    return model, df

model, df = get_model()


# Sidebar navigation

page = st.sidebar.selectbox("Choose a page", ["Premium Prediction", "Data Analysis"])


# Premium Prediction Page

if page == "Premium Prediction":
    st.markdown('<h2>📋 Customer Details</h2>', unsafe_allow_html=True)

    # Name input
    customer_name = st.text_input("Customer Name", placeholder="Enter full name")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 80, 35)
        gender = st.radio("Gender", ["Male", "Female"])
        annual_income = st.slider("Annual Income (₹)", 20000, 200000, 60000, 1000)
    with col2:
        health_score = st.slider("Health Score (1-100)", 1, 100, 75)
        smoking_status = st.radio("Smoking Status", ["Yes", "No"])
        previous_claims = st.slider("Previous Claims", 0, 10, 0)
    with col3:
        vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
        credit_score = st.slider("Credit Score", 300, 850, 700)
        policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
        location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    
    if st.button("Calculate Premium", use_container_width=True):
        input_data = pd.DataFrame([{
            'Age': age, 'Gender': gender, 'Annual_Income': annual_income,
            'Health_Score': health_score, 'Smoking_Status': smoking_status,
            'Previous_Claims': previous_claims, 'Vehicle_Age': vehicle_age,
            'Credit_Score': credit_score, 'Policy_Type': policy_type, 'Location': location
        }])
        
        prediction = model.predict(input_data)[0]

        # Show name with prediction in INR
        if customer_name.strip() != "":
            st.markdown(f'<p class="prediction-text">{customer_name}, your Predicted Premium is: ₹{prediction:,.2f}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="prediction-text">Predicted Premium: ₹{prediction:,.2f}</p>', unsafe_allow_html=True)

        # Insights
        insights = []
        if smoking_status == 'Yes': insights.append("🚭 Smoking increases premium")
        if health_score < 50: insights.append("⚕️ Low health score increases premium")
        if previous_claims > 0: insights.append("⚠️ Previous claims increase premium")
        if vehicle_age > 10: insights.append("🚗 Older vehicle increases premium")
        if credit_score < 600: insights.append("💳 Low credit score increases premium")
        if age > 60: insights.append("👵 Age factor increases premium")
        if policy_type == 'Premium': insights.append("🛡️ Premium policy increases cost")
        if not insights: insights.append("✅ All factors are within optimal ranges")
        
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)


# Data Analysis Page

else:
    st.markdown('<h2>📈 Data Analysis</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Records", len(df))
    with col2: st.metric("Average Premium", f"₹{df['Premium_Amount'].mean():,.0f}")
    with col3: st.metric("Average Age", f"{df['Age'].mean():.1f}")
    with col4: st.metric("Smokers", f"{len(df[df['Smoking_Status']=='Yes'])}")
    
    tab1, tab2, tab3 = st.tabs(["Demographics", "Financial", "Premium Analysis"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1: st.bar_chart(df['Age'].value_counts().sort_index())
        with col2: st.bar_chart(df['Gender'].value_counts())
    with tab2:
        st.bar_chart(pd.cut(df['Annual_Income'], bins=10).value_counts().reset_index(drop=True))
        st.bar_chart(pd.cut(df['Credit_Score'], bins=10).value_counts().reset_index(drop=True))
    with tab3:
        st.bar_chart(df.groupby('Policy_Type')['Premium_Amount'].mean())
        st.bar_chart(df.groupby('Smoking_Status')['Premium_Amount'].mean())
