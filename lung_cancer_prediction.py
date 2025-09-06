import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler

# Set page configuration
st.set_page_config(
    page_title="Lung Cancer Prediction App",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4682B4;
        border-bottom: 2px solid #1E90FF;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0066CC;
        transform: scale(1.05);
    }
    .prediction-box {
        background-color: #F0F8FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E90FF;
        margin-top: 1.5rem;
    }
    .risk-high {
        color: #FF4500;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .risk-low {
        color: #32CD32;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .info-box {
        background-color: #E6F2FF;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #F8F9FA;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained models
try:
    rf_model = load('rf_model.joblib')
    scaler = load('scaler.pkl') 
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {e}")

# ---------------- Dataset Preview Page ----------------
def dataset_preview_page():
    st.markdown('<h1 class="main-header">📊 DATASET PREVIEW</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">LUNG CANCER PREDICTION DATASET</h2>', unsafe_allow_html=True)

    # Link to dataset
    dataset_link = 'https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer'
    st.markdown(f'<div class="info-box">You can download the full dataset from <a href="{dataset_link}" target="_blank">Kaggle</a>.</div>', unsafe_allow_html=True)

    try:
        # Load a sample dataset for preview
        df = pd.read_csv('lung_data.csv')
        st.write('HERE IS A PREVIEW OF THE DATASET:')
        
        # Add some metrics about the dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Number of Features", len(df.columns))
        with col3:
            cancer_cases = df[df['LUNG_CANCER'] == 'YES'].shape[0] if 'LUNG_CANCER' in df.columns else "N/A"
            st.metric("Cancer Cases", cancer_cases)
            
        st.dataframe(df.head(20), use_container_width=True)
        
        # Show data description
        with st.expander("Dataset Description"):
            st.write("This dataset contains information about patients and their likelihood of having lung cancer.")
            if not df.empty:
                st.write("Columns and their descriptions:")
                for col in df.columns:
                    st.write(f"- **{col}**: {df[col].dtype}")
    except FileNotFoundError:
        st.error("❌ File 'lung_data.csv' not found. Please check the file path.")

# ---------------- Prediction Page ----------------
def prediction_page():
    st.markdown('<h1 class="main-header">🫁 LUNG CANCER PREDICTION APP</h1>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Fill in the patient details to predict the risk of lung cancer.</div>', unsafe_allow_html=True)
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="sub-header">Personal Information</h3>', unsafe_allow_html=True)
        GENDER = st.radio('Gender 👤', ['M', 'F'], horizontal=True)
        AGE = st.slider('Age 🎂', min_value=0, max_value=120, value=45)
        
        st.markdown('<h3 class="sub-header">Lifestyle Factors</h3>', unsafe_allow_html=True)
        SMOKING = st.radio('DO YOU SMOKE? 🚬', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        ALCOHOL_CONSUMING = st.radio('ALCOHOL CONSUMING 🍺', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        YELLOW_FINGERS = st.radio('YELLOW FINGERS ✋', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        PEER_PRESSURE = st.radio('PEER PRESSURE 👥', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
    
    with col2:
        st.markdown('<h3 class="sub-header">Symptoms</h3>', unsafe_allow_html=True)
        ANXIETY = st.radio('ANXIETY 😟', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        CHRONIC_DISEASE = st.radio('CHRONIC DISEASE 🏥', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        FATIGUE = st.radio('FATIGUE 😴', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        ALLERGY = st.radio('ALLERGY 🤧', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        WHEEZING = st.radio('WHEEZING 😤', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        COUGHING = st.radio('COUGHING 🤧', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        SHORTNESS_OF_BREATH = st.radio('SHORTNESS OF BREATH 🫁', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        SWALLOWING_DIFFICULTY = st.radio('SWALLOWING DIFFICULTY 😣', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        CHEST_PAIN = st.radio('CHEST PAIN ❤️‍🩹', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)

    # Center the predict button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button('PREDICT RISK 🔮', use_container_width=True)

    # When user clicks Predict button
    if predict_btn:
        # Create a dictionary for the input
        input_data = {
            'GENDER': [GENDER],
            'AGE': [AGE],
            'SMOKING': [SMOKING],
            'YELLOW_FINGERS': [YELLOW_FINGERS],
            'ANXIETY': [ANXIETY],
            'PEER_PRESSURE': [PEER_PRESSURE],
            'CHRONIC_DISEASE': [CHRONIC_DISEASE],
            'FATIGUE': [FATIGUE],
            'ALLERGY': [ALLERGY],
            'WHEEZING': [WHEEZING],
            'ALCOHOL_CONSUMING': [ALCOHOL_CONSUMING],
            'COUGHING': [COUGHING],
            'SHORTNESS_OF_BREATH': [SHORTNESS_OF_BREATH],
            'SWALLOWING_DIFFICULTY': [SWALLOWING_DIFFICULTY],
            'CHEST_PAIN': [CHEST_PAIN]
        }

        input_df = pd.DataFrame(input_data)

        # Get the correct feature names from the scaler (if available)
        if hasattr(scaler, "feature_names_in_"):
            model_columns = scaler.feature_names_in_.tolist()
        else:
            # Fallback: Use the expected column order (you might need to adjust this)
            model_columns = ['AGE', 'GENDER_M', 'GENDER_F', 'SMOKING', 'YELLOW_FINGERS',
                            'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE',
                            'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING',
                            'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
            st.warning("⚠️ Using fallback feature names. Check if this matches your model training.")

        # Create encoded dataframe with correct column order
        encoded_input_df = pd.DataFrame(0, index=input_df.index, columns=model_columns)
        
        # Set AGE
        if 'AGE' in model_columns:
            encoded_input_df['AGE'] = input_df['AGE']
        
        # Encode gender
        if 'GENDER_M' in model_columns:
            encoded_input_df['GENDER_M'] = 1 if GENDER == 'M' else 0
        if 'GENDER_F' in model_columns:
            encoded_input_df['GENDER_F'] = 1 if GENDER == 'F' else 0
        
        # Set binary variables
        binary_mapping = {
            'SMOKING': 'SMOKING',
            'YELLOW_FINGERS': 'YELLOW_FINGERS',
            'ANXIETY': 'ANXIETY',
            'PEER_PRESSURE': 'PEER_PRESSURE',
            'CHRONIC_DISEASE': 'CHRONIC_DISEASE',
            'FATIGUE': 'FATIGUE',
            'ALLERGY': 'ALLERGY',
            'WHEEZING': 'WHEEZING',
            'ALCOHOL_CONSUMING': 'ALCOHOL_CONSUMING',
            'COUGHING': 'COUGHING',
            'SHORTNESS_OF_BREATH': 'SHORTNESS_OF_BREATH',
            'SWALLOWING_DIFFICULTY': 'SWALLOWING_DIFFICULTY',
            'CHEST_PAIN': 'CHEST_PAIN'
        }
        
        for input_col, model_col in binary_mapping.items():
            if model_col in model_columns:
                encoded_input_df[model_col] = input_df[input_col]

        if scaler:
            try:
                # Ensure the column order matches exactly what the scaler expects
                encoded_input_df = encoded_input_df[model_columns]
                
                # Scale input
                input_df_scaled = scaler.transform(encoded_input_df)

                # Predict
                prediction = rf_model.predict(input_df_scaled)[0]
                prediction_proba = rf_model.predict_proba(input_df_scaled)[0]
                
                # Display prediction with styling
                risk_class = "HIGH RISK OF LUNG CANCER" if prediction == 1 else "LOW RISK OF LUNG CANCER"
                risk_color = "risk-high" if prediction == 1 else "risk-low"
                
                st.markdown(f'<div class="prediction-box"><h3>Prediction Result:</h3><p class="{risk_color}">{risk_class}</p>', unsafe_allow_html=True)
                
                # Show probability
                st.write(f"Probability of high risk: {prediction_proba[1]:.2%}")
                st.write(f"Probability of low risk: {prediction_proba[0]:.2%}")
                
                # Add disclaimer
                st.markdown("---")
                st.caption("⚠️ Disclaimer: This prediction is for informational purposes only and should not replace professional medical advice. Always consult with healthcare professionals for medical diagnoses.")

            except Exception as e:
                st.error(f"⚠️ Error while scaling or predicting: {e}")
                st.write("Encoded DataFrame columns:", encoded_input_df.columns.tolist())
                if hasattr(scaler, "feature_names_in_"):
                    st.write("Scaler feature names:", scaler.feature_names_in_.tolist())
        else:
            st.error("⚠️ Scaler not loaded. Please check scaler.pkl.")

# ---------------- About Page ----------------
def about_page():
    st.markdown('<h1 class="main-header">📚 ABOUT THE PROJECT</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">LUNG CANCER PREDICTION USING MACHINE LEARNING MODELS</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    This project aims to predict the likelihood of lung cancer based on patient health data 
    using a Random Forest model. The dataset includes risk factors such as smoking habits, 
    medical history, and respiratory symptoms.
    
    The goal is to assist healthcare professionals in identifying individuals 
    at high risk early, supporting preventive care and early diagnosis.
    </div>
    """, unsafe_allow_html=True)
    
    # Add more details about the project
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Project Objectives")
        st.markdown("""
        - Develop an accurate predictive model for lung cancer risk
        - Provide an easy-to-use interface for healthcare workers
        - Support early detection efforts
        - Educate about risk factors for lung cancer
        """)
    
    with col2:
        st.markdown("### 🔧 Technical Details")
        st.markdown("""
        - **Algorithm**: Random Forest Classifier
        - **Preprocessing**: MinMax Scaling
        - **Framework**: Streamlit for the web interface
        - **Language**: Python
        """)
    
    st.markdown("### 📊 Dataset Information")
    st.markdown("""
    The dataset contains information about patients including:
    - Demographic details (age, gender)
    - Lifestyle factors (smoking, alcohol consumption)
    - Medical symptoms (coughing, shortness of breath, etc.)
    - Diagnostic information
    """)
    
    # Add a disclaimer
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This application is intended for educational and informational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment.
    """)

# ---------------- Main Function ----------------
def main():
    # Custom sidebar design
    st.sidebar.markdown("<h1 style='text-align: center; color: #1E90FF;'>🫁 LUNG CANCER PREDICTION</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### 🗂️ NAVIGATION")
    menu_options = ['PREDICTION PAGE', 'DATASET PREVIEW', 'ABOUT THE PROJECT']
    choice = st.sidebar.radio("GO TO", menu_options, label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ INFORMATION")
    st.sidebar.info(
        "This app predicts lung cancer risk based on patient data. "
        "Select a page from the navigation menu to get started."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "For educational purposes only. Not a substitute for professional medical advice."
        "</div>", 
        unsafe_allow_html=True
    )

    if choice == 'PREDICTION PAGE':
        prediction_page()
    elif choice == 'DATASET PREVIEW':
        dataset_preview_page()
    elif choice == 'ABOUT THE PROJECT':
        about_page()

if __name__ == '__main__':
    main()
