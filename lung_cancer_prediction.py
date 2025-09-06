import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler

# Load the trained models
rf_model = load('rf_model.joblib')
scaler = load('scaler.pkl') 

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E88E5;
        margin-bottom: 30px;
    }
    .section-header {
        color: #0D47A1;
        border-bottom: 2px solid #64B5F6;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .stRadio > div {
        flex-direction: row;
        gap: 20px;
    }
    .stRadio > label {
        font-weight: 500;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
        border-left: 5px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Dataset Preview Page ----------------
def dataset_preview_page():
    st.title('📊 DATASET PREVIEW')
    st.header('LUNG CANCER PREDICTION DATASET')

    # Link to dataset
    dataset_link = 'https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer'
    st.write(f'You can download the full dataset from [Kaggle]({dataset_link}).')

    try:
        # Load a sample dataset for preview
        df = pd.read_csv('lung_data.csv')
        st.write('HERE IS A PREVIEW OF THE DATASET:')
        st.dataframe(df.head(20))
    except FileNotFoundError:
        st.error("❌ File 'lung_data.csv' not found. Please check the file path.")

# ---------------- Prediction Page ----------------
def prediction_page():
    st.markdown('<h1 class="main-title">🫁 LUNG CANCER PREDICTION APP</h1>', unsafe_allow_html=True)
    st.write('FILL IN THE PATIENT DETAILS TO PREDICT THE RISK OF LUNG CANCER.')
    
    # Personal Information Section
    st.markdown('<h2 class="section-header">Personal Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        GENDER = st.radio('Gender', ['M', 'F'], horizontal=True)
    with col2:
        AGE = st.slider('Age', min_value=0, max_value=120, value=45)
    
    st.markdown("---")
    
    # Symptoms Section
    st.markdown('<h2 class="section-header">Symptoms</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        ANXIETY = st.radio('ANXIETY', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        CHRONIC_DISEASE = st.radio('CHRONIC DISEASE', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        FATIGUE = st.radio('FATIGUE', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        ALLERGY = st.radio('ALLERGY', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        WHEEZING = st.radio('WHEEZING', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
    
    with col2:
        COUGHING = st.radio('COUGHING', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        SHORTNESS_OF_BREATH = st.radio('SHORTNESS OF BREATH', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        SWALLOWING_DIFFICULTY = st.radio('SWALLOWING DIFFICULTY', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        CHEST_PAIN = st.radio('CHEST PAIN', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
        YELLOW_FINGERS = st.radio('YELLOW FINGERS', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
    
    st.markdown("---")
    
    # Lifestyle Factors Section
    st.markdown('<h2 class="section-header">Lifestyle Factors</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        SMOKING = st.radio('DO YOU SMOKE?', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
    with col2:
        ALCOHOL_CONSUMING = st.radio('ALCOHOL CONSUMING', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)
    
    PEER_PRESSURE = st.radio('PEER PRESSURE', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes', horizontal=True)

    # When user clicks Predict button
    if st.button('PREDICT RISK', use_container_width=True):
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
                
                # Display prediction with styling
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                if prediction == 1:
                    st.error('🛑 HIGH RISK OF LUNG CANCER')
                    st.write("This prediction suggests a higher likelihood of lung cancer. Please consult with a healthcare professional for further evaluation.")
                else:
                    st.success('✅ LOW RISK OF LUNG CANCER')
                    st.write("This prediction suggests a lower likelihood of lung cancer. However, regular check-ups are still recommended for maintaining good health.")
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Error while scaling or predicting: {e}")
                st.write("Encoded DataFrame columns:", encoded_input_df.columns.tolist())
                if hasattr(scaler, "feature_names_in_"):
                    st.write("Scaler feature names:", scaler.feature_names_in_.tolist())
                # Show the actual values being sent
                st.write("Data being sent:", encoded_input_df.values)
        else:
            st.error("⚠️ Scaler not loaded. Please check scaler.pkl.")

# ---------------- About Page ----------------
def about_page():
    st.title('📚 ABOUT THE PROJECT')
    st.header('LUNG CANCER PREDICTION USING MACHINE LEARNING MODELS')
    st.write("""
    THIS PROJECT AIMS TO PREDICT THE LIKELIHOOD OF LUNG CANCER BASED ON PATIENT HEALTH DATA 
    USING A RANDOM FOREST MODEL. THE DATASET INCLUDES RISK FACTORS SUCH AS SMOKING HABITS, 
    MEDICAL HISTORY, AND RESPIRATORY SYMPTOMS.

    THE GOAL IS TO ASSIST HEALTHCARE PROFESSIONALS IN IDENTIFYING INDIVIDUALS 
    AT HIGH RISK EARLY, SUPPORTING PREVENTIVE CARE AND EARLY DIAGNOSIS.
    """)

# ---------------- Main Function ----------------
def main():
    st.sidebar.title('🗂️ NAVIGATION')
    menu_options = ['PREDICTION PAGE', 'DATASET PREVIEW', 'ABOUT THE PROJECT']
    choice = st.sidebar.selectbox('GO TO', menu_options)

    if choice == 'PREDICTION PAGE':
        prediction_page()
    elif choice == 'DATASET PREVIEW':
        dataset_preview_page()
    elif choice == 'ABOUT THE PROJECT':
        about_page()

if __name__ == '__main__':
    main()
