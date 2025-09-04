import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler

# Load the trained models
# lr_model = load('logreg_model.joblib')
rf_model = load('rf_model.joblib')
# svm_model = load('svm_model.joblib')
scaler = load('scaler.pkl') 
# scaler = MinMaxScaler()

# ---------------- Dataset Preview Page ----------------
def dataset_preview_page():
    st.title('📊 DATASET PREVIEW')
    st.header('LUNG CANCER PREDICTION DATASET')

    # Link to dataset
    dataset_link = 'https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer'
    st.write(f'You can download the full dataset from [Kaggle]({dataset_link}).')

    # Load a sample dataset for preview
    df = pd.read_csv('lung_data.csv')  # Update this with your dataset file
    st.write('HERE IS A PREVIEW OF THE DATASET:')
    st.dataframe(df.head(20))

# ---------------- Prediction Page ----------------
def prediction_page():
    st.title('🫁 LUNG CANCER PREDICTION APP')
    st.write('FILL IN THE PATIENT DETAILS TO PREDICT THE RISK OF LUNG CANCER.')

    # Input fields for user data
    AGE = st.number_input('AGE 🎂', min_value=0, max_value=120, value=50)
    GENDER = st.selectbox('GENDER 👤', ['M', 'F'])
    SMOKING = st.selectbox('DO YOU SMOKE? 🚬', ['YES', 'NO'])
    YELLOW_FINGERS = st.selectbox('YELLOW FINGERS ✋', ['YES', 'NO'])
    ANXIETY = st.selectbox('ANXIETY 😟', ['YES', 'NO'])
    PEER_PRESSURE = st.selectbox('PEER PRESSURE 👥', ['YES', 'NO'])
    CHRONIC_DISEASE = st.selectbox('CHRONIC DISEASE 🏥', ['YES', 'NO'])
    FATIGUE = st.selectbox('FATIGUE 😴', ['YES', 'NO'])
    ALLERGY = st.selectbox('ALLERGY 🤧', ['YES', 'NO'])
    WHEEZING = st.selectbox('WHEEZING 😤', ['YES', 'NO'])
    ALCOHOL_CONSUMPTION = st.selectbox('ALCOHOL CONSUMPTION 🍺', ['YES', 'NO'])
    COUGHING = st.selectbox('COUGHING 🤧', ['YES', 'NO'])
    SHORTNESS_OF_BREATH = st.selectbox('SHORTNESS OF BREATH 🫁', ['YES', 'NO'])
    SWALLOWING_DIFFICULTY = st.selectbox('SWALLOWING DIFFICULTY 😣', ['YES', 'NO'])
    CHEST_PAIN = st.selectbox('CHEST PAIN ❤️‍🩹', ['YES', 'NO'])

    # When user clicks Predict button
    if st.button('PREDICT 🔮'):
        # Create a dictionary for the input
        input_data = {
            'AGE': [AGE],
            'GENDER': [GENDER],
            'SMOKING': [SMOKING],
            'YELLOW_FINGERS': [YELLOW_FINGERS],
            'ANXIETY': [ANXIETY],
            'PEER_PRESSURE': [PEER_PRESSURE],
            'CHRONIC_DISEASE': [CHRONIC_DISEASE],
            'FATIGUE': [FATIGUE],
            'ALLERGY': [ALLERGY],
            'WHEEZING': [WHEEZING],
            'ALCOHOL_CONSUMPTION': [ALCOHOL_CONSUMPTION],
            'COUGHING': [COUGHING],
            'SHORTNESS_OF_BREATH': [SHORTNESS_OF_BREATH],
            'SWALLOWING_DIFFICULTY': [SWALLOWING_DIFFICULTY],
            'CHEST_PAIN': [CHEST_PAIN]
        }

        input_df = pd.DataFrame(input_data)

        # Define model columns
        model_columns = [
            'AGE',
            'GENDER_F', 'GENDER_M',
            'SMOKING_YES', 'SMOKING_NO',
            'YELLOW_FINGERS_YES', 'YELLOW_FINGERS_NO',
            'ANXIETY_YES', 'ANXIETY_NO',
            'PEER_PRESSURE_YES', 'PEER_PRESSURE_NO',
            'CHRONIC_DISEASE_YES', 'CHRONIC_DISEASE_NO',
            'FATIGUE_YES', 'FATIGUE_NO',
            'ALLERGY_YES', 'ALLERGY_NO',
            'WHEEZING_YES', 'WHEEZING_NO',
            'ALCOHOL_CONSUMPTION_YES', 'ALCOHOL_CONSUMPTION_NO',
            'COUGHING_YES', 'COUGHING_NO',
            'SHORTNESS_OF_BREATH_YES', 'SHORTNESS_OF_BREATH_NO',
            'SWALLOWING_DIFFICULTY_YES', 'SWALLOWING_DIFFICULTY_NO',
            'CHEST_PAIN_YES', 'CHEST_PAIN_NO'
        ]

        # Create encoded dataframe
        encoded_input_df = pd.DataFrame(0, index=input_df.index, columns=model_columns)
        encoded_input_df['AGE'] = input_df['AGE']

        # Hardcode categorical mappings
        categorical_data = {
            'GENDER': {'M': 'GENDER_M', 'F': 'GENDER_F'},
            'SMOKING': {'YES': 'SMOKING_YES', 'NO': 'SMOKING_NO'},
            'YELLOW_FINGERS': {'YES': 'YELLOW_FINGERS_YES', 'NO': 'YELLOW_FINGERS_NO'},
            'ANXIETY': {'YES': 'ANXIETY_YES', 'NO': 'ANXIETY_NO'},
            'PEER_PRESSURE': {'YES': 'PEER_PRESSURE_YES', 'NO': 'PEER_PRESSURE_NO'},
            'CHRONIC_DISEASE': {'YES': 'CHRONIC_DISEASE_YES', 'NO': 'CHRONIC_DISEASE_NO'},
            'FATIGUE': {'YES': 'FATIGUE_YES', 'NO': 'FATIGUE_NO'},
            'ALLERGY': {'YES': 'ALLERGY_YES', 'NO': 'ALLERGY_NO'},
            'WHEEZING': {'YES': 'WHEEZING_YES', 'NO': 'WHEEZING_NO'},
            'ALCOHOL_CONSUMPTION': {'YES': 'ALCOHOL_CONSUMPTION_YES', 'NO': 'ALCOHOL_CONSUMPTION_NO'},
            'COUGHING': {'YES': 'COUGHING_YES', 'NO': 'COUGHING_NO'},
            'SHORTNESS_OF_BREATH': {'YES': 'SHORTNESS_OF_BREATH_YES', 'NO': 'SHORTNESS_OF_BREATH_NO'},
            'SWALLOWING_DIFFICULTY': {'YES': 'SWALLOWING_DIFFICULTY_YES', 'NO': 'SWALLOWING_DIFFICULTY_NO'},
            'CHEST_PAIN': {'YES': 'CHEST_PAIN_YES', 'NO': 'CHEST_PAIN_NO'}
        }

        # Encode categorical
        for col in categorical_data:
            for column in categorical_data[col].values():
                encoded_input_df[column] = 0
            value = input_df[col].iloc[0]
            encoded_input_df[categorical_data[col][value]] = 1

        # Ensure all columns are present in same order as model
        encoded_input_df = encoded_input_df.reindex(columns=model_columns, fill_value=0)

        if scaler:
            try:
                # Match scaler feature names if available
                if hasattr(scaler, "feature_names_in_"):
                    encoded_input_df = encoded_input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
                    
                st.write("✅ Encoded Input DataFrame:", encoded_input_df)  # Debugging step

                # Scale input
                input_df_scaled = scaler.transform(encoded_input_df)

                # Predict
                prediction = rf_model.predict(input_df_scaled)[0]
                st.success(f'🌟 PREDICTION: {"HIGH RISK OF LUNG CANCER" if prediction == 1 else "LOW RISK OF LUNG CANCER"}')

            except Exception as e:
                st.error(f"⚠️ Error while scaling input: {e}")
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





