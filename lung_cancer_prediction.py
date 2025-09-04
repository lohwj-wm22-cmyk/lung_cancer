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
    GENDER = st.selectbox('Gender 👤', ['M', 'F'])
    AGE = st.number_input('Age 🎂', min_value=0, max_value=120, value=25)
    SMOKING = st.selectbox('DO YOU SMOKE? 🚬', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    YELLOW_FINGERS = st.selectbox('YELLOW FINGERS ✋', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    ANXIETY = st.selectbox('ANXIETY 😟', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    PEER_PRESSURE = st.selectbox('PEER PRESSURE 👥', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    CHRONIC_DISEASE = st.selectbox('CHRONIC DISEASE 🏥', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    FATIGUE = st.selectbox('FATIGUE 😴', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    ALLERGY = st.selectbox('ALLERGY 🤧', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    WHEEZING = st.selectbox('WHEEZING 😤', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    ALCOHOL_CONSUMPTION = st.selectbox('ALCOHOL CONSUMPTION 🍺', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    COUGHING = st.selectbox('COUGHING 🤧', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    SHORTNESS_OF_BREATH = st.selectbox('SHORTNESS OF BREATH 🫁', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    SWALLOWING_DIFFICULTY = st.selectbox('SWALLOWING DIFFICULTY 😣', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    CHEST_PAIN = st.selectbox('CHEST PAIN ❤️‍🩹', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

    # When user clicks Predict button
    if st.button('PREDICT 🔮'):
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
            'ALCOHOL_CONSUMPTION': [ALCOHOL_CONSUMPTION],
            'COUGHING': [COUGHING],
            'SHORTNESS_OF_BREATH': [SHORTNESS_OF_BREATH],
            'SWALLOWING_DIFFICULTY': [SWALLOWING_DIFFICULTY],
            'CHEST_PAIN': [CHEST_PAIN]
        }

        input_df = pd.DataFrame(input_data)

        # Define model columns
        model_columns = ['AGE','GENDER_M', 'GENDER_F', 'SMOKING_YES', 'SMOKING_NO',
                         'YELLOW_FINGERS_YES', 'YELLOW_FINGERS_NO','ANXIETY_YES', 'ANXIETY_NO',
                         'PEER_PRESSURE_YES', 'PEER_PRESSURE_NO','CHRONIC_DISEASE_YES', 'CHRONIC_DISEASE_NO',
                         'FATIGUE_YES', 'FATIGUE_NO','ALLERGY_YES', 'ALLERGY_NO','WHEEZING_YES', 'WHEEZING_NO',
                         'ALCOHOL_CONSUMPTION_YES', 'ALCOHOL_CONSUMPTION_NO','COUGHING_YES', 'COUGHING_NO',
                         'SHORTNESS_OF_BREATH_YES', 'SHORTNESS_OF_BREATH_NO',
                         'SWALLOWING_DIFFICULTY_YES', 'SWALLOWING_DIFFICULTY_NO',
                         'CHEST_PAIN_YES', 'CHEST_PAIN_NO']

        # Create encoded dataframe
        encoded_input_df = pd.DataFrame(0, index=input_df.index, columns=model_columns)
        encoded_input_df['AGE'] = input_df['AGE']

        # Helper function to convert 0/1 to 'No'/'Yes'
        def get_category_value(value):
            return 'YES' if value == 1 else 'NO'

        # Encode categorical variables
        # Gender encoding
        if GENDER == 'M':
            encoded_input_df['GENDER_M'] = 1
        else:
            encoded_input_df['GENDER_F'] = 1

        # Encode binary variables
        binary_vars = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                      'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 
                      'ALCOHOL_CONSUMPTION', 'COUGHING', 'SHORTNESS_OF_BREATH', 
                      'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']

        for var in binary_vars:
            value = input_df[var].iloc[0]
            yes_col = f"{var}_YES"
            no_col = f"{var}_NO"
            
            if yes_col in encoded_input_df.columns:
                encoded_input_df[yes_col] = 1 if value == 1 else 0
            if no_col in encoded_input_df.columns:
                encoded_input_df[no_col] = 1 if value == 0 else 0

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
                st.write("Encoded DataFrame columns:", encoded_input_df.columns.tolist())
                if hasattr(scaler, "feature_names_in_"):
                    st.write("Scaler feature names:", scaler.feature_names_in_.tolist())
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


