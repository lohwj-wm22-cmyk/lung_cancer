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

def dataset_preview_page():
    st.title('📊 Dataset Preview')
    st.header('LUNG CANCER PREDICTION DATASET')
    
    # Link to the dataset
    dataset_link = 'https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset/data'
    st.write(f'You can download the full dataset from [Kaggle]({dataset_link}).')
    
    # Load a sample dataset for preview
    df = pd.read_csv('lung_data.csv')  # Update this with the path to your sample data file
    st.write('Here is a preview of the dataset:')
    st.dataframe(df.head(20))

def prediction_page():
    st.title('🫁 LUNG CANCER PREDICTION')
    st.write('Fill in the details to predict the patient\'s health outcome.')

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
    if st.button('Predict 🔮'):
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

        # Convert the input to a DataFrame
        input_df = pd.DataFrame(input_data)

        # Define the model columns
        model_columns = ['AGE','GENDER_MALE', 'GENDER_FEMALE', 'SMOKING_YES', 'SMOKING_NO',
                         'YELLOW_FINGERS_YES', 'YELLOW_FINGERS_NO','ANXIETY_YES', 'ANXIETY_NO',
                         'PEER_PRESSURE_YES', 'PEER_PRESSURE_NO','CHRONIC_DISEASE_YES', 'CHRONIC_DISEASE_NO',
                         'FATIGUE_YES', 'FATIGUE_NO','ALLERGY_YES', 'ALLERGY_NO','WHEEZING_YES', 'WHEEZING_NO',
                         'ALCOHOL_CONSUMING_YES', 'ALCOHOL_CONSUMING_NO','COUGHING_YES', 'COUGHING_NO','SHORTNESS_OF_BREATH_YES', 
                         'SHORTNESS_OF_BREATH_NO','SWALLOWING_DIFFICULTY_YES', 'SWALLOWING_DIFFICULTY_NO','CHEST_PAIN_YES', 'CHEST_PAIN_NO']
        
        # Create a DataFrame to hold the encoded features
        encoded_input_df = pd.DataFrame(0, index=input_df.index, columns=model_columns)

        st.write(input_df.columns)

        # Ensure all column names in the dataset are stripped of spaces or match exactly
        input_df.columns = input_df.columns.str.replace(' ', '_')

        # Copy continuous variables
        encoded_input_df[['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 
                  'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 
                  'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']] = input_df[['AGE', 'SMOKING', 
                  'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE','FATIGUE','ALLERGY', 'WHEEZING',
                  'ALCOHOL_CONSUMING','COUGHING','SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY','CHEST_PAIN']]

        # Hardcode categorical mappings : LIMITATIONS 
        categorical_data = {
            'GENDER': {'Male': 'GENDER_MALE', 'Female': 'GENDER_FEMALE'},
            'SMOKING': {'Yes': 'SMOKING_YES', 'No': 'SMOKING_NO'},
            'YELLOW_FINGERS': {'Yes': 'YELLOW_FINGERS_YES', 'No': 'YELLOW_FINGERS_NO'},
            'ANXIETY': {'Yes': 'ANXIETY_YES', 'No': 'ANXIETY_NO'},
            'PEER_PRESSURE': {'Yes': 'PEER_PRESSURE_YES', 'No': 'PEER_PRESSURE_NO'},
            'CHRONIC_DISEASE': {'Yes': 'CHRONIC_DISEASE_YES', 'No': 'CHRONIC_DISEASE_NO'},
            'FATIGUE': {'Yes': 'FATIGUE_YES', 'No': 'FATIGUE_NO'},
            'ALLERGY': {'Yes': 'ALLERGY_YES', 'No': 'ALLERGY_NO'},
            'WHEEZING': {'Yes': 'WHEEZING_YES', 'No': 'WHEEZING_NO'},
            'ALCOHOL_CONSUMING': {'Yes': 'ALCOHOL_CONSUMING_YES', 'No': 'ALCOHOL_CONSUMING_NO'},
            'COUGHING': {'Yes': 'COUGHING_YES', 'No': 'COUGHING_NO'},
            'SHORTNESS_OF_BREATH': {'Yes': 'SHORTNESS_OF_BREATH_YES', 'No': 'SHORTNESS_OF_BREATH_NO'},
            'SWALLOWING_DIFFICULTY': {'Yes': 'SWALLOWING_DIFFICULTY_YES', 'No': 'SWALLOWING_DIFFICULTY_NO'},
            'CHEST_PAIN': {'Yes': 'CHEST_PAIN_YES', 'No': 'CHEST_PAIN_NO'}
        }

        # Populate categorical variables
        for col in categorical_data:
            # Set all columns to 0
            for column in categorical_data[col].values():
                encoded_input_df[column] = 0
            # Set the column for the specific input to 1
            value = input_df[col].iloc[0]
            encoded_input_df[categorical_data[col].get(value, '')] = 1

        # Ensure all columns are present
        encoded_input_df = encoded_input_df.reindex(columns=model_columns, fill_value=0)
        
        # st.write("encoded_input_df")
        # st.write(encoded_input_df)
        # print(encoded_input_df)

        # Check if scaler is fitted
        if scaler:
            # Scale the input data
            input_df_scaled = scaler.transform(encoded_input_df)

            # Predict using the Random Forest model
            rf_prediction = rf_model.predict(input_df_scaled)[0]

            # st.write("input_df_scaled")
            # st.write(input_df_scaled)
            
            # Display the prediction result
            st.success(f'🌟 Random Forest Prediction: {"HIGH RISK OF LUNG CANCER" if rf_prediction == 1 else "LOW RISK OF LUNG CANCER"}')
        else:
            st.error("⚠️ Scaler not loaded properly. Please check the scaler file.")

def about_page():
    st.title('📚 About the Project')
    st.header('LUNG CANCER PREDICTION USING MACHINE LEARNING MODELS')
    st.write("""
    THIS PROJECT AIMS TO PREDICT THE LIKELIHOOD OF LUNG CANCER BASED ON PATIENT HEALTH DATA 
    USING A RANDOM FOREST MODEL. THE DATASET INCLUDES RISK FACTORS SUCH AS SMOKING HABITS, 
    MEDICAL HISTORY, AND RESPIRATORY SYMPTOMS.
    
    THE GOAL IS TO ASSIST HEALTHCARE PROFESSIONALS IN IDENTIFYING INDIVIDUALS 
    AT HIGH RISK EARLY, SUPPORTING PREVENTIVE CARE AND EARLY DIAGNOSIS.
    """)

# Main function with sidebar navigation
def main():
    # Sidebar for navigation
    st.sidebar.title('🗂️ Navigation')
    menu_options = ['Prediction Page', 'Dataset Preview', 'About the Project']
    choice = st.sidebar.selectbox('Go to', menu_options)

    # Navigation based on user selection
    if choice == 'Prediction Page':
        prediction_page()
    elif choice == 'Dataset Preview':
        dataset_preview_page()
    elif choice == 'About the Project':
        about_page()

if __name__ == '__main__':
    main()

