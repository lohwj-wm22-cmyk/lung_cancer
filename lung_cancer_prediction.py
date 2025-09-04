import streamlit as st
import pandas as pd
from joblib import load

# Load the trained models
rf_model = load('rf_model.joblib')
scaler = load('scaler.pkl')  # Make sure the scaler is saved properly

# ---------------- Dataset Preview Page ----------------
def dataset_preview_page():
    st.title('📊 Dataset Preview')
    st.header('Lung Cancer Prediction Dataset')
    
    # Link to the dataset
    dataset_link = 'https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer'
    st.write(f'You can download the full dataset from [Kaggle]({dataset_link}).')
    
    # Load a sample dataset for preview
    df = pd.read_csv('lung_data.csv')  # Make sure the path is correct for your dataset
    st.write('Here is a preview of the dataset:')
    st.dataframe(df.head(20))

# ---------------- Prediction Page ----------------
def prediction_page():
    st.title('🩺 Patient Health Prediction App')
    st.write('Fill in the details to predict the patient\'s health outcome.')

    # Input fields for user data
    gender = st.selectbox('Gender 👤', ['Male', 'Female'])
    age = st.number_input('Age 🎂', min_value=0, max_value=120, value=25)
    smoking = st.selectbox('Do you smoke? 🚬', ['Yes', 'No'])
    yellow_fingers = st.selectbox('Yellow Fingers ✋', ['Yes', 'No'])
    anxiety = st.selectbox('Anxiety 😟', ['Yes', 'No'])
    peer_pressure = st.selectbox('Peer Pressure 👥', ['Yes', 'No'])
    chronic_disease = st.selectbox('Chronic Disease 🏥', ['Yes', 'No'])
    fatigue = st.selectbox('Fatigue 😴', ['Yes', 'No'])
    allergy = st.selectbox('Allergy 🤧', ['Yes', 'No'])
    wheezing = st.selectbox('Wheezing 😤', ['Yes', 'No'])
    alcohol_consuming = st.selectbox('Alcohol Consumption 🍺', ['Yes', 'No'])
    coughing = st.selectbox('Coughing 🤧', ['Yes', 'No'])
    shortness_of_breath = st.selectbox('Shortness of Breath 🫁', ['Yes', 'No'])
    swallowing_difficulty = st.selectbox('Swallowing Difficulty 😣', ['Yes', 'No'])
    chest_pain = st.selectbox('Chest Pain ❤️‍🩹', ['Yes', 'No'])

    # When user clicks Predict button
    if st.button('Predict 🔮'):
        # Create a dictionary for the input
        input_data = {
            'gender': [gender],
            'age': [age],
            'smoking': [smoking],
            'yellow_fingers': [yellow_fingers],
            'anxiety': [anxiety],
            'peer_pressure': [peer_pressure],
            'chronic_disease': [chronic_disease],
            'fatigue': [fatigue],
            'allergy': [allergy],
            'wheezing': [wheezing],
            'alcohol_consuming': [alcohol_consuming],
            'coughing': [coughing],
            'shortness_of_breath': [shortness_of_breath],
            'swallowing_difficulty': [swallowing_difficulty],
            'chest_pain': [chest_pain]
        }

        # Convert the input to a DataFrame
        input_df = pd.DataFrame(input_data)

        # Define the model columns (update as needed)
        model_columns = ['age', 'smoking', 'yellow_fingers', 'anxiety', 'peer_pressure', 'chronic_disease',
                         'fatigue', 'allergy', 'wheezing', 'alcohol_consuming', 'coughing', 'shortness_of_breath',
                         'swallowing_difficulty', 'chest_pain', 'gender_Male', 'gender_Female']

        # Create a DataFrame to hold the encoded features
        encoded_input_df = pd.DataFrame(0, index=input_df.index, columns=model_columns)

        # Encode categorical data
        encoded_input_df['age'] = input_df['age']
        encoded_input_df['smoking'] = input_df['smoking'].map({'Yes': 1, 'No': 0})
        encoded_input_df['yellow_fingers'] = input_df['yellow_fingers'].map({'Yes': 1, 'No': 0})
        encoded_input_df['anxiety'] = input_df['anxiety'].map({'Yes': 1, 'No': 0})
        encoded_input_df['peer_pressure'] = input_df['peer_pressure'].map({'Yes': 1, 'No': 0})
        encoded_input_df['chronic_disease'] = input_df['chronic_disease'].map({'Yes': 1, 'No': 0})
        encoded_input_df['fatigue'] = input_df['fatigue'].map({'Yes': 1, 'No': 0})
        encoded_input_df['allergy'] = input_df['allergy'].map({'Yes': 1, 'No': 0})
        encoded_input_df['wheezing'] = input_df['wheezing'].map({'Yes': 1, 'No': 0})
        encoded_input_df['alcohol_consuming'] = input_df['alcohol_consuming'].map({'Yes': 1, 'No': 0})
        encoded_input_df['coughing'] = input_df['coughing'].map({'Yes': 1, 'No': 0})
        encoded_input_df['shortness_of_breath'] = input_df['shortness_of_breath'].map({'Yes': 1, 'No': 0})
        encoded_input_df['swallowing_difficulty'] = input_df['swallowing_difficulty'].map({'Yes': 1, 'No': 0})
        encoded_input_df['chest_pain'] = input_df['chest_pain'].map({'Yes': 1, 'No': 0})

        # Encode gender
        encoded_input_df['gender_Male'] = input_df['gender'].map({'Male': 1, 'Female': 0})
        encoded_input_df['gender_Female'] = input_df['gender'].map({'Female': 1, 'Male': 0})

        # Ensure all columns are present in the same order as the model
        encoded_input_df = encoded_input_df.reindex(columns=model_columns, fill_value=0)

        # Scale the input data
        if scaler:
            input_df_scaled = scaler.transform(encoded_input_df)

            # Predict using the Random Forest model
            rf_prediction = rf_model.predict(input_df_scaled)[0]

            # Display the prediction result
            st.success(f'🌟 Lung Cancer Prediction: {"At risk of lung cancer" if rf_prediction == 1 else "Not at risk of lung cancer"}')
        else:
            st.error("⚠️ Scaler not loaded properly. Please check the scaler file.")

def about_page():
    st.title('📚 About the Project')
    st.header('Lung Cancer Prediction using Machine Learning Models')
    st.write("""
    This project aims to predict the risk of lung cancer based on patient data using a Random Forest model. 
    The dataset includes features such as smoking habits, age, medical history (hypertension, anxiety), 
    lifestyle factors (yellow fingers, fatigue), and others that help in predicting the likelihood of lung cancer.
    
    The model is trained using a lung cancer prediction dataset, and the goal is to assist healthcare professionals 
    in identifying high-risk individuals early on.
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
