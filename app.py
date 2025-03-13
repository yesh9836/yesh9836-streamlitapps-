import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the model
model = joblib.load('model.pkl')

# Define features and their levels
levels = {
    "Level 1": [
        "ag+1:629e", "feeling.nervous", "trouble.in.concentration",
        "having.trouble.in.sleeping", "social.media.addiction",
        "having.nightmares", "change.in.eating", "feeling.tired"
    ],
    "Level 2": [
        "sweating", "breathing.rapidly", "anger", "close.friend",
        "introvert", "feeling.negative", "avoids.people.or.activities",
        "blamming.yourself"
    ],
    "Level 3": [
        "hallucinations", "panic", "hopelessness",
        "suicidal.thought", "popping.up.stressful.memory"
    ]
}

# Apply custom background
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            color: white;
        }
        .stApp {
            background: linear-gradient(to right, #ff7e5f, #feb47b);
        }
        .stExpander {
            background-color: rgba(255, 255, 255, 0.2) !important;
            border-radius: 10px;
            padding: 10px;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            background-color: white !important;
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Mental Health Assessment Chatbot")

responses = {}

# Loop through levels and features within them
for level, features in levels.items():
    with st.expander(level):
        st.subheader(level)
        for feature in features:
            if feature == "ag+1:629e":
                name = st.text_input("What is your name?")
                age = st.number_input("How old are you?", min_value=0, max_value=120, step=1)
                responses["ag+1:629e"] = age
                continue
            question = f"Do you experience {feature.replace('.', ' ')}?"
            user_response = st.selectbox(
                question,
                options=["Select an option", "Yes", "No"],
                key=feature
            )
            responses[feature] = 1 if user_response == "Yes" else (0 if user_response == "No" else None)

if st.button("Submit"):
    if None in responses.values():
        st.warning("Please answer all the questions before submitting.")
    else:
        st.write(f"Thank you, {name}.")

        # Prediction
        input_data = pd.DataFrame([responses])
        prediction = model.predict(input_data)
        loaded_mapping = pd.read_csv('label_encoder_mappings.csv')
        
        for i in responses.keys():
            if responses[i] == 1:
                responses[i] = "yes"
            elif responses[i] == 0:
                responses[i] = "no"

        # Recreate the LabelEncoder
        encoder = LabelEncoder()
        encoder.classes_ = loaded_mapping['label'].values
        st.write(f"Your responses are: {responses}")
        st.write(f"The model predicts: {encoder.inverse_transform([prediction[0]])}")
