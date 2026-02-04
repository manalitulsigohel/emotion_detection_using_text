import streamlit as st
import joblib
import os

# No resource_path needed for Streamlit, it looks in the root folder
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Page Config
st.set_page_config(page_title="Emotion AI Pro", page_icon="🎭")

# UI Design
st.title("🎭 Emotion Analysis Engine")
st.markdown("---")

user_input = st.text_input("Tell me how you feel...", placeholder="I am having a wonderful day!")

# Color and Emoji Map
color_map = {
    "joy": "😊", "sadness": "😢", "anger": "😡",
    "fear": "😨", "love": "❤️", "surprise": "😲"
}

if st.button("Analyze Emotion"):
    if user_input.strip():
        # Predict
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0].lower()

        emoji = color_map.get(prediction, "✨")

        # Display results in a nice box
        st.success(f"### Detected Emotion: {prediction.upper()} {emoji}")
    else:
        st.warning("Please enter some text first!")