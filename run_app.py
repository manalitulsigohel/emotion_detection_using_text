import joblib

# Load the saved "brains"
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

print("🚀 Emotion AI Loaded and Ready!")

while True:
    text = input("\nHow are you feeling? (or 'quit'): ")
    if text.lower() == 'quit':
        break

    # Predict
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]

    print(f"The AI thinks you feel: {prediction.upper()}")