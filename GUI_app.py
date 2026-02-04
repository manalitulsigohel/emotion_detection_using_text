import tkinter as tk
from tkinter import messagebox
import joblib
import streamlit as st

# 1. Load the saved model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Create a dictionary for emojis
emoji_map = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}


def predict_emotion():
    text = entry.get()
    if not text.strip():
        messagebox.showwarning("Warning", "Please type something first!")
        return

    # Predict logic
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0].lower()  # match dictionary keys

    # Get the emoji from our map, or use a default if not found
    emoji = emoji_map.get(prediction, "✨")

    # Update the label with the word AND the emoji
    result_label.config(text=f"{prediction.upper()} {emoji}", fg="#2c3e50")



# 3. Build the Window
window = tk.Tk()
window.title("Emotion Detector AI")
window.geometry("400x300")
window.config(padx=20, pady=20)

# Add Widgets
header = tk.Label(window, text="AI Emotion Analysis", font=("Arial", 18, "bold"))
header.pack(pady=10)

instruction = tk.Label(window, text="Type how you feel below:")
instruction.pack()

entry = tk.Entry(window, font=("Arial", 12), width=30)
entry.pack(pady=10)

btn = tk.Button(window, text="Analyze Emotion", command=predict_emotion,
                bg="#2ecc71", fg="white", font=("Arial", 10, "bold"))
btn.pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 14, "bold"))
result_label.pack(pady=20)

window.mainloop()