import os
import sys
import customtkinter as ctk  # Changed from 'import ctk'
import joblib

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 1. Setup Appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# 2. Load the Brains (Using resource_path for deployment)
# Make sure these files are in the same folder as this script!
model = joblib.load(resource_path('emotion_model.pkl'))
vectorizer = joblib.load(resource_path('vectorizer.pkl'))

# 3. Dynamic Styles & Emojis
color_map = {
    "joy": ("#FFD700", "😊"), "sadness": ("#3498db", "😢"),
    "anger": ("#e74c3c", "😡"), "fear": ("#9b59b6", "😨"),
    "love": ("#ff69b4", "❤️"), "surprise": ("#2ecc71", "😲")
}

def analyze():
    text = entry.get()
    if text.strip():
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0].lower()
        color, emoji = color_map.get(prediction, ("#abb2b9", "✨"))

        result_label.configure(text=f"{prediction.upper()} {emoji}", text_color=color)
        analyze_btn.configure(fg_color=color)

def clear():
    entry.delete(0, 'end')
    result_label.configure(text="Waiting for input...", text_color="white")
    analyze_btn.configure(fg_color="#1f538d")

# 4. Build Modern UI
app = ctk.CTk()
app.title("Emotion AI Pro")
app.geometry("600x550")

title = ctk.CTkLabel(app, text="Emotion Analysis Engine", font=("Roboto", 24, "bold"))
title.pack(pady=20)

entry = ctk.CTkEntry(app, placeholder_text="Tell me how you feel...", width=400, height=45)
entry.pack(pady=10)

button_frame = ctk.CTkFrame(app, fg_color="transparent")
button_frame.pack(pady=10)

analyze_btn = ctk.CTkButton(button_frame, text="Analyze", command=analyze, width=150)
analyze_btn.grid(row=0, column=0, padx=10)

clear_btn = ctk.CTkButton(button_frame, text="Clear", command=clear, fg_color="#34495e", width=150)
clear_btn.grid(row=0, column=1, padx=10)

result_label = ctk.CTkLabel(app, text="Waiting for input...", font=("Roboto", 22, "bold"))
result_label.pack(pady=20)

app.mainloop()