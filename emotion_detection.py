import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("emotion_detection.csv")

X = data["text"]
y = data["emotion"]

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

print(X_vectorized.shape)

from sklearn.model_selection import train_test_split

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, data["emotion"], test_size=0.2, random_state=42
)

print("✅ Data split into training and testing sets")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

print("✅ Model trained successfully")

from sklearn.metrics import accuracy_score, f1_score

# Predict emotions on test data
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"✅ Accuracy: {accuracy}")
print(f"✅ F1-score: {f1}")

import joblib

# Save the model and the vectorizer (you need both!)
joblib.dump(model, 'emotion_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model and Vectorizer saved successfully!")


print("\n" + "=" * 30)
print("LIVE EMOTION DETECTOR READY")
print("=" * 30)

while True:
    user_text = input("\nType a sentence (or 'exit' to stop): ")

    if user_text.lower() == 'exit':
        print("Goodbye!")
        break

    # Transform the user's text into the format the model understands
    text_vector = vectorizer.transform([user_text])

    # Make the prediction
    prediction = model.predict(text_vector)[0]

    print(f"Detected Emotion: ✨ {prediction.upper()} ✨")