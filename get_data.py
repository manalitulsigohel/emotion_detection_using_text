import pandas as pd
from datasets import load_dataset

print("Downloading emotion data... please wait a moment.")

# We removed trust_remote_code because this dataset is now in a safer format
dataset = load_dataset("dair-ai/emotion", revision="main")

# Convert the 'train' part of the data to a Pandas table
df = pd.DataFrame(dataset['train'])

# Map the numbers to actual emotion words
label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
df['emotion'] = df['label'].map(label_map)

# Save only the columns we need
df[['text', 'emotion']].to_csv("emotion_detection.csv", index=False)

print(f"✅ Success! Created 'emotion_detection.csv' with {len(df)} rows.")
print("Now you can go back and run your 'emotion_detection.py' file!")