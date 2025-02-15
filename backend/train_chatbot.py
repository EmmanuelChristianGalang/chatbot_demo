import json
import nltk
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, Bidirectional, LSTM

# Download necessary NLP data
nltk.download("punkt")

# Load dataset
with open("datasets.json") as f:
    intents = json.load(f)

# Prepare data
sentences, labels, classes = [], [], []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])
    
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Tokenization
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding="post")

# Encode labels
label_map = {label: idx for idx, label in enumerate(classes)}
label_sequences = np.array([label_map[label] for label in labels])

# Save tokenizer and label map
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)

# Build Enhanced ANN Model
model = Sequential([
    Embedding(10000, 256, input_length=padded_sequences.shape[1]),
    Bidirectional(LSTM(128, return_sequences=True)),
    GlobalAveragePooling1D(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(len(classes), activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(padded_sequences, label_sequences, epochs=100, verbose=1)

# Save model
model.save("chatbot_model.h5")

print("✅ Training complete! Model and tokenizer saved.")
