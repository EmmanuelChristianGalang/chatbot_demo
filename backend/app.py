from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import numpy as np
import pickle
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load NLP tools
nltk.download("punkt")

# Load the trained model
chatbot_model = tf.keras.models.load_model("chatbot_model.h5")


print("✅ Model has been loaded successfully!")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load intents dataset
with open("datasets.json", "r") as f:
    intents = json.load(f)

# Create a mapping of intent tags to their indices
intent_map = {intent["tag"]: i for i, intent in enumerate(intents["intents"])}

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return tokens

def get_response(user_input):
    tokens = preprocess_text(user_input)
    input_data = tokenizer.texts_to_sequences([tokens])

    if not input_data or not input_data[0]:  
        return "I'm not sure how to respond to that."

    # ✅ Fix: Ensure proper padding and shape
    input_data = pad_sequences(input_data, maxlen=20, padding="post")  # 20 should match training maxlen

    prediction = chatbot_model.predict(input_data)
    predicted_class_index = np.argmax(prediction)

    intent_tag = list(intent_map.keys())[predicted_class_index]

    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return np.random.choice(intent["responses"])

    return "I'm not sure how to respond to that."

# Save images from assets folder
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets'))
    return send_from_directory(assets_dir, filename)

# Render the frontend
@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."})
    # Token store will be converted to sequence
    tokens = preprocess_text(user_message)  
    input_data = tokenizer.texts_to_sequences([tokens])  
     # Handle empty sequence
    if not input_data or not input_data[0]: 
        return jsonify({"response": "I'm not sure how to respond to that."})
    # For testing the response (accept any value)
    input_data = np.array(input_data, dtype=np.float32)  

    prediction = chatbot_model.predict(input_data)
    predicted_class_index = np.argmax(prediction)

    intent_tag = list(intent_map.keys())[predicted_class_index]

    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return jsonify({"response": np.random.choice(intent["responses"])})

    return jsonify({"response": "I'm not sure how to respond to that."})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)