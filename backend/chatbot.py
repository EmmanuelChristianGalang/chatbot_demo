from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import json
import nltk
from nltk.tokenize import word_tokenize
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download tokenizer data
nltk.download("punkt")

# Load trained chatbot model
model = load_model("chatbot_model.h5")

# Load tokenizer and label mappings
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_map.pkl", "rb") as f:
    intent_map = pickle.load(f)

with open("datasets.json", "r") as f:
    intents = json.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

def preprocess_text(text):
    """Tokenize and preprocess user input."""
    tokens = word_tokenize(text.lower())
    return tokens

def get_response(user_input):
    """Generate response based on user input."""
    tokens = preprocess_text(user_input)
    input_data = tokenizer.texts_to_sequences([tokens])

    if len(input_data[0]) == 0:
        return "I'm not sure how to respond to that."

    # Ensure correct input shape
    input_data = pad_sequences(input_data, maxlen=20, padding="post")
    
    prediction = model.predict(input_data)
    predicted_class_index = np.argmax(prediction)

    # Get intent tag
    intent_tag = list(intent_map.keys())[predicted_class_index]

    # Find response from dataset
    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            response = np.random.choice(intent["responses"])
            return response

    return "I'm not sure how to respond to that."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    tokens = preprocess_text(user_message)  
    input_data = tokenizer.texts_to_sequences([tokens])  

    if not input_data or not input_data[0]:  
        return jsonify({"response": "I'm not sure how to respond to that."})

    # Apply padding to match training input shape
    input_data = pad_sequences(input_data, maxlen=20, padding="post")  

 
    prediction = model.predict(input_data)
    predicted_class_index = np.argmax(prediction)

    intent_tag = list(intent_map.keys())[predicted_class_index]

    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return jsonify({"response": np.random.choice(intent["responses"])})

    return jsonify({"response": "I'm not sure how to respond to that."})



if __name__ == "__main__":
    app.run(debug=True)