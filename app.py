import json
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import logging
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from datetime import datetime

# Configure logging to write logs to a file
logging.basicConfig(filename='chatbot.log', level=logging.DEBUG)

nltk.download('punkt')

app = Flask(__name__, static_folder='static')

# Load the trained model and other necessary files
try:
    model = load_model("chat_model")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
except Exception as e:
    print("Error loading model or pickled files:", str(e))

# Load intents for debugging
with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

# Instantiate NLTK's Porter Stemmer
stemmer = PorterStemmer()

# Instantiate Tokenizer for preprocessing
tokenizer_for_server = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer_for_server.fit_on_texts(["<OOV>"])  # Fit on a dummy text to avoid issues

# Function to process and tokenize text
def preprocess_text(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token.lower()) for token in tokens]
    return ' '.join(stemmed_tokens)

# Function to check for bad words
def check_for_bad_words(text):
    bad_words = ['sex','sexulaity','drug','alcohol','smoke','smoking','ganja','rape','wine','whiskey','gun','adult image','adult video','bullet','hindu','muslim','christian','buddhism','jainism','bomb','grenade','abuse','kill','murder','kiss','kidnap','blood','horror','assault','sniper','ammo','pornography','nudity','nude','horny','porn','kama','kamasutra','kama sutra','intercourse','penis','vagina','vaginal','xxx','flirting','flirt','fuck','bitch','boob','boobs','boobies','sex photo']  # Add your bad words here
    for word in bad_words:
        if word in text:
            return True
    return False

# Function to predict responses
def chatbot_response(text):
    if check_for_bad_words(text):
        return "Sorry, I cannot provide information on this topic as this violates our terms and conditions."

    processed_text = preprocess_text(text)

    # Check if the user is asking for date or time
    if 'date' in processed_text:
        current_date = datetime.now().strftime("%d-%m-%Y")
        return f"The current date is {current_date}"
    elif 'time' in processed_text:
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"The current time is {current_time}"

    seq = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(seq, maxlen=20)
    pred = model.predict(padded)
    predicted_class_index = np.argmax(pred, axis=1)[0]
    predicted_tag = lbl_encoder.inverse_transform([predicted_class_index])[0]

    print("Input Text:", text)
    print("Processed Text:", processed_text)
    print("Predicted Class Index:", predicted_class_index)
    print("Predicted Tag:", predicted_tag)

    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            responses = intent['responses']
            return np.random.choice(responses)

    return "I'm sorry, I didn't understand that."

# Function to tokenize and update the tokenizer used during training
def tokenize_and_update(text):
    processed_text = preprocess_text(text)
    tokenizer_for_server.fit_on_texts([processed_text])

# Define routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        user_message = request.form.get("user_message")
        if not user_message:
            raise ValueError("Invalid user input")

        response = chatbot_response(user_message)
        tokenize_and_update(user_message)  # Update tokenizer during runtime
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error processing user message: {str(e)}")
        return jsonify({"response": "An error occurred."})


if __name__ == "__main__":
    app.run(debug=True)
