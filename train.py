import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import nltk

# Download WordNet resource
nltk.download('wordnet')

with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Encode training labels
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

# Encode responses
response_encoder = LabelEncoder()
response_encoder.fit(np.concatenate(responses))

# Create a dictionary to map encoded responses to original responses
response_mapping = {i: response for i, response in enumerate(response_encoder.classes_)}

# Data Augmentation
lemmatizer = nltk.stem.WordNetLemmatizer()

def augment_data(sentence):
    words = sentence.split()
    augmented_words = [word if np.random.rand() < 0.2 else lemmatizer.lemmatize(word, pos='v') for word in words]
    return ' '.join(augmented_words)

# Apply data augmentation to training sentences
augmented_sentences = [augment_data(sentence) for sentence in training_sentences]
training_sentences += augmented_sentences
training_labels = np.concatenate([training_labels]*2)  # Double the labels due to augmentation

# Tokenization and Padding
vocab_size = 1000
embedding_dim = 64  # Increased embedding dimension
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    padded_sequences, training_labels, test_size=0.1, random_state=42
)

# Model Architecture
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))  # Increased complexity
model.add(Dropout(0.5))  # Added dropout for regularization
model.add(Dense(64, activation='relu'))
model.add(Dense(len(labels), activation='softmax'))

# Model Compilation
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

# Model Training
epochs = 550
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

# Saving model
model.save("chat_model")

# Saving tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Saving label encoder for intent tags
with open('label_encoder.pickle', 'wb') as lbl_file:
    pickle.dump(lbl_encoder, lbl_file, protocol=pickle.HIGHEST_PROTOCOL)

# Saving label encoder for responses
with open('response_encoder.pickle', 'wb') as resp_file:
    pickle.dump(response_encoder, resp_file, protocol=pickle.HIGHEST_PROTOCOL)

# Saving response mapping
with open('response_mapping.json', 'w') as resp_map_file:
    json.dump(response_mapping, resp_map_file)
