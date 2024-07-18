import nltk
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import gdown
import os
import zipfile
from collections import defaultdict

LABEL_INT_MAPPING = {
    'BACKGROUND': 0,
    'OBJECTIVE': 1,
    'METHODS': 2,
    'RESULTS': 3,
    'CONCLUSIONS': 4,
}

INT_LABEL_MAPPING = {value: key for key, value in LABEL_INT_MAPPING.items()}

def download_and_extract_model(model_url, destination_path='poseidon_lstm'):
    if not os.path.exists(destination_path):
        print("Downloading model...")
        gdown.download(model_url, 'model.zip', quiet=False)
        print("Extracting model...")
        with zipfile.ZipFile('model.zip', 'r') as zip_ref:
            zip_ref.extractall()
        print("Model is ready.")
        os.remove('model.zip')

def sentences(abstract):
    nltk.download('punkt')
    return nltk.sent_tokenize(abstract)

def load_tf_model(path):
    print("Loading model...")
    model = keras.models.load_model(path)
    print('Model loaded.')
    return model


def predict(sentences, model):
    processed_sentences = []
    processed_chars = []
    positions = []

    for i, sentence in enumerate(sentences):
        # For sentences
        processed_sentence = tf.constant([sentence])  # Model expects batch dimension
        processed_sentences.append(processed_sentence)

        # For characters
        sentence_chars = " ".join(list(sentence))
        processed_chars.append(tf.constant([sentence_chars]))  # Model expects batch dimension

        # For positional encoding
        position_encoding = tf.constant([[i, len(sentences)]], dtype=tf.float32)
        positions.append(position_encoding)

    # Predict classes for each sentence
    predictions = []
    for sent, chars, pos in zip(processed_sentences, processed_chars, positions):
        pred = model.predict([sent, chars, pos])
        predicted_class = np.argmax(pred, axis=1)
        predictions.append(predicted_class.item())

    predictions = [INT_LABEL_MAPPING[label] for label in predictions]

    return predictions


def structure_sentences(sentences, predictions):
    # Group sentences by their predicted labels
    grouped_sentences = defaultdict(list)
    for sentence, prediction in zip(sentences, predictions):
        grouped_sentences[prediction].append(sentence)

    # Construct the structured text
    structured_text = ""
    for label, sentences in grouped_sentences.items():
        structured_text += f"{label}:\n" + " ".join(sentences) + "\n\n"

    return structured_text.strip()
