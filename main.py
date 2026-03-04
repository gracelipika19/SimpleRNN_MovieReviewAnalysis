import numpy as np
import tensorflow as tf
import streamlit as st
import re

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Vocabulary size used during training
VOCAB_SIZE = 10000
MAXLEN = 500

# Load word index
word_index = imdb.get_word_index()

# Reverse word index (for decoding if needed)
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Load trained model
model = load_model("simple_rnn_imdb.h5")


# Function to decode reviews (optional)
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


# Preprocess user input text
def preprocess_text(text):

    # Remove punctuation
    text = re.sub(r'[^a-zA-Z ]', '', text)

    # Lowercase
    words = text.lower().split()

    encoded_review = []

    for word in words:
        index = word_index.get(word, 2)

        # Restrict to top 10k words
        if index >= VOCAB_SIZE:
            index = 2

        encoded_review.append(index + 3)

    padded_review = sequence.pad_sequences([encoded_review], maxlen=MAXLEN)

    return padded_review


# Prediction function
def predict_sentiment(review):

    processed_input = preprocess_text(review)

    prediction = model.predict(processed_input)

    sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😞"

    confidence = float(prediction[0][0])

    return sentiment, confidence


# ---------------- STREAMLIT APP ---------------- #

st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write(
"""
This app predicts whether a movie review is **Positive or Negative**
using a **Simple RNN trained on the IMDB dataset**.
"""
)

# User input
user_input = st.text_area("Enter a movie review:")


# Button
if st.button("Classify Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter a movie review first.")

    else:
        sentiment, confidence = predict_sentiment(user_input)

        st.subheader("Prediction Result")

        st.write("**Sentiment:**", sentiment)

        st.write("**Confidence Score:**", round(confidence, 4))

        if sentiment.startswith("Positive"):
            st.success("The model predicts this review is positive.")
        else:
            st.error("The model predicts this review is negative.")


st.write("---")
st.caption("Built with TensorFlow + Streamlit")