from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
import re

model = load_model("imdb_model.h5")
word_index = imdb.get_word_index()

def preprocess_txt(txt, max_features=10000):
    words = re.sub(r'[^\w\s]', '', txt.lower()).split()

    encoded_review = []

    for word in words:
        idx = word_index.get(word)
        if idx is not None and (idx + 3) < max_features:
            encoded_review.append(idx + 3)
        else:
            encoded_review.append(2)  

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=500,
        padding='pre'
    )

    return padded_review

def predict_sentiment(review):
    preprocessed_review = preprocess_txt(review)
    prediction = model.predict(preprocessed_review)

    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment, prediction[0][0]


import streamlit as st

st.title("IMDB Review Analysis")
st.write("Enter a movie review o classify it as postive or negative")

ip_review = st.text_area("Enter Movie a review")

if st.button("Submit"):
    sent, pred_score = predict_sentiment(ip_review)
    st.write("Sentiment:", sent)
    st.write("Score:", pred_score * 100)
else:
    st.write("Enter a movie review to proceed")
