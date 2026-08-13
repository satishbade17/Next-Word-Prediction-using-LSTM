# ================================
# Import Libraries
# ================================

import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ================================
# Page Configuration
# ================================

st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="✍️",
    layout="centered"
)


# ================================
# Load Model
# ================================

@st.cache_resource
def load_lstm_model():

    model = load_model(
        "lstm_wikitext2_model.h5"
    )

    return model


model = load_lstm_model()


# ================================
# Load Tokenizer
# ================================

@st.cache_resource
def load_tokenizer():

    with open(
        "tokenizer.pkl",
        "rb"
    ) as f:

        tokenizer = pickle.load(f)

    return tokenizer


tokenizer = load_tokenizer()


# ================================
# Load Max Length
# ================================

@st.cache_resource
def load_max_len():

    with open(
        "max_len.pkl",
        "rb"
    ) as f:

        max_len = pickle.load(f)

    return max_len


max_len = load_max_len()


# ================================
# Vocabulary
# ================================

vocab_size = 10000


# ================================
# Create Index to Word Dictionary
# ================================

index_to_word = {}

for word, index in tokenizer.word_index.items():

    if index < vocab_size:

        index_to_word[index] = word


# ================================
# Predictor Function
# ================================
def predictor(text, top_n=5):

    text = text.lower()

    seq = tokenizer.texts_to_sequences([text])[0]

    if len(seq) == 0:
        return []

    seq = seq[-max_len:]

    seq = pad_sequences(
        [seq],
        maxlen=max_len,
        padding="pre"
    )

    prediction = model.predict(
        seq,
        verbose=0
    )[0]

    top_indices = np.argsort(prediction)[::-1]

    words = []

    for index in top_indices:

        word = index_to_word.get(index, "")

        # Remove unwanted tokens
        if word in [
            "<OOV>",
            "unk",
            "<unk>",
            "–",
            "-",
            "--"
        ]:
            continue

        if word:
            words.append(word)

        if len(words) == top_n:
            break

    return words


# ================================
# Streamlit UI
# ================================

st.title(
    "✍️ Next Word Prediction using LSTM"
)

st.write(
    "Enter a sentence and the model "
    "will predict the most likely next words."
)


# ================================
# Text Input
# ================================

text = st.text_input(
    "✍️ Enter text:",
    placeholder="Type something like: what is"
)


# ================================
# Number of Suggestions
# ================================

top_n = st.slider(
    "Number of suggestions",
    min_value=1,
    max_value=10,
    value=5
)


# ================================
# Prediction
# ================================

if text:

    predictions = predictor(
        text,
        top_n=top_n
    )

    st.subheader(
        "Next Word Suggestions"
    )

    if predictions:

        for i, word in enumerate(
            predictions,
            1
        ):

            st.write(
                f"**{i}. {word}**"
            )

    else:

        st.warning(
            "No prediction available. "
            "Try another sentence."
        )


# ================================
# Footer
# ================================

st.divider()

st.caption(
    "LSTM Next Word Prediction | WikiText-2 Dataset"
)