import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch
import re

# ==========================
# STREAMLIT APP CONFIG
# ==========================
st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")
st.title("🧠 Toxic Comment Classifier")
st.markdown("Predict toxic comments using a pre-trained **LSTM** or **BERT** model.")

# ==========================
# CLEANING FUNCTION
# ==========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ==========================
# MODEL LOADING
# ==========================
MODEL_TYPE = st.sidebar.radio("Select Model", ["LSTM", "BERT"])
SAVE_DIR = "saved_model"
LABELS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

st.sidebar.success("✅ Pre-trained model selected. Ready for prediction!")

if MODEL_TYPE == "LSTM":
    st.info("Using LSTM model for inference")
    model = tf.keras.models.load_model(os.path.join(SAVE_DIR, "lstm_model.h5"))
    with open(os.path.join(SAVE_DIR, "tokenizer.json")) as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

elif MODEL_TYPE == "BERT":
    st.info("Using BERT model for inference")
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR)
    model.eval()

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_lstm(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=150, padding="post")
    preds = model.predict(pad)[0]
    return preds

def predict_bert(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**enc).logits
        preds = torch.sigmoid(logits).numpy()[0]
    return preds

# ==========================
# USER INPUT SECTION
# ==========================
user_input = st.text_area("Enter a comment to classify")

if st.button("Predict") and user_input:
    cleaned_text = clean_text(user_input)
    if MODEL_TYPE == "LSTM":
        preds = predict_lstm(cleaned_text)
    else:
        preds = predict_bert(cleaned_text)

    results_df = pd.DataFrame({"Label": LABELS, "Probability": preds})
    st.subheader("🔹 Prediction Results")
    st.dataframe(results_df)
