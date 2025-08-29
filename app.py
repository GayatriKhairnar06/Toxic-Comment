import streamlit as st
import pickle, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load model, tokenizer, and labels
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("saved_models/bilstm_toxicity.h5")
    with open("saved_models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("saved_models/labels.json") as f:
        labels = json.load(f)
    return model, tokenizer, labels

model, tokenizer, labels = load_artifacts()

# -------------------------------
# Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")

st.title("💬 Toxic Comment Classification")
st.write("Enter a comment below and the model will classify it into multiple toxicity categories.")

# Text input
user_input = st.text_area("📝 Enter your comment here:")

# Prediction button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Preprocess input
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=128, padding="post", truncating="post")

        # Predict
        probs = model.predict(padded)[0]
        preds = (probs >= 0.5).astype(int)

        # Display results
        st.subheader("🔎 Prediction Results")
        results = {label: float(probs[i]) for i, label in enumerate(labels)}

        for label, score in results.items():
            st.write(f"**{label}**: {'✅ Toxic' if score >= 0.5 else '❌ Not Toxic'} (prob={score:.2f})")
