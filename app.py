import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
import json

# -----------------------------
# Load Model & Tokenizer
# -----------------------------
MODEL_DIR = "saved_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🤖")
st.title("🧠 Toxic Comment Detection")
st.write("This app predicts whether a comment is **Toxic** or **Non-Toxic**.")

# Input text box
user_input = st.text_area("Enter your comment here:", height=100)

# Prediction button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a comment before predicting.")
    else:
        # Tokenize the input
        inputs = tokenizer(
            user_input,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Assume binary classification: toxic vs non-toxic
        toxic_score = probs[0]  # First label score
        label = "🛑 Toxic" if toxic_score >= 0.5 else "✅ Non-Toxic"

        # Display results
        st.subheader("Prediction Result:")
        st.write(f"**Prediction:** {label}")
        st.progress(int(toxic_score * 100))
        st.write(f"**Confidence:** {toxic_score:.2%}")

# -----------------------------
# Upload Test Data (Optional)
# -----------------------------
st.write("---")
st.subheader("📂 Test Multiple Comments")
uploaded_file = st.file_uploader("Upload Excel file (Optional)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    if "comment_text" not in df.columns:
        st.error("The Excel file must contain a column named **'comment_text'**.")
    else:
        st.write("✅ File uploaded successfully!")
        df["Predicted_Label"] = ""

        # Process each comment
        for i, text in enumerate(df["comment_text"].astype(str)):
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            df.loc[i, "Predicted_Label"] = "Toxic" if probs[0] >= 0.5 else "Non-Toxic"

        st.write("### Prediction Results")
        st.dataframe(df)

        # Download the results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
