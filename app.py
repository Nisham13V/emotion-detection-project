import os
import streamlit as st
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("Emotion Detection from Text")
st.write("Enter a sentence to detect emotion")

user_input = st.text_area("Enter text")

CONFIDENCE_THRESHOLD = 0.30   # LOWERED (IMPORTANT)

# simple keyword fallback (industry practice)
KEYWORD_MAP = {
    "love": "love",
    "happy": "happy",
    "joy": "happy",
    "sad": "sad",
    "angry": "anger",
    "anger": "anger",
    "scared": "fear",
    "afraid": "fear",
    "fear": "fear",
    "surprised": "surprise",
    "wow": "surprise"
}

if st.button("Predict Emotion"):
    if user_input.strip():
        text_lower = user_input.lower()

        # 1️⃣ Keyword-based quick check
        for word, emotion in KEYWORD_MAP.items():
            if word in text_lower:
                st.success(f"Detected Emotion: **{emotion.upper()}** (keyword match)")
                break
        else:
            # 2️⃣ ML prediction
            probs = model.predict_proba([user_input])[0]
            max_prob = np.max(probs)
            predicted_emotion = model.classes_[np.argmax(probs)]

            if max_prob < CONFIDENCE_THRESHOLD:
                st.warning("Detected Emotion: NEUTRAL / UNKNOWN")
            else:
                st.success(
                    f"Detected Emotion: **{predicted_emotion.upper()}**\n"
                    f"Confidence: {max_prob:.2f}"
                )
    else:
        st.warning("Please enter some text.")
