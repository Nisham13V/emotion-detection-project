import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "emotions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "emotion_model.pkl")

data = pd.read_csv(DATA_PATH)

X = data["text"]
y = data["emotion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
joblib.dump(model, MODEL_PATH)

print("Model trained successfully")
