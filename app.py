
import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
for resource in [
    "stopwords",
    "wordnet",
    "omw-1.4",
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger_eng"
]:
    nltk.download(resource, quiet=True)

# Load saved model and vectorizer
@st.cache_resource
def load_files():
    model = joblib.load("bernoulli_model.pkl")
    vectorizer = joblib.load("binary_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_files()

# Preprocessing
stop_words = set(stopwords.words("english"))
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("nor")

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return "a"
    elif tag.startswith("V"):
        return "v"
    elif tag.startswith("N"):
        return "n"
    elif tag.startswith("R"):
        return "r"
    return "n"

def preprocess_review(text):
    text = str(text).lower()
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        word for word in text.split()
        if word not in stop_words
    ]

    pos_tags = nltk.pos_tag(words)

    words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]

    return " ".join(words)

# Streamlit page
st.set_page_config(
    page_title="Restaurant Review Sentiment Analysis",
    page_icon="🍽️"
)

st.title("🍽️ Restaurant Review Sentiment Analysis")

st.write(
    "Enter a restaurant review below to predict "
    "whether the sentiment is positive or negative."
)

review = st.text_area(
    "Enter your review:",
    placeholder="Example: The food was fantastic!"
)

if st.button("Predict Sentiment"):
    if review.strip():

        cleaned_review = preprocess_review(review)

        review_vector = vectorizer.transform(
            [cleaned_review]
        )

        prediction = model.predict(review_vector)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(review_vector)[0]
            class_index = list(model.classes_).index(prediction)
            confidence = probabilities[class_index] * 100

            st.write(
                f"Prediction confidence: {confidence:.2f}%"
            )

    else:
        st.warning("Please enter a review first.")

