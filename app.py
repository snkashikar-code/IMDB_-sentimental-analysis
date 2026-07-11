import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
# Download NLTK resources if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    filtered_and_lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_and_lemmatized_tokens)

# Load the saved TF-IDF vectorizer
try:
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
except FileNotFoundError:
    st.error("Error: tfidf_vectorizer.pkl not found. Please ensure the vectorizer is saved in the same directory.")
    st.stop()

# Load the saved Logistic Regression model
try:
    with open('best_sentiment_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: best_sentiment_model.pkl not found. Please ensure the model is saved in the same directory.")
    st.stop()

# Streamlit App Title
st.title("Sentiment Analysis App")
st.write("Enter a movie review below to predict its sentiment (Positive/Negative).")

# Text input from user
user_input = st.text_area("Enter your review here:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        processed_input = clean_and_lemmatize_text(user_input)

        # Transform the preprocessed text using the loaded vectorizer
        input_vectorized = vectorizer.transform([processed_input])

        # Predict sentiment
        prediction = model.predict(input_vectorized)
        prediction_proba = model.predict_proba(input_vectorized)

        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        positive_proba = prediction_proba[0][1] * 100
        negative_proba = prediction_proba[0][0] * 100

        st.success(f"Predicted Sentiment: **{sentiment}**")
        st.write(f"Confidence (Positive): {positive_proba:.2f}%")
        st.write(f"Confidence (Negative): {negative_proba:.2f}%")
    else:
        st.warning("Please enter some text to analyze.")
