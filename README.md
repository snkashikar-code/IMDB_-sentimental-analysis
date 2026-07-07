%%writefile README.md
# Sentiment Analysis App

This project implements a sentiment analysis model to classify movie reviews as either positive or negative. The model is deployed as a simple web application using Streamlit.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Details](#model-details)
- [Setup Instructions](#setup-instructions)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Deployment to Streamlit Cloud](#deployment-to-streamlit-cloud)

## Project Overview

The goal of this project is to build and deploy a sentiment analysis tool. It takes a movie review as input, preprocesses the text, and then uses a trained machine learning model to predict whether the sentiment expressed in the review is positive or negative.

## Model Details

- **Algorithm**: Logistic Regression (chosen for its superior F1-Score of 0.8869 compared to other models).
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer, limited to the top 5000 features.
- **Preprocessing**: The text data undergoes several cleaning steps:
    - HTML tag removal
    - Punctuation and special character removal
    - Lowercasing
    - Stop word removal
    - Lemmatization (reducing words to their base form)
- **Performance**: The model achieved an accuracy of approximately 88.45% on the test dataset.

## Setup Instructions

To run this project locally, follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone <your-github-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data**:
    The Streamlit app requires specific NLTK data. These will be downloaded automatically when the app runs for the first time if not present.

## Running the Streamlit App

Once the setup is complete, you can run the Streamlit application:

```bash
streamlit run app.py
```

This will open the application in your default web browser, usually at `http://localhost:8501`.

## Deployment to Streamlit Cloud

For easy online deployment, you can use [Streamlit Cloud](https://streamlit.io/cloud):

1.  Ensure all necessary files (`app.py`, `best_sentiment_model.pkl`, `tfidf_vectorizer.pkl`, `requirements.txt`) are committed to your GitHub repository.
2.  Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account.
3.  Click 'New app' and select your repository and the `app.py` file.
4.  Streamlit Cloud will handle the deployment, and your app will be live shortly.
