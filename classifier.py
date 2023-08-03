import pandas as pd
import numpy as np
import re
import string
import nltk
import base64
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import streamlit as st
import PyPDF2
from streamlit_lottie import st_lottie
import json

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub('http\S+\s*', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the tokens back to form the preprocessed text
    preprocessed_text = ' '.join(words)

    return preprocessed_text

# Function to read the text from a PDF file
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ''
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

# Function to display the PDF in Streamlit app
def show_pdf(file):
    # Read the contents of the file
    pdf_bytes = file.read()

    # Encode the PDF content as a base64 string
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

    # Generate the HTML code to display the PDF
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Display the PDF in the app
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to classify a given text
def classify_text(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Load the TF-IDF vectorizer and model
    tfidf_vectorizer = joblib.load('Models/tfidf_vectorizer.pkl')
    model = joblib.load('Models/model.pkl')

    # Vectorize the preprocessed text
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text]).toarray()

    # Make predictions using the model
    prediction = model.predict(text_tfidf)

    # Load the label encoder to map the numerical prediction back to the original label
    label_encoder = joblib.load('Models/label_encoder.pkl')
    predicted_category = label_encoder.inverse_transform(prediction)[0]

    return predicted_category

def footer():
    # Add page footer
    footer_html = """
        <style>
            .footer {
                font-size: 0.7em;
                text-align: center;
                padding: 1em;
            }
        </style>
        <div class="footer">
            <p>Made by Nour Chaker Ons Ghariani Cyrine Ben Messasoud</p>
        </div>
    """
    st.write("___")
    st.markdown(footer_html, unsafe_allow_html=True)

def classifier1():
    # Set up the Streamlit app
    st.title('CV Classifier')
    # Allow the user to upload a file
    cv_file = st.file_uploader('Upload your CV (PDF only)', type='pdf')
    # Once the user has uploaded a file and clicks the "Classify" button, read the text and classify it
    if cv_file is not None:

        if st.button('Classify'):
            text = read_pdf(cv_file)
            st.write('Classifying...')
            predicted_category = classify_text(text)
            st.write(f'The CV is classified as: {predicted_category}')
        else:
            st.write('')

        if st.button('Show CV'):
            st.write('Showing candidate''s CV...')
            show_pdf(cv_file)
        else:
            st.write('')
    footer()
def classifier():
    with open('classifier.json') as f:
        lottie_json = json.load(f)

    # Create two columns
    col1, col2 = st.columns(2)

    # Display image in the first column
    with col1:
        st_lottie(lottie_json,height=500)
    with col2:
        classifier1()

# Load the data, train the classifier, and start the Streamlit app
if __name__ == '__main__':
    # ... (load the data, train the classifier, and save the model from the previous code)

    # Start the Streamlit app
    classifier()
