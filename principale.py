from PIL import Image
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import json

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Error: Unable to access the URL {url}")
        return None
    # Check if the response headers indicate JSON content
    if 'application/json' not in r.headers['Content-Type']:
        print(f"Error: URL did not return JSON data")
        return None
    try:
        json_data = r.json()
    except ValueError as ve:
        print(f"Error: Unable to decode the response as JSON. Original exception: {ve}")
        return None
    return json_data
def espace():
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")





def principale():
    

    # Load the image
    '''image = Image.open('image1.png')'''
    
    with open('home.json') as f:
        lottie_json = json.load(f)

    # Create two columns
    col1, col2 = st.columns(2)

    # Display image in the first column
    with col1:
        st_lottie(lottie_json,height=500)

    # Display text in the second column
    with col2:
        espace()
        st.title("Your gateway to AI-powered, efficient, and bias-free CV classification, streamlining your talent acquisition process!")
    