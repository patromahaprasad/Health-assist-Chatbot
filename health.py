import streamlit as st
import nltk
import tensorflow as tf
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load AI model for text generation
chatbot = pipeline("text-generation", model="distilgpt2")

# Function to preprocess user input
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(filtered_tokens)

# Function to handle user queries
def healthcare_chatbot(user_input):
    user_input = preprocess_text(user_input)
    
    if "symptom" in user_input:
        return "Please consult a doctor for accurate medical advice."
    elif "appointment" in user_input:
        return "Would you like to schedule an appointment with a doctor?"
    elif "medication" in user_input:
        return "Always take prescribed medications. If you have concerns, consult your doctor."
    else:
        response = chatbot(user_input, max_length=100, num_return_sequences=1)
        return response[0]['generated_text']

# Streamlit UI
def main():
    st.title("🩺 AI-Powered Healthcare Assistant")
    st.write("Chat with our AI health assistant for quick advice!")
    
    user_input = st.text_input("How can I assist you today?")
    
    if st.button("Submit"):
        if user_input:
            st.write("🧑‍⚕️ **User:**", user_input)
            with st.spinner("Processing your query, please wait..."):
                response = healthcare_chatbot(user_input)
            st.success("🤖 **Assistant:** " + response)
        else:
            st.warning("Please enter a message to proceed.")

if __name__ == "__main__":
    main()
