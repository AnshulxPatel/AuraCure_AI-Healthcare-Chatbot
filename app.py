
#AuraCure
# Healthcare Chatbot using Streamlit, TensorFlow, and NLTK
# This chatbot helps answer basic healthcare-related questions.

import streamlit as st
import nltk
from transformers import pipeline
from nltk.tokenize import word_tokenize

# Download necessary data for tokenization
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Load the AI text generation model
chatbot = pipeline("text-generation", model="distilgpt2")

# Function to generate chatbot responses
def healthcare_chatbot(user_input):
    user_input = user_input.lower()  # Normalize input
    
    # Basic rule-based responses
    if "symptom" in user_input:
        return "Please consult a doctor for accurate advice."
    elif "appointment" in user_input:
        return "Would you like to schedule an appointment with a doctor?"
    elif "medication" in user_input:
        return "It is important to take prescribed medicines regularly. If you have concerns, consult your doctor."
    
    # AI-generated response for other queries
    response = chatbot(user_input, max_length=150, num_return_sequences=1)
    return response[0]['generated_text']

# Streamlit UI
def main():
    # Set page title and layout
    st.set_page_config(page_title="Healthcare Chatbot", page_icon="🤖", layout="wide")

    # Header
    st.markdown("<h1 style='text-align: center; color: white;'>AI Healthcare Assistant</h1>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("About the Chatbot")
        st.write("This chatbot provides basic healthcare guidance.")
        st.write("Ask about symptoms, appointments, or medications.")
        st.write("For other queries, AI will generate a response.")
        
        # Button to clear chat
        if st.button("Clear Chat"):
            st.experimental_rerun()

    # Chatbot Introduction
    st.info("Type your healthcare-related question below and get assistance.")

    # User input field
    user_input = st.text_input("How can I assist you today?", "")

    # Process user input
    if st.button("Submit", use_container_width=True):
        if user_input.strip():
            with st.spinner("Processing your query..."):
                response = healthcare_chatbot(user_input)
            st.success(f"Healthcare Assistant: {response}")
        else:
            st.warning("Please enter a question to receive a response.")

# Run the chatbot app
if __name__ == "__main__":
    main()
