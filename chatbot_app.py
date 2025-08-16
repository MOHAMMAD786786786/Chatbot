import streamlit as st
import requests
import os

# Read Gemini API key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")

def chat(message):
    url = "https://api.gemini.com/v1/engines/text-chat-001/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"prompt": message, "max_tokens": 100}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["choices"][0]["text"]

st.title("Chatbot")
st.write("Chat with your AI!")
user_input = st.text_input("You:")
if user_input:
    response = chat(user_input)
    st.text_area("Bot:", value=response, height=200)
