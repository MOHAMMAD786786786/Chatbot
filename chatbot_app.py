
import streamlit as st
import requests
import os

# Read Gemini API key from Streamlit secrets
API_KEY = os.getenv("GEMINI_API_KEY")

def chat(message):
    url = "https://api.gemini.com/v1/engines/text-chat-001/completions"  # adjust if needed
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"prompt": message, "max_tokens": 100}

    try:
        response = requests.post(url, headers=headers, json=data)
    except Exception as e:
        return f"Request failed: {e}"

    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    res_json = response.json()
    if "choices" in res_json and len(res_json["choices"]) > 0:
        return res_json["choices"][0].get("text", "No text returned")
    else:
        return f"Unexpected response format: {res_json}"

# Streamlit UI
st.title("Chatbot")
st.write("Chat with your AI!")
user_input = st.text_input("You:")

if user_input:
    response = chat(user_input)
    st.text_area("Bot:", value=response, height=200)
