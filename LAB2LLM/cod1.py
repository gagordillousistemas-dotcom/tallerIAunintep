from ollama import chat
from ollama import ChatResponse
import streamlit as st

st.title("Chat con Ollama - Clase 9")

pregunta = "¿Quien  es Oscar Pulgarin?"
st.write(f"Pregunta: {pregunta}")

# gemma4:latest / gemma2:2b

response: ChatResponse = chat(model='gemma2:2b', messages=[
  {
    'role': 'user',
    'content': pregunta,
  },
])

respuesta_final = response['message']['content']


st.write(respuesta_final)
