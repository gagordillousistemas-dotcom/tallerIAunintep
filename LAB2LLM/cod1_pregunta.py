from ollama import chat
from ollama import ChatResponse
import streamlit as st

st.title("Chat con Ollama - Clase 9 profe Pulgarin")

pregunta = st.text_input("Escribe tu pregunta aquí:")

if pregunta:
    st.write(f"**Tu pregunta:** {pregunta}")

    with st.spinner("Pensando..."):
        response: ChatResponse = chat(model='gemma2:2b', messages=[
            {
                'role': 'system',
                'content': 'Eres un asistente útil y amable. Si no tienes información segura o no sabes la respuesta a lo que te preguntan, responde exactamente: "No tengo información sobre ese tema en mi base de datos local, pero puedo ayudarte con otra consulta."'
            },
            {
                'role': 'user',
                'content': pregunta,
            },
        ])

    respuesta_final = response['message']['content']
    st.write(respuesta_final)