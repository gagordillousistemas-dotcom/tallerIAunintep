from google import genai
import os
from dotenv import load_dotenv
import streamlit as st
from ollama import chat
from ollama import ChatResponse


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

gemini_models = "gemini-3.8-flash"
ollama_models = "qwen2.5:1.5b"
st.title("Gemini API with Streamlit and Ollama")

st.session_state.pregunta = st.text_input("Escribe tu pregunta aquí:")

if st.session_state.pregunta is not None and st.session_state.pregunta != "":

    st.write(f"Tu pregunta es: {st.session_state.pregunta}")
    col1, col2 = st.columns(2)

    with col1:
        st.header(f"Modelo con API de Google y el modelo {gemini_models}")

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=st.session_state.pregunta,
        )

        st.write(f"La respuesta es: {response.text}")

    with col2:
        st.header(f"Modelo con Ollama y el modelo {ollama_models}")

        response: ChatResponse = chat(model='qwen2.5:1.5b', messages=[
          {
            'role': 'user',
            'content': st.session_state.pregunta
          },
        ])
        
        st.write(response['message']['content'])









