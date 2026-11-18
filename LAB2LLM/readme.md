# Taller de Modelos de Lenguaje Locales y en la Nube (Ollama y Gemini)

Este repositorio contiene el desarrollo y las pruebas realizadas para el taller de ejecución e integración de modelos de lenguaje utilizando Ollama en entorno local, la API de Google Gemini en la nube y Streamlit para la interfaz web.

## Contenido y modificaciones realizadas

El proyecto fue inicializado con uv para la gestión de dependencias y contiene los siguientes archivos:
cod1.py
Archivo original del taller que realiza una consulta fija ("¿Quien es Oscar Pulgarin?") mediante la librería de Ollama y muestra 
la respuesta en Streamlit. En este archivo se probó inicialmente el modelo gemma4:latest (6.6 GB), observando tiempos altos de carga
en la memoria del equipo, y posteriormente se modificó el parámetro del modelo a gemma2:2b (1.6 GB), logrando respuestas mucho más 
rápidas y fluidas. Además, se comprobó que al correr sin acceso a internet, el modelo local responde que no tiene información sobre
personas específicas fuera de su entrenamiento.
Este codigo lo modifiqué como de prueba cod1_pregunta.py
Archivo nuevo creado a partir de cod1.py para convertir la consulta estática en una aplicación interactiva. Se  se reemplazó la 
pregunta fija por una caja de texto con st.text_input para escribir cualquier pregunta desde el navegador, se añadió un indicador de 
carga con st.spinner y se configuró un mensaje de rol system para indicar al modelo que responda de forma controlada cuando no 
disponga de información en su base local.

cod2.py
Aplicación comparativa dividida en dos columnas que envía una misma pregunta de forma simultánea a la API de Google GenAI en la nube 
y a Ollama en local. En este código se configuró la lectura segura de la clave GEMINI_API_KEY desde un archivo .env, se actualizó el 
modelo de Google a gemini-3.8-flash y se probaron distintos modelos locales en Ollama, cambiando de gemma4:latest a gemma2:2b y 
posteriormente descargando y probando qwen2.5:1.5b (986 MB).

## Observaciones y conclusiones
Durante las pruebas se evidenciaron diferencias claras en el rendimiento según el tamaño de cada modelo en hardware local. 
Mientras gemma4:latest requiere mayor tiempo y memoria para cargar, modelos ligeros como gemma2:2b y qwen2.5:1.5b ofrecen una interacción ágil
en equipos sin tarjeta gráfica dedicada. 
Asimismo, se verificó que los modelos locales en Ollama funcionan totalmente desconectados de internet, por lo que desconocen datos 
en tiempo real como la fecha actual o información local no incluida en su entrenamiento, a menos que se les proporcione contexto 
explícito desde el código. Por otro lado, la conexión en la nube con Gemini ofrece mayor capacidad sin consumir recursos locales,
aunque depende de la disponibilidad de los servidores y de los límites del plan gratuito de la cuenta utilizada.
