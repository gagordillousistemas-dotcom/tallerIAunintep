# Laboratorio Práctico: Fundamentos y Optimización de LLMs

Institución: Unintep  
Asignatura: Inteligencia Artificial Clase 8 (Large Language Models)

## Descripción del Proyecto
Este repositorio contiene el desarrollo práctico de la guía de estudio sobre Grandes Modelos de Lenguaje (LLMs). A través de tres ejercicios en PyTorch, se demuestra el cuello de botella de las redes recurrentes tradicionales y cómo las arquitecturas modernas (Transformers, FlashAttention y LoRA) resuelven el procesamiento en paralelo y la eficiencia de memoria.

## Archivos del Repositorio
laboratorio_pr_ctico_llms.py: Script completo con los tres experimentos juntos.
paso1RNN.py: Prueba de desvanecimiento del gradiente en Redes Neuronales Recurrentes.
paso2Transformer.py: Implementación de atención multicabeza en paralelo.
paso3Eficiencia.py: Pruebas de optimización con FlashAttention y LoRA.

## Requisitos y Ejecución
El entorno se configuró con Python, uv, torch y numpy. Para correr el laboratorio completo desde la terminal:

    uv run python laboratorio_pr_ctico_llms.py

## Resultados Obtenidos

### 1. Desvanecimiento de Memoria en RNN (paso1RNN.py)
Se probó una red nn.RNN con una secuencia de 100 tokens propagando el gradiente hacia atrás desde el último token:
Gradiente en token t=99 (final): 1.48644316
Gradiente en token t=50 (mitad): 0.00000000
Gradiente en token t=0 (inicio): 0.00000000

Análisis: La señal de aprendizaje cae a cero al retroceder en la secuencia, comprobando por qué las RNN pierden el contexto del inicio en textos largos.

### 2. Atención en Paralelo Transformer (paso2Transformer.py)
Se ejecutó una capa MultiHeadAttention (d_model = 768, num_heads = 12) con un lote de 32 secuencias de 512 tokens:
Dimensiones de entrada: torch.Size([32, 512, 768])
Dimensiones de salida: torch.Size([32, 512, 768])

Análisis: El Transformer procesa toda la secuencia al mismo tiempo mediante multiplicación de matrices, eliminando los bucles secuenciales de las RNN.

### 3. Eficiencia Computacional: FlashAttention y LoRA (paso3Eficiencia.py)
FlashAttention: Se ejecutó F.scaled_dot_product_attention para evitar saturar la memoria HBM de la GPU con matrices de atención grandes (1024x1024).
LoRA (rango r=8) sobre una capa lineal de 4096 x 4096:
  Parámetros en Fine-Tuning tradicional: 16,781,312
  Parámetros entrenables con LoRA: 65,536
  Reducción de parámetros: 99.6095%

Análisis: Congelar los pesos originales y entrenar solo matrices de bajo rango permite ajustar modelos grandes ahorrando más del 99% de parámetros actualizables.
