import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def ejemplo_desvanecimiento_rnn():
    """
    Capítulo 1: Demuestra cómo las Redes Neuronales Recurrentes (RNN) pierden
    la memoria de los primeros tokens en secuencias largas (Vanishing Gradient).
    """
    print("="*60)
    print("EJEMPLO 1: DESVANECIMIENTO DE MEMORIA EN RNN")
    print("="*60)
    
    seq_len = 100 # Secuencia larga para evidenciar el cuello de botella
    input_size = 10
    hidden_size = 20
    
    # Inicializamos una RNN estándar
    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    
    # Creamos un tensor de entrada y le exigimos gradientes para rastrear el aprendizaje
    x = torch.randn(1, seq_len, input_size, requires_grad=True) 
    
    # Forward pass
    out, h_n = rnn(x)
    
    # Simulamos una tarea donde SOLO importa el último token (ej. predecir el sentimiento de todo el texto)
    # Calculamos una pérdida ficticia sumando los valores del último estado oculto
    loss = out[0, -1].sum() 
    
    # Backward pass: propagamos el error hacia atrás en el tiempo (BPTT)
    loss.backward()
    
    # Analizamos la magnitud del gradiente que logró llegar a los primeros y últimos tokens
    grad_norm_inicio = x.grad[0, 0].norm().item()
    grad_norm_mitad = x.grad[0, seq_len//2].norm().item()
    grad_norm_fin = x.grad[0, -1].norm().item()
    
    print(f"Magnitud de actualización (gradiente) en token t=0:  {grad_norm_inicio:.8f}")
    print(f"Magnitud de actualización (gradiente) en token t=50: {grad_norm_mitad:.8f}")
    print(f"Magnitud de actualización (gradiente) en token t=99: {grad_norm_fin:.8f}")
    print("\nConclusión: El gradiente en el paso t=0 es minúsculo en comparación con t=99.")
    print("Esto demuestra matemáticamente por qué las RNN 'olvidan' el inicio de oraciones largas.\n")

if __name__ == '__main__':
    ejemplo_desvanecimiento_rnn()