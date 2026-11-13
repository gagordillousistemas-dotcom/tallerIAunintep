import matplotlib.pyplot as plt

epocas = range(1, 11)

# Tus resultados reales de Pérdida de Validación (val_loss)
val_loss_base = [0.1652, 0.1383, 0.1257, 0.1174, 0.1159, 0.1218, 0.1113, 0.1310, 0.1189, 0.1229]
val_loss_exp1 = [0.1502, 0.1273, 0.1324, 0.1391, 0.1473, 0.1212, 0.1294, 0.1228, 0.1451, 0.1535]
val_loss_exp2 = [0.3830, 0.2819, 0.2448, 0.2260, 0.2137, 0.2051, 0.1986, 0.1932, 0.1891, 0.1862]
val_loss_exp3a = [2.6509, 2.4273, 2.4265, 2.4261, 2.4259, 2.4259, 2.4258, 2.4258, 2.4258, 2.4258]
val_loss_exp3b = [1.7046, 1.1244, 0.8016, 0.6310, 0.5299, 0.4645, 0.4196, 0.3871, 0.3628, 0.3437]

plt.figure(figsize=(12, 5))

# Gráfica 1: Comparación de Arquitectura (Base vs Exp 1 vs Exp 2)
plt.subplot(1, 2, 1)
plt.plot(epocas, val_loss_base, marker='o', label='Base (128, 64)')
plt.plot(epocas, val_loss_exp1, marker='s', label='Exp 1: +2 Capas (Overfitting)')
plt.plot(epocas, val_loss_exp2, marker='^', label='Exp 2: 16 y 8 neuronas (Underfitting)')
plt.title('Impacto de Arquitectura en Pérdida (Val Loss)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (Cross-Entropy)')
plt.xticks(epocas)
plt.legend()
plt.grid(True)

# Gráfica 2: Comparación de Tasa de Aprendizaje (Base vs Exp 3A vs Exp 3B)
plt.subplot(1, 2, 2)
plt.plot(epocas, val_loss_base, marker='o', label='Base (η = 0.001)')
plt.plot(epocas, val_loss_exp3a, marker='x', label='Exp 3A: η = 0.5 (Explosiva)')
plt.plot(epocas, val_loss_exp3b, marker='d', label='Exp 3B: η = 0.00001 (Lenta)')
plt.title('Impacto del Learning Rate η en Pérdida (Val Loss)')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (Cross-Entropy)')
plt.xticks(epocas)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("grafica_comparativa_ra2.png", dpi=300)
print("¡Gráfica guardada exitosamente como grafica_comparativa_ra2.png!")
plt.show()
