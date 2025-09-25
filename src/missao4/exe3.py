# Exercício para Alunos: Prever Temperatura
# Sua Tarefa: Você tem dados da temperatura em uma cidade com base na altitude (em metros). Crie um modelo para prever a temperatura a 2500 metros.
print("\n--- 1.3: Exercício - Prever Temperatura ---")

# Importações necessárias
import numpy as np
from sklearn.linear_model import LinearRegression

# X: Altitude em metros
altitudes = np.array([500, 1000, 1500, 2000, 3000]).reshape(-1, 1)
# y: Temperatura em Celsius
temperaturas = np.array([25, 20, 15, 10, 5])

# Crie e treine um modelo de Regressão Linear com estes dados.
modelo_temp = LinearRegression()
modelo_temp.fit(altitudes, temperaturas)

# Faça a previsão da temperatura para uma altitude de 2500 metros.
altitude_nova = np.array([[2500]])
temp_prevista = modelo_temp.predict(altitude_nova)
print(f"Temperatura prevista para 2500 metros: {temp_prevista[0]:.2f} °C")