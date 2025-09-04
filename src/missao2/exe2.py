"""EXEC-02 - Aprendizado Supervisionado (Regressão)
Crie um modelo que prevê o preço de um imóvel com base na sua área (m²)
e no número de quartos. Usem LinearRegression.
"""
# -----------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
import numpy as np

print("--- Exercício 2 -  Missão 2 (Aprendizado Supervisionado) ---")

# Dados: [área_m2, numero_quartos]
# Rótulos: preco_em_milhares_de_reais
X_imoveis = np.array([
    [60, 2], [75, 3], [80, 3],  # Imóveis menores
    [120, 3], [150, 4], [200, 4]  # Imóveis maiores
])

y_precos = np.array([150, 200, 230, 310, 400, 500])

# Crie uma instância do modelo LinearRegression.
modelo_regressao = LinearRegression()

# Treine o modelo com os dados de imóveis (X_imoveis, y_precos).
modelo_regressao.fit(X_imoveis, y_precos)

# Crie um novo imóvel para testar (ex: 100m², 3 quartos).
imovel_teste = np.array([[250, 5]])

# Faça a previsão do preço para o novo imóvel.
preco_previsto = modelo_regressao.predict(imovel_teste)

print(f"Previsão de preço para um imóvel de 250m² com 5 quartos: R$ {preco_previsto[0]:.2f} mil")
print("-" * 50, "\n")