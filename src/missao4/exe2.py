import numpy as np

# Exercício: Prever Pontuação em Jogo
# Sua Tarefa: Crie um modelo que prevê a pontuação final de um jogador com base no número de horas que ele jogou.

# EXERCÍCIO - PREVER PONTUAÇÃO EM JOGO
print("\n--- 1.2: Exercício - Prever Pontuação ---")
# X: Horas jogadas
horas_jogadas = np.array([1, 3, 5, 8, 10]).reshape(-1, 1)
# y: Pontuação final (em milhares)
pontuacao_final = np.array([10, 25, 60, 90, 110])

# TODO: Crie uma instância do modelo LinearRegression.
from sklearn.linear_model import LinearRegression
modelo_pontuacao = LinearRegression()

# TODO: Treine o modelo com os dados de horas jogadas e pontuação.
modelo_pontuacao.fit(horas_jogadas, pontuacao_final)

# TODO: Preveja a pontuação para um jogador que jogou por 7 horas.
horas_novas = np.array([[7]])
pontuacao_prevista = modelo_pontuacao.predict(horas_novas)

print(f"Dados das horas jogadas (X):\n{horas_jogadas.flatten()}")
print(f"Dados das pontuações finais (y):\n{pontuacao_final}")
print("-" * 20)
print(f"Para um jogador que jogou por 7 horas, a pontuação prevista é: {pontuacao_prevista[0]:.2f} mil pontos")