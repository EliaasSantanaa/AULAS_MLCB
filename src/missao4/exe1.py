print("Regressão Linear (Prever Gorjeta) ---")
import numpy as np
from sklearn.linear_model import LinearRegression

# X: Característica (Feature) -> Valor da conta em R$
# Precisamos formatar como uma matriz 2D, por isso o .reshape(-1, 1)
X_contas = np.array([10, 20, 30, 45, 50, 65, 70, 80]).reshape(-1, 1)

# y: Rótulo (Label) -> Valor da gorjeta em R$
y_gorjetas = np.array([1.5, 3.0, 4.0, 6.0, 7.5, 9.0, 10.0, 12.0])

# 1. Criar o modelo de Regressão Linear
modelo_gorjeta = LinearRegression()

# 2. Treinar o modelo com nossos dados
# O .fit() encontra a melhor linha que descreve a relação entre X e y
modelo_gorjeta.fit(X_contas, y_gorjetas)

# 3. Fazer uma previsão para um novo valor
# Qual seria a gorjeta para uma conta de R$ 55?
nova_conta = np.array([[55]]) # Precisa ser um array 2D
gorjeta_prevista = modelo_gorjeta.predict(nova_conta)

print(f"Dados das contas (X):\n{X_contas.flatten()}")
print(f"Dados das gorjetas (y):\n{y_gorjetas}")
print("-" * 20)
print(f"Para uma conta de R$ 55.00, a gorjeta prevista é: R$ {gorjeta_prevista[0]:.2f}")