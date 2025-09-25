# Árvores de Decisão
# Exercício: Aprovar Empréstimo
# Sua Tarefa: Crie uma árvore de decisão para aprovar (1) ou negar (0) um empréstimo com base na renda anual (em milhares) e se a pessoa possui casa própria (1=Sim, 0=Não).

import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
# EXERCÍCIO - APROVAR EMPRÉSTIMO
print("\n--- 3.2: Exercício - Aprovar Empréstimo ---")
# X: [renda_anual_milhar, casa_propria]
dados_credito = np.array([[50, 1], [30, 0], [80, 1], [40, 0], [120, 1], [70, 0]])
# y: 0=Negado, 1=Aprovado
decisao_credito = np.array([1, 0, 1, 0, 1, 1])


# Crie e treine um modelo DecisionTreeClassifier.
modelo_credito = DecisionTreeClassifier()
modelo_credito.fit(dados_credito, decisao_credito)


# Preveja a decisão para alguém com renda de 90 mil e casa própria.
novo_cliente = np.array([[90, 1]])
previsao_credito = modelo_credito.predict(novo_cliente)


resultado_credito = "Aprovado" if previsao_credito[0] == 1 else "Negado"
print(f"Decisão para o cliente [R$90k, Casa Própria]: {resultado_credito}")


# (Opcional): Plote a árvore de decisão para este modelo.
plt.figure(figsize=(8,5))
plot_tree(modelo_credito, feature_names=["Renda (mil)", "Casa Própria"], class_names=["Negado", "Aprovado"], filled=True)
plt.title("Árvore de Decisão - Aprovação de Empréstimo")
plt.show()