
# EXEMPLO  ÁRVORES DE DECISÃO
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np

print("\n--- 3.1: Exemplo - Árvore de Decisão (Jogar ou não?) ---")

# X: [Tempo (0=Sol, 1=Nublado, 2=Chuva), Umidade (0=Normal, 1=Alta)]
X_clima = np.array([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [1, 1]])
# y: 0=Não Joga, 1=Joga
y_decisao = np.array([1, 0, 1, 1, 0, 0])

# 1. Criar e treinar o modelo
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_clima, y_decisao)

# 2. Fazer uma previsão: Tempo=Sol (0), Umidade=Normal (0)
previsao_clima = modelo_arvore.predict(np.array([[0, 0]]))
resultado_clima = "Joga" if previsao_clima[0] == 1 else "Não Joga"
print(f"Para um dia de Sol e Umidade Normal, a decisão é: {resultado_clima}")

# 3. Visualizar a árvore (a parte mais legal!)
plt.figure(figsize=(8, 6))
plot_tree(modelo_arvore, feature_names=['Tempo', 'Umidade'], class_names=['Não Joga', 'Joga'], filled=True)
plt.title("Árvore de Decisão para Jogar")
plt.show()
