from sklearn.neighbors import KNeighborsClassifier
import numpy as np

print("\n- KNN (Classificar Frutas) ---")

# X: [peso em gramas, textura (0=lisa, 1=cascuda)]
X_frutas = np.array([
    [150, 0], [170, 0], [180, 0], # Maçãs
    [130, 1], [120, 1], [140, 1]  # Laranjas
])
# y: 0=Laranja, 1=Maçã
y_frutas = np.array([1, 1, 1, 0, 0, 0])

# 1. Criar o modelo KNN
# n_neighbors=3 significa que ele vai consultar os 3 vizinhos mais próximos.
modelo_frutas = KNeighborsClassifier(n_neighbors=3)

# 2. Treinar o modelo
modelo_frutas.fit(X_frutas, y_frutas)

# 3. Fazer uma previsão para uma nova fruta: 160g e textura lisa (0)
fruta_nova = np.array([[160, 0]])
previsao = modelo_frutas.predict(fruta_nova)

# Traduzindo a previsão numérica para texto
resultado = "Maçã" if previsao[0] == 1 else "Laranja"
print(f"Dados das Frutas:\n{X_frutas}")
print(f"Rótulos: {y_frutas}")
print("-" * 20)
print(f"Uma fruta de [160g, lisa] foi classificada como: {resultado}")