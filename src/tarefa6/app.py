# TAREFA 6 : # Encontrar Produtos "Âncora" - rode sem erros no VSCODE ou no Colab
# Sua Missão: identificar os 2 produtos que melhor representam suas categorias principais, para colocá-los em destaque na home page. Estes são os produtos "âncora".
# Dica: Os produtos âncora são os centros dos clusters!
# ------------------------------------------------------------------------------
import numpy as np
from sklearn.cluster import KMeans


print("\n--- Exercício Não Supervisionado ---")

# Dados: [preco_produto, nota_de_popularidade (0-10)]
dados_produtos = np.array([
    [10, 2], [15, 3], [12, 1],   # Categoria 1: Baratos e menos populares
    [200, 9], [180, 8], [210, 10] # Categoria 2: Caros e muito populares
])

# 1. Crie um modelo KMeans para encontrar 2 clusters
modelo_produtos = KMeans(n_clusters=2, random_state=42, n_init=10)

# 2. Treine o modelo com os dados dos produtos
modelo_produtos.fit(dados_produtos)

# 3. Os centros dos clusters são os nossos produtos "âncora" ideais
produtos_ancora = modelo_produtos.cluster_centers_

print(f"Características dos Produtos Âncora (Preço, Popularidade):\n{produtos_ancora}")
