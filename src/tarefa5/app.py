# TAREFA 5 : Agrupar frases de um chatbot de turismo - rode sem erros no VSCODE
# 1. Crie uma lista de frases sobre passagens, hospedagem, passeios, restaurantes
# 2. Vetorize as frases
# 3. Use KMeans com número de clusters à sua escolha
# 4. Imprima a qual cluster cada frase pertence

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np


# 1. Dataset
frases = [
    # Passagens
    "Quero comprar uma passagem para o Rio de Janeiro",
    "Tem promoção de passagens para Salvador?",
    "Como remarcar minha passagem?",
    # Hospedagem
    "Quero reservar hotel em Florianópolis",
    "O hotel tem café da manhã incluso?",
    "Quais hotéis têm piscina?",
    # Passeios
    "Quais passeios estão disponíveis em Foz do Iguaçu?",
    "Tem passeio de barco?",
    "Quero agendar um city tour",
    # Restaurantes
    "Onde encontro restaurantes italianos?",
    "Quais restaurantes aceitam reserva?",
    "Tem restaurante vegano na cidade?"
]

# 2. Vetorização
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(frases)

# 3. Modelo (4 clusters: passagens, hospedagem, passeios, restaurantes)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)

# 4. Saída
print("\nAgrupamento de frases do chatbot de turismo:")
for i, frase in enumerate(frases):
    print(f"'{frase}' => Cluster {kmeans.labels_[i]}")

# TAREFA 6 : # Encontrar Produtos "Âncora" - rode sem erros no VSCODE ou no Colab
# Sua Missão: identificar os 2 produtos que melhor representam suas categorias principais, para colocá-los em destaque na home page. Estes são os produtos "âncora".
# Dica: Os produtos âncora são os centros dos clusters!
# ------------------------------------------------------------------------------
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
