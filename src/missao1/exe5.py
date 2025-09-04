# TAREFA 5 : Agrupar frases de um chatbot de turismo - rode sem erros no VSCODE
# 1. Crie uma lista de frases sobre passagens, hospedagem, passeios, restaurantes
# 2. Vetorize as frases
# 3. Use KMeans com número de clusters à sua escolha
# 4. Imprima a qual cluster cada frase pertence

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans


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
