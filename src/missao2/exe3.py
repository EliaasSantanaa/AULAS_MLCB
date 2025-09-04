
# Aprendizado Não Supervisionado (Clusterização)
# -----------------------------------------------------------------------------

import numpy as np
from sklearn.cluster import KMeans

print("--- Exercício 3 -  Missão 2 (Aprendizado Supervisionado) ---")

# Dados: [valor_gasto_medio, frequencia_visitas_mensal]
# Não temos rótulos!
clientes = np.array([
    [20, 2], [25, 1], [22, 3], # Grupo de baixo valor/frequência
    [400, 7], [420, 8], [410, 6] # Grupo de alto valor/frequência
])

# Criamos o modelo KMeans e pedimos para ele encontrar 2 clusters. Parametrize da forma que seja melhor (ver outros exemplos)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

# '.fit_predict()' treina o modelo e já retorna o cluster de cada cliente.
clusters_encontrados = kmeans.fit_predict(clientes)

print(f"Dados dos clientes (sem rótulos):\n{clientes}")
print(f"Clusters encontrados pelo KMeans para cada cliente: {clusters_encontrados}")
print("Observe como o algoritmo separou corretamente os clientes nos grupos 0 e 1.")
print("-" * 50, "\n")