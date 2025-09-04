"""EXEC-04 - Aprendizado Não Supervisionado
Você é um analista de segurança. Sua missão é identificar transações fraudulentas
(anomalias) em um conjunto de dados. Uma transação anômala geralmente está muito
distante das outras. Use KMeans para agrupar as transações. A transação que ficar
isolada em seu próprio cluster é provavelmente a anomalia.
"""

import numpy as np
from sklearn.cluster import KMeans

print("--- Exercício 4 -  Missão 2 (Aprendizado Não Supervisionado) ---")

# Dados: [valor_transacao, hora_do_dia (0-23)]
transacoes = np.array([
    [15.50, 14], [30.00, 10], [12.75, 11],
    [50.20, 19], [25.00, 9],
    [200.00, 3]  # Uma transação muito alta e de madrugada -> suspeita
])

# Crie um modelo KMeans para encontrar 2 grupos.
# A ideia é que as transações normais fiquem em um grupo e a anômala fique sozinha no outro.
modelo_anomalia = KMeans(n_clusters=2, random_state=42)

# Treine e preveja os clusters para os dados de transações.
clusters_transacoes = modelo_anomalia.fit_predict(transacoes)

print(f"Clusters para as transações: {clusters_transacoes}")

# Identifica o cluster com menor número de elementos (possível anomalia)
unique, counts = np.unique(clusters_transacoes, return_counts=True)
cluster_counts = dict(zip(unique, counts))
anomaly_cluster = unique[np.argmin(counts)]
anomaly_indices = np.where(clusters_transacoes == anomaly_cluster)[0]

print(f"Contagem por cluster: {cluster_counts}")
print(f"Cluster de anomalia (menos elementos): {anomaly_cluster}")

if len(anomaly_indices) == 1:
    idx = int(anomaly_indices[0])
    valor, hora = transacoes[idx]
    print(f"Transação anômala identificada: index={idx}, valor={valor}, hora={hora}")
else:
    print("Nenhuma transação isolada encontrada (ou mais de uma). Revise os dados/cluster).")

print("-" * 50, "\n")
