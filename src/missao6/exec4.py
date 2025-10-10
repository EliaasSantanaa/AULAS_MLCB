# ETAPA DE SUPER DESAFIO: VALOR DE 0,30 DE BÔNUS NA AVALIAÇÃO SEMESTRAL

# Projeto 4: Classificação de Espécies de Flores Íris (0,15 PONTO)
# Cenário de Negócio:
# Você é um pesquisador botânico e coletou dados sobre as dimensões das pétalas e sépalas de várias flores do gênero Íris. Para acelerar futuras pesquisas, você quer um programa que identifique automaticamente a espécie de uma flor (setosa, versicolor ou virginica) a partir dessas medidas.
# Seu Objetivo:
# Construir um modelo de classificação multiclasse simples e eficiente.
# Dicas para a Construção:
# Dataset:
# Não precisa criar do zero! O Scikit-learn já vem com este dataset clássico.
# Use from sklearn.datasets import load_iris. A função load_iris() retorna um objeto onde iris.data são as features e iris.target são as classes.
# Bibliotecas Essenciais:
# load_iris do sklearn.datasets.
# train_test_split.
# Pipeline.
# StandardScaler.
# KNeighborsClassifier do sklearn.neighbors (um algoritmo clássico baseado em "proximidade").
# accuracy_score, classification_report.
# Estrutura do Projeto:
# Preparação: Carregue os dados e imediatamente os divida em conjuntos de treino e teste.
# Modelagem: O algoritmo KNN é muito sensível à escala das features. Portanto, crie um Pipeline que primeiro aplica o StandardScaler e depois o KNeighborsClassifier. Você pode experimentar com o parâmetro n_neighbors (um bom começo é 5).
# Treinamento: Treine o pipeline com .fit().
# Avaliação: Faça previsões e use o classification_report para ver a performance do modelo para cada uma das três espécies de flores. A acurácia geral é uma boa métrica aqui, pois o dataset é balanceado.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# carregar os dados
iris = load_iris()
X, y = iris.data, iris.target

# dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# criar pipeline com padronizador e knn (vizinhos = 5)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

# treinar o pipeline nos dados de treino
pipeline.fit(X_train, y_train)

# fazer previsões no conjunto de teste
predictions = pipeline.predict(X_test)

# calcular e imprimir acurácia e relatório de classificação
accuracy = accuracy_score(y_test, predictions)
print(f'acurácia geral: {accuracy * 100:.2f}%')
print('relatório de classificação:')
print(classification_report(y_test, predictions, target_names=iris.target_names))
