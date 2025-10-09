# ATIVIDADE 09/01/2025

# MISSÃO AC

# Projeto 1: Classificação de Risco de Crédito (0,10 PONTO)
# Cenário de Negócio: Você foi contratado por uma fintech para desenvolver um sistema automatizado de análise de crédito. O objetivo é minimizar o risco de perdas financeiras, identificando clientes com alta probabilidade de não pagarem um empréstimo.
# Sua tarefa é construir um modelo de Machine Learning que classifique os pedidos de empréstimo em "Baixo Risco" ou "Alto Risco".

# Seu Objetivo: Construir um pipeline de classificação que receba os dados de um cliente e preveja sua categoria de risco.

# Dicas para a Construção:
# Dataset:
# Crie um DataFrame com o Pandas.
# Features Numéricas: idade (ex: 25 a 65), salario_anual (ex: 30000 a 200000), anos_empregado (ex: 0 a 30), valor_emprestimo (ex: 5000 a 100000).
# Features Categóricas: tipo_moradia (com valores como 'Aluguel', 'Propria', 'Financiada').
# Alvo (Target): risco_inadimplencia (use 0 para Baixo Risco e 1 para Alto Risco).
# Desafio Extra: Use np.nan para introduzir alguns valores ausentes na coluna salario_anual para simular dados do mundo real.

# Bibliotecas Essenciais:
# pandas e numpy para manipulação de dados.
# train_test_split do sklearn.model_selection.
# Pipeline do sklearn.pipeline.
# SimpleImputer, StandardScaler, OneHotEncoder do sklearn.preprocessing.
# ColumnTransformer do sklearn.compose.
# LogisticRegression do sklearn.linear_model (um bom ponto de partida para classificação).
# accuracy_score, classification_report, roc_auc_score do sklearn.metrics.

# Estrutura do Projeto:
# Preparação: Separe seus dados em X (features) e y (alvo). Use train_test_split para criar os conjuntos de treino e teste (sugestão: 30% para teste).
# Pré-processamento: Defina um ColumnTransformer. Dentro dele, crie um pipeline para as colunas numéricas (com SimpleImputer e StandardScaler) e outro para as colunas categóricas (com OneHotEncoder).
# Modelagem: Crie um Pipeline final que una o ColumnTransformer e o modelo LogisticRegression.
# Treinamento: Use o método .fit() no seu pipeline final com os dados de treino.
# Avaliação: Use .predict() e .predict_proba() nos dados de teste. Calcule a acurácia, o classification_report e o roc_auc_score para entender a performance do seu modelo.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Criação do Dataset Simulado

np.random.seed(42)  # Reprodutibilidade

n = 300  # número de clientes

df = pd.DataFrame({
    'idade': np.random.randint(25, 65, n),
    'salario_anual': np.random.randint(30000, 200000, n).astype(float),
    'anos_empregado': np.random.randint(0, 30, n),
    'valor_emprestimo': np.random.randint(5000, 100000, n),
    'tipo_moradia': np.random.choice(['Aluguel', 'Propria', 'Financiada'], n),
    'risco_inadimplencia': np.random.choice([0, 1], n, p=[0.7, 0.3])  # maioria baixo risco
})

# Introduzindo valores ausentes em salario_anual
mask = np.random.choice([True, False], size=n, p=[0.1, 0.9])
df.loc[mask, 'salario_anual'] = np.nan

print("Visualização inicial do dataset:")
print(df.head(), "\n")

# Separar Features e Alvo
X = df.drop('risco_inadimplencia', axis=1)
y = df['risco_inadimplencia']

# Divisão em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Definir Colunas Numéricas e Categóricas
colunas_numericas = ['idade', 'salario_anual', 'anos_empregado', 'valor_emprestimo']
colunas_categoricas = ['tipo_moradia']

# Pipelines de Pré-processamento
pipeline_numerico = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

pipeline_categorico = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer 
preprocessador = ColumnTransformer([
    ('num', pipeline_numerico, colunas_numericas),
    ('cat', pipeline_categorico, colunas_categoricas)
])

# Pipeline Final com Modelo 
modelo = Pipeline([
    ('preprocessador', preprocessador),
    ('classificador', LogisticRegression(max_iter=1000))
])

# Treinamento
print("Treinando o modelo Logistic Regression...")
modelo.fit(X_train, y_train)
print("Treinamento concluído!\n")

# Avaliação do Modelo
y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

# Métricas
acuracia = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("=== Resultados do Modelo ===")
print(f"Acurácia: {acuracia:.3f}")
print(f"ROC AUC Score: {roc_auc:.3f}\n")

print("=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred, target_names=['Baixo Risco', 'Alto Risco']))

# Exemplo de Previsão 
novo_cliente = pd.DataFrame({
    'idade': [35],
    'salario_anual': [85000],
    'anos_empregado': [5],
    'valor_emprestimo': [20000],
    'tipo_moradia': ['Financiada']
})

predicao = modelo.predict(novo_cliente)[0]
proba = modelo.predict_proba(novo_cliente)[0][1]

print("=== Exemplo de Previsão ===")
print(novo_cliente)
print(f"\nPrevisão de risco: {'Alto Risco' if predicao == 1 else 'Baixo Risco'}")
print(f"Probabilidade de inadimplência: {proba:.2%}")

print("\n Projeto finalizado com sucesso!")
