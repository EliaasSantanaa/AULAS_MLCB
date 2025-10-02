# Regressão - Prevendo a Nota do Exame
# O Problema: Queremos prever a nota final que um aluno irá tirar em um exame, com base no número de horas que ele estudou.
# Tipo: Regressão, porque a "nota" é um valor numérico contínuo (pode ser 7.5, 8.2, 9.0, etc.).
# Algoritmo: Usaremos a LinearRegression, que tentará traçar a "melhor reta" que descreve a relação entre as horas de estudo e a nota.




# # --- REGRESSÃO ---

# Exercício 1: Regressão - Previsão de Preço de Carros Usados (0,20pt)
# Cenário: Sua tarefa é criar um modelo que estime o preço de venda de um carro usado com base em suas características, para ajudar os vendedores a precificarem seus anúncios de forma justa.
# Tipo: Regressão, pois o alvo (Preco) é um valor numérico contínuo.
# Desafio: Complete o esqueleto de código abaixo para treinar e avaliar um modelo de previsão de preços.

# --- EXERCÍCIO PRÁTICO DE REGRESSÃO ---
# Objetivo: Prever o preço de carros usados.

# --- 1. SETUP INICIAL ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 2. DATASET (Grande e com desafios reais) ---
# Dataset simulado de carros usados
data = {
    'Ano': [2018, 2015, 2019, 2020, 2017, 2014, 2018, 2016, 2021, 2013, 2019, 2017, 2015, 2020, 2018, 2016, 2022, 2014, 2019, 2017],
    'Quilometragem': [50000, 120000, 30000, 15000, 70000, 150000, 60000, 95000, 5000, 180000, 40000, 80000, 110000, 20000, 55000, 105000, 2000, 160000, 35000, 75000],
    'Potencia_Motor': [1.6, 2.0, 1.0, 1.4, 1.8, 2.0, 1.6, 1.8, 1.0, 2.2, 1.4, 1.6, 2.0, 1.0, 1.8, 2.0, 1.2, 2.2, 1.4, 1.8],
    'Tipo_Combustivel': ['Flex', 'Gasolina', 'Flex', 'Flex', 'Gasolina', 'Diesel', 'Flex', 'Gasolina', 'Flex', 'Diesel', 'Flex', 'Flex', 'Gasolina', 'Flex', 'Gasolina', 'Diesel', 'Flex', 'Diesel', 'Flex', 'Gasolina'],
    'Cambio': ['Manual', 'Automatico', 'Manual', 'Manual', 'Automatico', 'Manual', 'Automatico', 'Automatico', 'Manual', 'Automatico', 'Manual', 'Manual', 'Automatico', 'Manual', 'Automatico', 'Automatico', 'Manual', 'Automatico', 'Manual', 'Automatico'],
    'Preco': [45000, 38000, 55000, 65000, 52000, 35000, 48000, 46000, 75000, 32000, 58000, 50000, 40000, 68000, 53000, 44000, 82000, 30000, 60000, 51000]
}
df_carros = pd.DataFrame(data)

# Adicionando valores ausentes para o desafio
df_carros.loc[5, 'Quilometragem'] = np.nan
df_carros.loc[11, 'Potencia_Motor'] = np.nan

print("--- Análise Inicial: Dataset de Carros ---")
df_carros.info()
print("\nValores ausentes:")
print(df_carros.isnull().sum())

# --- INÍCIO DO TRABALHO ---

# --- 3. PREPARAÇÃO DOS DADOS ---

# 3.1. Separe as features (X) do alvo (y)
# Dica: 'y' é a coluna 'Preco'. 'X' são todas as outras.
X = df_carros.drop('Preco', axis=1)
y = df_carros['Preco']

# 3.2. Divida os dados em conjuntos de treino e teste
# Dica: Use 20% para teste e random_state=42 para consistência.
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.3. Crie um pipeline de pré-processamento
# Dica: Use ColumnTransformer para aplicar transformações diferentes em colunas diferentes.
# Para colunas numéricas: um SimpleImputer (para preencher NaN) e um StandardScaler.
# Para colunas categóricas: um OneHotEncoder.

colunas_numericas = ['Ano', 'Quilometragem', 'Potencia_Motor']
colunas_categoricas = ['Tipo_Combustivel', 'Cambio']

# Pipeline para dados numéricos
pipeline_numerico = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para dados categóricos
pipeline_categorico = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Junte os pipelines em um ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', pipeline_numerico, colunas_numericas),
        ('cat', pipeline_categorico, colunas_categoricas)
    ])

# --- 4. TREINAMENTO DO MODELO ---

# 4.1. Crie o modelo de Regressão Linear
modelo_reg = LinearRegression()

# 4.2. Crie o pipeline final que une o pré-processador e o modelo
# Dica: Este pipeline fará todo o trabalho: pré-processar e depois treinar.
pipeline_final_reg = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', modelo_reg)])

# 4.3. Treine o pipeline completo com os dados de treino
# Dica: Use o método .fit() no pipeline final com X_treino e y_treino.
pipeline_final_reg.fit(X_treino, y_treino)

print("\n Modelo de Regressão treinado com sucesso!")

# --- 5. AVALIAÇÃO DO MODELO ---

# 5.1. Faça previsões com os dados de teste
# Dica: Use o método .predict() no pipeline treinado com X_teste.
previsoes_preco = pipeline_final_reg.predict(X_teste)

# 5.2. Calcule as métricas de avaliação
mae = mean_absolute_error(y_teste, previsoes_preco)
r2 = r2_score(y_teste, previsoes_preco)

print("\n--- Resultados da Avaliação (Regressão) ---")
print(f"Erro Absoluto Médio (MAE): R$ {mae:.2f}")
print(f"Score R²: {r2:.2f}")

# 5.3. Visualize os resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_teste, previsoes_preco)
plt.plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], '--r', linewidth=2)
plt.xlabel("Preços Reais")
plt.ylabel("Preços Previstos")
plt.title("Preços Reais vs. Previsões do Modelo de Regressão")
plt.show()
