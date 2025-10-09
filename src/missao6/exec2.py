# Projeto 2: Previsão de Consumo de Energia (0,10 PONTO)
# Cenário de Negócio:
# Você é um cientista de dados em uma companhia de energia. Para otimizar a distribuição e evitar desperdícios, a empresa precisa prever o consumo diário de energia de grandes edifícios.
# Sua tarefa é criar um modelo que estime o consumo em kWh com base em dados de calendário e clima.

# Seu Objetivo:Construir um modelo de regressão que preveja um valor numérico de consumo de energia.

# Dicas para a Construção:
# Dataset: Crie um DataFrame com o Pandas.
# Features Iniciais: data (use pd.to_datetime para criar uma série de datas), temperatura_media (ex: 0 a 35), dia_util (use 1 para sim e 0 para não).
# Alvo (Target): consumo_energia_kwh (um valor numérico, ex: 1000 a 5000).
# Bibliotecas Essenciais:
# pandas e numpy.
# train_test_split.
# Pipeline.
# StandardScaler.
# RandomForestRegressor do sklearn.ensemble (um modelo robusto para esse tipo de problema).
# mean_absolute_error, r2_score do sklearn.metrics.
# Estrutura do Projeto:
# Engenharia de Atributos (Feature Engineering): Esta é a etapa chave! A partir da coluna data, crie novas features numéricas: mes, dia_da_semana, dia_do_ano. Use os acessadores .dt do Pandas (ex: df['data'].dt.month). Após criar essas novas colunas, remova a coluna data original.
# Preparação: Com as novas features criadas, separe X e y. Divida em treino e teste.
# Modelagem: Como todas as suas features agora são numéricas, seu pipeline pode ser mais simples. Crie um Pipeline que contenha um StandardScaler e o RandomForestRegressor.
# Treinamento: Treine o pipeline com os dados de treino.
# Avaliação: Faça previsões nos dados de teste e avalie seu modelo usando o Erro Absoluto Médio (MAE) e o R² Score. Interprete o que o MAE significa no contexto do problema (kWh).

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data_inicio = '2023-01-01'
data_fim = '2024-12-31'
datas = pd.to_datetime(pd.date_range(start=data_inicio, end=data_fim, freq='D'))
n_dias = len(datas)

np.random.seed(42)

temperatura_media = np.random.uniform(5, 30, size=n_dias).round(1)

dia_da_semana = datas.dayofweek
dia_util = np.where((dia_da_semana >= 0) & (dia_da_semana <= 4), 1, 0) # 1 se for Segunda a Sexta, 0 caso contrário

base_consumo = 2500
impacto_temp = temperatura_media * 50
impacto_util = dia_util * 1500
ruido = np.random.normal(loc=0, scale=300, size=n_dias)
consumo_energia_kwh = base_consumo + impacto_temp + impacto_util + ruido
consumo_energia_kwh = np.maximum(1000, consumo_energia_kwh).round(0)

df = pd.DataFrame({
    'data': datas,
    'temperatura_media': temperatura_media,
    'dia_util': dia_util,
    'consumo_energia_kwh': consumo_energia_kwh
})

print("--- DataFrame Inicial ---")
print(df.head())
print("-" * 30)

print("--- Engenharia de Atributos ---")

df['mes'] = df['data'].dt.month
df['dia_da_semana'] = df['data'].dt.dayofweek
df['dia_do_ano'] = df['data'].dt.dayofyear

df.drop('data', axis=1, inplace=True)

print(df.head())
print("-" * 30)

X = df.drop('consumo_energia_kwh', axis=1)
y = df['consumo_energia_kwh']

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Formato de X_treino: {X_treino.shape}")
print(f"Formato de X_teste: {X_teste.shape}")
print("-" * 30)

print("--- Criação do Pipeline de Modelagem ---")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rfr', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("Pipeline criado com StandardScaler e RandomForestRegressor.")
print("-" * 30)

print("--- Treinamento do Modelo ---")
pipeline.fit(X_treino, y_treino)
print("Treinamento concluído.")
print("-" * 30)

print("--- Avaliação do Modelo ---")

y_pred = pipeline.predict(X_teste)
mae = mean_absolute_error(y_teste, y_pred)
r2 = r2_score(y_teste, y_pred)

print(f"Erro Absoluto Médio (MAE): {mae:.2f} kWh")
print(f"R² Score: {r2:.4f}")

print("\n--- Interpretação do MAE ---")
print(f"O MAE de {mae:.2f} kWh significa que, em média, a previsão de consumo do modelo")
print(f"está errada por cerca de {mae:.2f} Quilowatts-hora. No contexto do negócio,")
print("é o desvio médio esperado entre o consumo real e o consumo previsto.")
print("Um valor baixo (em relação à média de consumo) indica um modelo com boa precisão.")