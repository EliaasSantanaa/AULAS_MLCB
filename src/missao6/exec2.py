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
