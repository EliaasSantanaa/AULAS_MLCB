# Projeto 5: Previsão de Qualidade de Vinho (0,15 PONTO)
# Cenário de Negócio:
# Você é um enólogo em uma grande vinícola e quer usar dados para aprimorar o processo de produção. Você tem acesso a um grande dataset com as análises físico-químicas de vinhos e uma nota de qualidade (de 0 a 10) atribuída por especialistas.
# Sua tarefa é criar um modelo que preveja essa nota de qualidade.

# Seu Objetivo:
# Construir um modelo de regressão para prever a nota de um vinho com base em suas características.
# Dicas para a Construção:

# Dataset:
# Este é um dataset público famoso. Você pode carregá-lo diretamente com o Pandas.
# URL: 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
# Importante: Os dados neste CSV são separados por ponto e vírgula. Use o parâmetro sep=';' no pd.read_csv( ).
# O alvo é a coluna quality. Todas as outras são features.
# Bibliotecas Essenciais:
# pandas.
# train_test_split.
# Pipeline.
# StandardScaler.
# SVR do sklearn.svm (Support Vector Regressor, um algoritmo poderoso).
# mean_absolute_error, r2_score.
# Estrutura do Projeto:
# Preparação: Carregue o dataset, separe X e y, e divida em treino e teste.
# Modelagem: O SVR, assim como o KNN, é muito sensível à escala dos dados. É essencial usar um StandardScaler. Crie um Pipeline contendo o StandardScaler e o SVR.
# Treinamento: Treine o pipeline. Dependendo do tamanho dos dados, o SVR pode levar um pouco mais de tempo para treinar do que a Regressão Linear.
# Avaliação: Faça previsões e avalie com MAE e R² Score. Reflita: um erro médio de 0.5 na nota de um vinho é aceitável para o seu negócio?

# Importação das Bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# Carregar o Dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'


df = pd.read_csv(url, sep=';')

print("Visualização inicial do dataset:")
print(df.head(), "\n")

# Separar Features e Alvo
X = df.drop('quality', axis=1)
y = df['quality']

# Dividir em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Tamanho do conjunto de treino:", X_train.shape)
print("Tamanho do conjunto de teste:", X_test.shape, "\n")

# Criar o Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=100, gamma='scale'))
])

# Treinar o Modelo
print("Treinando o modelo SVR...")
pipeline.fit(X_train, y_train)
print("Treinamento concluído!\n")

# Fazer Previsões
y_pred = pipeline.predict(X_test)

# Avaliar o Modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Resultados do Modelo ===")
print(f"Erro Médio Absoluto (MAE): {mae:.3f}")
print(f"R² Score: {r2:.3f}")


print("\nInterpretação:")
print("- O MAE indica o erro médio entre a nota real e a prevista.")
print("- Um MAE em torno de 0.5 significa que o modelo erra, em média, meio ponto na nota do vinho.")
print("- O R² mostra o quanto o modelo explica da variação das notas de qualidade.")
print("- Quanto mais próximo de 1, melhor o ajuste.\n")

#  Busca de Melhores Hiperparâmetros
print("Executando GridSearchCV para ajustar hiperparâmetros (pode demorar alguns minutos)...")

param_grid = {
    'svr__C': [10, 100],
    'svr__gamma': ['scale'],
    'svr__kernel': ['rbf']
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

print("\n=== Melhores Parâmetros Encontrados ===")
print(grid.best_params_)
print(f"Melhor R² durante a validação cruzada: {grid.best_score_:.3f}\n")

# Avaliar o Melhor 
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)

mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print("=== Avaliação do Melhor Modelo ===")
print(f"MAE: {mae_best:.3f}")
print(f"R²: {r2_best:.3f}")

print("\n Projeto finalizado com sucesso!")
