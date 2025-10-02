# Prevendo Preços de Imóveis (0,10pt)
# Sua missão é: Treinar um modelo de Machine Learning com os dados de treino.
# As variáveis abaixo já existem e estão prontas para serem usadas:
# X_treino_final: As características (tamanho, quartos, cidade) dos imóveis de treino, já limpas e transformadas.
# y_treino: Os preços reais dos imóveis de treino.
# X_teste_final: As características dos imóveis de teste, que o modelo nunca viu.
# y_teste: Os preços reais dos imóveis de teste. Usaremos isso no final para comparar com as previsões do nosso modelo.

# Instruções:
# Analise o código e as dicas.
# Preencha as linhas que contêm # SEU CÓDIGO AQUI.

# Seu Desafio: Complete o Código Abaixo
# Copie este bloco de código e preencha as seções marcadas com # SEU CÓDIGO AQUI. Use as dicas para te guiar.

# --- Bloco de Código do Exercício ---
# Importe as bibliotecas necessárias para o modelo e para a avaliação
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Suponha que estas variáveis já existem e foram geradas pelo código anterior
# (Para que este código rode de forma independente, vamos recriá-las aqui rapidamente)
# --- Início do Bloco de Contexto (não precisa alterar) ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

dados = {
    'tamanho_m2': [50, 70, 100, 120, 65, np.nan, 95, 88, 110, 150],
    'n_quartos': [1, 2, 3, 3, 2, 2, 3, 2, 3, 4],
    'cidade': ['SP', 'RJ', 'SP', 'BH', 'RJ', 'SP', 'BH', 'RJ', 'SP', 'BH'],
    'preco': [150000, 210000, 300000, 350000, 190000, 180000, 280000, 250000, 320000, 450000]
}
df = pd.DataFrame(dados)
imputer = SimpleImputer(strategy='mean')
df['tamanho_m2'] = imputer.fit_transform(df[['tamanho_m2']])
X = df.drop('preco', axis=1)
y = df['preco']
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)
colunas_numericas = ['tamanho_m2', 'n_quartos']
colunas_categoricas = ['cidade']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_treino_cat = encoder.fit_transform(X_treino[colunas_categoricas])
X_teste_cat = encoder.transform(X_teste[colunas_categoricas])
scaler = StandardScaler()
X_treino_num = scaler.fit_transform(X_treino[colunas_numericas])
X_teste_num = scaler.transform(X_teste[colunas_numericas])
X_treino_final = np.hstack([X_treino_num, X_treino_cat])
X_teste_final = np.hstack([X_teste_num, X_teste_cat])
# --- Fim do Bloco de Contexto ---

# --- INÍCIO DO TRABALHO ---

# 1. Crie uma instância do modelo de Regressão Linear
# Dica: Assim como fizemos na primeira aula.
modelo_preco_casas = LinearRegression()


# 2. Treine o modelo com os dados de TREINO
# Dica: Use o método .fit() com os dados de treino (X_treino_final e y_treino).
modelo_preco_casas.fit(X_treino_final, y_treino)

print("Modelo treinado com sucesso!")

# 3. Faça previsões nos dados de TESTE
# Dica: Use o método .predict() no conjunto de teste (X_teste_final).
previsoes = modelo_preco_casas.predict(X_teste_final)
print("\n--- Resultados da Avaliação ---")
# Vamos comparar as previsões com os valores reais (y_teste)
for i in range(len(previsoes)):
    print(f"Imóvel {i+1}: Preço Real = R${y_teste.iloc[i]:.2f} | Previsão do Modelo = R${previsoes[i]:.2f}")

# 4. Avalie a performance do modelo
# Usaremos duas métricas:
# - Erro Absoluto Médio (MAE): A média da diferença (em R$) entre o preço real e a previsão. Quanto menor, melhor.
# - R² Score: Indica o quão bem o modelo explica a variação dos dados. Varia de 0 a 1. Quanto mais perto de 1, melhor.

# Dica: Use as funções mean_absolute_error() e r2_score(). Elas recebem (y_teste, previsoes) como argumentos.
mae = mean_absolute_error(y_teste, previsoes)
r2 = r2_score(y_teste, previsoes)

print(f"\nErro Absoluto Médio (MAE): R$ {mae:.2f}")
print(f"Score R²: {r2:.2f}")

# --- FIM DO SEU TRABALHO ---

