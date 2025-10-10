# Projeto 3: Detecção de E-mails Spam (0,10 PONTO)
# Cenário de Negócio:
# Você está desenvolvendo um novo cliente de e-mail e quer oferecer um filtro de spam inteligente. Você tem um dataset de e-mails, onde cada um está rotulado como "spam" ou "ham" (não spam). O desafio é que os dados são texto puro.
# Seu Objetivo:
# Construir um pipeline de NLP (Processamento de Linguagem Natural) para classificar e-mails.
# Dicas para a Construção:
# Dataset:
# Crie um pequeno DataFrame com duas colunas: texto e categoria.
# Exemplos de texto: "oferta imperdível clique aqui agora", "ganhe dinheiro fácil", "relatório de vendas anexo", "oi, tudo bem? reunião amanhã".
# categoria: 'spam' ou 'ham'.
# Bibliotecas Essenciais:
# pandas.
# train_test_split.
# Pipeline.
# TfidfVectorizer do sklearn.feature_extraction.text. Esta é a ferramenta chave que converterá o texto em números que o modelo entende.
# MultinomialNB do sklearn.naive_bayes. Um algoritmo de classificação clássico e muito eficaz para problemas de texto.
# accuracy_score, classification_report.
# Estrutura do Projeto:
# Preparação: Crie o dataset, separe X (a coluna texto) e y (a coluna categoria), e divida em treino e teste.
# Modelagem de NLP: Este pipeline é diferente. Ele não usará StandardScaler. As etapas serão:
# TfidfVectorizer: Esta etapa transforma as frases em um vetor numérico baseado na frequência e relevância das palavras.
# MultinomialNB: O classificador que aprenderá com esses vetores.
# Crie um Pipeline com essas duas etapas.
# Treinamento: Treine o pipeline. O .fit() irá aplicar o vetorizador e depois treinar o classificador.
# Avaliação: Faça previsões. Analise o classification_report. Para um filtro de spam, qual é mais importante: alta precisão ou alto recall na classe 'spam'? 
# (Pense no custo de um e-mail importante ser classificado como spam vs. um spam passar pelo filtro).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 1. Criando o dataset com frases e a categoria (spam ou ham)
dados = {
    'texto': [
        "oferta imperdível clique aqui agora",
        "ganhe dinheiro fácil",
        "relatório de vendas anexo",
        "oi, tudo bem? reunião amanhã",
        "promoção exclusiva só hoje",
        "não perca essa chance única",
        "vamos almoçar amanhã?",
        "clique para ganhar um prêmio",
        "relatório financeiro enviado",
        "encontro marcado para segunda"
    ],
    'categoria': [
        'spam',
        'spam',
        'ham',
        'ham',
        'spam',
        'spam',
        'ham',
        'spam',
        'ham',
        'ham'
    ]
}

# montagem da tabela
df = pd.DataFrame(dados)

# mostrando nosso conjunto de dados
print("Dataset:\n", df)

# 2. separando as frases (X) e os rótulos (y)
X = df['texto']  # Textos dos e-mails
y = df['categoria']  # Labels: spam ou ham

# Dividindo em 80% treino e 20% teste, para avaliar depois como nos exercicios anteriores
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Criando o pipeline que transforma texto e treina o modelo
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Biblioteca importada que transf texto em números importantes
    ('modelo', MultinomialNB())    # Modelo que aprende a distinguir o que será spam
])

# 4. treinando o pipeline com os dados de treino
pipeline.fit(X_treino, y_treino)

# 5. previsões com os dados de teste
y_pred = pipeline.predict(X_teste)

# 6. Avaliando o modelo com um relatório detalhado
#print("\nRelatório de Classificação:\n", classification_report(y_teste, y_pred)) 

# 6. Avaliando o modelo com um relatório detalhado,
# usando zero_division=0 para evitar warnings e deixando a saída mais limpa
print("\nRelatório de Classificação:\n",
      classification_report(y_teste, y_pred, zero_division=0, digits=2, target_names=['ham', 'spam']))
