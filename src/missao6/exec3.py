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

