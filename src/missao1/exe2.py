#### TAREFA 2 :  Criar um classificador de mensagens para um bot de atendimento acadêmico - rode sem erros no VSCODE  ou no Colab ###

# Criar um classificador de mensagens para um bot de atendimento acadêmico.
# Instruções:
# 1. Crie uma lista de frases (ex: dúvidas sobre matrícula, notas, eventos, biblioteca)
# 2. Crie a lista de rótulos correspondentes
# 3. Vetorize as frases com CountVectorizer
# 4. Treine um modelo Naive Bayes ou outro de sua escolha
# 5. Teste com uma nova frase e imprima o resultado

# início código
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Dataset
frases = [
    "Quando será a próxima reunião do conselho?",
    "Como posso acessar o Wi-Fi do campus?",
    "Quais são os horários do restaurante universitário?",
    "Onde encontro o calendário acadêmico?",
    "Como solicitar segunda via do crachá?",
    "Tem vaga no estacionamento hoje?",
    "Qual o prazo para entrega do TCC?",
    "Como participar do grupo de pesquisa?"
]
rotulos = [
    "reuniao",
    "wifi",
    "restaurante",
    "calendario",
    "crachá",
    "estacionamento",
    "tcc",
    "pesquisa"
]

# 2. Vetorização
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(frases)

# 3. Modelo
modelo = MultinomialNB()
modelo.fit(X, rotulos)

# 4. Previsão interativa
while True:
    nova_frase = input("\nDigite uma mensagem (ou 'sair' para encerrar): ")
    if nova_frase.lower() == "sair":
        break
    X_novo = vectorizer.transform([nova_frase])
    rotulo_previsto = modelo.predict(X_novo)
    print(f"Classificação prevista: {rotulo_previsto[0]}")
