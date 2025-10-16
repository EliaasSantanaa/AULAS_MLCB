# Exercício 3: Análise de Sentimento com RNN (LSTM) (0,20 pontos)

# Objetivo: Construir uma Rede Neural Recorrente (LSTM) para classificar o sentimento de avaliações de filmes (positivo/negativo) usando o dataset IMDB.

# Passos:
# 1. Carregar e Pré-processar os Dados: Carregue o dataset IMDB (reviews de filmes, já pré-processadas em sequências de inteiros). Padronize o comprimento das sequências (padding).
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# 2. Construir a Rede LSTM: Crie um modelo sequencial. Adicione uma camada de Embedding para representar as palavras como vetores densos. Em seguida, adicione uma camada LSTM e uma camada Dense de saída com ativação sigmoid (para classificação binária).

# 3. Compilar e Treinar: Compile o modelo com um otimizador e função de perda (binary_crossentropy). Treine o modelo.

# 4. Avaliar e Testar: Avalie o modelo e tente prever o sentimento de algumas frases de teste que você mesmo criar.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

num_words = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = keras.Sequential([
    layers.Embedding(input_dim=num_words, output_dim=64, input_length=maxlen),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {test_acc:.4f}")
print(f"Perda no conjunto de teste: {test_loss:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia do Modelo LSTM')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda do Modelo LSTM')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.tight_layout()
plt.show()

word_index = keras.datasets.imdb.get_word_index()
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = '<PAD>'
reverse_word_index[1] = '<START>'
reverse_word_index[2] = '<UNK>'
reverse_word_index[3] = '<UNUSED>'

def encode_review(text):
    words = text.lower().split()
    encoded = [1]
    for word in words:
        index = word_index.get(word, 2)
        if index < num_words:
            encoded.append(index)
    return keras.preprocessing.sequence.pad_sequences([encoded], maxlen=maxlen)

test_sentences = [
    "I loved this movie, it was amazing and very emotional",
    "The film was terrible and boring, I hated it",
    "An average movie with some good moments but weak story",
    "Absolutely fantastic! The best movie I've seen in years",
    "It was so bad that I wanted to leave the cinema"
]

for sentence in test_sentences:
    encoded = encode_review(sentence)
    prediction = model.predict(encoded, verbose=0)[0][0]
    sentiment = "Positivo 😀" if prediction > 0.5 else "Negativo 😠"
    print(f"\nFrase: {sentence}")
    print(f"Sentimento previsto: {sentiment} ({prediction:.2f})")
