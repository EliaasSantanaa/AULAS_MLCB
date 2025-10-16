# ETAPA 2: DESENVOLVER EXERCICIOS DE DEEP LEARNING (0,40pts)
# Exercício 1: Classificação de Imagens com MLP (0,10)
# Objetivo: Construir e treinar uma Rede Neural Perceptron Multicamadas (MLP) para classificar imagens do dataset Fashion MNIST.

# Passos:
# 1. Carregar e Pré-processar os Dados: Carregue o dataset Fashion MNIST (disponível no Keras ou TensorFlow Datasets). Normalize os valores dos pixels para o intervalo [0, 1] e "achate" (flatten) as imagens para um vetor unidimensional.

# 2. Construir a MLP: Crie um modelo sequencial com Keras. Adicione uma camada de entrada, algumas camadas densas (Dense) com função de ativação ReLU e uma camada de saída com função de ativação softmax (para classificação multiclasse).

# 3. Compilar o Modelo: Configure o modelo com um otimizador (ex: adam), função de perda (ex: sparse_categorical_crossentropy) e métricas (ex: accuracy).

# 4. Treinar o Modelo: Treine o modelo com os dados de treinamento e valide-o com os dados de teste. Monitore a acurácia e a perda.

# 5. Avaliar e Visualizar: Avalie o modelo no conjunto de teste final e visualize algumas previsões para entender onde o modelo acerta e erra.


# Passo 1: Carregar e Pré-processar os Dados
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Carregar o dataset Fashion MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Verificar a forma dos dados
print(f"Forma dos dados de treinamento: {x_train.shape}")
print(f"Forma dos rótulos de treinamento: {y_train.shape}")
print(f"Forma dos dados de teste: {x_test.shape}")
print(f"Forma dos rótulos de teste: {y_test.shape}")

# Normalizar os valores dos pixels para [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Achatar as imagens (28x28 -> 784)
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

print(f"Forma após flatten: {x_train_flat.shape}")


# Passo 2: Construir a MLP
# Definir os nomes das classes para visualização
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Construir o modelo MLP
model = keras.Sequential([
    # Camada de entrada (flatten já foi feito manualmente)
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    
    # Camadas ocultas
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    
    # Camada de saída (10 classes)
    keras.layers.Dense(10, activation='softmax')
])

# Visualizar a arquitetura do modelo
model.summary()


# Passo 3: Compilar o Modelo
# Compilar o modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Passo 4: Treinar o Modelo
# Definir callbacks para monitoramento
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Treinar o modelo
history = model.fit(
    x_train_flat, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(x_test_flat, y_test),
    callbacks=[early_stopping],
    verbose=1
)


# Passo 5: Avaliar e Visualizar
# Avaliar o modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {test_accuracy:.4f}")
print(f"Perda no conjunto de teste: {test_loss:.4f}")

# Plotar histórico de treinamento
plt.figure(figsize=(12, 4))

# Gráfico de acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia do Modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

# Gráfico de perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.show()

# Fazer previsões no conjunto de teste
predictions = model.predict(x_test_flat)
predicted_classes = np.argmax(predictions, axis=1)

# Visualizar algumas previsões
def plot_predictions(images, true_labels, predicted_labels, class_names, num_images=10):
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        
        # Cor do texto: verde se correto, vermelho se errado
        color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
        
        plt.title(f'Verd: {class_names[true_labels[i]]}\nPred: {class_names[predicted_labels[i]]}', 
                 color=color, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualizar as primeiras 10 imagens do teste
plot_predictions(x_test, y_test, predicted_classes, class_names)

# Matriz de confusão
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Calcular matriz de confusão
cm = confusion_matrix(y_test, predicted_classes)

# Plotar matriz de confusão
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, predicted_classes, target_names=class_names))

# Analisar exemplos específicos
def analyze_predictions(true_labels, predicted_labels, probabilities, class_names, num_examples=5):
    # Encontrar exemplos corretos e incorretos
    correct_indices = np.where(true_labels == predicted_labels)[0]
    incorrect_indices = np.where(true_labels != predicted_labels)[0]
    
    print(f"\nExemplos de acertos:")
    for i in range(min(3, len(correct_indices))):
        idx = correct_indices[i]
        print(f"Imagem {idx}: {class_names[true_labels[idx]]} - "
              f"Confiança: {probabilities[idx][predicted_labels[idx]]:.4f}")
    
    print(f"\nExemplos de erros:")
    for i in range(min(3, len(incorrect_indices))):
        idx = incorrect_indices[i]
        print(f"Imagem {idx}: Verdadeiro: {class_names[true_labels[idx]]} - "
              f"Predito: {class_names[predicted_labels[idx]]} - "
              f"Confiança: {probabilities[idx][predicted_labels[idx]]:.4f}")

analyze_predictions(y_test, predicted_classes, predictions, class_names)

