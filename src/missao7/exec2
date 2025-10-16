# Exercício 2: Reconhecimento de Dígitos com CNN (0,10pts)

# Passo 1: Carregar e Pré-processar os Dados

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Carregar o dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Verificar a forma dos dados
print(f"Forma dos dados de treinamento: {x_train.shape}")
print(f"Forma dos rótulos de treinamento: {y_train.shape}")
print(f"Forma dos dados de teste: {x_test.shape}")
print(f"Forma dos rótulos de teste: {y_test.shape}")

# Normalizar os valores dos pixels para [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Redimensionar para o formato esperado pelas camadas convolucionais (28, 28, 1)
x_train_cnn = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test_cnn = x_test.reshape(x_test.shape[0], 28, 28, 1)

print(f"Forma após redimensionamento - Treino: {x_train_cnn.shape}")
print(f"Forma após redimensionamento - Teste: {x_test_cnn.shape}")

# Visualizar algumas imagens do dataset
def plot_sample_images(images, labels, num_images=10):
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_sample_images(x_train_cnn, y_train)

# Passo 2: Construir a CNN
# Construir o modelo CNN
model_cnn = keras.Sequential([
    # Primeira camada convolucional
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Segunda camada convolucional
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Terceira camada convolucional
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Achatar a saída para camadas densas
    keras.layers.Flatten(),
    
    # Camadas densas
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),  # Regularização para evitar overfitting
    
    # Camada de saída (10 classes - dígitos 0-9)
    keras.layers.Dense(10, activation='softmax')
])

# Visualizar a arquitetura do modelo
model_cnn.summary()

# Passo 3: Compilar e Treinar
# Compilar o modelo
model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks para melhor treinamento
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001
    )
]

# Treinar o modelo
history_cnn = model_cnn.fit(
    x_train_cnn, y_train,
    batch_size=128,
    epochs=30,
    validation_data=(x_test_cnn, y_test),
    callbacks=callbacks,
    verbose=1
)

# Passo 4: Avaliar e Analisar
# Avaliar o modelo no conjunto de teste
test_loss_cnn, test_accuracy_cnn = model_cnn.evaluate(x_test_cnn, y_test, verbose=0)
print(f"\n=== RESULTADOS DA CNN ===")
print(f"Acurácia no conjunto de teste: {test_accuracy_cnn:.4f}")
print(f"Perda no conjunto de teste: {test_loss_cnn:.4f}")

# Plotar histórico de treinamento
def plot_training_history(history):
    plt.figure(figsize=(15, 5))
    
    # Gráfico de acurácia
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Treinamento', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validação', linewidth=2)
    plt.title('Acurácia do Modelo CNN', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de perda
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Treinamento', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validação', linewidth=2)
    plt.title('Perda do Modelo CNN', fontsize=14)
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico da taxa de aprendizado
    if 'lr' in history.history:
        plt.subplot(1, 3, 3)
        plt.plot(history.history['lr'], linewidth=2, color='purple')
        plt.title('Taxa de Aprendizado', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_training_history(history_cnn)

# Fazer previsões
predictions_cnn = model_cnn.predict(x_test_cnn)
predicted_classes_cnn = np.argmax(predictions_cnn, axis=1)

# Matriz de confusão
def plot_confusion_matrix_cm(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão - CNN', fontsize=16)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()

plot_confusion_matrix_cm(y_test, predicted_classes_cnn, range(10))

# Relatório de classificação detalhado
print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y_test, predicted_classes_cnn))

# Visualizar previsões corretas e incorretas
def visualize_predictions(images, true_labels, predicted_labels, probabilities, num_examples=10):
    # Encontrar exemplos corretos e incorretos
    correct_indices = np.where(true_labels == predicted_labels)[0]
    incorrect_indices = np.where(true_labels != predicted_labels)[0]
    
    print(f"Total de acertos: {len(correct_indices)}/{len(true_labels)}")
    print(f"Total de erros: {len(incorrect_indices)}/{len(true_labels)}")
    
    # Plotar alguns exemplos corretos
    plt.figure(figsize=(15, 6))
    plt.suptitle('Exemplos de Previsões Corretas', fontsize=16)
    
    for i in range(min(5, len(correct_indices))):
        idx = correct_indices[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        confidence = probabilities[idx][predicted_labels[idx]]
        plt.title(f'Verd: {true_labels[idx]}, Pred: {predicted_labels[idx]}\nConf: {confidence:.4f}', 
                 color='green', fontsize=10)
        plt.axis('off')
    
    # Plotar alguns exemplos incorretos
    for i in range(min(5, len(incorrect_indices))):
        idx = incorrect_indices[i]
        plt.subplot(2, 5, i + 6)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        confidence = probabilities[idx][predicted_labels[idx]]
        plt.title(f'Verd: {true_labels[idx]}, Pred: {predicted_labels[idx]}\nConf: {confidence:.4f}', 
                 color='red', fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_predictions(x_test_cnn, y_test, predicted_classes_cnn, predictions_cnn)
