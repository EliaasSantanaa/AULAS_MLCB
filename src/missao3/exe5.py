
"""
Exercício 5: O Robô com Bateria
Cenário: O agente agora é um robô com uma bateria que começa em 100%. Cada passo consome 10% da bateria. Se a bateria chegar a 0, o episódio acaba.
Sua Missão: Adicionar e gerenciar um segundo componente do estado: a energia.
"""
import numpy as np
import time
print("\n--- Exercício 5: O Robô com Bateria ---")
posicao_agente = 0
objetivo = 9
recompensa_total = 0
bateria = 100 # Novo estado!


for passo in range(20):
    print(f"Passo {passo + 1}: Posição={posicao_agente}, Bateria={bateria}%")

    acao = np.random.choice(['esquerda', 'direita'])
    if acao == 'direita':
        posicao_agente += 1
    else:
        posicao_agente -= 1
    posicao_agente = np.clip(posicao_agente, 0, 9)

    # Cada passo consome 10% da bateria
    bateria -= 10

    # Condições de término
    if posicao_agente == objetivo:
        recompensa_total += 100
        print("O agente chegou ao objetivo!")
        break
    elif bateria <= 0:
        recompensa_total -= 30
        print("A bateria acabou!")
        break
    # Senão, recompensa do passo é 0 (penalidade já está no gasto da bateria)

    time.sleep(0.5)

print(f"Simulação finalizada! Posição: {posicao_agente}, Bateria: {bateria}%, Recompensa: {recompensa_total}")
