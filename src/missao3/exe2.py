
import numpy as np
import time

# Exercício 2: O Agente Indeciso
# Cenário: Agora o agente pode se mover para 'esquerda' ou 'direita' em um corredor de 10 posições (0 a 9). Ele não pode atravessar as paredes (posições < 0 ou > 9).
# Sua Missão: Implementar a lógica de movimento para ambas as direções e garantir que o agente permaneça dentro dos limites do corredor.
print("\n--- Exercício 2: O Agente Indeciso ---")
posicao_agente = 5
objetivo = 9
recompensa_total = 0


for passo in range(15):
    acao = np.random.choice(['esquerda', 'direita'])
    print(f"Passo {passo + 1}: Posição={posicao_agente}, Ação='{acao}'")

    # Movimento
    if acao == 'direita':
        posicao_agente += 1
    elif acao == 'esquerda':
        posicao_agente -= 1

    # Limites
    posicao_agente = max(0, min(9, posicao_agente))

    # Recompensa
    if posicao_agente == objetivo:
        recompensa_total += 20
        print("O agente chegou ao objetivo!")
        break
    else:
        recompensa_total -= 1

    time.sleep(0.5)

print(f"Simulação finalizada! Posição final: {posicao_agente}, Recompensa: {recompensa_total}")

