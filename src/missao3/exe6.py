
"""
Exercício 6: O Vento Traiçoeiro
Cenário: O ambiente agora é estocástico (imprevisível). Há uma chance de 30% de que a ação do agente falhe por causa de um "vento forte". Se a ação falhar, o agente não se move, mas ainda paga o custo de -1 ponto.
Sua Missão: Introduzir a aleatoriedade do ambiente na lógica de transição de estado.
"""
import numpy as np
import time
print("\n--- Exercício 6: O Vento Traiçoeiro ---")
posicao_agente = 0
objetivo = 9
recompensa_total = 0


for passo in range(20):
    acao = np.random.choice(['esquerda', 'direita'])
    print(f"Passo {passo + 1}: Posição={posicao_agente}, Tentando ir para '{acao}'")

    # Simula o vento
    vento_soprou = np.random.rand() < 0.3

    if not vento_soprou:
        if acao == 'direita':
            posicao_agente += 1
        else:
            posicao_agente -= 1
        posicao_agente = np.clip(posicao_agente, 0, 9)
    else:
        print("Vento forte! O agente não se moveu.")

    # Recompensa
    if posicao_agente == objetivo:
        recompensa_total += 50
        print("O agente chegou ao objetivo!")
        break
    else:
        recompensa_total -= 1

    time.sleep(0.5)

print(f"Simulação finalizada! Posição final: {posicao_agente}, Recompensa: {recompensa_total}")