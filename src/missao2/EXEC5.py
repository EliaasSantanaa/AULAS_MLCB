import time

POSICAO_INICIAL = 0
POSICAO_COMIDA = 6
recompensa_total = 0

posicao_agente = POSICAO_INICIAL

print("--- Simulação do PERSONAGEM EXPLORADOR ---")
print(f"O agente começa na posição {posicao_agente} e quer chegar na comida na posição {POSICAO_COMIDA}.")
print("-" * 30)

for passo in range(12):
    print(f"Passo {passo + 1}:")
    
    # O agente pode escolher a ação (aqui sempre 'direita')
    acao_agente = 'direita'
    print(f"   -> Ação do Agente: '{acao_agente}'")

    # Atualiza a posição do agente
    if acao_agente == 'direita':
        posicao_agente += 1
    elif acao_agente == 'esquerda':
        posicao_agente -= 1

    # Calcula a recompensa
    if posicao_agente == POSICAO_COMIDA:
        recompensa_do_passo = 30
    else:
        recompensa_do_passo = -2

    recompensa_total += recompensa_do_passo

    print(f"   <- Resposta do Ambiente: Nova Posição={posicao_agente}, Recompensa={recompensa_do_passo}")

    if posicao_agente == POSICAO_COMIDA:
        print("\nO personagem encontrou a comida!")
        break

    time.sleep(1)

print("-" * 30)
print(f"Simulação Finalizada! Recompensa total acumulada: {recompensa_total}")