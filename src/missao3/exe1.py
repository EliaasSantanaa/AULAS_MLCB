import time
posicao_agente = 0
objetivo = 6
recompensa_total = 0
for passo in range(10):
     print(f"Passo {passo + 1}: Posição atual = {posicao_agente}")
     # TODO 1: Atualize a 'posicao_agente' para que ele avance 1 passo.
     posicao_agente += 1

     # TODO 2: Verifique se o agente alcançou o 'objetivo'.
     if posicao_agente == objetivo:
         print(" >> Objetivo alcançado!")
         recompensa_total += 10
         break
     else:
         # Se não chegou, ele perde 1 ponto de 'recompensa_total' pelo esforço.
         recompensa_total -= 1
         time.sleep(0.5)
         print(f"Simulação finalizada! Recompensa total: {recompensa_total}") # Resultado esperado: 5