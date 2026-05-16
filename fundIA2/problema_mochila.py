from algoritmo_genetico import *

quantidade_itens, capacidade, itens = carregar_dados_mochila("mochila.txt")

maior_razao = max(item['valor'] / item['peso'] for item in itens)
fator_penalidade = maior_razao + 1

melhor_global, historico = algoritimo_genetico(500, 100, quantidade_itens, capacidade, itens, fator_penalidade)

print(melhor_global)