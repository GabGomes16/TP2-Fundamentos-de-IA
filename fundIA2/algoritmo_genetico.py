import random
import matplotlib.pyplot as plt
import numpy as np

def carregar_dados_mochila(caminho_arquivo):
    # O bloco 'with' garante que o arquivo será fechado automaticamente após a leitura
    with open(caminho_arquivo, 'r') as arquivo:
        linhas = arquivo.readlines()
        num_itens = int(linhas[0].strip())
        capacidade = int(linhas[1].strip())
        itens = []
        for i in range(2, len(linhas)):

            valores = linhas[i].strip().split()
            if len(valores) == 2:
                valor_item = int(valores[0])
                peso_item = int(valores[1])
                
                id_item = i - 2 
                
                itens.append({
                    'id': id_item,
                    'valor': valor_item,
                    'peso': peso_item
                })
                
    return num_itens, capacidade, itens

def gera_populacao(tam_populacao, num_itens):
    return [[random.choice([0, 1]) for _ in range(num_itens)] for _ in range(tam_populacao)]

def funcao_objetivo(cromossomo, itens):
    lucro = 0
    for gene, item in zip(cromossomo, itens):
        lucro +=  item['valor'] * gene

    return lucro

def adequacao(cromossomo, itens, capacidade, fator_penalidade):
    lucro_total = sum(item['valor'] * gene for gene, item in zip(cromossomo, itens))
    peso_total = sum(item['peso'] * gene for gene, item in zip(cromossomo, itens))
            
    if peso_total <= capacidade:
        return lucro_total
    
    else:
        peso_excedente = peso_total - capacidade
        
        multa = peso_excedente * fator_penalidade
        fitness_final = lucro_total - multa
        
        return max(1, fitness_final)

def selecao(populacao, valores_fitness, tamanho_torneio=3):
    pais_selecionados = []
    tamanho_populacao = len(populacao)
    
    for _ in range(tamanho_populacao):
        indices_competidores = random.sample(range(tamanho_populacao), tamanho_torneio)
        
        indice_vencedor = max(indices_competidores, key=lambda idx: valores_fitness[idx])
        
        pais_selecionados.append(populacao[indice_vencedor])
        
    return pais_selecionados

quantidade_itens, capacidade, itens = carregar_dados_mochila("mochila.txt")

maior_razao = max(item['valor'] / item['peso'] for item in itens)
fator_penalidade = maior_razao + 1

populacao = gera_populacao(100, quantidade_itens)
valores_fitness = []
for cromossomo in populacao:
    valores_fitness.append(adequacao(cromossomo, itens, capacidade, fator_penalidade))


print(selecao(populacao, valores_fitness))

# print(fator_penalidade)
# print(f"Dados do último item: {meus_dados['itens'][-1]}")
# teste = gera_populacao(4,2)
# print(teste)