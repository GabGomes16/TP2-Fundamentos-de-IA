import random

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

def calcula_peso(cromossomo, itens):
    peso = 0
    for gene, item in zip(cromossomo, itens):
        peso +=  item['peso'] * gene

    return peso

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

def crossover(pai_a, pai_b):
    taxa_cruzamento = 0.8

    if random.random() > taxa_cruzamento:
        return pai_a, pai_b
        
    else:
        filho_a = []
        filho_b = []
        
        for index in range(len(pai_a)):
            
            chance_media = random.choice([0, 1])
            
            if chance_media == 0:
                filho_a.append(pai_a[index])
                filho_b.append(pai_b[index])
            else:
                filho_a.append(pai_b[index])
                filho_b.append(pai_a[index])

        return filho_a, filho_b

def mutacao(taxa_mutacao, filho):
    
    for gene in range(len(filho)):
        if random.random() < taxa_mutacao:
            
            filho[gene] = int(not filho[gene])
            
    return filho    

def algoritimo_genetico(num_geracoes, tamanho_populacao, quantidade_itens, capacidade, itens, fator_penalidade):
    populacao = gera_populacao(tamanho_populacao, quantidade_itens)
    lucro_maximo = sum(item['valor'] for item in itens)
    taxa_mutacao = 1 / quantidade_itens
    melhor_global = {'cromossomo': None, 'fitness': -1, "peso": -1}
    pior_global = {'cromossomo': None, 'fitness': lucro_maximo, "peso": capacidade}
    historico_fitness = []
    
    for geracao in range(num_geracoes):
        # 1. Avaliação
        valores_fitness = [adequacao(cromossomo, itens, capacidade, fator_penalidade) for cromossomo in populacao]

        # 2. Registro e Elitismo
        melhor_fitness = max(valores_fitness)
        indice_melhor = valores_fitness.index(melhor_fitness)

        pior_fitness = min(valores_fitness)
        indice_pior = valores_fitness.index(pior_fitness)

        historico_fitness.append(melhor_fitness)

        if melhor_fitness > melhor_global['fitness']:
            melhor_global['fitness'] = melhor_fitness
            melhor_global['cromossomo'] = populacao[indice_melhor].copy()
            melhor_global['peso'] = calcula_peso(populacao[indice_melhor], itens)
            
        if pior_fitness < pior_global['fitness']:
            # CORREÇÃO APLICADA AQUI
            pior_global['fitness'] = pior_fitness 
            pior_global['cromossomo'] = populacao[indice_pior].copy()
            pior_global['peso'] = calcula_peso(populacao[indice_pior], itens)
            
        # 3. Seleção
        pais_selecionados = selecao(populacao, valores_fitness)
        nova_populacao = []

        # 4. Reprodução (Crossover e Mutação em pares)
        for i in range(0, len(pais_selecionados), 2):
            pai_a = pais_selecionados[i]
            pai_b = pais_selecionados[i + 1]

            filho_a, filho_b = crossover(pai_a, pai_b)

            nova_populacao.append(mutacao(taxa_mutacao, filho_a))
            nova_populacao.append(mutacao(taxa_mutacao, filho_b))

        # 5. Avaliação dos Filhos e Substituição do Pior (Garantia do Elitismo)
        fitness_filhos = [adequacao(c, itens, capacidade, fator_penalidade) for c in nova_populacao]

        pior_fitness_filhos = min(fitness_filhos)
        indice_do_pior_filho = fitness_filhos.index(pior_fitness_filhos)
        
        nova_populacao[indice_do_pior_filho] = melhor_global['cromossomo'].copy()
        
        # 6. Atualização de Geração
        populacao = nova_populacao

        # 7. Critério de Parada
        if len(historico_fitness) > 50 and historico_fitness[-1] == historico_fitness[-51]:
            # print("Encontrou otimo local")
            break
    
    return melhor_global, pior_global, historico_fitness