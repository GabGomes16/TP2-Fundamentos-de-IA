from evolucao_diferencial import *
import itertools

def executar_evolucao_diferencial(tamanho_populacao, num_variaveis, valor_min, valor_max, fator_escala, cr, max_geracoes):
    populacao = popular(tamanho_populacao, num_variaveis, valor_min, valor_max)
    
    # Define o que é "bom o suficiente" para parar de procurar
    tolerancia_convergencia = 1e-3 
    geracao_convergencia = max_geracoes # Assume o pior caso (não convergiu)

    for geracao in range(max_geracoes):
        
        pop_mutante = mutacao_diferencial(populacao, fator_escala)
        pop_teste = cruzamento(populacao, pop_mutante, cr)
        populacao = selecao(populacao, pop_teste)
        
        # Pega a nota do melhor indivíduo desta geração
        fitness_da_geracao = [funcao_objetivo(ind) for ind in populacao]
        melhor_atual = np.min(fitness_da_geracao)
        
        # Se chegou num valor excelente, anota a geração e para o loop
        if melhor_atual <= tolerancia_convergencia:
            geracao_convergencia = geracao + 1
            break # Interrompe as gerações para economizar processamento
            
    fitness_final = [funcao_objetivo(ind) for ind in populacao]
    melhor_indice = np.argmin(fitness_final) 

    return populacao[melhor_indice], fitness_final[melhor_indice], geracao_convergencia

def grid_search_estatistico(lista_populacao, lista_f, lista_cr, num_variaveis, valor_min, valor_max, max_geracoes, num_rodadas=30):
    
    combinacoes = list(itertools.product(lista_populacao, lista_f, lista_cr))
    print(f"Iniciando testes. Serão {len(combinacoes)} configurações x {num_rodadas} rodadas cada.\n")
    
    # Cabeçalho da Tabela
    print(f"| {'Pop':<4} | {'F':<4} | {'CR':<4} | {'Fit Médio':<12} | {'Fit Desvio':<12} | {'Gen Média':<10} | {'Gen Desvio':<10} |")
    print("-" * 81)

    resultados_finais = []

    for tamanho_pop, f, cr in combinacoes:
        
        historico_fitness = []
        historico_geracoes = []
        
        # Roda o algoritmo 30 vezes para a mesma configuração
        for rodada in range(num_rodadas):
            _, fit_final, gen_conv = executar_evolucao_diferencial(
                tamanho_pop, num_variaveis, valor_min, valor_max, f, cr, max_geracoes
            )
            historico_fitness.append(fit_final)
            historico_geracoes.append(gen_conv)
            
        # Calcula as estatísticas das 30 rodadas usando NumPy
        media_fit = np.mean(historico_fitness)
        std_fit = np.std(historico_fitness)
        media_gen = np.mean(historico_geracoes)
        std_gen = np.std(historico_geracoes)
        
        resultados_finais.append({
            'pop': tamanho_pop, 'f': f, 'cr': cr, 
            'fit_media': media_fit, 'gen_media': media_gen
        })
        
        # Imprime a linha da tabela formatada
        print(f"| {tamanho_pop:<4} | {f:<4.2f} | {cr:<4.2f} | {media_fit:<12.6f} | {std_fit:<12.6f} | {media_gen:<10.2f} | {std_gen:<10.2f} |")

    # Encontra a melhor configuração baseada na Média do Fitness
    melhor_config = min(resultados_finais, key=lambda x: x['fit_media'])
    
    print("-" * 81)
    print(f"\nA CONFIGURAÇÃO MAIS ESTÁVEL:")
    print(f"População: {melhor_config['pop']}, F: {melhor_config['f']}, CR: {melhor_config['cr']}")
    print(f"Isso atingiu um Fitness Médio de {melhor_config['fit_media']:.6f} em ~{melhor_config['gen_media']:.0f} gerações.")



VARS, V_MIN, V_MAX, GERACOES, RODADAS = 10, -5.0, 5.0, 200, 30

# As variações que você quer testar 
testes_populacao = [10] 
testes_fator_escala = [0.1, 0.5, 0.9]
testes_cr = [0.1, 0.5, 0.9]

grid_search_estatistico(testes_populacao, testes_fator_escala, testes_cr, VARS, V_MIN, V_MAX, GERACOES, RODADAS)