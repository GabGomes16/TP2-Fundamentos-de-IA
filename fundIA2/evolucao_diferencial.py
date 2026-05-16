import numpy as np

def funcao_objetivo(individuo):
    return np.sum((np.square(individuo)))

def popular(tamanho_populacao, numero_variaveis, valor_min, valor_max):
    populacao = np.random.uniform(valor_min, valor_max, (tamanho_populacao, numero_variaveis))
    return populacao

def mutacao_diferencial(populacao_original, fator_escala = 0.8):

    tamanho_populacao,num_variaveis = populacao_original.shape

    populacao_mutante = np.zeros((tamanho_populacao, num_variaveis))

    for i in range(tamanho_populacao):
        indices_possiveis = [idx for idx in range(tamanho_populacao) if idx != i]

        sorteados = np.random.choice(indices_possiveis, size = 2, replace = False)

        gamma = populacao_original[sorteados[0]]
        beta = populacao_original[sorteados[1]]

        populacao_mutante[i] = populacao_original[i] + fator_escala * (beta - gamma)

    return populacao_mutante

def cruzamento(populacao_original, populacao_mutante, cr):

    tamanho_populacao,num_variaveis = populacao_original.shape

    populacao_teste = np.zeros((tamanho_populacao, num_variaveis))
    for i in range(tamanho_populacao):

        indice_garantido = np.random.randint(0, num_variaveis)
        vetor_r = np.random.rand(num_variaveis)
        mascara = vetor_r <= cr
        mascara[indice_garantido] = True

        populacao_teste[i] = np.where(mascara, populacao_mutante[i], populacao_original[i])

    return populacao_teste

def selecao(populacao_original, populacao_teste):
    tamanho_populacao, num_variaveis = populacao_original.shape

    populacao_nova = np.zeros((tamanho_populacao, num_variaveis))
    for i in range(tamanho_populacao):
        fit_pop_original = funcao_objetivo(populacao_original[i])
        fit_pop_teste = funcao_objetivo(populacao_teste[i])

        if fit_pop_teste <= fit_pop_original:
            populacao_nova[i] = populacao_teste[i]
        else: 
            populacao_nova[i] = populacao_original[i]

    return populacao_nova