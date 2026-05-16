"""
Evolução Diferencial (ED) para minimização da função esfera de 10 variáveis:
    f(x) = sum_{i=1}^{10} x_i^2,   -5 <= x_i <= 5

Estratégia: DE/rand/1/bin (mutação rand/1 + recombinação binomial)
Seleção: elitista 1-para-1 (greedy)
"""

import numpy as np


# ----------------------------------------------------------------------
# Função objetivo
# ----------------------------------------------------------------------
def funcao_esfera(x: np.ndarray) -> float:
    """f(x) = soma dos quadrados das componentes (mínimo global em 0)."""
    return float(np.sum(x ** 2))


# ----------------------------------------------------------------------
# Algoritmo de Evolução Diferencial
# ----------------------------------------------------------------------
def differentialEvolution(
    funcao_objetivo=funcao_esfera,
    dim: int = 10,
    limites: tuple = (-5.0, 5.0),
    tam_populacao: int = 30,
    F: float = 0.5,
    CR: float = 0.9,
    max_geracoes: int = 1000,
    tolerancia: float = 1e-6,
    semente: int | None = None,
):
    """
    Executa o algoritmo de Evolução Diferencial (DE/rand/1/bin).

    Parâmetros
    ----------
    funcao_objetivo : callable
        Função a ser minimizada.
    dim : int
        Dimensionalidade do vetor de busca (10 para o problema-esfera).
    limites : (min, max)
        Limites inferior e superior das variáveis.
    tam_populacao : int
        Número de indivíduos por geração (NP). Recomenda-se NP >= 4.
    F : float
        Fator de escala da mutação diferencial (tipicamente em [0, 2]).
    CR : float
        Probabilidade de recombinação binomial (em [0, 1]).
    max_geracoes : int
        Número máximo de gerações.
    tolerancia : float
        Critério de parada: para quando o melhor fitness <= tolerancia.
    semente : int | None
        Semente do gerador aleatório (p/ reprodutibilidade).

    Retorna
    -------
    dict com:
        melhor_solucao : np.ndarray   – melhor vetor encontrado
        melhor_fitness : float        – f(melhor_solucao)
        geracoes       : int          – nº de gerações executadas
        historico      : list[float]  – melhor fitness a cada geração
        convergiu      : bool         – atingiu tolerancia?
    """
    rng = np.random.default_rng(semente)
    li, ls = limites

    # População inicial uniforme nos limites
    populacao = rng.uniform(li, ls, size=(tam_populacao, dim))
    fitness = np.array([funcao_objetivo(ind) for ind in populacao])

    historico = [float(fitness.min())]
    convergiu = False
    geracao_final = 0

    for geracao in range(1, max_geracoes + 1):
        for i in range(tam_populacao):
            # ---------- 1. MUTAÇÃO (DE/rand/1) -------------------------
            # escolher 3 índices distintos diferentes de i
            indices = list(range(tam_populacao))
            indices.remove(i)
            r1, r2, r3 = rng.choice(indices, size=3, replace=False)
            vetor_mutante = populacao[r1] + F * (populacao[r2] - populacao[r3])

            # garantir que o mutante respeite os limites (clipping)
            vetor_mutante = np.clip(vetor_mutante, li, ls)

            # ---------- 2. RECOMBINAÇÃO BINOMIAL -----------------------
            vetor_alvo = populacao[i]
            j_rand = rng.integers(0, dim)            # ao menos 1 gene do mutante
            mascara = rng.random(dim) < CR
            mascara[j_rand] = True
            vetor_teste = np.where(mascara, vetor_mutante, vetor_alvo)

            # ---------- 3. SELEÇÃO (1-vs-1, elitista) ------------------
            f_teste = funcao_objetivo(vetor_teste)
            if f_teste <= fitness[i]:
                populacao[i] = vetor_teste
                fitness[i] = f_teste

        melhor_atual = float(fitness.min())
        historico.append(melhor_atual)
        geracao_final = geracao

        # critério de parada
        if melhor_atual <= tolerancia:
            convergiu = True
            break

    idx_melhor = int(np.argmin(fitness))
    return {
        "melhor_solucao": populacao[idx_melhor].copy(),
        "melhor_fitness": float(fitness[idx_melhor]),
        "geracoes": geracao_final,
        "historico": historico,
        "convergiu": convergiu,
    }


# ----------------------------------------------------------------------
# Execução de demonstração
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    resultado = differentialEvolution(
        tam_populacao=30, F=0.5, CR=0.9,
        max_geracoes=1000, tolerancia=1e-6, semente=42,
    )
    print(f"Convergiu      : {resultado['convergiu']}")
    print(f"Gerações       : {resultado['geracoes']}")
    print(f"Melhor fitness : {resultado['melhor_fitness']:.3e}")
    print(f"Melhor x       : {resultado['melhor_solucao']}")