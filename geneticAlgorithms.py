"""
Algoritmo Genético (AG) para o Problema da Mochila 0-1.

Modelo:
    maximizar   sum_{i=1..n} p_i * x_i
    sujeito a   sum_{i=1..n} w_i * x_i <= C
                x_i ∈ {0, 1}

- Cromossomo: vetor binário de tamanho n.
- Tratamento de inviabilidade: penalidade quadrática proporcional ao
  excesso de peso (mais agressiva que linear, evita que cromossomos
  inviáveis com alto valor sobrevivam).
- Seleção: torneio binário (k=3 por padrão).
- Cruzamento: 2 pontos (mais diverso que 1 ponto, mantém boa exploração).
- Mutação: bit-flip independente em cada gene.
- Elitismo: o melhor indivíduo é sempre copiado para a geração seguinte.
"""

import numpy as np
from pathlib import Path


# ----------------------------------------------------------------------
# Leitura da instância
# ----------------------------------------------------------------------
def read_instance(caminho: str | Path):
    """Lê arquivo no formato:  n \\n C \\n  p_i w_i  (um por linha)."""
    with open(caminho, "r") as f:
        tokens = f.read().split()
    n = int(tokens[0])
    C = int(tokens[1])
    valores = np.empty(n, dtype=np.int64)
    pesos = np.empty(n, dtype=np.int64)
    for i in range(n):
        valores[i] = int(tokens[2 + 2 * i])
        pesos[i] = int(tokens[3 + 2 * i])
    return n, C, valores, pesos


# ----------------------------------------------------------------------
# Aptidão (fitness) com penalidade
# ----------------------------------------------------------------------
def fitness(individuo: np.ndarray, valores, pesos, C):
    """Retorna (fitness, valor_total, peso_total).

    Penalidade quadrática para excedente de peso:
        fitness = valor_total - alpha * excesso^2
    onde alpha é grande o suficiente para que toda solução inviável
    perca para qualquer solução viável.
    """
    valor_total = int(np.dot(individuo, valores))
    peso_total = int(np.dot(individuo, pesos))
    if peso_total <= C:
        return float(valor_total), valor_total, peso_total
    excesso = peso_total - C
    # fator alpha grande: máximo valor possível dividido pela menor
    # diferença significativa, garante que viáveis sempre vençam
    alpha = float(valores.sum())
    penalizado = valor_total - alpha * (excesso ** 2)
    return float(penalizado), valor_total, peso_total


# ----------------------------------------------------------------------
# Operadores genéticos
# ----------------------------------------------------------------------
def selecao_torneio(populacao, fits, k, rng):
    """Torneio de tamanho k: retorna o índice do vencedor."""
    candidatos = rng.integers(0, len(populacao), size=k)
    melhor = candidatos[0]
    for c in candidatos[1:]:
        if fits[c] > fits[melhor]:
            melhor = c
    return int(melhor)


def cruzamento_dois_pontos(p1, p2, rng):
    """Cruzamento de 2 pontos: gera 2 filhos."""
    n = len(p1)
    if n < 2:
        return p1.copy(), p2.copy()
    a, b = sorted(rng.choice(n - 1, size=2, replace=False) + 1)
    f1 = np.concatenate([p1[:a], p2[a:b], p1[b:]])
    f2 = np.concatenate([p2[:a], p1[a:b], p2[b:]])
    return f1, f2


def mutacao_bitflip(individuo, taxa, rng):
    """Inverte cada bit com probabilidade `taxa`."""
    mascara = rng.random(len(individuo)) < taxa
    individuo[mascara] = 1 - individuo[mascara]
    return individuo


# ----------------------------------------------------------------------
# Algoritmo Genético principal
# ----------------------------------------------------------------------
def geneticAlgorithms(
    valores: np.ndarray,
    pesos: np.ndarray,
    C: int,
    tam_populacao: int = 100,
    prob_cruzamento: float = 0.9,
    prob_mutacao: float = 0.02,
    tam_torneio: int = 3,
    max_geracoes: int = 500,
    geracoes_sem_melhora: int = 100,
    elitismo: bool = True,
    semente: int | None = None,
):
    """
    Resolve o problema da mochila com AG.

    Parâmetros
    ----------
    valores, pesos : np.ndarray   – arrays de tamanho n
    C              : int          – capacidade da mochila
    tam_populacao  : int          – tamanho da população
    prob_cruzamento: float        – probabilidade de cruzamento (pc)
    prob_mutacao   : float        – probabilidade de mutação por bit (pm)
    tam_torneio    : int          – tamanho do torneio
    max_geracoes   : int          – limite de gerações
    geracoes_sem_melhora : int    – critério de parada por estagnação
    elitismo       : bool         – preserva o melhor indivíduo
    semente        : int | None   – reprodutibilidade

    Retorna
    -------
    dict com a melhor solução e estatísticas da evolução.
    """
    rng = np.random.default_rng(semente)
    n = len(valores)

    # População inicial: cada bit ativo com prob ~ C / soma(pesos), o
    # que enviesa a inicialização para soluções não-triviais e quase
    # sempre viáveis. Reduz o tempo gasto consertando lixo aleatório.
    p_inicial = min(0.5, C / float(pesos.sum()))
    populacao = (rng.random((tam_populacao, n)) < p_inicial).astype(np.int8)

    fits = np.empty(tam_populacao)
    valores_brutos = np.empty(tam_populacao, dtype=np.int64)
    pesos_brutos = np.empty(tam_populacao, dtype=np.int64)
    for i in range(tam_populacao):
        fits[i], valores_brutos[i], pesos_brutos[i] = fitness(
            populacao[i], valores, pesos, C
        )

    # Estatísticas de busca
    melhor_idx = int(np.argmax(fits))
    melhor_individuo = populacao[melhor_idx].copy()
    melhor_fit = fits[melhor_idx]
    melhor_valor = int(valores_brutos[melhor_idx]) if pesos_brutos[melhor_idx] <= C else 0
    melhor_peso = int(pesos_brutos[melhor_idx]) if pesos_brutos[melhor_idx] <= C else 0

    historico = [melhor_valor]
    estagnacao = 0
    geracao_final = 0
    geracao_melhor = 0

    for geracao in range(1, max_geracoes + 1):
        nova_pop = np.empty_like(populacao)
        idx = 0

        # Elitismo
        if elitismo:
            nova_pop[0] = populacao[int(np.argmax(fits))]
            idx = 1

        # Geração de descendentes
        while idx < tam_populacao:
            i1 = selecao_torneio(populacao, fits, tam_torneio, rng)
            i2 = selecao_torneio(populacao, fits, tam_torneio, rng)

            if rng.random() < prob_cruzamento:
                f1, f2 = cruzamento_dois_pontos(populacao[i1], populacao[i2], rng)
            else:
                f1, f2 = populacao[i1].copy(), populacao[i2].copy()

            f1 = mutacao_bitflip(f1, prob_mutacao, rng)
            f2 = mutacao_bitflip(f2, prob_mutacao, rng)

            nova_pop[idx] = f1
            idx += 1
            if idx < tam_populacao:
                nova_pop[idx] = f2
                idx += 1

        populacao = nova_pop
        for i in range(tam_populacao):
            fits[i], valores_brutos[i], pesos_brutos[i] = fitness(
                populacao[i], valores, pesos, C
            )

        # Atualiza melhor (apenas viáveis contam para "melhor valor")
        idx_top = int(np.argmax(fits))
        f_top = fits[idx_top]
        if f_top > melhor_fit:
            melhor_fit = f_top
            melhor_individuo = populacao[idx_top].copy()
            if pesos_brutos[idx_top] <= C:
                melhor_valor = int(valores_brutos[idx_top])
                melhor_peso = int(pesos_brutos[idx_top])
            geracao_melhor = geracao
            estagnacao = 0
        else:
            estagnacao += 1

        historico.append(melhor_valor)
        geracao_final = geracao

        if estagnacao >= geracoes_sem_melhora:
            break

    return {
        "melhor_individuo": melhor_individuo,
        "melhor_valor": melhor_valor,
        "melhor_peso": melhor_peso,
        "melhor_fitness": float(melhor_fit),
        "viavel": int(melhor_peso) <= int(C),
        "geracoes": geracao_final,
        "geracao_melhor": geracao_melhor,
        "historico": historico,
    }


# ----------------------------------------------------------------------
# Execução de demonstração
# ----------------------------------------------------------------------
if __name__ == "__main__":
    n, C, valores, pesos = read_instance("mochila.txt")
    print(f"Instância: n={n}, C={C}, soma_pesos={pesos.sum()}, "
          f"soma_valores={valores.sum()}")

    res = geneticAlgorithms(
        valores, pesos, C,
        tam_populacao=100, prob_cruzamento=0.9, prob_mutacao=0.02,
        tam_torneio=3, max_geracoes=500, geracoes_sem_melhora=100,
        semente=42,
    )
    print(f"Viável         : {res['viavel']}")
    print(f"Gerações       : {res['geracoes']} (melhor encontrado em {res['geracao_melhor']})")
    print(f"Valor (lucro)  : {res['melhor_valor']}")
    print(f"Peso usado     : {res['melhor_peso']} / {C}")
    print(f"Itens escolhidos: {int(res['melhor_individuo'].sum())} de {n}")