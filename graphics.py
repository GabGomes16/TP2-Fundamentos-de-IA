"""
Gera gráficos de convergência para uma seleção de configurações,
ilustrando o comportamento típico de cada algoritmo.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from differentialEvolution import differentialEvolution
from geneticAlgorithms import geneticAlgorithms, read_instance


# ----------------------------------------------------------------------
# 1) Convergência da ED – efeito do fator F (com CR=0.9, NP=50)
# ----------------------------------------------------------------------
def grafico_ed():
    fig, ax = plt.subplots(figsize=(8, 5))
    Fs = [0.3, 0.5, 0.8]
    cores = ["#1f77b4", "#2ca02c", "#d62728"]
    for F, cor in zip(Fs, cores):
        # média de 10 execuções
        historicos = []
        max_len = 0
        for s in range(10):
            res = differentialEvolution(
                tam_populacao=50, F=F, CR=0.9,
                max_geracoes=600, tolerancia=1e-10,
                semente=2000 + s,
            )
            historicos.append(res["historico"])
            max_len = max(max_len, len(res["historico"]))
        # padroniza tamanhos repetindo último valor
        for i, h in enumerate(historicos):
            if len(h) < max_len:
                historicos[i] = h + [h[-1]] * (max_len - len(h))
        media = np.mean(historicos, axis=0)
        ax.semilogy(media, label=f"F = {F}", color=cor, linewidth=2)
    ax.set_xlabel("Geração")
    ax.set_ylabel("Melhor fitness (escala log)")
    ax.set_title("Evolução Diferencial – convergência média (10 runs, NP=50, CR=0.9)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergencia_ed.png", dpi=130)
    plt.close()
    print("Salvo: convergencia_ed.png")


# ----------------------------------------------------------------------
# 2) Convergência do AG – efeito da taxa de mutação
# ----------------------------------------------------------------------
def grafico_ag():
    n, C, valores, pesos = read_instance("mochila.txt")
    fig, ax = plt.subplots(figsize=(8, 5))
    pms = [0.01, 0.05, 0.10]
    cores = ["#1f77b4", "#2ca02c", "#d62728"]
    OTIMO = 15768
    for pm, cor in zip(pms, cores):
        historicos = []
        max_len = 0
        for s in range(10):
            res = geneticAlgorithms(
                valores, pesos, C,
                tam_populacao=100, prob_cruzamento=0.9, prob_mutacao=pm,
                max_geracoes=400, geracoes_sem_melhora=400,  # roda completo
                semente=2000 + s,
            )
            historicos.append(res["historico"])
            max_len = max(max_len, len(res["historico"]))
        for i, h in enumerate(historicos):
            if len(h) < max_len:
                historicos[i] = h + [h[-1]] * (max_len - len(h))
        media = np.mean(historicos, axis=0)
        ax.plot(media, label=f"pm = {pm}", color=cor, linewidth=2)
    ax.axhline(OTIMO, color="black", linestyle="--", linewidth=1, label=f"Ótimo = {OTIMO}")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Melhor valor da mochila (lucro)")
    ax.set_title("Algoritmo Genético – convergência média (10 runs, NP=100, pc=0.9)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergencia_ag.png", dpi=130)
    plt.close()
    print("Salvo: convergencia_ag.png")


if __name__ == "__main__":
    grafico_ed()
    grafico_ag()