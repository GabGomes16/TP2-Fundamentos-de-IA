"""
Experimentos comparativos:
  - Evolução Diferencial (ED) sobre f(x) = sum x_i^2  (10 variáveis)
  - Algoritmo Genético  (AG) sobre o Problema da Mochila 0-1

Para cada configuração de parâmetros, executa N_RUNS vezes (com sementes
diferentes) e calcula média e desvio padrão das métricas relevantes.

Saídas:
  - resultados_ed.csv
  - resultados_ag.csv
  - tabelas formatadas no console
"""

import time
import csv
import numpy as np

from differentialEvolution import differentialEvolution
from geneticAlgorithms import geneticAlgorithms, read_instance


# ----------------------------------------------------------------------
# Configurações globais
# ----------------------------------------------------------------------
N_RUNS = 30                 # nº de execuções por configuração
SEMENTE_BASE = 1000         # sementes diferentes p/ cada execução

# ----------------------------------------------------------------------
# Experimentos – Evolução Diferencial
# ----------------------------------------------------------------------
def experimentos_ed():
    """Roda grid de configurações para a ED."""
    print("\n" + "=" * 78)
    print("EXPERIMENTOS – EVOLUÇÃO DIFERENCIAL (função esfera, 10D)")
    print("=" * 78)

    configs = []
    for NP in [20, 50]:
        for F in [0.3, 0.5, 0.8]:
            for CR in [0.3, 0.7, 0.9]:
                configs.append({"NP": NP, "F": F, "CR": CR})

    resultados = []
    inicio_total = time.time()

    for cfg in configs:
        fits, geracoes, sucessos = [], [], 0
        t0 = time.time()
        for r in range(N_RUNS):
            res = differentialEvolution(
                tam_populacao=cfg["NP"],
                F=cfg["F"], CR=cfg["CR"],
                max_geracoes=2000,
                tolerancia=1e-6,
                semente=SEMENTE_BASE + r,
            )
            fits.append(res["melhor_fitness"])
            geracoes.append(res["geracoes"])
            sucessos += int(res["convergiu"])
        dt = time.time() - t0

        linha = {
            "NP": cfg["NP"], "F": cfg["F"], "CR": cfg["CR"],
            "fit_med":  float(np.mean(fits)),
            "fit_std":  float(np.std(fits)),
            "ger_med":  float(np.mean(geracoes)),
            "ger_std":  float(np.std(geracoes)),
            "taxa_suc": sucessos / N_RUNS,
            "tempo_s":  dt,
        }
        resultados.append(linha)
        print(f"  NP={cfg['NP']:3d}  F={cfg['F']:.1f}  CR={cfg['CR']:.1f}  "
              f"| fit médio={linha['fit_med']:.2e} (±{linha['fit_std']:.1e})  "
              f"| gerações={linha['ger_med']:6.1f} (±{linha['ger_std']:5.1f})  "
              f"| sucessos={sucessos}/{N_RUNS}  | t={dt:.1f}s")

    print(f"\nTempo total ED: {time.time() - inicio_total:.1f}s")

    with open("resultados_ed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
        w.writeheader()
        w.writerows(resultados)
    print("Resultados salvos em resultados_ed.csv")
    return resultados


# ----------------------------------------------------------------------
# Experimentos – Algoritmo Genético (mochila)
# ----------------------------------------------------------------------
def experimentos_ag():
    """Roda grid de configurações para o AG na instância de mochila.txt."""
    print("\n" + "=" * 78)
    print("EXPERIMENTOS – ALGORITMO GENÉTICO (mochila 0-1)")
    print("=" * 78)

    n, C, valores, pesos = read_instance("mochila.txt")
    print(f"Instância: n={n}, C={C}\n")

    configs = []
    for NP in [50, 100]:
        for pc in [0.7, 0.9]:
            for pm in [0.01, 0.05, 0.10]:
                configs.append({"NP": NP, "pc": pc, "pm": pm})

    resultados = []
    inicio_total = time.time()

    for cfg in configs:
        valores_finais, geracoes_melhor, viaveis = [], [], 0
        t0 = time.time()
        for r in range(N_RUNS):
            res = geneticAlgorithms(
                valores, pesos, C,
                tam_populacao=cfg["NP"],
                prob_cruzamento=cfg["pc"],
                prob_mutacao=cfg["pm"],
                tam_torneio=3,
                max_geracoes=500,
                geracoes_sem_melhora=100,
                elitismo=True,
                semente=SEMENTE_BASE + r,
            )
            valores_finais.append(res["melhor_valor"])
            geracoes_melhor.append(res["geracao_melhor"])
            viaveis += int(res["viavel"])
        dt = time.time() - t0

        linha = {
            "NP": cfg["NP"], "pc": cfg["pc"], "pm": cfg["pm"],
            "valor_med":  float(np.mean(valores_finais)),
            "valor_std":  float(np.std(valores_finais)),
            "valor_max":  int(np.max(valores_finais)),
            "ger_melhor_med": float(np.mean(geracoes_melhor)),
            "ger_melhor_std": float(np.std(geracoes_melhor)),
            "viaveis":    viaveis,
            "tempo_s":    dt,
        }
        resultados.append(linha)
        print(f"  NP={cfg['NP']:3d}  pc={cfg['pc']:.2f}  pm={cfg['pm']:.2f}  "
              f"| valor médio={linha['valor_med']:8.1f} (±{linha['valor_std']:6.1f})  "
              f"| max={linha['valor_max']:5d}  "
              f"| ger.melhor={linha['ger_melhor_med']:5.1f}  "
              f"| viáveis={viaveis}/{N_RUNS}  | t={dt:.1f}s")

    print(f"\nTempo total AG: {time.time() - inicio_total:.1f}s")

    with open("resultados_ag.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
        w.writeheader()
        w.writerows(resultados)
    print("Resultados salvos em resultados_ag.csv")
    return resultados


# ----------------------------------------------------------------------
if __name__ == "__main__":
    res_ed = experimentos_ed()
    res_ag = experimentos_ag()