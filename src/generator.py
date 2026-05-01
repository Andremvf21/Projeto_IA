"""
generator.py — Geração de dados sintéticos

Baseado em: Sihag et al. (2024), Expert Systems With Applications 252, 124106
Simula as 10 variáveis-alvo (fatores de risco para queda) descritas na Tabela 9
do artigo, com prevalências próximas às reportadas no paper.

Variáveis-alvo (Tabela 9 do paper):
    trMar     — Distúrbio de marcha          (prevalência: 82.5%)
    peurTom   — Medo de cair                 (prevalência: 75.6%)
    trEq      — Distúrbio de equilíbrio      (prevalência: 73.4%)
    sarcopen  — Fraqueza muscular            (prevalência: 62.0%)
    nbchu2    — Nº de quedas nos últimos 6m  (prevalência: 57.9%)
    demence   — Demência                     (prevalência: 42.1%)
    osteopor  — Osteoporose                  (prevalência: 33.2%)
    dep       — Depressão                    (prevalência: 27.8%)
    ADLlt5    — Perda de autonomia (ADL < 5) (prevalência: 22.9%)
    parkOuSP  — Parkinson ou síndrome P.     (prevalência: 17.1%)
"""

import os
import numpy as np
import pandas as pd


def generate_data(n: int = 1745, missing_rate: float = 0.05,
                  seed: int = 42) -> pd.DataFrame:
    """
    Gera base sintética com n pacientes.

    Parâmetros
    ----------
    n            : número de pacientes (paper usa 1745 após limpeza)
    missing_rate : proporção de NaN introduzidos em variáveis auxiliares
                   (o paper reporta até 70% de missing em algumas variáveis;
                    use um valor baixo para testes simples)
    seed         : semente para reprodutibilidade
    """
    rng = np.random.default_rng(seed)  # gerador moderno — mais seguro que seed global

    # ------------------------------------------------------------------
    # 1. Variáveis sociodemográficas e de saúde (independentes)
    # ------------------------------------------------------------------
    age     = rng.integers(65, 96, n)             # 65–95 anos (paper: média 81)
    sexe    = rng.integers(0, 2, n)               # 0=feminino, 1=masculino (paper: 72% F)
    trVision = rng.choice([0, 1], n, p=[0.75, 0.25])  # distúrbio visual

    # Variáveis do Markov Blanket que influenciam os alvos
    # (estimadas a partir das figuras 5 e 6 do paper)
    # Probabilidades calibradas para atingir as prevalências da Tabela 9
    myopat   = rng.choice([0, 1], n, p=[0.65, 0.35])   # miopatia
    evitsort  = rng.choice([0, 1], n, p=[0.55, 0.45])   # evita sair de casa
    montDesc  = rng.choice([0, 1], n, p=[0.55, 0.45])   # dificuldade em escadas
    sortSeul  = rng.choice([0, 1], n, p=[0.65, 0.35])   # sai sozinho
    TUGgt20   = rng.choice([0, 1], n, p=[0.60, 0.40])   # TUG > 20s (mobilidade)
    agonDopa  = rng.choice([0, 1], n, p=[0.62, 0.38])   # agonista dopaminérgico
    akines    = rng.choice([0, 1], n, p=[0.68, 0.32])   # acinesia
    BMIlt19   = rng.choice([0, 1], n, p=[0.80, 0.20])   # IMC < 19 (baixo peso)
    htNivEtu  = rng.choice([0, 1], n, p=[0.55, 0.45])   # nível educacional
    a1medSed  = rng.choice([0, 1], n, p=[0.60, 0.40])   # sedativo prescrito
    arth      = rng.choice([0, 1], n, p=[0.60, 0.40])   # artrite
    gt2psych  = rng.choice([0, 1], n, p=[0.65, 0.35])   # > 2 psicotrópicos
    a1AntiDep = rng.choice([0, 1], n, p=[0.72, 0.28])   # antidepressivo
    conduit   = rng.choice([0, 1], n, p=[0.75, 0.25])   # dirigia antes
    LSAi4     = rng.choice([0, 1], n, p=[0.62, 0.38])   # escore LSA < 4
    maisRet   = rng.choice([0, 1], n, p=[0.75, 0.25])   # vive em casa de repouso
    traAnOst  = rng.choice([0, 1], n, p=[0.75, 0.25])   # tratamento antiosteoporose
    syndCer   = rng.choice([0, 1], n, p=[0.85, 0.15])   # síndrome cerebelar

    # ------------------------------------------------------------------
    # 2. Fatores de risco — alvos do paper (Tabela 9)
    #    Cada um é influenciado por variáveis do seu Markov Blanket
    #    (Figuras 5 e 6 do paper). A lógica causal é simplificada para
    #    dados sintéticos, mas preserva a direção das relações.
    # ------------------------------------------------------------------

    # trEq: distúrbio de equilíbrio (73.4%)
    # Causas: trVision, age > 80
    p_trEq = np.where(
        (trVision == 1) | (age > 80), 0.85, 0.55
    )
    trEq = rng.binomial(1, p_trEq)

    # trMar: distúrbio de marcha (82.5%)
    # Causas: trEq, myopat, TUGgt20
    p_trMar = 0.65 + 0.15 * trEq + 0.08 * myopat + 0.08 * TUGgt20
    p_trMar = np.clip(p_trMar, 0.05, 0.97)
    trMar = rng.binomial(1, p_trMar)

    # peurTom: medo de cair (75.6%)
    # Causas: evitsort, sortSeul (inverso), trMar
    p_peurTom = 0.58 + 0.10 * evitsort + 0.10 * (1 - sortSeul) + 0.08 * trMar
    p_peurTom = np.clip(p_peurTom, 0.05, 0.97)
    peurTom = rng.binomial(1, p_peurTom)

    # sarcopen: fraqueza muscular/sarcopenia (62%)
    # Causas: myopat, TUGgt20, BMIlt19, trMar
    p_sarcopen = 0.36 + 0.22 * myopat + 0.14 * TUGgt20 + 0.10 * BMIlt19 + 0.08 * trMar
    p_sarcopen = np.clip(p_sarcopen, 0.05, 0.97)
    sarcopen = rng.binomial(1, p_sarcopen)

    # nbchu2: número de quedas nos últimos 6 meses (57.9%)
    # Causas: trMar, trEq, TUGgt20
    p_nbchu2 = 0.30 + 0.15 * trMar + 0.12 * trEq + 0.10 * TUGgt20
    p_nbchu2 = np.clip(p_nbchu2, 0.05, 0.97)
    nbchu2 = rng.binomial(1, p_nbchu2)

    # parkOuSP: Parkinson ou síndrome parkinsoniana (17.1%)
    # Causas: akines, agonDopa
    p_parkOuSP = 0.10 + 0.15 * akines + 0.08 * agonDopa
    p_parkOuSP = np.clip(p_parkOuSP, 0.02, 0.97)
    parkOuSP = rng.binomial(1, p_parkOuSP)

    # demence: demência (42.1%)
    # Causas: age, htNivEtu, parkOuSP
    p_demence = 0.25 + 0.15 * (age > 82).astype(int) + 0.08 * htNivEtu + 0.05 * parkOuSP
    p_demence = np.clip(p_demence, 0.05, 0.97)
    demence = rng.binomial(1, p_demence)

    # osteopor: osteoporose (33.2%)
    # Causas: sexe (F), BMIlt19, traAnOst
    p_osteopor = 0.22 + 0.12 * (1 - sexe) + 0.08 * BMIlt19 + 0.05 * traAnOst
    p_osteopor = np.clip(p_osteopor, 0.05, 0.97)
    osteopor = rng.binomial(1, p_osteopor)

    # dep: depressão (27.8%)
    # Causas: a1medSed, arth, gt2psych
    p_dep = 0.16 + 0.07 * a1medSed + 0.06 * arth + 0.06 * gt2psych
    p_dep = np.clip(p_dep, 0.02, 0.97)
    dep = rng.binomial(1, p_dep)

    # ADLlt5: perda de autonomia — ADL < 5 (22.9%)
    # Causas: demence, parkOuSP, conduit, LSAi4
    p_ADLlt5 = 0.10 + 0.12 * demence + 0.10 * parkOuSP + 0.04 * (1 - conduit) + 0.05 * LSAi4
    p_ADLlt5 = np.clip(p_ADLlt5, 0.02, 0.97)
    ADLlt5 = rng.binomial(1, p_ADLlt5)

    # ------------------------------------------------------------------
    # 3. Montar DataFrame
    # ------------------------------------------------------------------
    df = pd.DataFrame({
        # Sociodemográficas
        'age':       age,
        'sexe':      sexe,
        # Variáveis auxiliares (Markov Blanket)
        'trVision':  trVision,
        'myopat':    myopat,
        'evitsort':  evitsort,
        'montDesc':  montDesc,
        'sortSeul':  sortSeul,
        'TUGgt20':   TUGgt20,
        'agonDopa':  agonDopa,
        'akines':    akines,
        'BMIlt19':   BMIlt19,
        'htNivEtu':  htNivEtu,
        'a1medSed':  a1medSed,
        'arth':      arth,
        'gt2psych':  gt2psych,
        'a1AntiDep': a1AntiDep,
        'conduit':   conduit,
        'LSAi4':     LSAi4,
        'maisRet':   maisRet,
        'traAnOst':  traAnOst,
        'syndCer':   syndCer,
        # ── 10 fatores de risco (alvos do paper) ──
        'trEq':      trEq,
        'trMar':     trMar,
        'peurTom':   peurTom,
        'sarcopen':  sarcopen,
        'nbchu2':    nbchu2,
        'demence':   demence,
        'parkOuSP':  parkOuSP,
        'osteopor':  osteopor,
        'dep':       dep,
        'ADLlt5':    ADLlt5,
    })

    # ------------------------------------------------------------------
    # 4. Introduzir missing values nas variáveis auxiliares
    #    (o paper reporta até 70% de missing em 11 variáveis)
    # ------------------------------------------------------------------
    aux_cols = [
        'trVision', 'myopat', 'evitsort', 'montDesc', 'sortSeul',
        'TUGgt20', 'agonDopa', 'akines', 'BMIlt19', 'htNivEtu', 'a1medSed'
    ]
    for col in aux_cols:
        mask = rng.random(n) < missing_rate
        df.loc[mask, col] = np.nan

    # ------------------------------------------------------------------
    # 5. Salvar
    # ------------------------------------------------------------------
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/base_sintetica.csv', index=False)

    # Relatório de prevalências (comparar com Tabela 9 do paper)
    targets = ['trMar', 'peurTom', 'trEq', 'sarcopen', 'nbchu2',
               'demence', 'osteopor', 'dep', 'ADLlt5', 'parkOuSP']
    paper   = [82.5, 75.6, 73.4, 62.0, 57.9, 42.1, 33.2, 27.8, 22.9, 17.1]

    print("✅ Base gerada em data/base_sintetica.csv")
    print(f"   Pacientes: {n} | Variáveis: {len(df.columns)}\n")
    print(f"{'Alvo':<12} {'Gerado':>8} {'Paper':>8}")
    print("-" * 30)
    for t, p in zip(targets, paper):
        prev = df[t].mean() * 100
        print(f"{t:<12} {prev:>7.1f}% {p:>7.1f}%")

    return df


if __name__ == "__main__":
    generate_data()
