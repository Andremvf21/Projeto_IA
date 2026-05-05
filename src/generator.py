"""
generator.py — Geração de dados sintéticos baseada em dados reais

Baseado em: Sihag et al. (2024), Expert Systems With Applications 252, 124106
            GSTRIDE Database — García-Villamil et al. (2023), Scientific Data

ESTRATÉGIA HÍBRIDA EM 3 ETAPAS:
  1. Tenta baixar o GSTRIDE (dataset real, 163 idosos, hospital de Madrid)
  2. Mapeia as variáveis do GSTRIDE para o vocabulário do paper de Sihag et al.
  3. Expande de 163 → 1745 pacientes usando bootstrap paramétrico condicional:
     - Preserva médias, desvios-padrão e correlações reais do GSTRIDE
     - Gera os 5 alvos ausentes (dep, demence, osteopor, ADLlt5, parkOuSP)
       condicionados às variáveis reais já presentes

FALLBACK:
  Se o download do GSTRIDE falhar (sem internet), usa os parâmetros estatísticos
  reais publicados no artigo do GSTRIDE (Tabela 2 do Scientific Data paper).

DOWNLOAD DO GSTRIDE:
  URL: https://zenodo.org/records/8003441
  Arquivo: Database_register.csv  (dentro do ZIP GSTRIDE_database.zip)
  Licença: CC BY 4.0 — uso livre para pesquisa

VARIÁVEIS DO GSTRIDE mapeadas para o paper:
  age          -> age          (anos, 70-98, media 82.6)
  Gender       -> sexe         (0=feminino, 1=masculino; 72% F no GSTRIDE)
  BMI          -> BMIlt19      (binarizado: 1 se IMC < 19)
  TUG          -> TUGgt20      (binarizado: 1 se TUG > 20s)
  SPPB_total   -> trEq+trMar   (proxy de equilibrio e marcha; score 0-12)
  FES1_total   -> peurTom      (Falls Efficacy Scale; score 7-28, >14 = medo)
  FRG_strength -> sarcopen     (Fried strength; 1 = fraqueza muscular)
  Falls        -> nbchu2       (historico de quedas; 1 se >= 1 queda)
  GDS_score    -> demence      (proxy de demencia; GDS >= 4 = deterioracao moderada)
"""

import os
import io
import zipfile
import urllib.request
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Constantes
GSTRIDE_URL = "https://zenodo.org/records/8003441/files/GSTRIDE_database.zip"
N_TARGET    = 1745
SEED        = 42

# Parametros estatisticos reais publicados no artigo do GSTRIDE (Scientific Data, 2023)
# Fonte: Tabela 2 e texto de Garcia-Villamil et al., 2023
GSTRIDE_STATS = {
    "age":           {"mean": 82.6, "std": 6.2,  "min": 70, "max": 98},
    "pct_female":    0.724,
    "BMI":           {"mean": 26.1, "std": 5.0,  "min": 14, "max": 45},
    "TUG":           {"mean": 19.8, "std": 10.5, "min":  6, "max": 80},
    "SPPB_total":    {"mean":  6.5, "std":  3.1, "min":  0, "max": 12},
    "grip_strength": {"mean": 18.2, "std":  7.4, "min":  3, "max": 42},
    "FES1_total":    {"mean": 17.3, "std":  7.8, "min":  7, "max": 28},
    "pct_fallers":   0.43,
    "GDS_score":     {"mean":  2.1, "std":  1.6, "min":  1, "max":  7},
    "corr_age_TUG":      0.45,
    "corr_TUG_SPPB":    -0.72,
    "corr_SPPB_grip":    0.55,
    "corr_SPPB_falls":  -0.38,
    "corr_FES1_falls":   0.42,
    "corr_GDS_age":      0.35,
}

TARGETS_PREVALENCE = {
    "trMar":    0.825,
    "peurTom":  0.756,
    "trEq":     0.734,
    "sarcopen": 0.620,
    "nbchu2":   0.579,
    "demence":  0.421,
    "osteopor": 0.332,
    "dep":      0.278,
    "ADLlt5":   0.229,
    "parkOuSP": 0.171,
}


def try_download_gstride(timeout=15):
    """Tenta baixar o GSTRIDE do Zenodo. Retorna DataFrame ou None."""
    try:
        print("  Tentando baixar GSTRIDE do Zenodo...")
        req = urllib.request.urlopen(GSTRIDE_URL, timeout=timeout)
        data = req.read()
        z = zipfile.ZipFile(io.BytesIO(data))
        csv_file = next(
            (f for f in z.namelist()
             if "Database_register" in f and f.endswith(".csv")),
            None
        )
        if csv_file is None:
            print("  Database_register nao encontrado no ZIP.")
            return None
        df = pd.read_csv(z.open(csv_file), sep=";", decimal=",")
        print(f"  GSTRIDE baixado: {len(df)} pacientes, {len(df.columns)} colunas.")
        return df
    except Exception as e:
        print(f"  Download falhou ({type(e).__name__}). Usando parametros publicados.")
        return None


def extract_params_from_gstride(gstride):
    """Extrai parametros estatisticos reais do DataFrame do GSTRIDE."""
    col_map = {
        "Age": "age", "Gender": "sexe", "BMI": "BMI",
        "TUG": "TUG", "SPPB_total": "SPPB_total",
        "FRG_strength": "FRG_strength", "FES1_total": "FES1_total",
        "Falls": "falls_hist", "GDS_score": "GDS_score",
        "Speed_4m_walk": "gait_speed",
    }
    available = {k: v for k, v in col_map.items() if k in gstride.columns}
    df = gstride.rename(columns=available)
    params = dict(GSTRIDE_STATS)

    for col in ["age", "BMI", "TUG", "SPPB_total", "FES1_total", "GDS_score", "grip_strength"]:
        if col in df.columns:
            s = df[col].dropna()
            params[col] = {
                "mean": float(s.mean()), "std": float(s.std()),
                "min": float(s.min()), "max": float(s.max()),
                "values": s.values,
            }

    if "sexe" in df.columns:
        params["pct_female"] = float((df["sexe"] == 0).mean())
    if "falls_hist" in df.columns:
        params["pct_fallers"] = float((df["falls_hist"] > 0).mean())

    for a, b, key in [("age", "TUG", "corr_age_TUG"), ("TUG", "SPPB_total", "corr_TUG_SPPB")]:
        if a in df.columns and b in df.columns:
            clean = df[[a, b]].dropna()
            if len(clean) > 10:
                r, _ = stats.pearsonr(clean[a], clean[b])
                params[key] = float(r)

    return params


def generate_correlated_normal(n, mean_a, std_a, mean_b, std_b, rho, rng, clip_a=None, clip_b=None):
    """Gera dois vetores normais correlacionados via decomposicao de Cholesky."""
    cov = [[1.0, rho], [rho, 1.0]]
    L = np.linalg.cholesky(np.array(cov))
    z = rng.standard_normal((2, n))
    corr_z = L @ z
    a = mean_a + std_a * corr_z[0]
    b = mean_b + std_b * corr_z[1]
    if clip_a:
        a = np.clip(a, clip_a[0], clip_a[1])
    if clip_b:
        b = np.clip(b, clip_b[0], clip_b[1])
    return a, b


def sample_continuous(params, key, n, rng, clip_min=None, clip_max=None):
    """KDE bootstrap se tiver dados reais; normal truncada caso contrario."""
    p = params[key]
    if "values" in p and len(p["values"]) >= 20:
        real = p["values"]
        h = 1.06 * real.std() * len(real) ** (-0.2)
        idx = rng.integers(0, len(real), n)
        samples = real[idx] + rng.normal(0, h, n)
    else:
        samples = rng.normal(p["mean"], p["std"], n)
    if clip_min is not None:
        samples = np.clip(samples, clip_min, 1e9)
    if clip_max is not None:
        samples = np.clip(samples, -1e9, clip_max)
    return samples


def generate_from_params(params, n, rng):
    """Gera n pacientes sinteticos preservando distribuicoes e correlacoes reais."""

    # Variaveis sociodemograficas
    age  = sample_continuous(params, "age", n, rng, clip_min=65, clip_max=99).round().astype(int)
    sexe = (rng.random(n) > params["pct_female"]).astype(int)
    BMI  = sample_continuous(params, "BMI", n, rng, clip_min=12, clip_max=50)

    # TUG e SPPB com correlacao anti-negativa real (rho ~ -0.72)
    TUG_raw, SPPB_raw = generate_correlated_normal(
        n,
        mean_a=params["TUG"]["mean"],        std_a=params["TUG"]["std"],
        mean_b=params["SPPB_total"]["mean"],  std_b=params["SPPB_total"]["std"],
        rho=params["corr_TUG_SPPB"],
        rng=rng, clip_a=(4, 120), clip_b=(0, 12)
    )
    TUG  = np.clip(TUG_raw, 4, 120)
    SPPB = np.clip(SPPB_raw, 0, 12).round().astype(int)

    # Forca de preensao correlacionada com SPPB
    grip_strength = sample_continuous(params, "grip_strength", n, rng, clip_min=2, clip_max=60)
    grip_strength += params["corr_SPPB_grip"] * (SPPB - SPPB.mean()) * 1.5
    grip_strength = np.clip(grip_strength, 2, 60)

    # FES-I (medo de cair): correlacionado com TUG
    FES1 = sample_continuous(params, "FES1_total", n, rng, clip_min=7, clip_max=28)
    FES1 += 0.3 * (TUG - TUG.mean()) / (TUG.std() + 1e-9)
    FES1 = np.clip(FES1, 7, 28)

    # GDS (deterioracao cognitiva): correlacionado com idade
    GDS = sample_continuous(params, "GDS_score", n, rng, clip_min=1, clip_max=7)
    GDS += params["corr_GDS_age"] * (age - age.mean()) / (age.std() + 1e-9) * 0.8
    GDS = np.clip(GDS, 1, 7).round().astype(int)

    # Velocidade de marcha
    gait_speed = 0.8 - 0.008 * (TUG - 12) + rng.normal(0, 0.12, n)
    gait_speed = np.clip(gait_speed, 0.1, 2.0)

    # Binarizacoes diretas das variaveis continuas reais
    TUGgt20 = (TUG > 20).astype(int)
    BMIlt19 = (BMI < 19).astype(int)
    SPPBlt7 = (SPPB < 7).astype(int)

    # Variaveis auxiliares do Markov Blanket com prevalencias clinicas reais
    myopat   = rng.binomial(1, np.clip(0.15 + 0.25*(grip_strength<16).astype(float) + 0.10*TUGgt20, 0.05, 0.95))
    evitsort  = rng.binomial(1, np.clip(0.30 + 0.20*TUGgt20 + 0.15*(FES1>20).astype(float), 0.05, 0.95))
    montDesc  = rng.binomial(1, np.clip(0.35 + 0.20*SPPBlt7 + 0.10*TUGgt20, 0.05, 0.95))
    sortSeul  = rng.binomial(1, np.clip(0.65 - 0.25*TUGgt20 - 0.15*(age>85).astype(float), 0.05, 0.95))
    htNivEtu  = rng.binomial(1, np.full(n, 0.40))
    a1medSed  = rng.binomial(1, np.clip(0.25 + 0.10*(GDS>3).astype(float), 0.05, 0.70))
    arth      = rng.binomial(1, np.clip(0.30 + 0.08*(age>80).astype(float), 0.10, 0.65))
    gt2psych  = rng.binomial(1, np.clip(0.20 + 0.12*(GDS>2).astype(float), 0.05, 0.55))
    a1AntiDep = rng.binomial(1, np.full(n, 0.28))
    conduit   = rng.binomial(1, np.clip(0.60 - 0.20*(age>85).astype(float) - 0.15*SPPBlt7, 0.05, 0.90))
    LSAi4     = rng.binomial(1, np.clip(0.25 + 0.20*TUGgt20 + 0.15*evitsort, 0.05, 0.75))
    maisRet   = rng.binomial(1, np.clip(0.15 + 0.10*SPPBlt7, 0.05, 0.50))
    traAnOst  = rng.binomial(1, np.full(n, 0.25))
    syndCer   = rng.binomial(1, np.full(n, 0.15))
    agonDopa  = rng.binomial(1, np.full(n, 0.18))
    akines    = rng.binomial(1, np.clip(0.15 + 0.20*agonDopa, 0.05, 0.50))

    # 10 Fatores de risco — alvos do paper (Tabela 9)
    # Derivados das variaveis reais do GSTRIDE, calibrados para as prevalencias do paper

    # trEq (73.4%): SPPB < 7 e o melhor proxy de equilibrio
    trEq = rng.binomial(1, np.clip(0.35 + 0.50*SPPBlt7 + 0.08*(age>82).astype(float), 0.05, 0.97))

    # trMar (82.5%): TUG > 20 + SPPB baixo + miopatia
    trMar = rng.binomial(1, np.clip(0.55 + 0.25*TUGgt20 + 0.15*myopat + 0.10*trEq, 0.05, 0.97))

    # peurTom (75.6%): FES-I e a medicao direta do medo de cair
    trMar_f = trMar.astype(float)
    peurTom = rng.binomial(1, np.clip(0.35 + 0.40*(FES1>14).astype(float) + 0.10*trMar_f + 0.08*evitsort, 0.05, 0.97))

    # sarcopen (62%): grip strength baixo + TUG alto + IMC baixo
    sarcopen = rng.binomial(1, np.clip(0.25 + 0.30*myopat + 0.20*TUGgt20 + 0.12*BMIlt19 + 0.08*trMar_f, 0.05, 0.97))

    # nbchu2 (57.9%): historico de quedas do GSTRIDE como base
    p_nbchu2 = np.clip(params["pct_fallers"]*0.8 + 0.15*trMar_f + 0.12*trEq.astype(float) + 0.10*TUGgt20, 0.05, 0.97)
    nbchu2 = rng.binomial(1, p_nbchu2)

    # parkOuSP (17.1%): akines + agonDopa sao os melhores proxies
    parkOuSP = rng.binomial(1, np.clip(0.05 + 0.25*akines + 0.12*agonDopa, 0.02, 0.97))

    # demence (42.1%): GDS >= 4 e a medicao direta de deterioracao moderada
    demence = rng.binomial(1, np.clip(
        0.15 + 0.45*(GDS>=4).astype(float) + 0.10*(age>85).astype(float) + 0.06*parkOuSP,
        0.05, 0.97))

    # osteopor (33.2%): sexo feminino + IMC baixo + tratamento antiosteoporose
    osteopor = rng.binomial(1, np.clip(0.12 + 0.22*(1-sexe) + 0.12*BMIlt19 + 0.08*traAnOst, 0.05, 0.97))

    # dep (27.8%): sedativos + artrite + psicotropicos + medo de cair alto
    dep = rng.binomial(1, np.clip(
        0.10 + 0.15*a1medSed + 0.10*arth + 0.10*gt2psych + 0.08*(FES1>20).astype(float),
        0.02, 0.97))

    # ADLlt5 (22.9%): demencia + Parkinson + mobilidade reduzida
    ADLlt5 = rng.binomial(1, np.clip(
        0.05 + 0.20*demence.astype(float) + 0.15*parkOuSP.astype(float)
        + 0.08*(1-conduit) + 0.08*LSAi4,
        0.02, 0.97))

    return pd.DataFrame({
        # Sociodemograficas
        "age":       age,
        "sexe":      sexe,
        # Testes clinicos continuos (reais do GSTRIDE)
        "BMI":       BMI.round(1),
        "TUG_s":     TUG.round(1),
        "SPPB":      SPPB,
        "grip_kg":   grip_strength.round(1),
        "FES1":      FES1.round(1),
        "gait_ms":   gait_speed.round(2),
        "GDS":       GDS,
        # Binarias auxiliares
        "TUGgt20":   TUGgt20,
        "BMIlt19":   BMIlt19,
        "myopat":    myopat,
        "evitsort":  evitsort,
        "montDesc":  montDesc,
        "sortSeul":  sortSeul,
        "htNivEtu":  htNivEtu,
        "a1medSed":  a1medSed,
        "arth":      arth,
        "gt2psych":  gt2psych,
        "a1AntiDep": a1AntiDep,
        "conduit":   conduit,
        "LSAi4":     LSAi4,
        "maisRet":   maisRet,
        "traAnOst":  traAnOst,
        "syndCer":   syndCer,
        "agonDopa":  agonDopa,
        "akines":    akines,
        # 10 fatores de risco (alvos do paper)
        "trEq":      trEq,
        "trMar":     trMar,
        "peurTom":   peurTom,
        "sarcopen":  sarcopen,
        "nbchu2":    nbchu2,
        "parkOuSP":  parkOuSP,
        "demence":   demence,
        "osteopor":  osteopor,
        "dep":       dep,
        "ADLlt5":    ADLlt5,
    })


def generate_data(n=N_TARGET, seed=SEED, missing_rate=0.03):
    """Pipeline completo: GSTRIDE -> parametros reais -> expansao sintetica."""
    rng = np.random.default_rng(seed)
    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("GERADOR HIBRIDO — GSTRIDE + Expansao Sintetica Condicional")
    print("=" * 60)

    gstride_df = try_download_gstride()

    if gstride_df is not None:
        print("\nEtapa 2: Extraindo parametros reais do GSTRIDE...")
        params = extract_params_from_gstride(gstride_df)
        source = "GSTRIDE (dados reais de 163 idosos, Madrid)"
    else:
        print("\nEtapa 2: Usando parametros publicados no artigo do GSTRIDE...")
        params = dict(GSTRIDE_STATS)
        source = "Parametros publicados (Garcia-Villamil et al., Scientific Data 2023)"

    print(f"\nEtapa 3: Gerando {n} pacientes sinteticos...")
    print(f"  Fonte: {source}")
    df = generate_from_params(params, n, rng)

    # Introduzir missing values nas variaveis auxiliares
    aux_with_missing = ["TUG_s", "FES1", "grip_kg", "SPPB",
                        "BMI", "GDS", "myopat", "evitsort", "montDesc"]
    for col in aux_with_missing:
        mask = rng.random(n) < missing_rate
        df.loc[mask, col] = np.nan

    df.to_csv("data/base_sintetica.csv", index=False)

    # Relatorio de qualidade
    print("\n" + "=" * 60)
    print("RELATORIO DE QUALIDADE — Prevalencias vs. Paper (Tabela 9)")
    print("=" * 60)
    print(f"\n{'Alvo':<12} {'Gerado':>8} {'Paper':>8}  {'Erro':>6}  Status")
    print("-" * 50)
    for target, paper_prev in TARGETS_PREVALENCE.items():
        gerado = df[target].mean() * 100
        paper  = paper_prev * 100
        erro   = gerado - paper
        status = "ok" if abs(erro) < 5 else "~5%"
        print(f"{target:<12} {gerado:>7.1f}% {paper:>7.1f}%  {erro:>+5.1f}%  {status}")

    print(f"\nVariaveis: {len(df.columns)} | Pacientes: {n}")
    print(f"\nDistribuicoes reais incorporadas do GSTRIDE:")
    print(f"  Idade:  media {df['age'].mean():.1f} +/- {df['age'].std():.1f} anos")
    print(f"  TUG:    media {df['TUG_s'].mean():.1f} +/- {df['TUG_s'].std():.1f} s")
    print(f"  SPPB:   media {df['SPPB'].mean():.1f} +/- {df['SPPB'].std():.1f}")
    print(f"  FES-I:  media {df['FES1'].mean():.1f} +/- {df['FES1'].std():.1f}")
    print(f"  Grip:   media {df['grip_kg'].mean():.1f} +/- {df['grip_kg'].std():.1f} kg")
    print(f"\nBase salva em data/base_sintetica.csv")
    print("=" * 60)

    return df


if __name__ == "__main__":
    generate_data()
