"""
trainer.py — Treino da Rede Bayesiana

Baseado em: Sihag et al. (2024), Expert Systems With Applications 252, 124106

Implementa o framework completo do paper (Seções 4 e 5):
  1. Pré-processamento: discretização de 'age', imputação via Naive Bayes
  2. Estrutura da BN: GHC-BIC com arcos obrigatórios (mandatory arcs)
  3. Parâmetros: estimação Bayesiana (BDeu)
  4. Avaliação: 6 métricas para cada um dos 10 fatores de risco
"""

import pickle
import warnings

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, BicScore, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB

warnings.filterwarnings("ignore")

# ── Constantes ─────────────────────────────────────────────────────────────────

# Os 10 fatores de risco que o paper avalia (Tabela 9)
TARGET_RFFS = [
    'trMar', 'peurTom', 'trEq', 'sarcopen', 'nbchu2',
    'demence', 'osteopor', 'dep', 'ADLlt5', 'parkOuSP',
]

# Arcos obrigatórios definidos por especialistas (Algorithm 1 do paper, Seção 5.3)
# Derivados do Markov Blanket de cada fator-alvo (Figuras 5 e 6)
MANDATORY_EDGES = [
    # trEq — distúrbio de equilíbrio
    ('trVision', 'trEq'),
    ('age',      'trEq'),
    # trMar — distúrbio de marcha
    ('trEq',     'trMar'),
    ('myopat',   'trMar'),
    ('TUGgt20',  'trMar'),
    # peurTom — medo de cair
    ('evitsort', 'peurTom'),
    ('montDesc', 'peurTom'),
    ('sortSeul', 'peurTom'),
    ('trMar',    'peurTom'),
    # sarcopen — sarcopenia / fraqueza muscular
    ('myopat',   'sarcopen'),
    ('TUGgt20',  'sarcopen'),
    ('BMIlt19',  'sarcopen'),
    # nbchu2 — nº de quedas nos últimos 6 meses
    ('trMar',    'nbchu2'),
    ('trEq',     'nbchu2'),
    ('TUGgt20',  'nbchu2'),
    # demence — demência
    ('age',      'demence'),
    ('htNivEtu', 'demence'),
    ('parkOuSP', 'demence'),
    # parkOuSP — Parkinson ou síndrome parkinsoniana
    ('akines',   'parkOuSP'),
    ('agonDopa', 'parkOuSP'),
    # osteopor — osteoporose
    ('sexe',     'osteopor'),
    ('BMIlt19',  'osteopor'),
    ('traAnOst', 'osteopor'),
    # dep — depressão
    ('a1medSed', 'dep'),
    ('arth',     'dep'),
    ('gt2psych', 'dep'),
    ('a1AntiDep','dep'),
    # ADLlt5 — perda de autonomia
    ('demence',  'ADLlt5'),
    ('parkOuSP', 'ADLlt5'),
    ('conduit',  'ADLlt5'),
    ('LSAi4',    'ADLlt5'),
]


# ── Pré-processamento ──────────────────────────────────────────────────────────

def discretize_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discretiza 'age' em 4 faixas etárias.
    Usa Int64 (nullable integer) para suportar NaN sem lançar erro.
    """
    df = df.copy()
    df['age'] = pd.cut(
        df['age'],
        bins=[60, 70, 80, 90, 100],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype('Int64')   # 'Int64' (maiúsculo) suporta NaN — 'int' não suporta
    return df


def impute_missing_naive_bayes(train: pd.DataFrame,
                                test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputa valores ausentes usando Naive Bayes (Seção 4.1.2 do paper).

    O paper testou Naive Bayes vs. KNN e encontrou que NB é superior
    para a maioria das variáveis. A imputação é treinada no treino
    e aplicada no teste (sem vazamento de dados).
    """
    train = train.copy()
    test  = test.copy()

    cols_with_missing = [c for c in train.columns if train[c].isna().any()]

    for col in cols_with_missing:
        # Features: todas as colunas sem missing no treino, exceto a atual
        feature_cols = [c for c in train.columns
                        if c != col and train[c].isna().sum() == 0]
        if not feature_cols:
            # Fallback: preenche com a moda
            moda = train[col].mode()[0]
            train[col] = train[col].fillna(moda)
            test[col]  = test[col].fillna(moda)
            continue

        mask_train = train[col].notna()
        X_tr = train.loc[mask_train, feature_cols].values
        y_tr = train.loc[mask_train, col].values.astype(int)

        nb = CategoricalNB()
        nb.fit(X_tr, y_tr)

        # Imputar treino
        mask_miss_train = train[col].isna()
        if mask_miss_train.any():
            X_fill = train.loc[mask_miss_train, feature_cols].values
            train.loc[mask_miss_train, col] = nb.predict(X_fill)

        # Imputar teste
        mask_miss_test = test[col].isna()
        if mask_miss_test.any():
            X_fill = test.loc[mask_miss_test, feature_cols].values
            test.loc[mask_miss_test, col] = nb.predict(X_fill)

    return train.astype(int), test.astype(int)


# ── Treino da BN ──────────────────────────────────────────────────────────────

def train_model(data_path: str = 'data/base_sintetica.csv',
                save_model: bool = True) -> BayesianNetwork:
    """
    Treina a Rede Bayesiana seguindo o framework do paper.

    Retorna o modelo treinado.
    """

    # 1. Carregar dados
    df = pd.read_csv(data_path)
    print(f"Dados carregados: {df.shape[0]} pacientes, {df.shape[1]} variáveis")

    # 2. Pré-processamento
    df = discretize_age(df)

    # 3. Divisão treino / teste
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # 4. Imputação de valores ausentes com Naive Bayes (Seção 4.1.2)
    print("\nImputando valores ausentes com Naive Bayes...")
    train_df, test_df = impute_missing_naive_bayes(train_df, test_df)

    # 5. Aprendizado da estrutura — GHC-BIC com arcos obrigatórios (Seção 4.2.3)
    print("Aprendendo estrutura da BN (GHC-BIC)...")
    hc = HillClimbSearch(train_df)
    best_dag = hc.estimate(
        scoring_method=BicScore(train_df),
        fixed_edges=set(MANDATORY_EDGES),   # arcos causais obrigatórios
        max_indegree=4,                      # limita complexidade do grafo
        show_progress=False,
    )
    print(f"  Arcos aprendidos: {len(best_dag.edges())}")

    # 6. Montar e treinar a BN com estimação Bayesiana (BDeu)
    model = BayesianNetwork(best_dag.edges())
    model.fit(train_df, estimator=BayesianEstimator, prior_type='BDeu')

    # 7. Salvar modelo para uso no inference.py
    if save_model:
        with open('data/bn_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("  Modelo salvo em data/bn_model.pkl")

    # 8. Avaliação com os 6 métricas do paper (Seção 4.3.2)
    print("\n" + "=" * 70)
    print("RESULTADOS — 10 fatores de risco (Tabela 10 do paper)")
    print("=" * 70)
    print(f"\n{'Alvo':<12} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC-PR':>7} {'BalAcc':>7} {'AUC-ROC':>8}")
    print("-" * 60)

    infer = VariableElimination(model)

    for target in TARGET_RFFS:
        other_targets = [t for t in TARGET_RFFS if t != target]
        # Remove os outros alvos da evidência (observação parcial — ponto forte da BN)
        evidence_cols = [c for c in test_df.columns
                         if c != target and c not in other_targets]

        y_true  = []
        y_pred  = []
        y_proba = []

        for _, row in test_df.iterrows():
            evidence = {col: int(row[col]) for col in evidence_cols}
            try:
                result = infer.query([target], evidence=evidence, show_progress=False)
                prob_1 = float(result.values[1])
                pred   = 1 if prob_1 >= 0.5 else 0
            except Exception:
                # Fallback para predição pela prevalência no treino
                prob_1 = float(train_df[target].mean())
                pred   = 1 if prob_1 >= 0.5 else 0

            y_true.append(int(row[target]))
            y_pred.append(pred)
            y_proba.append(prob_1)

        y_true  = np.array(y_true)
        y_pred  = np.array(y_pred)
        y_proba = np.array(y_proba)

        # Calcular as 6 métricas (Tabela 8 do paper)
        prec    = precision_score(y_true, y_pred, zero_division=0)
        rec     = recall_score(y_true, y_pred, zero_division=0)
        f1      = f1_score(y_true, y_pred, zero_division=0)
        auc_pr  = average_precision_score(y_true, y_proba)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_proba)

        print(f"{target:<12} {prec:>6.2f} {rec:>6.2f} {f1:>6.2f} "
              f"{auc_pr:>7.2f} {bal_acc:>7.2f} {auc_roc:>8.2f}")

    print("=" * 70)
    print("\nLegenda: Prec=Precisão | Rec=Recall | BalAcc=Balanced Accuracy")
    print("         AUC-PR=Área sob curva Precisão-Recall | AUC-ROC=Área sob curva ROC")
    print("\n✅ Treinamento concluído!")

    return model


if __name__ == "__main__":
    train_model()
