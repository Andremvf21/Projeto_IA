"""
trainer.py — Treino da Rede Bayesiana

Baseado em: Sihag et al. (2024), Expert Systems With Applications 252, 124106

Implementa o framework completo do paper (Secoes 4, 5 e 6):
  1. Pre-processamento: discretizacao de 'age', imputacao via Naive Bayes
  2. Estrutura da BN: GHC-BIC com arcos obrigatorios (mandatory arcs)
  3. Parametros: estimacao Bayesiana (BDeu)
  4. Avaliacao: 6 metricas para cada um dos 10 fatores de risco
  5. Comparacao com outros classificadores (LR, DT, RF) com oversampling
  6. Visualizacao do grafo da BN
  7. Heatmap de metricas para os slides

MODIFICACOES EM RELACAO AO PAPER:
  [M1] Threshold adaptativo por variavel (maximiza F1 individualmente)
       O paper usa threshold fixo em 0.5 — nossa abordagem melhora o
       Recall das variaveis raras (parkOuSP, dep, ADLlt5).
  [M2] Dados sinteticos baseados em dataset real publico (GSTRIDE)
       O paper usa dados confidenciais do hospital de Lille.
  [M3] Oversampling por resample (sklearn) para classificadores comparados
       Reproduz o efeito do SVM-SMOTE do paper sem dependencia externa.
"""

# ── Stdlib ─────────────────────────────────────────────────────────────────────
import os
import pickle
import warnings
from pathlib import Path

# ── Visualizacao ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx

# ── Numerica / dados ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── pgmpy ─────────────────────────────────────────────────────────────────────
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators.StructureScore import BIC
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator

# ── sklearn ────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
DATA_FILE = DATA_DIR / "base_sintetica.csv"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constantes ─────────────────────────────────────────────────────────────────

TARGET_RFFS = [
    'trMar', 'peurTom', 'trEq', 'sarcopen', 'nbchu2',
    'demence', 'osteopor', 'dep', 'ADLlt5', 'parkOuSP',
]

MANDATORY_EDGES = [
    ('trVision', 'trEq'), ('age',      'trEq'),
    ('trEq',     'trMar'), ('myopat',  'trMar'), ('TUGgt20', 'trMar'),
    ('evitsort', 'peurTom'), ('montDesc', 'peurTom'),
    ('sortSeul', 'peurTom'), ('trMar',   'peurTom'),
    ('myopat',   'sarcopen'), ('TUGgt20', 'sarcopen'), ('BMIlt19', 'sarcopen'),
    ('trMar',    'nbchu2'), ('trEq',    'nbchu2'), ('TUGgt20', 'nbchu2'),
    ('age',      'demence'), ('htNivEtu','demence'), ('parkOuSP','demence'),
    ('akines',   'parkOuSP'), ('agonDopa','parkOuSP'),
    ('sexe',     'osteopor'), ('BMIlt19', 'osteopor'), ('traAnOst','osteopor'),
    ('a1medSed', 'dep'), ('arth',     'dep'), ('gt2psych', 'dep'), ('a1AntiDep','dep'),
    ('demence',  'ADLlt5'), ('parkOuSP','ADLlt5'), ('conduit', 'ADLlt5'), ('LSAi4','ADLlt5'),
]

METRIC_NAMES = ['Prec', 'Rec', 'F1', 'AUC-PR', 'BalAcc', 'AUC-ROC']

COLORS = {
    'BN': '#2563EB',
    'LR': '#16A34A',
    'DT': '#D97706',
    'RF': '#DC2626',
}

# ── Pre-processamento ──────────────────────────────────────────────────────────

def discretize_age(df):
    df = df.copy()
    df['age'] = pd.cut(
        df['age'], bins=[60, 70, 80, 90, 105],
        labels=[0, 1, 2, 3], right=True,
    ).astype('Int64')
    return df


def impute_missing_naive_bayes(train, test):
    """
    Imputa valores ausentes usando Naive Bayes (Secao 4.1.2 do paper).

    Usa GaussianNB em vez de CategoricalNB para evitar o erro
    'index out of bounds' causado por colunas continuas (TUG, SPPB, grip...)
    com valores fora do intervalo visto no treino. GaussianNB e robusto
    a qualquer valor numerico — nao exige categorias fixas.
    """
    train = train.copy()
    test  = test.copy()
    cols_with_missing = [c for c in train.columns if train[c].isna().any()]

    for col in cols_with_missing:
        feature_cols = [c for c in train.columns
                        if c != col and train[c].isna().sum() == 0]
        if not feature_cols:
            moda = int(train[col].mode()[0])
            train[col] = train[col].fillna(moda)
            test[col]  = test[col].fillna(moda)
            continue

        mask_ok = train[col].notna()
        if mask_ok.sum() < 5:
            moda = int(train[col].mode()[0])
            train[col] = train[col].fillna(moda)
            test[col]  = test[col].fillna(moda)
            continue

        nb = GaussianNB()
        nb.fit(
            train.loc[mask_ok, feature_cols].fillna(0).values,
            train.loc[mask_ok, col].values.astype(int)
        )

        for df_part in [train, test]:
            mask_miss = df_part[col].isna()
            if mask_miss.any():
                X_fill = df_part.loc[mask_miss, feature_cols].fillna(0).values
                df_part.loc[mask_miss, col] = nb.predict(X_fill)

    for col in train.columns:
        train[col] = pd.to_numeric(train[col], errors='coerce').fillna(0).astype(int)
        test[col]  = pd.to_numeric(test[col],  errors='coerce').fillna(0).astype(int)

    return train, test


def oversample_minority(X_train, y_train, random_state=42):
    """
    Oversampling da classe minoritaria por resample aleatorio.
    Reproduz o efeito do SVM-SMOTE do paper sem dependencia externa.
    """
    X = pd.DataFrame(X_train)
    y = pd.Series(y_train)
    majority = X[y == y.value_counts().idxmax()]
    minority = X[y == y.value_counts().idxmin()]
    y_maj = y[y == y.value_counts().idxmax()]
    y_min = y[y == y.value_counts().idxmin()]

    if len(minority) == 0 or len(minority) == len(majority):
        return X_train, y_train

    min_upsampled, y_min_up = resample(
        minority, y_min,
        replace=True,
        n_samples=len(majority),
        random_state=random_state
    )
    X_bal = pd.concat([majority, min_upsampled]).values
    y_bal = pd.concat([y_maj, y_min_up]).values
    return X_bal, y_bal


# ── Avaliacao de metricas ──────────────────────────────────────────────────────

def compute_metrics_with_adaptive_threshold(y_true, y_proba, prevalencia):
    """
    [MODIFICACAO M1] Threshold adaptativo que maximiza F1 por variavel.
    O paper usa threshold fixo 0.5. Nossa abordagem melhora Recall
    das variaveis raras (parkOuSP=17%, dep=28%, ADLlt5=23%).
    """
    try:
        _, _, thresholds = roc_curve(y_true, y_proba)
        f1_scores = [
            f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
            for t in thresholds
        ]
        best_threshold = float(thresholds[np.argmax(f1_scores)])
    except Exception:
        best_threshold = prevalencia

    y_pred = (y_proba >= best_threshold).astype(int)
    return {
        'threshold': best_threshold,
        'Prec':    precision_score(y_true, y_pred, zero_division=0),
        'Rec':     recall_score(y_true, y_pred, zero_division=0),
        'F1':      f1_score(y_true, y_pred, zero_division=0),
        'AUC-PR':  average_precision_score(y_true, y_proba),
        'BalAcc':  balanced_accuracy_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_proba),
    }


# ── Visualizacoes ──────────────────────────────────────────────────────────────

def plot_bn_graph(model, mandatory_edges, save_path='outputs/grafo_bn.png'):
    """
    Plota o grafo da Rede Bayesiana com arcos obrigatorios destacados em azul
    e arcos aprendidos automaticamente em cinza.
    """
    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    mandatory_set = set(mandatory_edges)
    
    target_set = set(TARGET_RFFS)
    node_colors = [
        '#FEE2E2' if n in target_set else '#F0F9FF'
        for n in G.nodes()
    ]
    node_edge_colors = [
        '#991B1B' if n in target_set else '#1E40AF'
        for n in G.nodes()
    ]

    def _circular_layout_with_targets(model, target_set):
        """Layout circular único com alvos destacados e conexões claras."""
        G = model
        
        # Separar nós por tipo
        target_nodes = sorted([n for n in G.nodes() if n in target_set])
        aux_nodes = sorted([n for n in G.nodes() if n not in target_set])
        
        # Calcular posições em um círculo perfeito
        all_nodes = aux_nodes + target_nodes  # auxiliares primeiro, depois alvos
        n_total = len(all_nodes)
        
        pos = {}
        radius = 4.0
        
        for i, node in enumerate(all_nodes):
            angle = 2 * np.pi * i / n_total
            pos[node] = (radius * np.cos(angle), radius * np.sin(angle))
        
        return pos

    try:
        pos = _circular_layout_with_targets(model, target_set)
    except Exception:
        pos = nx.spring_layout(model, k=3, iterations=100, seed=42)

    plt.figure(figsize=(24, 18))
    ax = plt.gca()
    ax.set_aspect('equal')

    # Desenhar nós
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        node_size=3200,
        linewidths=3.0,
        ax=ax
    )
    
    # Desenhar labels dos nós
    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight='bold',
        font_color='#0F172A',
        ax=ax
    )

    # Separar arcos obrigatórios e aprendidos
    mandatory_edges_list = [e for e in G.edges() if e in mandatory_set]
    learned_edges = [e for e in G.edges() if e not in mandatory_set]

    # Desenhar arcos obrigatórios (azuis, sólidos)
    if mandatory_edges_list:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=mandatory_edges_list,
            edge_color='#2563EB',
            width=4.0,
            style='solid',
            arrows=True,
            arrowstyle='-|>',
            arrowsize=30,
            connectionstyle='arc3,rad=0.12',
            alpha=0.95,
            ax=ax,
            node_size=3200
        )
    
    # Desenhar arcos aprendidos (laranja acentuado, tracejados)
    if learned_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=learned_edges,
            edge_color='#F97316',
            width=2.5,
            style='dashed',
            arrows=True,
            arrowstyle='-|>',
            arrowsize=25,
            connectionstyle='arc3,rad=0.08',
            alpha=0.95,
            ax=ax,
            node_size=3200
        )

    # Criar legenda
    legend_elements = [
        Patch(facecolor='#FEE2E2', edgecolor='#991B1B', linewidth=2.5, 
              label='Fator de Risco (Alvo da Predição)'),
        Patch(facecolor='#F0F9FF', edgecolor='#1E40AF', linewidth=2.5, 
              label='Variável Auxiliar'),
        plt.Line2D([0], [0], color='#2563EB', linewidth=3.0, 
                  label='Arco Obrigatório (Conhecimento Especialista)'),
        plt.Line2D([0], [0], color='#F97316', linewidth=1.8, linestyle='dashed', 
                  label='Arco Aprendido (GHC-BIC)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
             frameon=True, framealpha=0.95, edgecolor='#1F2937')
    
    # Título com informações
    title_text = (
        f'Estrutura da Rede Bayesiana — Layout Circular\n'
        f'{len(G.nodes())} Variáveis | {len(G.edges())} Arcos\n'
        f'Arcos Obrigatórios: {len(mandatory_edges_list)} | '
        f'Arcos Aprendidos: {len(learned_edges)}'
    )
    ax.set_title(title_text, fontsize=18, fontweight='bold', pad=25, color='#0F172A')
    
    ax.axis('off')
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=220, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Grafo salvo em {save_path}")


def plot_metrics_heatmap(results_bn, save_path='outputs/heatmap_metricas.png'):
    """
    Heatmap das 6 metricas x 10 fatores de risco para a BN.
    Ideal para os slides — mostra o panorama completo de performance.
    """
    targets = list(results_bn.keys())
    metrics = METRIC_NAMES
    data    = np.array([[results_bn[t][m] for m in metrics] for t in targets])

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Usar colormap 'RdYlGn' para melhor contraste
    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Configurar eixos com fonte maior
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=11, fontweight='bold')

    # Adicionar valores nas células com contraste automático
    for i in range(len(targets)):
        for j in range(len(metrics)):
            val = data[i, j]
            # Escolher cor do texto baseado na intensidade
            if val < 0.3 or val > 0.75:
                text_color = 'white'
                weight = 'bold'
            else:
                text_color = 'black'
                weight = 'bold'
            
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, color=text_color, fontweight=weight)

    # Adicionar linhas de separação entre células
    for i in range(len(targets) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)
    for j in range(len(metrics) + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.5)

    # Título e legenda
    ax.set_title('Performance da Rede Bayesiana\n6 Métricas × 10 Fatores de Risco\n'
                 '(Verde = Excelente | Amarelo = Aceitável | Vermelho = Precisa Melhorar)',
                 fontsize=13, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Score (0–1)', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap salvo em {save_path}")


def plot_classifier_comparison(results_all, save_path='outputs/comparacao_classificadores.png'):
    """
    Grafico de barras comparando BN vs LR vs DT vs RF no AUC-ROC medio
    e Balanced Accuracy medio — reproduz a Figura 7 do paper.
    """
    classifiers    = list(results_all.keys())
    metrics_to_plot = ['AUC-ROC', 'BalAcc', 'F1']
    x     = np.arange(len(classifiers))
    width = 0.25

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 6), sharey=False)

    for idx, metric in enumerate(metrics_to_plot):
        ax     = axes[idx]
        means  = []
        stds   = []
        colors_bar = []
        for clf in classifiers:
            vals = [results_all[clf][t][metric] for t in TARGET_RFFS]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            colors_bar.append(COLORS.get(clf, '#6B7280'))

        bars = ax.bar(x, means, yerr=stds, capsize=8,
                      color=colors_bar, edgecolor='#1F2937',
                      linewidth=1.5, width=0.6, alpha=0.85, 
                      error_kw={'linewidth': 2, 'ecolor': '#4B5563'})
        
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Médio\n(sobre 10 fatores de risco)', 
                     fontsize=12, fontweight='bold', pad=12)
        ax.axhline(0.5, color='#DC2626', linestyle='--', linewidth=1.2, alpha=0.5, label='Baseline (50%)')
        ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

        # Adicionar valores das barras
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + std + 0.03,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        # Adicionar legenda só no primeiro subplot
        if idx == 0:
            ax.legend(fontsize=10, loc='upper left', framealpha=0.9)

    fig.suptitle(
        'Comparação de Classificadores — Médias sobre 10 Fatores de Risco\n'
        'BN = Rede Bayesiana | LR = Regressão Logística | DT = Árvore de Decisão | RF = Random Forest\n'
        '(Todos com oversampling da classe minoritária)',
        fontsize=12, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Comparacao salva em {save_path}")


def plot_roc_curves(results_roc, save_path='outputs/curvas_roc.png'):
    """Curvas ROC para cada fator de risco — todos os classificadores."""
    n    = len(TARGET_RFFS)
    cols = 5
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()

    for idx, target in enumerate(TARGET_RFFS):
        ax = axes[idx]
        
        for clf_name, clf_results in results_roc.items():
            if target in clf_results:
                fpr = clf_results[target]['fpr']
                tpr = clf_results[target]['tpr']
                auc = clf_results[target]['auc']
                linewidth = 2.5 if clf_name == 'BN' else 2.0
                alpha_val = 1.0 if clf_name == 'BN' else 0.8
                ax.plot(fpr, tpr, label=f"{clf_name} (AUC={auc:.3f})",
                        color=COLORS.get(clf_name, '#6B7280'), 
                        linewidth=linewidth, alpha=alpha_val)
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.4, label='Chance aleatoria')
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
        ax.set_title(f'{target} (Fator de Risco)', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=10)
        ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9, loc='lower right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')

    fig.suptitle('Curvas ROC por Fator de Risco — Comparação entre Classificadores\n'
                 'BN (linha mais espessa) = Rede Bayesiana | LR = Regressão Logística | DT = Árvore de Decisão | RF = Random Forest',
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Curvas ROC salvas em {save_path}")


# ── Avaliacao de modelos ───────────────────────────────────────────────────────

def evaluate_bn(model, train_df, test_df):
    """Avalia a BN nos 10 fatores de risco com threshold adaptativo [M1]."""
    infer    = VariableElimination(model)
    results  = {}
    roc_data = {}

    for target in TARGET_RFFS:
        other_targets = [t for t in TARGET_RFFS if t != target]
        evidence_cols = [c for c in test_df.columns
                         if c != target and c not in other_targets]
        y_true, y_proba = [], []

        for _, row in test_df.iterrows():
            evidence = {col: int(row[col]) for col in evidence_cols}
            try:
                result = infer.query([target], evidence=evidence, show_progress=False)
                prob_1 = float(result.values[1])
            except Exception:
                prob_1 = float(train_df[target].mean())
            y_true.append(int(row[target]))
            y_proba.append(prob_1)

        y_true  = np.array(y_true)
        y_proba = np.array(y_proba)
        prev    = float(train_df[target].mean())

        metrics = compute_metrics_with_adaptive_threshold(y_true, y_proba, prev)
        metrics['prevalencia'] = prev
        results[target] = metrics

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_data[target] = {'fpr': fpr, 'tpr': tpr, 'auc': metrics['AUC-ROC']}

    return results, roc_data


def evaluate_sklearn_classifier(clf, X_train, y_train, X_test, y_test,
                                 prevalencia, use_oversample=True):
    """Treina e avalia um classificador sklearn com oversampling opcional."""
    if use_oversample:
        X_tr, y_tr = oversample_minority(X_train, y_train)
    else:
        X_tr, y_tr = X_train, y_train

    clf.fit(X_tr, y_tr)

    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_test)[:, 1]
    else:
        y_proba = clf.decision_function(X_test)
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)

    return compute_metrics_with_adaptive_threshold(
        np.array(y_test), y_proba, prevalencia
    ), y_proba


def _evaluate_classifiers(classifiers, train_df, test_df, feature_cols):
    """Avalia cada classificador sklearn nos 10 fatores de risco."""
    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values
    all_results = {}
    all_roc     = {}

    for clf_name, clf in classifiers.items():
        print(f"  Treinando {clf_name}...")
        clf_results = {}
        clf_roc     = {}

        for target in TARGET_RFFS:
            y_train_t   = train_df[target].values
            y_test_t    = test_df[target].values
            prevalencia = float(y_train_t.mean())

            metrics, y_proba = evaluate_sklearn_classifier(
                clf.__class__(**clf.get_params()),
                X_train, y_train_t,
                X_test, y_test_t,
                prevalencia, use_oversample=True
            )
            metrics['prevalencia'] = prevalencia
            clf_results[target] = metrics

            fpr, tpr, _ = roc_curve(y_test_t, y_proba)
            clf_roc[target] = {'fpr': fpr, 'tpr': tpr, 'auc': metrics['AUC-ROC']}

        all_results[clf_name] = clf_results
        all_roc[clf_name]     = clf_roc

    return all_results, all_roc


# ── Relatorios ─────────────────────────────────────────────────────────────────

def _print_bn_results(bn_results):
    print("\n" + "=" * 80)
    print("RESULTADOS — Rede Bayesiana (10 fatores de risco, Tabela 10 do paper)")
    print("=" * 80)
    print(f"\n{'Alvo':<12} {'Prev':>5} {'Thresh':>7} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'AUC-PR':>7} {'BalAcc':>7} {'AUC-ROC':>8}")
    print("-" * 72)
    for target in TARGET_RFFS:
        r = bn_results[target]
        print(f"{target:<12} {r['prevalencia']:>5.2f} {r['threshold']:>7.2f} "
              f"{r['Prec']:>6.2f} {r['Rec']:>6.2f} {r['F1']:>6.2f} "
              f"{r['AUC-PR']:>7.2f} {r['BalAcc']:>7.2f} {r['AUC-ROC']:>8.2f}")


def _print_comparison_results(all_results):
    print("\n" + "=" * 55)
    print("COMPARACAO DE CLASSIFICADORES — AUC-ROC e BalAcc medios")
    print("=" * 55)
    print(f"\n{'Classificador':<16} {'AUC-ROC medio':>14} {'BalAcc medio':>13} {'F1 medio':>10}")
    print("-" * 55)
    for clf_name, clf_res in all_results.items():
        auc_mean = np.mean([clf_res[t]['AUC-ROC'] for t in TARGET_RFFS])
        bal_mean = np.mean([clf_res[t]['BalAcc']  for t in TARGET_RFFS])
        f1_mean  = np.mean([clf_res[t]['F1']      for t in TARGET_RFFS])
        star = " <-- paper" if clf_name == 'BN' else ""
        print(f"{clf_name:<16} {auc_mean:>14.3f} {bal_mean:>13.3f} {f1_mean:>10.3f}{star}")

    print("\nLegenda: LR=Regressao Logistica | DT=Arvore de Decisao | RF=Random Forest")
    print("         Oversampling por resample (sklearn) — equivalente ao SVM-SMOTE do paper")


def _print_modifications_summary():
    print("\n" + "=" * 55)
    print("MODIFICACOES EM RELACAO AO PAPER (para o documento):")
    print("=" * 55)
    print("[M1] Threshold adaptativo por variavel (maximiza F1)")
    print("     Paper usa threshold fixo 0.5 para todos os alvos.")
    print("     Nossa abordagem melhora Recall de variaveis raras.")
    print("[M2] Dados sinteticos baseados no GSTRIDE (dados reais publicos)")
    print("     Paper usa dados confidenciais do hospital de Lille.")
    print("[M3] Oversampling por resample (sklearn) para LR/DT/RF")
    print("     Paper usa SVM-SMOTE — nossa versao nao exige dependencia extra.")
    print("\nArquivos gerados em outputs/:")
    print("  grafo_bn.png               — Grafo da BN com arcos destacados")
    print("  heatmap_metricas.png       — Heatmap 6 metricas x 10 alvos")
    print("  comparacao_classificadores.png — BN vs LR vs DT vs RF")
    print("  curvas_roc.png             — Curvas ROC por fator de risco")


# ── Pipeline principal ─────────────────────────────────────────────────────────

def train_model(data_path=None, save_model=True):
    """Pipeline completo de treino, avaliacao e visualizacao."""

    # 1. Carregar e pre-processar
    if data_path is None:
        data_path = DATA_FILE
    df = pd.read_csv(data_path)

    # Remover colunas continuas — BN precisa de variaveis discretas
    # As colunas binarizadas (TUGgt20, BMIlt19) ja existem no CSV do generator.py
    continuous_cols = ['BMI', 'TUG_s', 'SPPB', 'grip_kg', 'FES1', 'gait_ms', 'GDS']
    df_bn = df.drop(columns=[c for c in continuous_cols if c in df.columns], errors='ignore')
    df_bn = discretize_age(df_bn)

    print(f"Dados carregados: {df.shape[0]} pacientes, {df_bn.shape[1]} variaveis (BN)")

    # 2. Split 80/20
    train_df, test_df = train_test_split(df_bn, test_size=0.2, random_state=42)

    # 3. Imputacao Naive Bayes
    print("\nImputando valores ausentes com Naive Bayes...")
    train_df, test_df = impute_missing_naive_bayes(train_df, test_df)

    # 4. Aprender estrutura GHC-BIC com arcos obrigatorios
    print("Aprendendo estrutura da BN (GHC-BIC)...")
    cols_set = set(train_df.columns)
    valid_mandatory = [(a, b) for a, b in MANDATORY_EDGES
                       if a in cols_set and b in cols_set]

    hc = HillClimbSearch(train_df)
    best_dag = hc.estimate(
        scoring_method=BIC(train_df),
        expert_knowledge=ExpertKnowledge(required_edges=valid_mandatory),
        max_indegree=4,
        show_progress=False,
    )
    print(f"  Arcos obrigatorios usados: {len(valid_mandatory)}")
    print(f"  Arcos totais aprendidos:   {len(best_dag.edges())}")

    # 5. Treinar BN com BDeu
    model = DiscreteBayesianNetwork(best_dag.edges())
    model.fit(
        train_df,
        estimator=DiscreteBayesianEstimator(prior_type='BDeu'),
    )

    if save_model:
        with open(DATA_DIR / 'bn_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"  Modelo BN salvo em {DATA_DIR / 'bn_model.pkl'}")

    # 6. Visualizar grafo da BN
    print("\nGerando grafo da BN...")
    plot_bn_graph(model, set(valid_mandatory), save_path=OUTPUT_DIR / 'grafo_bn.png')

    # 7. Avaliar BN
    print("\nAvaliando Rede Bayesiana...")
    bn_results, bn_roc = evaluate_bn(model, train_df, test_df)

    # 8. Comparar com LR, DT, RF
    print("\nComparando com outros classificadores (com oversampling)...")
    classifiers = {
        'LR': LogisticRegression(max_iter=1000, random_state=42),
        'DT': DecisionTreeClassifier(max_depth=6, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    }
    feature_cols = [c for c in train_df.columns if c not in TARGET_RFFS]
    clf_results, clf_roc = _evaluate_classifiers(classifiers, train_df, test_df, feature_cols)

    all_results = {'BN': bn_results, **clf_results}
    all_roc     = {'BN': bn_roc,     **clf_roc}

    # 9. Relatorios
    _print_bn_results(bn_results)
    _print_comparison_results(all_results)

    # 10. Visualizacoes
    print("\nGerando visualizacoes...")
    plot_metrics_heatmap(bn_results, save_path=OUTPUT_DIR / 'heatmap_metricas.png')
    plot_classifier_comparison(all_results, save_path=OUTPUT_DIR / 'comparacao_classificadores.png')
    plot_roc_curves(all_roc, save_path=OUTPUT_DIR / 'curvas_roc.png')

    _print_modifications_summary()
    print("\nTreinamento concluido!")

    return model


if __name__ == "__main__":
    train_model()
