"""
Continual Learning Experiment — 3 tasks secuenciales: SPCN vs MLP Naive vs EWC
================================================================================
Protocolo Split-MNIST extendido, class-incremental (MNIST real via sklearn,
con fallback sintetico):
  Task 1: digitos 0-3  (4 clases)
  Task 2: digitos 4-6  (3 clases nuevas)
  Task 3: digitos 7-9  (3 clases nuevas)

Class-incremental: un UNICO head compartido de 10 logits (no un head por
task). Durante el entrenamiento de cada task se calcula la loss solo sobre
los logits de las clases presentes en esa task (masked cross-entropy /
masked softmax), pero el forward pass nunca se enmascara — el modelo ve
los 10 logits y debe aprender a no activar los que no corresponden. En
evaluacion el argmax es siempre sobre los 10 logits, sin mascara: esto es
el test "honesto" que expone el catastrophic forgetting y la confusion
entre clases de distintas tasks.

Esto es deliberadamente mas dificil que un protocolo task-incremental con
heads separados por task — un head por task le daria a cualquier modelo
(incluido el SPCN) un boost artificial de BWT que no refleja resistencia
real al olvido en la representacion compartida.

No se modifica la arquitectura ni los hiperparametros del SPCN respecto a
continual_learning_experiment.py — solo se generaliza el head/loss a 10
clases compartidas y el bucle de entrenamiento a N tasks secuenciales.
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from continual_learning_experiment import DEVICE, load_data, PCLayer

OUT_DIR = 'output/continual'
N_CLASSES_TOTAL = 10
TASK_SPLITS = ((0, 4), (4, 7), (7, 10))
TASK_LABELS_RANGE = ['0-3', '4-6', '7-9']
N_TASKS = 3


# ─────────────────────────────────────────────────────────────────
# MLP class-incremental (naive / EWC) — single shared 10-way head
# ─────────────────────────────────────────────────────────────────

class MLP10(nn.Module):
    def __init__(self, input_dim=784, hidden=256, n_classes=N_CLASSES_TOTAL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def masked_ce(logits, y_global, class_lo, class_hi):
    """Cross-entropy donde solo las columnas [class_lo, class_hi) participan
    de la softmax/loss (logits de otras clases no se enmascaran en el forward,
    solo se excluyen del denominador de esta loss puntual)."""
    sub_logits = logits[:, class_lo:class_hi]
    y_local = y_global - class_lo
    return nn.functional.cross_entropy(sub_logits, y_local)


def train_mlp_task(model, X_tr, y_tr_global, class_lo, class_hi, epochs=10,
                    lr=1e-3, bs=128, extra_loss=None):
    opt = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_tr, y_tr_global), batch_size=bs, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = masked_ce(logits, yb, class_lo, class_hi)
            if extra_loss:
                loss = loss + extra_loss(model)
            loss.backward()
            opt.step()


def eval_mlp_unmasked(model, X_te, y_te_global, bs=256):
    """Argmax sobre los 10 logits, sin mascara — test honesto de class-incremental."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(X_te), bs):
            xb, yb = X_te[i:i+bs], y_te_global[i:i+bs]
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / total


class EWC10:
    """EWC sobre la MLP10 de 10 clases, Fisher calculado con masked CE
    restringida a la task de referencia."""
    def __init__(self, model, X_tr, y_tr_global, class_lo, class_hi,
                 ewc_lambda=400.0, n_samples=300):
        self.lam = ewc_lambda
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._fisher(model, X_tr, y_tr_global, class_lo, class_hi, n_samples)

    def _fisher(self, model, X_tr, y_tr_global, class_lo, class_hi, n):
        F = {nm: torch.zeros_like(p) for nm, p in model.named_parameters() if p.requires_grad}
        model.eval()
        idx = torch.randperm(len(X_tr))[:n]
        for i in idx:
            model.zero_grad()
            logits = model(X_tr[i:i+1])
            loss = masked_ce(logits, y_tr_global[i:i+1], class_lo, class_hi)
            loss.backward()
            for nm, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    F[nm] += p.grad.detach() ** 2
        for nm in F:
            F[nm] /= n
        model.train()
        return F

    def penalty(self, model):
        loss = 0.0
        for nm, p in model.named_parameters():
            if p.requires_grad and nm in self.fisher:
                loss = loss + (self.fisher[nm] * (p - self.params[nm]) ** 2).sum()
        return self.lam * loss


def run_mlp_multitask(data, n_tasks, epochs, use_ewc, ewc_lambda=400.0):
    """Entrena un MLP10 (naive o EWC) secuencialmente sobre n_tasks con
    head unico compartido. Devuelve acc_matrix[i][j] = accuracy (unmasked,
    10-way) en task j+1 evaluado en test, tras terminar de entrenar task i+1."""
    model = MLP10().to(DEVICE)
    ewc_terms = []
    acc_matrix = [[None] * n_tasks for _ in range(n_tasks)]
    label = 'mlp_ewc' if use_ewc else 'mlp_naive'

    for t in range(n_tasks):
        lo, hi = TASK_SPLITS[t]
        Xtr, ytr_local = data[f'task{t+1}']['train']
        ytr_global = ytr_local + lo   # volver a labels globales 0-9

        def extra_loss(m, terms=ewc_terms):
            if not terms:
                return 0.0
            return sum(e.penalty(m) for e in terms)

        train_mlp_task(model, Xtr, ytr_global, lo, hi, epochs=epochs,
                        extra_loss=extra_loss if use_ewc else None)

        if use_ewc:
            ewc_terms.append(EWC10(model, Xtr, ytr_global, lo, hi, ewc_lambda=ewc_lambda))

        for j in range(t + 1):
            jlo, jhi = TASK_SPLITS[j]
            Xte, yte_local = data[f'task{j+1}']['test']
            yte_global = yte_local + jlo
            acc_matrix[t][j] = eval_mlp_unmasked(model, Xte, yte_global)

        print(f"  [{label}] after T{t+1}: " +
              "  ".join(f"T{j+1}={acc_matrix[t][j]:.3f}" for j in range(t + 1)))

    return acc_matrix


# ─────────────────────────────────────────────────────────────────
# SPCN class-incremental — single shared 10-way head
# ─────────────────────────────────────────────────────────────────

def run_spcn_multitask(data, n_tasks, epochs):
    """SPCN con head unico de 10 clases. El cuerpo predictive-coding
    (2 PCLayers, misma regla local, mismos hiperparametros que en
    continual_learning_experiment.py) se entrena de forma continua sin
    reset entre tasks. El head se actualiza con masked softmax: el
    gradiente de cross-entropy solo se calcula sobre las clases de la
    task actual, pero los 10 logits siempre se computan."""
    np_d = data['np']

    layers = [PCLayer(784, 128, lr=0.004, clr=0.002), PCLayer(128, 64, lr=0.004, clr=0.002)]
    hW = np.random.randn(N_CLASSES_TOTAL, 64) * 0.01
    hb = np.zeros(N_CLASSES_TOTAL)
    hlr = 0.003
    protos = np.zeros((N_CLASSES_TOTAL, 64))
    proto_n = np.zeros(N_CLASSES_TOTAL)

    def forward(x):
        states = [x]
        for L in layers:
            states.append(L.forward(states[-1]))
        return states

    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def predict_unmasked(x):
        states = forward(x)
        logits = hW @ states[-1] + hb
        return logits.argmax()

    def evaluate(X, y_global):
        correct = sum(predict_unmasked(X[i]) == y_global[i] for i in range(len(y_global)))
        return correct / len(y_global)

    def train_sample(x, label_global, class_lo, class_hi):
        states = forward(x)
        for i, L in enumerate(layers):
            L.hebbian_update(states[i], states[i + 1])

        top = states[-1]
        a = 0.008
        protos[label_global] = (1 - a) * protos[label_global] + a * top
        proto_n[label_global] += 1

        if proto_n[label_global] > 2:
            dists = np.linalg.norm(protos - top, axis=1)
            dists[label_global] = np.inf
            # solo competimos contra clases ya vistas (proto_n > 0) para que
            # el negativo no sea un prototipo en cero de una clase aun no vista
            seen_mask = proto_n > 0
            dists[~seen_mask] = np.inf
            neg = dists.argmin()
            if np.isfinite(dists[neg]):
                sp, sn = protos[label_global], protos[neg]
                top_layer = layers[-1]
                top_input = states[-2]
                top_layer.contrastive_update(top_input, sp, sn)

        # Head update: masked softmax cross-entropy grad sobre [class_lo, class_hi)
        logits = hW @ top + hb
        sub_logits = logits[class_lo:class_hi]
        probs_sub = softmax(sub_logits)
        full_probs = np.zeros(N_CLASSES_TOTAL)
        full_probs[class_lo:class_hi] = probs_sub
        oh = np.zeros(N_CLASSES_TOTAL)
        oh[label_global] = 1.0
        grad = full_probs - oh   # grad es 0 fuera de [class_lo, class_hi)
        hW[:] -= hlr * np.outer(grad, top)
        hb[:] -= hlr * grad

    acc_matrix = [[None] * n_tasks for _ in range(n_tasks)]

    for t in range(n_tasks):
        lo, hi = TASK_SPLITS[t]
        Xtr, ytr_local = np_d[f'X{t+1}tr'], np_d[f'y{t+1}tr']
        ytr_global = ytr_local + lo

        for ep in range(epochs):
            idx = np.random.permutation(len(Xtr))
            for i in idx:
                train_sample(Xtr[i], ytr_global[i], lo, hi)

        for j in range(t + 1):
            jlo, jhi = TASK_SPLITS[j]
            Xte, yte_local = np_d[f'X{j+1}te'], np_d[f'y{j+1}te']
            yte_global = yte_local + jlo
            acc_matrix[t][j] = evaluate(Xte, yte_global)

        print(f"  [spcn] after T{t+1}: " +
              "  ".join(f"T{j+1}={acc_matrix[t][j]:.3f}" for j in range(t + 1)))

    return acc_matrix


# ─────────────────────────────────────────────────────────────────
# METRICAS
# ─────────────────────────────────────────────────────────────────

def cl_metrics_multitask(acc_matrix, n_tasks):
    """acc_matrix[i][j] = acc en task j+1 tras entrenar hasta task i+1 (j<=i).

    BWT_avg: promedio sobre todos los pares j < i de
        (acc_matrix[i][j] - acc_matrix[j][j])
    Plasticity_avg: promedio de acc_matrix[t][t] (peak de cada task,
      medido inmediatamente despues de entrenarla).
    """
    peak = [acc_matrix[t][t] for t in range(n_tasks)]
    bwt_pairs = []
    for i in range(n_tasks):
        for j in range(i):
            bwt_pairs.append(acc_matrix[i][j] - peak[j])
    bwt_avg = float(np.mean(bwt_pairs)) if bwt_pairs else 0.0
    plasticity_avg = float(np.mean(peak))

    final_accs = [acc_matrix[n_tasks - 1][j] for j in range(n_tasks)]
    retention_ratios = [final_accs[j] / max(peak[j], 1e-6) for j in range(n_tasks - 1)]
    retention_avg = float(np.mean(retention_ratios)) if retention_ratios else 1.0

    return {
        'peak_per_task': peak,
        'final_per_task': final_accs,
        'BWT_avg': bwt_avg,
        'Plasticity_avg': plasticity_avg,
        'Retention_ratio_avg': retention_avg,
    }


# ─────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────

def plot_3tasks(results, metrics, timing, path=f'{OUT_DIR}/cl_3tasks_results.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    BG, CARD, GRID, W = '#0d1117', '#161b22', '#21262d', '#c9d1d9'
    MODELS = ['mlp_naive', 'mlp_ewc', 'spcn']
    LABELS = {'mlp_naive': 'MLP Naive', 'mlp_ewc': 'MLP + EWC', 'spcn': 'SPCN (local PC)'}
    COLORS = {'mlp_naive': '#f85149', 'mlp_ewc': '#e3b341', 'spcn': '#3fb950'}
    n_tasks = N_TASKS

    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    def style_ax(ax, title):
        ax.set_facecolor(CARD)
        ax.set_title(title, color=W, fontsize=11, pad=8, fontweight='bold')
        ax.tick_params(colors=W, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.yaxis.label.set_color(W)
        ax.xaxis.label.set_color(W)

    # A: accuracy de T1 a lo largo de las 3 fases
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, 'Retención Task 1\n(a través de T1→T2→T3, 10-way)')
    x = np.arange(n_tasks)
    w = 0.22
    for i, m in enumerate(MODELS):
        vals = [results[m][t][0] if results[m][t][0] is not None else np.nan for t in range(n_tasks)]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=LABELS[m], color=COLORS[m],
                      alpha=0.85, edgecolor=BG, linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{v:.2f}', ha='center', va='bottom', color=W, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Después\nde T{t+1}' for t in range(n_tasks)], color=W)
    ax.set_ylabel('Accuracy (Task 1, 10-way)')
    ax.set_ylim(0, 1.1)
    ax.legend(facecolor=CARD, labelcolor=W, fontsize=8, framealpha=0.8)

    # B: Plasticity promedio
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'Plasticidad Promedio\n(acc. peak en task nueva, 10-way)')
    vals2 = [metrics[m]['Plasticity_avg'] for m in MODELS]
    bars2 = ax2.bar([LABELS[m] for m in MODELS], vals2, color=[COLORS[m] for m in MODELS],
                    alpha=0.85, edgecolor=BG, linewidth=0.5)
    for bar, v in zip(bars2, vals2):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{v:.2f}', ha='center', va='bottom', color=W, fontsize=10)
    ax2.set_ylabel('Accuracy promedio (peak por task)')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='x', labelsize=8)

    # C: BWT promedio vs Plasticity promedio
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, 'Tradeoff BWT vs Plasticity\n(promedio sobre 3 tasks, 10-way)')
    for m in MODELS:
        ax3.scatter(metrics[m]['BWT_avg'], metrics[m]['Plasticity_avg'], s=250, c=COLORS[m],
                    zorder=5, edgecolors=W, linewidth=1.2, label=LABELS[m])
        ax3.annotate(LABELS[m], (metrics[m]['BWT_avg'], metrics[m]['Plasticity_avg']),
                     textcoords='offset points', xytext=(6, 4), color=COLORS[m], fontsize=8)
    ax3.axvline(0, color=W, linestyle='--', alpha=0.3, linewidth=1)
    ax3.set_xlabel('BWT promedio (0 = sin olvido, < 0 = olvido)')
    ax3.set_ylabel('Plasticity promedio')
    ax3.xaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)

    # D: tabla resumen con matriz completa
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor(CARD)
    ax4.axis('off')
    ax4.set_title('Matriz de Accuracy Final por Task (10-way, class-incremental) y Métricas Promedio',
                  color=W, fontsize=12, pad=10, fontweight='bold', loc='left')

    rows = []
    for m in MODELS:
        am = results[m]
        row_final = [f'{am[n_tasks-1][j]:.3f}' if am[n_tasks-1][j] is not None else '—' for j in range(n_tasks)]
        rows.append([LABELS[m], *row_final,
                     f"{metrics[m]['BWT_avg']:+.3f}",
                     f"{metrics[m]['Plasticity_avg']:.3f}",
                     f"{metrics[m]['Retention_ratio_avg']:.3f}",
                     f"{timing[m]:.1f}s"])

    col_labels = ['Modelo', f'Final T1 ({TASK_LABELS_RANGE[0]})',
                  f'Final T2 ({TASK_LABELS_RANGE[1]})',
                  f'Final T3 ({TASK_LABELS_RANGE[2]})',
                  'BWT avg', 'Plasticity avg', 'Retention avg', 'Tiempo']
    tbl = ax4.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center',
                    bbox=[0, -0.1, 1, 1.1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor('#1f6feb')
            cell.set_text_props(color=W, fontweight='bold')
        elif col == 0:
            cell.set_facecolor('#21262d')
            cell.set_text_props(color=COLORS[MODELS[row - 1]], fontweight='bold')
        else:
            cell.set_facecolor(CARD)
            cell.set_text_props(color=W)

    fig.suptitle('Continual Learning Benchmark — 3 Tasks Secuenciales (Class-Incremental, 10-way)\n'
                 f'T1: dígitos {TASK_LABELS_RANGE[0]}  ·  T2: dígitos {TASK_LABELS_RANGE[1]}  ·  T3: dígitos {TASK_LABELS_RANGE[2]}',
                 color=W, fontsize=14, y=0.98, fontweight='bold')

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Plot guardado: {path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_data(n_train=5000, n_test=1000, task_splits=TASK_SPLITS)
    epochs = 10

    results = {}
    timing = {}

    print("\n[1/3] MLP Naive (3 tasks, class-incremental 10-way)")
    t0 = time.time()
    results['mlp_naive'] = run_mlp_multitask(data, N_TASKS, epochs, use_ewc=False)
    timing['mlp_naive'] = time.time() - t0

    print("\n[2/3] MLP + EWC (3 tasks, class-incremental 10-way)")
    t0 = time.time()
    results['mlp_ewc'] = run_mlp_multitask(data, N_TASKS, epochs, use_ewc=True)
    timing['mlp_ewc'] = time.time() - t0

    print("\n[3/3] SPCN (3 tasks, class-incremental 10-way, local predictive coding)")
    t0 = time.time()
    results['spcn'] = run_spcn_multitask(data, N_TASKS, epochs)
    timing['spcn'] = time.time() - t0

    metrics = {m: cl_metrics_multitask(results[m], N_TASKS) for m in results}

    print("\n" + "=" * 62)
    print("METRICAS FINALES — 3 TASKS (class-incremental)")
    print("=" * 62)
    for m, mx in metrics.items():
        print(f"\n  {m.upper()}")
        for k, v in mx.items():
            tag = ' ← KEY' if k in ('BWT_avg', 'Retention_ratio_avg') else ''
            print(f"    {k:24s}: {v}{tag}")

    with open(f'{OUT_DIR}/cl_3tasks_metrics.json', 'w') as f:
        json.dump({
            'task_splits': TASK_SPLITS,
            'accuracy_matrix': results,
            'metrics': metrics,
            'timing': timing,
        }, f, indent=2)

    plot_3tasks(results, metrics, timing)
    print("\nListo!")
