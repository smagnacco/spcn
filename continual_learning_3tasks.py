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
from utils import specialization_index, population_specialization

OUT_DIR = 'output/continual'
N_CLASSES_TOTAL = 10
TASK_SPLITS = ((0, 4), (4, 7), (7, 10))
TASK_LABELS_RANGE = ['0-3', '4-6', '7-9']
N_TASKS = 3
SPCN_SPARSE_FRAC = 0.15    # k_active = 15% de out_dim por PCLayer (Cambio 1)
SPCN_RECRUIT_BETA = 1.0    # sesgo de reclutamiento por commitment normalizado [0,1] (Fix correcto)
SPCN_CLR = 0.01            # contrastive learning rate, subido de 0.002 (un cambio a la vez, hlr intacto)
SPCN_PROTECT_GAMMA = 2.0   # rigidez por consistencia sqrt(freq); punto operativo del barrido (Plast~EWC)


# ─────────────────────────────────────────────────────────────────
# MLP class-incremental (naive / EWC) — single shared 10-way head
# ─────────────────────────────────────────────────────────────────

class MLP10(nn.Module):
    def __init__(self, input_dim=784, hidden=256, n_classes=N_CLASSES_TOTAL):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x))

    def get_hidden(self, x):
        """Activaciones de la penultima capa (encoder), para specialization_index."""
        with torch.no_grad():
            return self.encoder(x).cpu().numpy()


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
    10-way) en task j+1 evaluado en test, tras terminar de entrenar task i+1.
    Tambien devuelve task_activations[t] = activaciones del encoder (penultima
    capa) sobre el test set de la task t, recogidas justo despues de
    entrenar esa task (peak), para specialization_index."""
    model = MLP10().to(DEVICE)
    ewc_terms = []
    acc_matrix = [[None] * n_tasks for _ in range(n_tasks)]
    task_activations = [None] * n_tasks
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

        Xte_t, _ = data[f'task{t+1}']['test']
        task_activations[t] = model.get_hidden(Xte_t)

        for j in range(t + 1):
            jlo, jhi = TASK_SPLITS[j]
            Xte, yte_local = data[f'task{j+1}']['test']
            yte_global = yte_local + jlo
            acc_matrix[t][j] = eval_mlp_unmasked(model, Xte, yte_global)

        print(f"  [{label}] after T{t+1}: " +
              "  ".join(f"T{j+1}={acc_matrix[t][j]:.3f}" for j in range(t + 1)))

    return acc_matrix, task_activations


# ─────────────────────────────────────────────────────────────────
# SPCN class-incremental — single shared 10-way head
# ─────────────────────────────────────────────────────────────────

def run_spcn_multitask(data, n_tasks, epochs, sparse_frac=SPCN_SPARSE_FRAC,
                       recruit_beta=SPCN_RECRUIT_BETA, protect_gamma=SPCN_PROTECT_GAMMA):
    """SPCN con head unico de 10 clases. El cuerpo predictive-coding
    (2 PCLayers, misma regla local, mismos hiperparametros que en
    continual_learning_experiment.py) se entrena de forma continua sin
    reset entre tasks. El head se actualiza con masked softmax: el
    gradiente de cross-entropy solo se calcula sobre las clases de la
    task actual, pero los 10 logits siempre se computan.

    sparse_frac aplica el top-k mask (Cambio 1) a cada PCLayer: solo
    k_active = round(sparse_frac * out_dim) unidades sobreviven por
    forward pass, y solo esas reciben update Hebbiano/contrastivo.
    Arquitectura (in/h1/h2), clr/prototypes y distance_lambda intactos.

    Devuelve tambien task_activations[t] = activaciones de la top layer
    (h2, post top-k mask) sobre el test set de la task t, recogidas justo
    despues de entrenarla, para specialization_index."""
    np_d = data['np']

    k1 = max(1, round(sparse_frac * 128)) if sparse_frac else None
    k2 = max(1, round(sparse_frac * 64)) if sparse_frac else None
    layers = [PCLayer(784, 128, lr=0.004, clr=SPCN_CLR, k_active=k1,
                      recruit_beta=recruit_beta, protect_gamma=protect_gamma),
              PCLayer(128, 64, lr=0.004, clr=SPCN_CLR, k_active=k2,
                      recruit_beta=recruit_beta, protect_gamma=protect_gamma)]
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

    def collect_top_activations(X):
        return np.array([forward(x)[-1] for x in X])

    # Instrumentacion (metrica honesta): contador por-task de cuantas veces
    # cada unidad estuvo en el top-k, por layer. Se resetea por task. El set
    # "consistente" = unidades en top-k en >= frac de las muestras de la task.
    topk_count_this_task = [np.zeros(128), np.zeros(64)]
    n_samples_this_task = [0]

    def train_sample(x, label_global, class_lo, class_hi):
        states = forward(x)
        for i, L in enumerate(layers):
            L.hebbian_update(states[i], states[i + 1])
            topk_count_this_task[i][L._last_active_idx] += 1.0
            L.mark_commitment()   # tracker de commitment inmune a homeostasis
        n_samples_this_task[0] += 1

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
    train_acc_per_task = [None] * n_tasks   # train acc de cada task al final de su fase
    task_activations = [None] * n_tasks
    consistent_sets = []   # consistent_sets[t] = [set_L1, set_L2] de la task t
    # Umbral relativo a la frecuencia uniforme esperada (k/out_dim): una unidad
    # es "del nucleo" si aparece en el top-k >= OVERUSE_MULT veces lo esperado
    # por azar. Un umbral absoluto (ej 0.5) da sets vacios porque con top-k
    # por-muestra ninguna neurona individual supera ~0.22 de las muestras.
    OVERUSE_MULT = 1.5
    uniform_frac = [k1 / 128, k2 / 64]

    def jaccard(a, b):
        if not a and not b:
            return 0.0
        return len(a & b) / len(a | b)

    for t in range(n_tasks):
        lo, hi = TASK_SPLITS[t]
        Xtr, ytr_local = np_d[f'X{t+1}tr'], np_d[f'y{t+1}tr']
        ytr_global = ytr_local + lo

        topk_count_this_task[0][:] = 0
        topk_count_this_task[1][:] = 0
        n_samples_this_task[0] = 0

        # Congelar protection_strength pre-task: la proteccion de esta task
        # usa este snapshot (solo protege nucleos consistentes de tasks PREVIAS)
        for L in layers:
            L.freeze_protection()

        for ep in range(epochs):
            idx = np.random.permutation(len(Xtr))
            for i in idx:
                train_sample(Xtr[i], ytr_global[i], lo, hi)

        Xte_t = np_d[f'X{t+1}te']
        task_activations[t] = collect_top_activations(Xte_t)

        # Train accuracy de esta task al final de su fase (10-way unmasked),
        # para distinguir under-fitting (clr bajo) de overfitting (neuronas
        # frescas memorizando pocas muestras): train alto + test bajo =
        # overfitting; train bajo + test bajo = under-fitting.
        train_acc_per_task[t] = evaluate(Xtr, ytr_global)

        # Metrica honesta: set "consistente" = unidades que estuvieron en el
        # top-k en >= CONSISTENT_FRAC de las muestras de esta task.
        n_s = n_samples_this_task[0]
        task_sets = []
        for li in range(2):
            counts = topk_count_this_task[li]
            thresh = OVERUSE_MULT * uniform_frac[li] * n_s
            consistent = set(np.where(counts >= thresh)[0].tolist())
            task_sets.append(consistent)
            # Consolidar la consistencia de esta task en la senal de proteccion:
            # protection_strength = max(prev, sqrt(freq)). La compresion sqrt
            # levanta la COLA de la distribucion (freq 0.25 -> 0.5) para que las
            # muchas neuronas de baja-media consistencia que igual sostienen la
            # representacion de la task queden protegidas, sin tocar las de
            # freq~0 (sqrt(0)=0) que son las frescas disponibles para tasks
            # futuras. Sin umbral/binarizacion: proteccion graduada continua.
            freq_in_task = counts / max(n_s, 1)
            layers[li].consolidate_protection(np.sqrt(freq_in_task))
        consistent_sets.append(task_sets)

        # Jaccard contra T1 (set consistente) por layer, para detectar si los
        # nucleos de cada task se separan (objetivo: Jaccard < 0.5)
        if t > 0:
            j_l1 = jaccard(task_sets[0], consistent_sets[0][0])
            j_l2 = jaccard(task_sets[1], consistent_sets[0][1])
            print(f"  [spcn] T{t+1} nucleo consistente — "
                  f"L1: {len(task_sets[0])} units (Jaccard vs T1 = {j_l1:.2f})  "
                  f"L2: {len(task_sets[1])} units (Jaccard vs T1 = {j_l2:.2f})")
        else:
            print(f"  [spcn] T1 nucleo consistente — "
                  f"L1: {len(task_sets[0])} units  L2: {len(task_sets[1])} units")

        for j in range(t + 1):
            jlo, jhi = TASK_SPLITS[j]
            Xte, yte_local = np_d[f'X{j+1}te'], np_d[f'y{j+1}te']
            yte_global = yte_local + jlo
            acc_matrix[t][j] = evaluate(Xte, yte_global)

        print(f"  [spcn] after T{t+1}: " +
              "  ".join(f"T{j+1}={acc_matrix[t][j]:.3f}" for j in range(t + 1)) +
              f"   (train T{t+1}={train_acc_per_task[t]:.3f})")

    # Resumen Jaccard entre nucleos consistentes de pares de tasks (L2 = top layer)
    overlap_summary = {}
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            overlap_summary[f'T{i+1}_vs_T{j+1}'] = {
                'jaccard_L1': jaccard(consistent_sets[i][0], consistent_sets[j][0]),
                'jaccard_L2': jaccard(consistent_sets[i][1], consistent_sets[j][1]),
            }

    return acc_matrix, task_activations, overlap_summary


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


def specialization_metrics(task_activations, n_tasks, threshold=0.5):
    """Para cada par de tasks (i, j) calcula specialization_index sobre las
    activaciones de la capa recogida (top layer / encoder), y resume con
    population_specialization. task_activations[t] son las activaciones
    sobre el test set de la task t, recogidas justo despues de entrenarla
    (mismo punto en el tiempo en que se mide peak_per_task)."""
    pair_results = {}
    si_values = []
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            si = specialization_index(task_activations[i], task_activations[j])
            pop_spec = population_specialization(si, threshold=threshold)
            pair_results[f'T{i+1}_vs_T{j+1}'] = {
                'population_specialization': pop_spec,
                'mean_abs_SI': float(np.mean(np.abs(si))),
            }
            si_values.append(pop_spec)

    return {
        'pairs': pair_results,
        'population_specialization_avg': float(np.mean(si_values)) if si_values else 0.0,
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
    activations = {}

    print("\n[1/3] MLP Naive (3 tasks, class-incremental 10-way)")
    t0 = time.time()
    results['mlp_naive'], activations['mlp_naive'] = run_mlp_multitask(data, N_TASKS, epochs, use_ewc=False)
    timing['mlp_naive'] = time.time() - t0

    print("\n[2/3] MLP + EWC (3 tasks, class-incremental 10-way)")
    t0 = time.time()
    results['mlp_ewc'], activations['mlp_ewc'] = run_mlp_multitask(data, N_TASKS, epochs, use_ewc=True)
    timing['mlp_ewc'] = time.time() - t0

    print(f"\n[3/3] SPCN (3 tasks, class-incremental 10-way, local predictive coding, "
          f"sparse_frac={SPCN_SPARSE_FRAC})")
    t0 = time.time()
    results['spcn'], activations['spcn'], spcn_overlap_log = run_spcn_multitask(data, N_TASKS, epochs)
    timing['spcn'] = time.time() - t0

    metrics = {m: cl_metrics_multitask(results[m], N_TASKS) for m in results}
    spec_metrics = {m: specialization_metrics(activations[m], N_TASKS) for m in activations}

    print("\n" + "=" * 62)
    print("METRICAS FINALES — 3 TASKS (class-incremental)")
    print("=" * 62)
    for m, mx in metrics.items():
        print(f"\n  {m.upper()}")
        for k, v in mx.items():
            tag = ' ← KEY' if k in ('BWT_avg', 'Retention_ratio_avg') else ''
            print(f"    {k:24s}: {v}{tag}")
        print(f"    population_specialization_avg: {spec_metrics[m]['population_specialization_avg']:.4f}")

    with open(f'{OUT_DIR}/cl_3tasks_metrics.json', 'w') as f:
        json.dump({
            'task_splits': TASK_SPLITS,
            'spcn_sparse_frac': SPCN_SPARSE_FRAC,
            'accuracy_matrix': results,
            'metrics': metrics,
            'specialization_metrics': spec_metrics,
            'timing': timing,
        }, f, indent=2)

    plot_3tasks(results, metrics, timing)
    print("\nListo!")
