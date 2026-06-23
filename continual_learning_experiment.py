"""
Continual Learning Experiment: SPCN vs MLP Naive vs EWC
========================================================
Protocolo Split-MNIST (MNIST real via sklearn, con fallback sintetico):
  Task 1: digitos 0-4  (5 clases)
  Task 2: digitos 5-9  (5 clases nuevas)

Metricas:
  - Accuracy por tarea despues de cada fase
  - BWT: Backward Transfer (retencion / olvido)
  - FWT: Forward Transfer (transferencia positiva)
  - Plasticity: accuracy final en Task 2
"""

import numpy as np
import json
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

OUT_DIR = 'output/continual'

# ─────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────

def _load_real_mnist(n_train_total, n_test_total, seed=42):
    from sklearn.datasets import fetch_openml
    print("Descargando MNIST real (sklearn fetch_openml)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    train_idx = idx[:n_train_total]
    test_idx = idx[n_train_total:n_train_total + n_test_total]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _load_synthetic_mnist(n_train_total, n_test_total, seed=42):
    print("Fallback: generando datos sinteticos estructurados...")
    rng = np.random.RandomState(seed)

    def gen(n):
        y = rng.randint(0, 10, size=n)
        X = np.zeros((n, 784), dtype=np.float32)
        for i, label in enumerate(y):
            region = label % 4
            base = np.zeros((28, 28), dtype=np.float32)
            r0, r1 = (0, 14) if region in (0, 1) else (14, 28)
            c0, c1 = (0, 14) if region in (0, 2) else (14, 28)
            base[r0:r1, c0:c1] = 1.0
            noise = rng.normal(0, 0.1, size=(28, 28)).astype(np.float32)
            X[i] = np.clip(base + noise, 0, 1).reshape(-1)
        return X, y

    Xtr, ytr = gen(n_train_total)
    Xte, yte = gen(n_test_total)
    return Xtr, ytr, Xte, yte


def load_data(n_train=5000, n_test=1000, task_splits=((0, 5), (5, 10))):
    """Carga MNIST real (fetch_openml); si falla por red, cae a sintetico.

    n_train / n_test son por task. task_splits define los rangos de
    digitos [lo, hi) de cada task (por defecto: split-MNIST 2 tasks).
    """
    n_tasks = len(task_splits)
    n_train_total = n_train * n_tasks * 3   # margen para filtrar por clase
    n_test_total = n_test * n_tasks * 3

    try:
        X_all, y_all, Xt_all, yt_all = _load_real_mnist(n_train_total, n_test_total)
    except Exception as e:
        print(f"  fetch_openml fallo ({type(e).__name__}: {e}) -> usando sintetico")
        X_all, y_all, Xt_all, yt_all = _load_synthetic_mnist(n_train_total, n_test_total)

    def split_task(X, y, Xt, yt, lo, hi, relabel=False):
        m_tr = (y >= lo) & (y < hi)
        m_te = (yt >= lo) & (yt < hi)
        Xtr, ytr = X[m_tr][:n_train], y[m_tr][:n_train]
        Xte, yte = Xt[m_te][:n_test], yt[m_te][:n_test]
        if relabel:
            ytr = ytr - lo
            yte = yte - lo
        return Xtr, ytr, Xte, yte

    def tt(X, y):
        return torch.FloatTensor(X).to(DEVICE), torch.LongTensor(y).to(DEVICE)

    tasks = {}
    np_d = {}
    for i, (lo, hi) in enumerate(task_splits, start=1):
        Xtr, ytr, Xte, yte = split_task(X_all, y_all, Xt_all, yt_all, lo, hi, relabel=True)
        print(f"  Task{i} (digitos {lo}-{hi-1}): {len(Xtr)} train / {len(Xte)} test")
        tasks[f'task{i}'] = {'train': tt(Xtr, ytr), 'test': tt(Xte, yte)}
        np_d[f'X{i}tr'], np_d[f'y{i}tr'] = Xtr, ytr
        np_d[f'X{i}te'], np_d[f'y{i}te'] = Xte, yte

    tasks['np'] = np_d
    tasks['n_tasks'] = n_tasks
    tasks['task_splits'] = task_splits
    return tasks

# ─────────────────────────────────────────────────────────────────
# 2. MLP (naive fine-tuning)
# ─────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden=256, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(model, X_tr, y_tr, epochs=10, lr=1e-3, bs=128, extra_loss=None,
              retention_probe=None):
    """retention_probe: callable(epoch_idx) llamado al final de cada epoch,
    para loggear accuracy en una tarea anterior mientras se entrena esta."""
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            if extra_loss:
                loss = loss + extra_loss(model)
            loss.backward()
            opt.step()
        if retention_probe:
            retention_probe(ep)

def eval_mlp(model, X_te, y_te, bs=256):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(X_te), bs):
            xb, yb = X_te[i:i+bs], y_te[i:i+bs]
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += len(yb)
    return correct / total

# ─────────────────────────────────────────────────────────────────
# 3. EWC
# ─────────────────────────────────────────────────────────────────

class EWC:
    """Elastic Weight Consolidation — Kirkpatrick et al. 2017."""
    def __init__(self, model, X_tr, y_tr, ewc_lambda=400.0, n_samples=300):
        self.lam    = ewc_lambda
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._fisher(model, X_tr, y_tr, n_samples)

    def _fisher(self, model, X_tr, y_tr, n):
        F = {nm: torch.zeros_like(p) for nm, p in model.named_parameters() if p.requires_grad}
        model.eval()
        idx = torch.randperm(len(X_tr))[:n]
        for i in idx:
            model.zero_grad()
            loss = nn.CrossEntropyLoss()(model(X_tr[i:i+1]), y_tr[i:i+1])
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
                loss += (self.fisher[nm] * (p - self.params[nm]) ** 2).sum()
        return self.lam * loss

# ─────────────────────────────────────────────────────────────────
# 4. SPCN
# ─────────────────────────────────────────────────────────────────

class PCLayer:
    """Capa de Predictive Coding con homeostasis y update contrastivo."""
    def __init__(self, in_dim, out_dim, lr=0.004, clr=0.002, lam=0.05, target=0.15, ema=0.005):
        self.W   = np.random.randn(out_dim, in_dim) / np.sqrt(in_dim)
        self.b   = np.zeros(out_dim)
        self.lr  = lr
        self.clr = clr
        self.lam = lam
        self.target = target
        self.ema_rate = ema
        self.ema_state = np.full(out_dim, target)

    def _sig(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def forward(self, x):
        return self._sig(self.W @ x + self.b)

    def hebbian_update(self, x, state):
        pred = self.forward(x)
        err  = state - pred
        self.W += self.lr * np.outer(err, x)
        self.b += self.lr * err
        # Homeostasis
        self.ema_state += self.ema_rate * (state - self.ema_state)
        self.b += self.ema_rate * (self.target - self.ema_state)

    def contrastive_update(self, x, s_pos, s_neg):
        diff = np.tanh(s_pos - s_neg)
        self.W += self.clr * np.outer(diff, x)
        self.b += self.clr * diff


class SPCN:
    """
    Red de Predictive Coding de 2 capas ocultas + head lineal.
    El aprendizaje es local — sin backprop global.
    Los pesos existentes son protegidos por la penalizacion de homeostasis
    y el target de firing rate, lo que naturalmente ralentiza el olvido.
    """
    def __init__(self, in_dim=784, h1=128, h2=64, n_classes=5,
                 lr=0.004, clr=0.002):
        self.layers   = [PCLayer(in_dim, h1, lr, clr), PCLayer(h1, h2, lr, clr)]
        self.n_cls    = n_classes
        self.hW       = np.random.randn(n_classes, h2) * 0.01
        self.hb       = np.zeros(n_classes)
        self.hlr      = 0.003
        self.protos   = np.zeros((n_classes, h2))
        self.proto_n  = np.zeros(n_classes)

    def _forward(self, x):
        states = [x]
        for L in self.layers:
            states.append(L.forward(states[-1]))
        return states

    def _softmax(self, z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def predict(self, x):
        s = self._forward(x)
        return (self.hW @ s[-1] + self.hb).argmax()

    def evaluate(self, X, y):
        return sum(self.predict(X[i]) == y[i] for i in range(len(y))) / len(y)

    def train_sample(self, x, label):
        states = self._forward(x)

        # ── Hebbian local ──
        for i, L in enumerate(self.layers):
            L.hebbian_update(states[i], states[i+1])

        # ── Actualizar prototipo ──
        top = states[-1]
        a   = 0.008
        self.protos[label]  = (1-a)*self.protos[label] + a*top
        self.proto_n[label] += 1

        # ── Contrastivo: solo top layer ──
        if self.proto_n[label] > 2:
            dists = np.linalg.norm(self.protos - top, axis=1)
            dists[label] = np.inf
            neg = dists.argmin()
            sp  = self.protos[label]
            sn  = self.protos[neg]
            # Aplicar contrastivo solo en top layer con su input correcta
            top_layer = self.layers[-1]
            top_input = states[-2]  # input de la top layer
            top_layer.contrastive_update(top_input, sp, sn)

        # ── Head update (cross-entropy grad) ──
        logits = self.hW @ top + self.hb
        probs  = self._softmax(logits)
        oh     = np.zeros(self.n_cls); oh[label] = 1.0
        grad   = probs - oh
        self.hW -= self.hlr * np.outer(grad, top)
        self.hb -= self.hlr * grad

    def train_epoch(self, X, y):
        idx = np.random.permutation(len(X))
        for i in idx:
            self.train_sample(X[i], y[i])

# ─────────────────────────────────────────────────────────────────
# 5. EXPERIMENTO
# ─────────────────────────────────────────────────────────────────

def run_experiment(data, epochs=10, ewc_lambda=400.0):
    res    = defaultdict(lambda: defaultdict(dict))
    timing = {}
    retention_curves = {}   # model -> list of T1 accuracy, one per T2 epoch
    np_d   = data['np']

    X1tr, y1tr = data['task1']['train']
    X1te, y1te = data['task1']['test']
    X2tr, y2tr = data['task2']['train']
    X2te, y2te = data['task2']['test']

    # ── MLP Naive ──
    print("\n[1/3] MLP Naive")
    mlp = MLP().to(DEVICE)
    t0  = time.time()
    train_mlp(mlp, X1tr, y1tr, epochs=epochs)
    res['mlp_naive']['after_t1']['t1'] = eval_mlp(mlp, X1te, y1te)
    res['mlp_naive']['after_t1']['t2'] = eval_mlp(mlp, X2te, y2te)
    print(f"  After T1 → T1:{res['mlp_naive']['after_t1']['t1']:.3f}  T2:{res['mlp_naive']['after_t1']['t2']:.3f}")

    curve = []
    def probe_naive(ep):
        curve.append(eval_mlp(mlp, X1te, y1te))
    train_mlp(mlp, X2tr, y2tr, epochs=epochs, retention_probe=probe_naive)
    retention_curves['mlp_naive'] = curve
    res['mlp_naive']['after_t2']['t1'] = eval_mlp(mlp, X1te, y1te)
    res['mlp_naive']['after_t2']['t2'] = eval_mlp(mlp, X2te, y2te)
    timing['mlp_naive'] = time.time() - t0
    print(f"  After T2 → T1:{res['mlp_naive']['after_t2']['t1']:.3f}  T2:{res['mlp_naive']['after_t2']['t2']:.3f}")

    # ── MLP EWC ──
    print("\n[2/3] MLP + EWC")
    mlp_ewc = MLP().to(DEVICE)
    t0 = time.time()
    train_mlp(mlp_ewc, X1tr, y1tr, epochs=epochs)
    res['mlp_ewc']['after_t1']['t1'] = eval_mlp(mlp_ewc, X1te, y1te)
    res['mlp_ewc']['after_t1']['t2'] = eval_mlp(mlp_ewc, X2te, y2te)
    print(f"  After T1 → T1:{res['mlp_ewc']['after_t1']['t1']:.3f}  T2:{res['mlp_ewc']['after_t1']['t2']:.3f}")
    ewc = EWC(mlp_ewc, X1tr, y1tr, ewc_lambda=ewc_lambda)

    curve = []
    def probe_ewc(ep):
        curve.append(eval_mlp(mlp_ewc, X1te, y1te))
    train_mlp(mlp_ewc, X2tr, y2tr, epochs=epochs, extra_loss=ewc.penalty,
              retention_probe=probe_ewc)
    retention_curves['mlp_ewc'] = curve
    res['mlp_ewc']['after_t2']['t1'] = eval_mlp(mlp_ewc, X1te, y1te)
    res['mlp_ewc']['after_t2']['t2'] = eval_mlp(mlp_ewc, X2te, y2te)
    timing['mlp_ewc'] = time.time() - t0
    print(f"  After T2 → T1:{res['mlp_ewc']['after_t2']['t1']:.3f}  T2:{res['mlp_ewc']['after_t2']['t2']:.3f}")

    # ── SPCN ──
    print("\n[3/3] SPCN (predictive coding local)")
    spcn = SPCN(in_dim=784, h1=128, h2=64, n_classes=5)
    t0 = time.time()

    for ep in range(epochs):
        spcn.train_epoch(np_d['X1tr'], np_d['y1tr'])
        if ep % max(1, epochs//4) == 0:
            a = spcn.evaluate(np_d['X1te'], np_d['y1te'])
            print(f"  T1 ep{ep}: {a:.3f}")

    res['spcn']['after_t1']['t1'] = spcn.evaluate(np_d['X1te'], np_d['y1te'])
    res['spcn']['after_t1']['t2'] = spcn.evaluate(np_d['X2te'], np_d['y2te'])
    print(f"  After T1 → T1:{res['spcn']['after_t1']['t1']:.3f}  T2:{res['spcn']['after_t1']['t2']:.3f}")

    # SPCN continua aprendiendo sin reset — el aprendizaje local es la proteccion
    curve = []
    for ep in range(epochs):
        spcn.train_epoch(np_d['X2tr'], np_d['y2tr'])
        curve.append(spcn.evaluate(np_d['X1te'], np_d['y1te']))
        if ep % max(1, epochs//4) == 0:
            a = spcn.evaluate(np_d['X2te'], np_d['y2te'])
            print(f"  T2 ep{ep}: {a:.3f}")
    retention_curves['spcn'] = curve

    res['spcn']['after_t2']['t1'] = spcn.evaluate(np_d['X1te'], np_d['y1te'])
    res['spcn']['after_t2']['t2'] = spcn.evaluate(np_d['X2te'], np_d['y2te'])
    timing['spcn'] = time.time() - t0
    print(f"  After T2 → T1:{res['spcn']['after_t2']['t1']:.3f}  T2:{res['spcn']['after_t2']['t2']:.3f}")

    return res, timing, retention_curves

# ─────────────────────────────────────────────────────────────────
# 6. METRICAS CL
# ─────────────────────────────────────────────────────────────────

def cl_metrics(res):
    metrics = {}
    for m, r in res.items():
        t1_t1 = r['after_t1']['t1']
        t1_t2 = r['after_t2']['t1']
        t2_t1 = r['after_t1']['t2']   # FWT proxy
        t2_t2 = r['after_t2']['t2']
        metrics[m] = {
            'T1_peak':          t1_t1,
            'T1_retained':      t1_t2,
            'BWT':              t1_t2 - t1_t1,
            'FWT_proxy':        t2_t1,
            'Plasticity_T2':    t2_t2,
            'Retention_ratio':  t1_t2 / max(t1_t1, 1e-6),
        }
    return metrics

# ─────────────────────────────────────────────────────────────────
# 7. PLOT
# ─────────────────────────────────────────────────────────────────

def plot(res, metrics, timing, path=f'{OUT_DIR}/cl_results.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    BG   = '#0d1117'
    CARD = '#161b22'
    GRID = '#21262d'
    W    = '#c9d1d9'

    MODELS  = ['mlp_naive', 'mlp_ewc', 'spcn']
    LABELS  = {'mlp_naive': 'MLP Naive', 'mlp_ewc': 'MLP + EWC', 'spcn': 'SPCN (local PC)'}
    COLORS  = {'mlp_naive': '#f85149', 'mlp_ewc': '#e3b341', 'spcn': '#3fb950'}

    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

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

    # ── A: Retencion T1 (grouped bars) ──
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, 'Retención Task 1\n(anti-forgetting)')
    x   = np.arange(2)
    w   = 0.22
    for i, m in enumerate(MODELS):
        vals = [res[m]['after_t1']['t1'], res[m]['after_t2']['t1']]
        bars = ax.bar(x + (i-1)*w, vals, w, label=LABELS[m],
                      color=COLORS[m], alpha=0.85, edgecolor=BG, linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{v:.2f}', ha='center', va='bottom', color=W, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Después\nde Task 1', 'Después\nde Task 2'], color=W)
    ax.set_ylabel('Accuracy (Task 1)')
    ax.set_ylim(0, 1.1)
    ax.legend(facecolor=CARD, labelcolor=W, fontsize=8, framealpha=0.8)

    # ── B: Plasticity T2 ──
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'Plasticidad Task 2\n(adaptación a clases nuevas)')
    vals2 = [metrics[m]['Plasticity_T2'] for m in MODELS]
    bars2 = ax2.bar([LABELS[m] for m in MODELS], vals2,
                    color=[COLORS[m] for m in MODELS], alpha=0.85,
                    edgecolor=BG, linewidth=0.5)
    for bar, v in zip(bars2, vals2):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{v:.2f}', ha='center', va='bottom', color=W, fontsize=10)
    ax2.set_ylabel('Accuracy (Task 2)')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='x', labelsize=8)

    # ── C: BWT scatter ──
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, 'Tradeoff BWT vs Plasticity\n(espacio de continual learning)')
    for m in MODELS:
        ax3.scatter(metrics[m]['BWT'], metrics[m]['Plasticity_T2'],
                    s=250, c=COLORS[m], zorder=5,
                    edgecolors=W, linewidth=1.2, label=LABELS[m])
        ax3.annotate(LABELS[m], (metrics[m]['BWT'], metrics[m]['Plasticity_T2']),
                     textcoords='offset points', xytext=(6, 4),
                     color=COLORS[m], fontsize=8)
    ax3.axvline(0, color=W, linestyle='--', alpha=0.3, linewidth=1)
    ax3.set_xlabel('BWT  (0 = sin olvido,  < 0 = olvido)')
    ax3.set_ylabel('Plasticity (accuracy T2)')
    # Regiones
    ax3.text(0.02, 0.02, 'olvido', transform=ax3.transAxes, color='#f85149', fontsize=8, alpha=0.7)
    ax3.text(0.55, 0.02, 'retención', transform=ax3.transAxes, color='#3fb950', fontsize=8, alpha=0.7)
    ax3.xaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)

    # ── D: Tabla resumen ──
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor(CARD)
    ax4.axis('off')
    ax4.set_title('Resumen Completo de Métricas de Continual Learning', color=W,
                  fontsize=12, pad=10, fontweight='bold', loc='left')

    rows = ['T1 Peak Acc', 'T1 Retained (after T2)', 'BWT ↑ mejor',
            'FWT proxy (T2 sin verla)', 'Plasticity T2', 'Retention Ratio', f'Tiempo (s)']
    keys = ['T1_peak','T1_retained','BWT','FWT_proxy','Plasticity_T2','Retention_ratio']

    table_data = []
    for k, row_label in zip(keys, rows[:-1]):
        row = [row_label]
        for m in MODELS:
            v = metrics[m][k]
            fmt = f'{v:+.3f}' if 'BWT' in k else f'{v:.3f}'
            row.append(fmt)
        table_data.append(row)
    # Tiempos
    time_row = ['Tiempo (s)'] + [f"{timing[m]:.1f}s" for m in MODELS]
    table_data.append(time_row)

    col_labels = ['Métrica'] + [LABELS[m] for m in MODELS]
    tbl = ax4.table(cellText=table_data, colLabels=col_labels,
                    loc='center', cellLoc='center', bbox=[0, -0.1, 1, 1.1])
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
            cell.set_text_props(color='#8b949e')
        else:
            m = MODELS[col - 1]
            cell.set_facecolor(CARD)
            cell.set_text_props(color=COLORS[m], fontweight='bold')

    fig.suptitle('Continual Learning Benchmark: SPCN vs MLP Naive vs MLP+EWC\n'
                 'Split-MNIST  ·  Task 1: dígitos 0–4  |  Task 2: dígitos 5–9',
                 color=W, fontsize=14, y=0.98, fontweight='bold')

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Plot guardado: {path}")

def plot_retention_curve(retention_curves, path=f'{OUT_DIR}/retention_curve.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    BG   = '#0d1117'
    CARD = '#161b22'
    GRID = '#21262d'
    W    = '#c9d1d9'

    MODELS = ['mlp_naive', 'mlp_ewc', 'spcn']
    LABELS = {'mlp_naive': 'MLP Naive', 'mlp_ewc': 'MLP + EWC', 'spcn': 'SPCN (local PC)'}
    COLORS = {'mlp_naive': '#f85149', 'mlp_ewc': '#e3b341', 'spcn': '#3fb950'}

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(CARD)

    for m in MODELS:
        curve = retention_curves[m]
        epochs_x = np.arange(1, len(curve) + 1)
        ax.plot(epochs_x, curve, marker='o', color=COLORS[m], label=LABELS[m],
                linewidth=2, markersize=5)

    ax.set_title('Curva de Retención: Accuracy en Task 1 durante entrenamiento de Task 2',
                 color=W, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Época de entrenamiento en Task 2', color=W)
    ax.set_ylabel('Accuracy en Task 1 (test)', color=W)
    ax.tick_params(colors=W, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor=CARD, labelcolor=W, fontsize=10, framealpha=0.85)

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Plot guardado: {path}")

# ─────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_data(n_train=5000, n_test=1000)
    res, timing, retention_curves = run_experiment(data, epochs=10, ewc_lambda=400.0)
    metrics = cl_metrics(res)

    print("\n" + "="*62)
    print("METRICAS FINALES")
    print("="*62)
    for m, mx in metrics.items():
        print(f"\n  {m.upper()}")
        for k, v in mx.items():
            tag = ' ← KEY' if k in ('BWT','Retention_ratio') else ''
            print(f"    {k:30s}: {v:+.4f}{tag}")

    with open(f'{OUT_DIR}/cl_metrics.json', 'w') as f:
        json.dump({'metrics': metrics, 'timing': timing}, f, indent=2)

    plot(res, metrics, timing)
    plot_retention_curve(retention_curves)
    print("\nListo!")
