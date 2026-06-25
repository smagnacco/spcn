"""
Barrido de protect_gamma + frontera de Pareto (BWT vs Plasticity)
==================================================================
Corre el SPCN bajo el protocolo class-incremental de 3 tasks para varios
valores de protect_gamma (la rigidez por consistencia sqrt(freq), "EWC
local sin Fisher"), y dibuja la frontera de Pareto BWT-vs-Plasticidad del
SPCN contra los puntos de referencia MLP naive y MLP+EWC.

Esta es la figura central del finding de fase 4: muestra que la frontera
del SPCN escalar queda DENTRO de la de EWC — ningun gamma iguala a EWC en
ambos ejes a la vez (techo estructural de la representacion escalar).

Reutiliza run_spcn_multitask / run_mlp_multitask / cl_metrics_multitask del
experimento principal, sin reimplementar nada.
"""

import json
import numpy as np

from continual_learning_3tasks import (
    load_data, run_spcn_multitask, run_mlp_multitask, cl_metrics_multitask,
    TASK_SPLITS, N_TASKS, OUT_DIR,
)

GAMMAS = [0.0, 1.0, 2.0, 5.0, 10.0]
EPOCHS = 10
SEED = 0


def run_sweep():
    data = load_data(n_train=5000, n_test=1000, task_splits=TASK_SPLITS)

    # Baselines (no dependen de gamma) — una sola corrida
    print("\n=== Baselines MLP ===")
    np.random.seed(SEED)
    naive_acc, _ = run_mlp_multitask(data, N_TASKS, EPOCHS, use_ewc=False)
    np.random.seed(SEED)
    ewc_acc, _ = run_mlp_multitask(data, N_TASKS, EPOCHS, use_ewc=True)
    baselines = {
        'mlp_naive': cl_metrics_multitask(naive_acc, N_TASKS),
        'mlp_ewc': cl_metrics_multitask(ewc_acc, N_TASKS),
    }

    print("\n=== Barrido SPCN protect_gamma ===")
    sweep = []
    for gamma in GAMMAS:
        np.random.seed(SEED)   # misma init -> comparacion limpia entre gammas
        acc_matrix, _, overlap = run_spcn_multitask(
            data, N_TASKS, EPOCHS, protect_gamma=gamma)
        m = cl_metrics_multitask(acc_matrix, N_TASKS)
        j_l2 = float(np.mean([v['jaccard_L2'] for v in overlap.values()])) if overlap else 0.0
        sweep.append({
            'gamma': gamma,
            'acc_matrix': acc_matrix,
            'metrics': m,
            'jaccard_L2_avg': j_l2,
        })
        print(f"  gamma={gamma:>4}: peak={[round(v,2) for v in m['peak_per_task']]}  "
              f"final={[round(v,2) for v in m['final_per_task']]}  "
              f"BWT={m['BWT_avg']:+.3f}  Plast={m['Plasticity_avg']:.3f}  "
              f"Ret={m['Retention_ratio_avg']:.3f}  Jacc_L2={j_l2:.2f}")

    return baselines, sweep


def plot_pareto(baselines, sweep, path=f'{OUT_DIR}/pareto_frontier.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    BG, CARD, GRID, W = '#0d1117', '#161b22', '#21262d', '#c9d1d9'
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    ax.set_facecolor(CARD)

    # SPCN: curva sobre gamma (frontera de Pareto medida)
    bwt = [s['metrics']['BWT_avg'] for s in sweep]
    plast = [s['metrics']['Plasticity_avg'] for s in sweep]
    gammas = [s['gamma'] for s in sweep]
    ax.plot(bwt, plast, '-o', color='#3fb950', linewidth=2, markersize=8,
            zorder=4, label='SPCN (local PC, protect_gamma sweep)')
    for b, p, g in zip(bwt, plast, gammas):
        ax.annotate(f'γ={g:g}', (b, p), textcoords='offset points',
                    xytext=(7, 4), color='#3fb950', fontsize=8)

    # Baselines como puntos de referencia
    for key, color, lab in [('mlp_naive', '#f85149', 'MLP Naive'),
                            ('mlp_ewc', '#e3b341', 'MLP + EWC')]:
        m = baselines[key]
        ax.scatter(m['BWT_avg'], m['Plasticity_avg'], s=300, c=color,
                   edgecolors=W, linewidth=1.5, zorder=5, marker='*', label=lab)
        ax.annotate(lab, (m['BWT_avg'], m['Plasticity_avg']),
                    textcoords='offset points', xytext=(8, -12), color=color, fontsize=9)

    ax.set_title('Frontera de Pareto: Retención (BWT) vs Plasticidad\n'
                 'La frontera del SPCN escalar queda DENTRO de la de EWC — '
                 'ningún γ iguala a EWC en ambos ejes',
                 color=W, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('BWT promedio  (→ derecha = menos olvido, mejor retención)', color=W)
    ax.set_ylabel('Plasticity promedio  (↑ = aprende mejor cada task nueva)', color=W)
    ax.tick_params(colors=W, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(facecolor=CARD, labelcolor=W, fontsize=9, framealpha=0.85, loc='lower left')

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Plot guardado: {path}")


if __name__ == '__main__':
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    baselines, sweep = run_sweep()

    with open(f'{OUT_DIR}/cl_3tasks_sweep.json', 'w') as f:
        json.dump({
            'gammas': GAMMAS,
            'seed': SEED,
            'spcn_config': {'clr': 0.01, 'recruit_beta': 1.0, 'sparse_frac': 0.15,
                            'protection_signal': 'sqrt(freq) consolidated max across tasks'},
            'baselines': baselines,
            'sweep': sweep,
        }, f, indent=2)
    print(f"Sweep guardado: {OUT_DIR}/cl_3tasks_sweep.json")

    plot_pareto(baselines, sweep)
    print("\nListo!")
