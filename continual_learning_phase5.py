"""
Continual Learning — FASE 5, ETAPA 5a: Activación Compleja con FASE ASIGNADA
============================================================================
Heredado de la fase 4 (continual_learning_3tasks.py): protocolo idéntico
(3 tasks class-incremental, head único 10-way, masked CE en train, eval
unmasked, 5000/1000 por task, seed=0). NO se tocan los scripts de fase 4 —
este archivo es independiente.

HIPÓTESIS (fase 5, §6 de FINDINGS_PHASE4.md): el techo de la fase 4 es de
REPRESENTACIÓN — un escalar por neurona no puede preservar el rol de una
neurona en una task y adaptarlo a otra a la vez. Una activación COMPLEJA
z = r·e^(iθ) separa esos dos roles: magnitud r = intensidad (lo que el escalar
ya codificaba), fase θ = contexto/task. Con fase, dos tasks viven en
subespacios ortogonales por construcción (binding-by-phase: Singer; von der
Malsburg; HRR: Plate 1995).

ETAPA 5a — validar la maquinaria compleja con la fase ASIGNADA (upper bound),
ANTES de pedir que la fase emerja (5b/5c). Si con fase asignada PERFECTA el
forgetting no mejora vs el escalar, hay un bug en la aritmética compleja.

──────────────────────────────────────────────────────────────────────────
DISEÑO DE 5a (corregido tras el catch del readout):

  Bug original: si la fase es UNIFORME por capa (θ_t igual para toda neurona),
  e^(iθ_t) factoriza —  W@(r·e^(iθ_t)) = e^(iθ_t)·(W@r) — y si el readout lee
  magnitud |·|, la fase desaparece: 5a colapsaría a la escalar de fase 4 y
  concluiríamos falsamente que "la fase no sirve" cuando nunca entró en juego.

  Corrección (dos cambios):

  1. FASE POR-NEURONA, no uniforme por capa. La fase de cada neurona la fija la
     task que la RECLUTÓ, leída del commitment tracker que ya existe en fase 4:
       - neurona en el núcleo consistente de T1  → fase 0°
       - neurona reclutada (núcleo) por T2        → fase 120°
       - neurona reclutada (núcleo) por T3        → fase 240°
     Regla de asignación (documentada): una neurona se ASIGNA a la primera task
     en cuyo núcleo consistente entra (first-writer-wins, el mismo orden por el
     que el núcleo protegido de T1 queda disjunto en fase 4). Una vez asignada,
     su fase es fija. Las neuronas NO asignadas (frescas, o tocadas de pasada y
     nunca consistentes) emiten en la fase de la TASK ACTUAL en entrenamiento.
     Como el allocation de fase 4 ya separa núcleos (Jaccard T1-T2 = 0.00), las
     fases quedan separadas por neurona automáticamente.

  2. READOUT = PROYECCIÓN A FASE DE TASK (no magnitud). La capa siguiente / head,
     evaluando para la task t, leen Re( z·e^(−iθ_t) ): proyectan la suma compleja
     sobre la fase de t. Una neurona que T1 escribió en 0° y T2 reescribió en
     120°: el readout de T1 (proyección a 0°) ve la contribución de T1 plena y
     atenúa la de T2 por cos(120°) = −0.5. ESA es la protección por fase.

  Con fase POR-NEURONA + readout proyectado, e^(iθ) ya NO factoriza (las fases
  difieren entre neuronas de la misma capa), así que la proyección diferencial
  tiene efecto real. Es exactamente el grado de libertad que al escalar le falta.

  Magnitud, top-k, recruitment, commitment, protección y la parte de magnitud
  del contrastivo: IDÉNTICOS a fase 4, operando sobre r. La fase está FIJA en
  5a (asignada, no se aprende). Head lee la magnitud (la proyección a fase),
  idéntico a fase 4.
──────────────────────────────────────────────────────────────────────────
"""

import json
import time
import numpy as np

from continual_learning_experiment import load_data, PCLayer
from continual_learning_3tasks import (
    N_CLASSES_TOTAL, TASK_SPLITS, TASK_LABELS_RANGE, N_TASKS,
    SPCN_SPARSE_FRAC, SPCN_RECRUIT_BETA, SPCN_CLR,
    cl_metrics_multitask, specialization_metrics,
)

OUT_DIR = 'output/continual'

# Fases asignadas por task: equiespaciadas en el círculo (radianes).
# T1 → 0°, T2 → 120°, T3 → 240°.
TASK_PHASES = np.array([2 * np.pi * t / N_TASKS for t in range(N_TASKS)])

# Umbral de "núcleo consistente" — el MISMO que usa fase 4 para los sets
# consistentes (×1.5 sobre la frecuencia uniforme k/out_dim). Una neurona se
# ASIGNA a la fase de una task si entra en el núcleo consistente de esa task.
OVERUSE_MULT = 1.5


# ─────────────────────────────────────────────────────────────────
# Capa compleja: magnitud = PCLayer de fase 4; fase = asignada por task.
# ─────────────────────────────────────────────────────────────────

class ComplexPCLayer(PCLayer):
    """PCLayer de fase 4 extendida con activación compleja z = r·e^(iθ),
    versión 5a' (fase como estrategia ALTERNATIVA al allocation).

    Pesos REALES (§2). La fase vive en un ACUMULADOR COMPLEJO por neurona:
        c_i  +=  (masa de update de la task t en i) · e^(iθ_t)
    actualizado cada vez que la neurona i recibe un update durante la task t.

    Emisión:
        z_i = r_i · (|c_i| / max|c|) · e^(i·arg(c_i))
      - r_i (magnitud): IDÉNTICA a fase 4 — sigmoid(proj + b), top-k,
        recruitment, commitment, protección por consistencia. Toda la
        maquinaria de magnitud de fase 4 sin cambios.
      - arg(c_i): fase emitida. Una neurona usada solo por T1 → 0°; una
        compartida T1+T2 → ~60° con |c_i| REDUCIDO (interferencia destructiva:
        e^(i0)+e^(i2π/3) tiene módulo < suma de módulos). Ese costo de magnitud
        por compartir es la tensión que al escalar le faltaba — sin loss
        agregado, la aritmética sola recompensa fases distintas por task.

    Neuronas frescas (|c_i|=0, nunca entrenadas): emiten en la fase de la task
    actual con factor 1 (no se las penaliza antes de tener historia).

    SEPARACIÓN DE EFECTOS (crítica para validez): self.cmod (|c_i| normalizado)
    se computa pero su aplicación a la magnitud emitida se controla con
    apply_cmod. Con apply_cmod=False medimos el efecto de FASE puro (cosine)
    sin que |c_i| toque lo que leen head/top-k (que sería protección-escalar
    encubierta). Con apply_cmod=True está el mecanismo completo.
    """

    def __init__(self, *args, apply_cmod=True, **kwargs):
        super().__init__(*args, **kwargs)
        out_dim = self.W.shape[0]
        # Acumulador complejo de fase por neurona. 0 = sin historia (fresca).
        self.c = np.zeros(out_dim, dtype=np.complex128)
        # Fase de la task que se entrena/evalúa AHORA (para readout proyectado y
        # para taggear los updates). La setea el bucle por task.
        self.current_task_phase = 0.0
        # Si True, |c_i| normalizado modula la magnitud emitida (mecanismo
        # completo). Si False, no la toca (aísla el efecto de fase del de magnitud).
        self.apply_cmod = apply_cmod
        self._last_proj = np.zeros(out_dim)

    def accumulate_phase(self, active_idx, update_mass):
        """Acumula c_i += update_mass · e^(iθ_actual) sobre las neuronas que
        recibieron update en este sample. update_mass[k] ≥ 0 es la magnitud del
        update de la neurona activa k (proxy de cuánto la entrenó esta task).
        Así una neurona muy entrenada por T1 acumula mucho peso en 0°, y si T2
        la entrena también, suma peso en 120° → arg(c_i) se mueve, |c_i| sube
        menos (cancelación)."""
        phase_factor = np.exp(1j * self.current_task_phase)
        self.c[active_idx] += update_mass * phase_factor

    def cmod_norm(self):
        """|c_i| normalizado a [0,1] por el máximo de la capa. Neuronas frescas
        (|c|=0) reciben 1.0 (sin penalizar antes de tener historia)."""
        mag = np.abs(self.c)
        mx = mag.max()
        if mx < 1e-12:
            return np.ones_like(mag)
        norm = mag / mx
        norm[mag < 1e-12] = 1.0   # frescas: factor 1
        return norm

    def emitted_phase(self):
        """Fase emitida por neurona = arg(c_i); frescas (|c|≈0) → fase de la
        task actual (participan en la fase en curso hasta tener historia)."""
        phi = np.angle(self.c)
        fresh = np.abs(self.c) < 1e-12
        phi[fresh] = self.current_task_phase
        return phi

    def forward_complex(self, z_in):
        """Forward complejo. z_in: vector complejo de la capa previa (real para
        la capa 1, fase 0). Devuelve (z_out, r): z_out complejo, r la magnitud
        real (la activación de fase 4) que usan head, top-k y los updates.

        Paso 1 — readout proyectado a fase de task: proj = Re(e^(−iθ_t)·(W@z_in)).
        Para la capa 1, z_in real ⇒ proj = W@x (identidad de fase 4 a 0°).

        Paso 2 — magnitud + top-k = fase 4: r = sigmoid(proj+b), recruitment
        sobre score = act−β·commit (β=0 en 5a' ⇒ recruitment OFF, núcleos se
        solapan), top-k duro.

        Paso 3 — emisión compleja: z_out[i] = r[i]·cmod[i]·e^(i·arg(c_i)).
        cmod aplica solo si apply_cmod (separación de efectos)."""
        Z = self.W @ z_in
        proj = np.real(np.exp(-1j * self.current_task_phase) * Z)

        a = self._sig(proj + self.b)
        if self.k < len(a):
            a_norm = (a - a.min()) / (np.ptp(a) + 1e-8)
            c_norm = self.commitment / (self.commitment.max() + 1e-8)
            score = a_norm - self.recruit_beta * c_norm
            idx = np.argpartition(score, -self.k)[-self.k:]
            mask = np.zeros_like(a)
            mask[idx] = 1.0
            a = a * mask
            self._last_active_idx = idx
        else:
            self._last_active_idx = np.arange(len(a))
        r = a

        phi = self.emitted_phase()
        cmod = self.cmod_norm() if self.apply_cmod else np.ones_like(r)
        z_out = r * cmod * np.exp(1j * phi)
        self._last_proj = proj
        return z_out, r

    # --- updates: idénticos a fase 4, operan sobre la magnitud / proyección ---

    def hebbian_update_complex(self, x_proj_input, state):
        """Update Hebbiano de fase 4 sobre la magnitud (pesos REALES, sin
        cambios) + acumulación del tag de fase. x_proj_input es el input real
        (proyección a fase de la capa previa); state es la magnitud objetivo.

        Tras el update de pesos de fase 4, acumula en c_i el tag e^(iθ_actual)
        ponderado por la MASA con que esta task usó cada neurona activa. La masa
        es la ACTIVACIÓN de magnitud r de la neurona cuando está en el top-k
        (state[active], la señal que define su rol), NO |plast·err|: en este
        cuerpo PC el err Hebbiano es ~0 sobre las activas (state≈pred por
        construcción — el aprendizaje vive en homeostasis + contrastivo), así
        que |err| no mide cuánto entrenó la task a la neurona. r en el top-k sí:
        es el firing que define la pertenencia de la neurona a la task. Si otra
        task ya la había usado en otra fase, las contribuciones se SUPERPONEN en
        c_i → arg(c_i) se mueve y |c_i| cancela parcialmente."""
        pred = self._sig(self._last_proj + self.b)
        err = state - pred
        active = self._last_active_idx
        plast = self._plasticity(active)
        self.W[active] += self.lr * plast[:, None] * np.outer(err[active], x_proj_input)
        self.b[active] += self.lr * plast * err[active]
        self.ema_state[active] += self.ema_rate * (state[active] - self.ema_state[active])
        self.b[active] += self.ema_rate * (self.target - self.ema_state[active])
        # Masa de fase = activación r de la neurona en el top-k (≥0). plast la
        # modula: una neurona protegida (plast<1) acumula MENOS fase nueva — la
        # protección de magnitud y el anclaje de fase quedan acoplados de forma
        # consistente (una neurona congelada para T1 no se re-fasea hacia T2).
        update_mass = plast * state[active]
        self.accumulate_phase(active, update_mass)


# ─────────────────────────────────────────────────────────────────
# SPCN complejo — head único 10-way, fase asignada (5a)
# ─────────────────────────────────────────────────────────────────

def run_spcn_complex_multitask(data, n_tasks, epochs,
                               sparse_frac=SPCN_SPARSE_FRAC,
                               recruit_beta=0.0,
                               protect_gamma=2.0,
                               apply_cmod=True):
    """SPCN complejo 5a' (fase como estrategia ALTERNATIVA al allocation).
    Cuerpo = 2 ComplexPCLayer. Magnitud/top-k/commitment/protección/contrastivo
    de magnitud = fase 4. Fase en acumulador complejo c_i por neurona, tageada
    e^(iθ_t) por la masa del update de cada task. Readout = proyección a fase.

    recruit_beta=0 ⇒ recruitment OFF ⇒ núcleos se SOLAPAN (deseado: le da a la
    fase neuronas compartidas para separar — el coseno −0.5 vive ahí).
    apply_cmod: si True, |c_i| modula la magnitud emitida (mecanismo completo);
    si False, la fase NO toca la magnitud (aísla efecto de fase del de magnitud).

    Devuelve:
      acc_matrix[i][j], task_activations[t] (magnitud top-layer, para spec),
      overlap_summary (Jaccard de núcleos),
      phase_diag: coseno complejo entre tasks, no-factorización, |c_i| mono/multi.
    """
    np_d = data['np']
    k1 = max(1, round(sparse_frac * 128)) if sparse_frac else None
    k2 = max(1, round(sparse_frac * 64)) if sparse_frac else None
    layers = [ComplexPCLayer(784, 128, lr=0.004, clr=SPCN_CLR, k_active=k1,
                             recruit_beta=recruit_beta, protect_gamma=protect_gamma,
                             apply_cmod=apply_cmod),
              ComplexPCLayer(128, 64, lr=0.004, clr=SPCN_CLR, k_active=k2,
                             recruit_beta=recruit_beta, protect_gamma=protect_gamma,
                             apply_cmod=apply_cmod)]
    hW = np.random.randn(N_CLASSES_TOTAL, 64) * 0.01
    hb = np.zeros(N_CLASSES_TOTAL)
    hlr = 0.003
    protos = np.zeros((N_CLASSES_TOTAL, 64))      # prototipos de MAGNITUD (real)
    proto_n = np.zeros(N_CLASSES_TOTAL)

    def set_task_phase(phase):
        for L in layers:
            L.current_task_phase = phase

    def forward_complex(x):
        """Forward complejo completo. Devuelve (z_states, r_states):
        z_states[k] = activación compleja de la capa k (z_states[0] = x real),
        r_states[k] = magnitud real de la capa k (r_states[0] = x)."""
        z = x.astype(np.complex128)
        z_states = [z]
        r_states = [x]
        for L in layers:
            z_out, r = L.forward_complex(z_states[-1])
            z_states.append(z_out)
            r_states.append(r)
        return z_states, r_states

    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def top_magnitude_for_task(x, task_phase):
        """Magnitud de la top layer leída con readout proyectado a task_phase.
        Es lo que ve el head: Re( e^(−iθ_t)·z_top ) sobre la top layer. Para
        clasificar la task t fijamos la fase de readout a θ_t."""
        set_task_phase(task_phase)
        _, r_states = forward_complex(x)
        return r_states[-1]

    def evaluate_task(X, y_global, task_phase):
        """Evalúa accuracy sobre un test set leyendo con readout a task_phase.
        argmax 10-way unmasked sobre los logits del head (head lee la magnitud
        proyectada)."""
        correct = 0
        for i in range(len(y_global)):
            r_top = top_magnitude_for_task(X[i], task_phase)
            logits = hW @ r_top + hb
            if logits.argmax() == y_global[i]:
                correct += 1
        return correct / len(y_global)

    # Instrumentación de núcleos consistentes (idéntica a fase 4).
    topk_count_this_task = [np.zeros(128), np.zeros(64)]
    n_samples_this_task = [0]
    # ¿Qué tasks tocaron (top-k) cada neurona alguna vez? Para el sanity de
    # interferencia destructiva: |c_i| de neuronas mono-task vs multi-task.
    trained_by = [np.zeros((n_tasks, 128)), np.zeros((n_tasks, 64))]
    cur_task_idx = [0]

    def train_sample(x, label_global, class_lo, class_hi, task_phase):
        set_task_phase(task_phase)
        z_states, r_states = forward_complex(x)

        # Inputs reales para la regla de magnitud (pesos reales de fase 4): la
        # proyección a fase de la capa previa. Para la capa 0 es x; para capas
        # internas es Re(e^(−iθ_t)·z_prev), la misma señal que generó el proj.
        for i, L in enumerate(layers):
            z_prev = z_states[i]
            x_proj = np.real(np.exp(-1j * task_phase) * z_prev)
            L.hebbian_update_complex(x_proj, r_states[i + 1])
            topk_count_this_task[i][L._last_active_idx] += 1.0
            trained_by[i][cur_task_idx[0], L._last_active_idx] += 1.0
            L.mark_commitment()
        n_samples_this_task[0] += 1

        top = r_states[-1]                  # magnitud top layer (real)
        a = 0.008
        protos[label_global] = (1 - a) * protos[label_global] + a * top
        proto_n[label_global] += 1

        if proto_n[label_global] > 2:
            dists = np.linalg.norm(protos - top, axis=1)
            dists[label_global] = np.inf
            seen_mask = proto_n > 0
            dists[~seen_mask] = np.inf
            neg = dists.argmin()
            if np.isfinite(dists[neg]):
                sp, sn = protos[label_global], protos[neg]
                top_layer = layers[-1]
                z_prev = z_states[-2]
                top_input = np.real(np.exp(-1j * task_phase) * z_prev)
                # Contrastivo de MAGNITUD: regla de fase 4 sin cambios. La fase
                # está fija en 5a (no se aprende), así que NO hay update de fase.
                top_layer.contrastive_update(top_input, sp, sn)

        # Head update: masked softmax CE sobre [class_lo, class_hi). El head lee
        # la magnitud (top), idéntico a fase 4.
        logits = hW @ top + hb
        sub_logits = logits[class_lo:class_hi]
        probs_sub = softmax(sub_logits)
        full_probs = np.zeros(N_CLASSES_TOTAL)
        full_probs[class_lo:class_hi] = probs_sub
        oh = np.zeros(N_CLASSES_TOTAL); oh[label_global] = 1.0
        grad = full_probs - oh
        hW[:] -= hlr * np.outer(grad, top)
        hb[:] -= hlr * grad

    acc_matrix = [[None] * n_tasks for _ in range(n_tasks)]
    train_acc_per_task = [None] * n_tasks
    task_activations = [None] * n_tasks
    consistent_sets = []
    uniform_frac = [k1 / 128, k2 / 64]

    def jaccard(s, t):
        if not s and not t:
            return 0.0
        return len(s & t) / len(s | t)

    # Para sanity (b): activación compleja promedio del núcleo de cada task en
    # la top layer, leída en SU fase de entrenamiento (sin proyectar: guardamos
    # el complejo z crudo para medir el producto interno complejo entre tasks).
    task_complex_top = [None] * n_tasks

    for t in range(n_tasks):
        lo, hi = TASK_SPLITS[t]
        theta_t = TASK_PHASES[t]
        Xtr, ytr_local = np_d[f'X{t+1}tr'], np_d[f'y{t+1}tr']
        ytr_global = ytr_local + lo

        topk_count_this_task[0][:] = 0
        topk_count_this_task[1][:] = 0
        n_samples_this_task[0] = 0
        cur_task_idx[0] = t

        for L in layers:
            L.freeze_protection()

        for ep in range(epochs):
            idx = np.random.permutation(len(Xtr))
            for i in idx:
                train_sample(Xtr[i], ytr_global[i], lo, hi, theta_t)

        # Núcleos consistentes de esta task (umbral de fase 4) + consolidación
        # de protección (idéntica f4). La FASE ya NO se asigna por núcleo: vive
        # en el acumulador c_i, que se llenó durante el entrenamiento de la task.
        n_s = n_samples_this_task[0]
        task_sets = []
        for li in range(2):
            counts = topk_count_this_task[li]
            thresh = OVERUSE_MULT * uniform_frac[li] * n_s
            consistent = set(np.where(counts >= thresh)[0].tolist())
            task_sets.append(consistent)
            freq_in_task = counts / max(n_s, 1)
            layers[li].consolidate_protection(np.sqrt(freq_in_task))
        consistent_sets.append(task_sets)

        # Activaciones para spec (magnitud top layer, leída en fase de la task,
        # mismo punto temporal que peak) + complejo crudo promedio del núcleo.
        Xte_t = np_d[f'X{t+1}te']
        set_task_phase(theta_t)
        mags = []
        cplx = []
        for xi in Xte_t:
            z_states, r_states = forward_complex(xi)
            mags.append(r_states[-1])
            cplx.append(z_states[-1])
        task_activations[t] = np.array(mags)
        task_complex_top[t] = np.mean(np.array(cplx), axis=0)   # complejo promedio

        train_acc_per_task[t] = evaluate_task(Xtr, ytr_global, theta_t)

        if t > 0:
            j_l1 = jaccard(task_sets[0], consistent_sets[0][0])
            j_l2 = jaccard(task_sets[1], consistent_sets[0][1])
            print(f"  [spcn-cplx] T{t+1} núcleo — L1: {len(task_sets[0])} units "
                  f"(Jaccard vs T1={j_l1:.2f})  L2: {len(task_sets[1])} units "
                  f"(Jaccard vs T1={j_l2:.2f})")
        else:
            print(f"  [spcn-cplx] T1 núcleo — L1: {len(task_sets[0])} units  "
                  f"L2: {len(task_sets[1])} units")

        # Eval de cada task vista: cada una leída con su PROPIA fase de readout.
        for j in range(t + 1):
            jlo, jhi = TASK_SPLITS[j]
            Xte, yte_local = np_d[f'X{j+1}te'], np_d[f'y{j+1}te']
            yte_global = yte_local + jlo
            acc_matrix[t][j] = evaluate_task(Xte, yte_global, TASK_PHASES[j])

        print(f"  [spcn-cplx] after T{t+1}: " +
              "  ".join(f"T{j+1}={acc_matrix[t][j]:.3f}" for j in range(t + 1)) +
              f"   (train T{t+1}={train_acc_per_task[t]:.3f})")

    overlap_summary = {}
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            overlap_summary[f'T{i+1}_vs_T{j+1}'] = {
                'jaccard_L1': jaccard(consistent_sets[i][0], consistent_sets[j][0]),
                'jaccard_L2': jaccard(consistent_sets[i][1], consistent_sets[j][1]),
            }

    # ── Diagnóstico de fase ──
    phase_diag = phase_diagnostics(layers, task_complex_top, trained_by, n_tasks)

    return acc_matrix, task_activations, overlap_summary, phase_diag


# ─────────────────────────────────────────────────────────────────
# SANITY: interferencia compleja + no-factorización de la fase
# ─────────────────────────────────────────────────────────────────

def phase_diagnostics(layers, task_complex_top, trained_by, n_tasks,
                      overuse_mult=OVERUSE_MULT):
    """Sanity checks 5a' (decisivos):

    1. Coseno complejo (Re del producto interno hermítico normalizado) entre las
       activaciones compleja promedio de cada par de tasks en la top layer:
         (a) AGREGADO sobre toda la capa.
         (b) restringido a las neuronas COMPARTIDAS por el par (entrenadas por
             ambas tasks) — donde la fase debe imponer ≈ −0.5. Es el número
             decisivo: con núcleos solapados, ¿la fase separa lo compartido?

    2. No-factorización: arg(c_i) debe VARIAR entre neuronas de cada capa (cada
       una con distinta historia de uso). Si una capa colapsa a una sola fase,
       e^(iθ) factoriza → BUG. Reportamos cuántas fases distintas hay por capa
       (solo neuronas con historia, |c_i|>0).

    3. Interferencia destructiva: |c_i| de neuronas MONO-task vs MULTI-task.
       Una neurona entrenada por T1 y T2 por igual debe tener |c_i| MENOR que
       una mono-task (e^(i0)+e^(i2π/3) cancela parcialmente). top layer.
    """
    # Top layer es la última; trained_by[-1] tiene shape (n_tasks, out_dim_top).
    top_L = layers[-1]
    tb_top = trained_by[-1]                     # (n_tasks, out_dim)
    # ¿Qué neuronas tocó cada task? (umbral: tocada en > 0 samples basta para
    # "usada"; para "núcleo" usaríamos el umbral de consistencia, pero shared
    # se define por uso real, no por consistencia).
    used = tb_top > 0                           # (n_tasks, out_dim) bool

    # 1. Coseno complejo entre tasks (agregado + en compartidas).
    complex_cos = {}
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            vi, vj = task_complex_top[i], task_complex_top[j]
            denom = (np.linalg.norm(vi) * np.linalg.norm(vj)) + 1e-12
            ip_all = np.vdot(vi, vj) / denom
            shared = used[i] & used[j]
            if shared.sum() > 0:
                vis, vjs = vi[shared], vj[shared]
                den_s = (np.linalg.norm(vis) * np.linalg.norm(vjs)) + 1e-12
                ip_sh = np.vdot(vis, vjs) / den_s
                sh = {'real_cos': float(np.real(ip_sh)), 'abs': float(np.abs(ip_sh)),
                      'n_shared': int(shared.sum())}
            else:
                sh = {'real_cos': None, 'abs': None, 'n_shared': 0}
            complex_cos[f'T{i+1}_vs_T{j+1}'] = {
                'aggregate': {'real_cos': float(np.real(ip_all)), 'abs': float(np.abs(ip_all))},
                'shared_only': sh,
            }

    # 2. No-factorización: fases distintas (de arg(c_i)) por capa.
    nonfactor = []
    for li, L in enumerate(layers):
        mag = np.abs(L.c)
        has_hist = mag > 1e-9
        ph = np.angle(L.c[has_hist])
        uniq = sorted(set(np.round(np.degrees(ph)).astype(int).tolist()))
        nonfactor.append({
            'layer': li + 1,
            'n_with_history': int(has_hist.sum()),
            'n_fresh': int((~has_hist).sum()),
            'n_distinct_phases': len(uniq),
            'phase_spread_deg': float(np.degrees(np.ptp(ph))) if has_hist.sum() > 1 else 0.0,
            'factorizes': len(uniq) <= 1,   # True = BUG (una sola fase)
        })

    # 3. Interferencia destructiva: |c_i| mono vs multi-task (top layer).
    n_tasks_used = used.sum(axis=0)             # cuántas tasks usaron cada neurona
    mag_top = np.abs(top_L.c)
    mono = n_tasks_used == 1
    multi = n_tasks_used >= 2
    destructive = {
        'mean_abs_c_mono_task': float(mag_top[mono].mean()) if mono.sum() else None,
        'mean_abs_c_multi_task': float(mag_top[multi].mean()) if multi.sum() else None,
        'n_mono': int(mono.sum()),
        'n_multi': int(multi.sum()),
    }

    return {
        'complex_cos_between_tasks': complex_cos,
        'non_factorization': nonfactor,
        'destructive_interference': destructive,
    }


# ─────────────────────────────────────────────────────────────────
# BARRIDO de protect_gamma + reporte
# ─────────────────────────────────────────────────────────────────

def run_sweep(data, gammas=(0.0, 1.0, 2.0, 5.0, 10.0), epochs=10, seed=0):
    rows = {}
    diags = {}
    for g in gammas:
        np.random.seed(seed)   # misma init para todos los gamma (como fase 4)
        print(f"\n{'='*62}\nSPCN COMPLEJO 5a' (recruit OFF, c_i) — protect_gamma = {g}\n{'='*62}")
        t0 = time.time()
        acc, acts, overlap, phase_diag = run_spcn_complex_multitask(
            data, N_TASKS, epochs, protect_gamma=g)
        dt = time.time() - t0
        m = cl_metrics_multitask(acc, N_TASKS)
        spec = specialization_metrics(acts, N_TASKS)
        rows[g] = {
            'acc_matrix': acc,
            'metrics': m,
            'overlap': overlap,
            'specialization': spec,
            'time_s': dt,
        }
        diags[g] = phase_diag
        print(f"  → BWT_avg={m['BWT_avg']:+.3f}  Plast_avg={m['Plasticity_avg']:.3f}  "
              f"Retention_avg={m['Retention_ratio_avg']:.3f}  ({dt:.1f}s)")
    return rows, diags


def print_report(rows, diags, gammas):
    print("\n" + "=" * 80)
    print("FASE 5a' — SPCN COMPLEJO (recruit OFF, acumulador c_i) — barrido protect_gamma")
    print("=" * 80)
    print(f"\n{'gamma':>6} | {'Peak/task':^28} | {'BWT_avg':>8} | {'Plast_avg':>9} | {'Reten_avg':>9}")
    print("-" * 80)
    for g in gammas:
        m = rows[g]['metrics']
        peak = "[" + ", ".join(f"{p:.3f}" for p in m['peak_per_task']) + "]"
        print(f"{g:>6.1f} | {peak:^28} | {m['BWT_avg']:>+8.3f} | "
              f"{m['Plasticity_avg']:>9.3f} | {m['Retention_ratio_avg']:>9.3f}")

    # Referencia fase 4 (NO re-corrida — de FINDINGS_PHASE4.md). La comparación
    # clave de 5a' es contra la FRONTERA DE ALLOCATION escalar de fase 4 (el
    # barrido completo), no solo γ=0.
    print("\n  Referencia ESCALAR fase 4 (FINDINGS_PHASE4, mismo protocolo/seed):")
    print("    γ=0:   BWT=-0.685  Plast=0.855  Reten=0.258")
    print("    γ=1:   BWT=-0.575  Plast=0.833  Reten=0.368")
    print("    γ=2:   BWT=-0.518  Plast=0.792  Reten=0.430")
    print("    γ=5:   BWT=-0.390  Plast=0.639  Reten=0.542")
    print("    γ=10:  BWT=-0.264  Plast=0.491  Reten=0.671")
    print("    MLP Naive: BWT=-0.548 Plast=0.840   MLP+EWC: BWT=-0.345 Plast=0.786")

    # ── Sanity checks (decisivos) ──
    print("\n" + "=" * 80)
    print("SANITY CHECKS DECISIVOS (γ=0)")
    print("=" * 80)
    d0 = diags[0.0]
    bwt0 = rows[0.0]['metrics']['BWT_avg']
    plast0 = rows[0.0]['metrics']['Plasticity_avg']
    print(f"\n(a) BWT(5a', γ=0) = {bwt0:+.3f}   Plast = {plast0:.3f}")
    print(f"      vs escalar γ=0 (-0.685): {'MEJOR' if bwt0 > -0.685 else 'NO mejor'}")

    print("\n(b) DECISIVO — coseno (Re) entre activación compleja promedio de tasks")
    print("    en la top layer. shared_only = solo neuronas COMPARTIDAS por el par")
    print("    (donde la fase DEBE imponer ≈ −0.5):")
    for pair, v in d0['complex_cos_between_tasks'].items():
        agg = v['aggregate']
        sh = v['shared_only']
        sh_str = (f"Re={sh['real_cos']:+.3f} |c|={sh['abs']:.3f} (n={sh['n_shared']})"
                  if sh['real_cos'] is not None else "sin neuronas compartidas")
        print(f"      {pair}:  agg Re={agg['real_cos']:+.3f}  |  shared {sh_str}")

    print("\n(b-no-factorización) arg(c_i) debe variar entre neuronas de cada capa:")
    for nf in d0['non_factorization']:
        flag = 'BUG: factoriza' if nf['factorizes'] else 'OK: fases distintas'
        print(f"      L{nf['layer']}: {nf['n_with_history']} con historia, "
              f"{nf['n_fresh']} frescas, {nf['n_distinct_phases']} fases distintas, "
              f"spread={nf['phase_spread_deg']:.0f}° → {flag}")

    print("\n(c interferencia destructiva) |c_i| mono-task vs multi-task (top layer):")
    di = d0['destructive_interference']
    mono = di['mean_abs_c_mono_task']; multi = di['mean_abs_c_multi_task']
    if mono is not None and multi is not None:
        verdict = 'OK: multi < mono (cancela)' if multi < mono else 'NO cancela'
        print(f"      mono (n={di['n_mono']}): |c|={mono:.3f}   "
              f"multi (n={di['n_multi']}): |c|={multi:.3f}   → {verdict}")
    else:
        print(f"      mono (n={di['n_mono']})  multi (n={di['n_multi']}) — "
              f"falta un grupo, no se puede comparar")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    GAMMAS = (0.0, 1.0, 2.0, 5.0, 10.0)
    SEED = 0
    np.random.seed(SEED)

    data = load_data(n_train=5000, n_test=1000, task_splits=TASK_SPLITS)
    rows, diags = run_sweep(data, gammas=GAMMAS, epochs=10, seed=SEED)
    print_report(rows, diags, GAMMAS)

    # Persistir resultados (convertir keys float a str para JSON)
    out = {
        'phase': "5a'",
        'description': 'SPCN complejo, recruit OFF + acumulador de fase c_i por neurona '
                       '(fase como estrategia alternativa al allocation)',
        'task_splits': TASK_SPLITS,
        'task_phases_deg': [float(np.degrees(p)) for p in TASK_PHASES],
        'recruit_beta': 0.0,
        'apply_cmod': True,
        'seed': SEED,
        'sweep': {str(g): rows[g] for g in GAMMAS},
        'phase_diagnostics': {str(g): diags[g] for g in GAMMAS},
        'reference_phase4_scalar_frontier': {
            '0.0': {'BWT_avg': -0.685, 'Plasticity_avg': 0.855, 'Retention_avg': 0.258},
            '1.0': {'BWT_avg': -0.575, 'Plasticity_avg': 0.833, 'Retention_avg': 0.368},
            '2.0': {'BWT_avg': -0.518, 'Plasticity_avg': 0.792, 'Retention_avg': 0.430},
            '5.0': {'BWT_avg': -0.390, 'Plasticity_avg': 0.639, 'Retention_avg': 0.542},
            '10.0': {'BWT_avg': -0.264, 'Plasticity_avg': 0.491, 'Retention_avg': 0.671},
            'mlp_naive': {'BWT_avg': -0.548, 'Plasticity_avg': 0.840, 'Retention_avg': 0.317},
            'mlp_ewc': {'BWT_avg': -0.345, 'Plasticity_avg': 0.786, 'Retention_avg': 0.444},
        },
    }
    with open(f'{OUT_DIR}/cl_phase5a_prime_sweep.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResultados guardados: {OUT_DIR}/cl_phase5a_prime_sweep.json")
    print("Listo (5a').")
