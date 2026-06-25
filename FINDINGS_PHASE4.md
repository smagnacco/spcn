# Phase 4 Findings: La Frontera de Pareto del SPCN Escalar es Interior a la de EWC

## 1. Objetivo

Continual learning **class-incremental** (Split-MNIST, 3 tasks secuenciales),
comparando el SPCN (predictive coding local, sin backprop) contra dos
baselines de referencia: un MLP naive (fine-tuning secuencial sin protección)
y un MLP+EWC (Elastic Weight Consolidation, Kirkpatrick et al. 2017).

El protocolo es deliberadamente el más duro: un **único head compartido de 10
logits** para los tres modelos (no un head por task). Durante el entrenamiento
de cada task la cross-entropy se calcula solo sobre los logits de las clases
presentes en esa task (masked CE), pero el forward pass nunca se enmascara; en
evaluación el argmax es siempre sobre los 10 logits sin máscara. Un head
task-incremental separado por task le daría a cualquier modelo —incluido el
SPCN— un boost artificial de BWT que no refleja resistencia real al olvido en
la representación compartida.

Tasks: T1 = dígitos 0-3 (4 clases), T2 = dígitos 4-6 (3 clases), T3 = dígitos
7-9 (3 clases). 5000 train / 1000 test por task.

La métrica de interés **no es accuracy absoluta**, sino el tradeoff:

- **BWT promedio** (Backward Transfer): cuánto cae cada task vieja al aprender
  las nuevas. → 0 = sin olvido; más negativo = más olvido.
- **Plasticity promedio**: accuracy peak en cada task nueva, medido justo
  después de entrenarla.
- **Retention ratio**: accuracy final / accuracy peak, normalizado por el
  punto de partida.

El objeto del hallazgo es la **frontera de Pareto BWT-vs-Plasticity** del
SPCN, no un punto único.

## 2. Punto de partida

El SPCN original colapsaba en este protocolo: peak por task ~0.25, y T1
terminaba en accuracy **literal 0.000** tras ver las tres tasks (documentado
en `output/continual/README.md`, fase 3). No solo olvidaba: olvidaba todo, y
ni siquiera aprendía las tasks lo suficiente como para tener algo que olvidar.

La fase 4 parte de una observación de neurociencia: Cai et al. (Nature 2026)
reportan **selectividad poblacional esparsa** en córtex — solo una fracción
pequeña de neuronas responde fuertemente a cada feature, y distintos contextos
reclutan poblaciones distintas. La pregunta que abre esta fase: ¿puede una
regla **puramente local** (sin backprop, sin controlador central) producir esa
especialización por task, y traducirla en resistencia al olvido?

**Distinción clave de diseño.** El paper *observa* esparsidad emergente en
córtex; nosotros *imponemos* esparsidad vía un top-k duro (solo las `k_active`
unidades de mayor activación sobreviven el forward pass; el resto se fuerza a 0
y no recibe update). El top-k impuesto es nuestro **inductive bias**, elegido
para producir especialización análoga a la observación de Cai et al. — **no es
lo que el paper propone**. Es nuestro diseño inspirado en su observación.

## 3. Cadena de intervenciones (la historia de 6 pasos)

Cada paso fue motivado por el resultado del anterior. Todos respetan la
restricción dura: **el SPCN no usa backprop en ningún momento**, y el cuerpo
predictive-coding (2 PCLayers, regla Hebbiana + contrastiva local,
homeostasis vía EMA) no cambió su forma — los pasos añaden mecanismos de
allocation/protección, no gradiente global.

| # | Intervención | Qué se cambió | Qué número dio | Qué reveló |
|---|--------------|---------------|----------------|------------|
| 1 | **Top-k sparse** | `k_active = 15%` de out_dim por PCLayer; solo el top-k sobrevive y recibe update (no firing → no plasticity). | Peak por task sube de ~0.25 a ~0.6-0.9. | La esparsidad impuesta da capacidad: las tasks dejan de pisarse en el mismo conjunto denso de unidades. El SPCN *puede* aprender cada task. |
| 2 | **clr ↑ (0.002→0.01)** | Sube el contrastive learning rate (un cambio a la vez, hlr intacto). | Peak T1 = 0.89, train≈test (0.898 vs 0.892). | El cuello era under-fitting, no overfitting: train alto + test alto descarta memorización. La regla local aprende bien si la dejan moverse. |
| 3 | **Recruitment bias** (`recruit_beta`) | El top-k se elige sobre `score = act_norm − β·commitment_norm`: reserva neuronas poco comprometidas para tasks futuras. | Núcleos consistentes de tasks distintas empiezan a separarse. | La allocation se vuelve *dependiente de actividad histórica*: una neurona ya "ocupada" por una task pierde prioridad en la siguiente. |
| 4 | **Commitment tracker** | Contador por-unidad de cuántas veces estuvo en el top-k, **inmune a la homeostasis** (solo crece). | Permite distinguir "ocupación" (commitment crudo) de "importancia" (consistencia). | La homeostasis empuja `ema_state` hacia el target y borra la señal de qué neuronas ya están comprometidas; el commitment la preserva entre tasks. |
| 5 | **Protección "EWC local sin Fisher"** (`protect_gamma`) | Rigidez por unidad: `plasticity = 1/(1 + γ·consistencia_pre_task)`. La consistencia top-k (`sqrt(freq)`, consolidada como `max` across tasks, congelada pre-task) hace de **proxy local de importancia**. | Recupera retención (ver §5) sin ningún gradiente global. | Es EWC sin matriz de Fisher: la importancia de un peso se deriva solo de la actividad local, no de un gradiente del loss. |
| 6 | **Barrido `protect_gamma`** | γ ∈ {0, 1, 2, 5, 10}, misma init. | La frontera completa (§5). | El paso decisivo: ningún γ iguala a EWC en ambos ejes. El techo es estructural, no de tuning. |

## 4. Resultado positivo: allocation ortogonal por aprendizaje local

Con recruitment bias + protección (pasos 3-5), el SPCN produce **allocation
ortogonal entre el núcleo protegido y los núcleos nuevos**, vía aprendizaje
local sin backprop ni controlador central:

```
Núcleos consistentes (top layer, γ=2.0):
  T1_vs_T2:  Jaccard L2 = 0.00   ← núcleo de T1 disjunto del de T2
  T1_vs_T3:  Jaccard L2 = 0.00   ← núcleo de T1 disjunto del de T3
  T2_vs_T3:  Jaccard L2 = 0.576  ← T2 y T3 comparten ~58% del núcleo
```

El núcleo de la primera task (la que se consolida y protege) queda **totalmente
disjunto** de los núcleos de las tasks posteriores. Esto sin máscaras
explícitas (PackNet) ni un controlador que asigne capacidad (HAT): solo la
regla local de recruitment, que penaliza reclutar neuronas ya comprometidas, y
la protección que congela las consolidadas. El "EWC local sin Fisher" usa el
**commitment como proxy local de importancia** en vez de la información de
Fisher, que requiere gradientes globales.

**Matiz honesto (no sobre-vender).** La ortogonalidad es **parcial, no total**.
T1 queda disjunto de todo lo demás porque es lo que se protege; pero T2 y T3
compiten por el mismo pool de neuronas frescas y terminan compartiendo ~58% del
núcleo (Jaccard 0.576). El mecanismo protege contra el núcleo *consolidado*, no
particiona el espacio entero a priori. La capa inferior (L1) reporta 0 unidades
"consistentes" bajo el umbral de over-uso (×1.5 sobre uniforme): la
especialización medible vive en la top layer.

## 5. Resultado negativo / techo estructural (el hallazgo central)

A pesar de la allocation ortogonal, **la frontera de Pareto del SPCN escalar es
enteramente interior a la de EWC**. Barrido completo de `protect_gamma`
(seed=0, misma init, class-incremental 10-way):

| γ | Peak/task | Final/task | **BWT_avg** | **Plasticity_avg** | Retention_avg | Jaccard_L2_avg |
|------|----------------------------|----------------------------|-------------|--------------------|----------------|----------------|
| 0.0  | [0.892, 0.886, 0.788]      | [0.240, 0.218, 0.788]      | −0.685      | 0.855              | 0.258          | 0.063 |
| 1.0  | [0.892, 0.859, 0.749]      | [0.277, 0.366, 0.749]      | −0.575      | 0.833              | 0.368          | 0.131 |
| 2.0  | [0.892, 0.843, 0.641]      | [0.325, 0.417, 0.641]      | −0.518      | 0.792              | 0.430          | 0.192 |
| 5.0  | [0.892, 0.732, 0.292]      | [0.458, 0.418, 0.292]      | −0.390      | 0.639              | 0.542          | 0.253 |
| 10.0 | [0.892, 0.480, 0.101]      | [0.545, 0.351, 0.101]      | −0.264      | 0.491              | 0.671          | 0.303 |

Puntos de referencia (mismo protocolo, mismo seed):

| Modelo    | BWT_avg | Plasticity_avg | Retention_avg |
|-----------|---------|----------------|----------------|
| MLP Naive | −0.548  | 0.840          | 0.317 |
| **MLP+EWC** | **−0.345** | **0.786**     | **0.444** |

**Lectura.** A medida que γ sube, el SPCN intercambia plasticidad por retención
de forma monótona y limpia: BWT mejora (−0.685 → −0.264) mientras Plasticity
cae (0.855 → 0.491). Es una frontera de Pareto bien formada. Pero **ningún γ
domina a EWC en ambos ejes a la vez**:

- A **plasticidad comparable** a EWC (Plast ≈ 0.79, en γ=2.0), el SPCN tiene
  BWT = −0.518 vs −0.345 de EWC: **olvida ~50% más**.
- A **BWT comparable** a EWC (≈ −0.345, entre γ=5 y γ=10), el SPCN ha sacrificado
  tanta plasticidad (Plast ≈ 0.49-0.64) que ya no aprende las tasks nuevas:
  T3 cae a 0.10-0.29.

El punto EWC cae **fuera y arriba-a-la-derecha** de toda la curva del SPCN. Ver
`output/continual/pareto_frontier.png`: la curva verde (SPCN sobre γ) y la
estrella amarilla (EWC) — la estrella no es alcanzable por ningún punto verde.

**Causa identificada: la activación escalar.** Cada neurona del SPCN emite **un
solo valor** (su activación post-sigmoide top-k). Ese escalar es a la vez "qué
tan fuerte respondo" y "para qué task soy". No hay grados de libertad internos
para separar las dos cosas. La protección, entonces, solo puede operar sobre
ese único valor: subir `protect_gamma` **congela el rol completo** de una
neurona —toda su contribución, a cualquier task— porque no hay un canal interno
que distinga "preservá mi rol en T1" de "dejame adaptar para T3". Proteger es
binario-en-la-práctica: o la neurona queda rígida (y no ayuda a la task nueva),
o queda plástica (y pierde la vieja). Por eso la frontera es interior: el escalar
no puede preservar un rol y adaptarse a otro **simultáneamente en la misma
unidad**. EWC, con un peso real por conexión y una Fisher por peso, tiene
muchos más grados de libertad para repartir esa tensión.

**El solapamiento T2-T3 es evidencia adicional del techo.** La allocation
ortogonal del §4 mitiga la tensión escalar reservando *neuronas distintas* por
task —y de hecho lo logra mientras hay neuronas frescas: T1, la task protegida,
queda con Jaccard L2 = 0.00 contra todo lo demás. Pero la capacidad es finita
(64 unidades en la top layer, k=10 activas). En cuanto la demanda de tasks supera
al pool de neuronas frescas, la estrategia de "una población por task" deja de
ser viable: T2 y T3 **vuelven a compartir el 58% del núcleo** (Jaccard 0.576).
Y en cuanto dos tasks comparten unidades, esas unidades vuelven a enfrentar la
tensión escalar irresoluble —preservar el rol de una *o* adaptarse a la otra,
no ambas. El solapamiento medido no es ruido: es exactamente la firma de que el
escape por allocation se agotó y el techo de la representación escalar reaparece.
Con más tasks, el solapamiento solo puede crecer y la frontera cerrarse más.

### Dos techos distintos, no una contradicción

Conviene ser preciso sobre la relación con `FINDINGS_PHASE2.md`, porque son
**dos techos estructurales distintos** y es fácil confundirlos:

- **Techo de fase 2 (eje de capacidad).** En clasificación estática, la regla
  predictive-coding local + un prior top-down genérico no tiene ningún mecanismo
  que *junte* representaciones de la misma clase y *separe* las de clases
  distintas. Es un techo sobre **qué función** puede aprender la regla local:
  sin un término discriminativo, no separa clases por encima del azar.
- **Techo de fase 4 (eje de representación escalar).** Aquí la regla local
  **sí** aprende cada task (top-k + contrastivo resuelven el techo de fase 2
  para una task aislada: peak ~0.89). El techo nuevo es otro: la **activación
  escalar** no tiene grados de libertad internos para que una misma neurona
  preserve su rol en una task y se adapte a otra. Es un techo sobre **cómo
  coexisten múltiples funciones** en el mismo sustrato a lo largo del tiempo.

No se contradicen: son **capas distintas del mismo problema de fondo** —la
pobreza representacional de un escalar local. La fase 4 levantó el techo de
capacidad de la fase 2 (la regla ya clasifica una task), y al hacerlo expuso el
techo de la representación escalar que estaba debajo. Resolver un techo no
borra el otro; los apila. Por eso la dirección de fase 5 ataca la
representación misma (§6), no la regla de aprendizaje ni el protocolo.

## 6. Próximo paso motivado (fase 5): activación compleja

El techo no es de tuning ni de regla de aprendizaje — es de **representación**.
Un escalar por neurona no tiene estructura interna donde separar "intensidad de
respuesta" de "contexto/task". La dirección que esto abre:

**Activación compleja** — cada neurona emite un número complejo en vez de un
escalar:

- **magnitud** = intensidad de respuesta (lo que el escalar ya codificaba),
- **fase** = contexto / task.

Con fase, dos tasks pueden vivir en **subespacios ortogonales por construcción**
(fases separadas → producto interno ~0), no por protección retrospectiva. La
protección dejaría de ser "congelar la magnitud" (que mata la plasticidad) y
pasaría a ser "respetar la fase" (que deja la magnitud libre para adaptarse).
Una misma neurona podría sostener su rol en T1 *en la fase de T1* y aprender T3
*en la fase de T3*, sin que una pise a la otra — exactamente el grado de
libertad que al escalar le falta.

Conexión teórica: esto es **binding-by-phase** (Singer; von der Malsburg) y
**Holographic Reduced Representations** (Plate 1995) — esquemas donde la fase
liga features a roles/contextos y permite superponer múltiples ligaduras en el
mismo sustrato sin interferencia destructiva. La hipótesis de fase 5: la fase
da el canal que la frontera escalar no tiene, y la frontera del SPCN podría
entonces tocar (o cruzar) la de EWC sin introducir backprop global.

## Restricciones respetadas

- El SPCN **no usa backprop** en ningún paso. Toda la protección y la
  allocation se derivan de actividad local (top-k, commitment, consistencia).
- No se "mejoró" el SPCN ajustando hacia un número objetivo: el barrido de §5
  es la curva que produce el código existente sobre 5 valores de γ con la misma
  init. El resultado documentado es la frontera medida, no una selección.
- El protocolo es el más duro disponible (head compartido 10-way, eval
  unmasked) para los tres modelos por igual.

## Qué NO dice este finding

El SPCN **no le gana a EWC**. El resultado no es "el local learning resuelve
continual learning". El resultado es: **(a)** la frontera de Pareto del SPCN
escalar, medida; **(b)** el mecanismo del techo (la activación escalar no puede
preservar y adaptar un rol a la vez); y **(c)** la dirección que ese mecanismo
motiva (estructura interna vía fase). Allocation ortogonal sin controlador
central es un resultado positivo real, pero acotado: no cierra la brecha con
EWC por sí solo.

## Archivos

- `output/continual/pareto_frontier.png` — **figura central**: frontera
  BWT-vs-Plasticity del SPCN (curva sobre γ) con MLP Naive y MLP+EWC como
  puntos de referencia. La frontera del SPCN es interior a EWC.
- `output/continual/cl_3tasks_sweep.json` — barrido completo de `protect_gamma`
  + baselines (tabla de §5).
- `output/continual/cl_3tasks_results.png` / `cl_3tasks_metrics.json` — corrida
  principal a γ=2.0 (punto operativo Plast≈EWC), matriz de accuracy y
  specialization metrics.
- `continual_learning_3tasks.py` — experimento principal (protocolo
  class-incremental, SPCN + MLP naive/EWC).
- `continual_learning_sweep.py` — barrido de γ + plot de la frontera.
- `FINDINGS_PHASE2.md` — el techo estructural análogo en el eje de capacidad.
