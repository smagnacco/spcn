# Phase 5 Findings: La Fase y la Allocation Espacial son el Mismo Grado de Libertad

## 1. Hipótesis

La fase 4 cerró con un techo identificado como **de representación, no de
mecanismo**: una activación escalar no tiene grados de libertad internos para
que una misma neurona preserve su rol en una task y se adapte a otra a la vez
(FINDINGS_PHASE4 §5-6). La fase 5 pone a prueba la dirección que ese techo
motivaba:

> **Activación compleja** `z = r·e^(iθ)` — magnitud `r` = intensidad de
> respuesta (lo que el escalar ya codificaba), fase `θ` = contexto/task. Con
> fase, dos tasks podrían vivir en subespacios ortogonales **por construcción**
> (producto interno ~0), y la protección dejaría de ser "congelar la magnitud"
> (que mata plasticidad) para pasar a ser "respetar la fase" (que deja la
> magnitud libre). Conexión teórica: binding-by-phase (Singer; von der Malsburg)
> y Holographic Reduced Representations (Plate 1995).

Predicción falsable: la frontera de Pareto del SPCN complejo se mueve **hacia
afuera** de la escalar de fase 4, hacia (o más allá de) el punto de EWC.

Protocolo idéntico a fase 4 para comparación directa: 3 tasks class-incremental
(T1=0-3, T2=4-6, T3=7-9), head único 10-way, masked CE en train, eval unmasked,
5000/1000 por task, seed=0. Baselines (MLP naive, MLP+EWC) **no** re-corridos —
se usan los números de FINDINGS_PHASE4.

## 2. Diseño y la corrección que forzó el propio instrumento

La fase 5 se subdividió en 5a (fase **asignada** = upper bound teórico), 5b
(fase emergente con presión de separación) y 5c (fase puramente emergente). El
plan: validar primero que la maquinaria compleja funciona con la fase correcta
*antes* de pedir que la fase emerja. **El proyecto se detuvo en 5a**: la fase
asignada —el techo teórico— ya no separa tasks. No tiene sentido probar fase
emergente cuando la asignada no anda.

### 2a. El bug de factorización que descubrió el sanity (b)

El primer 5a asignaba una fase **uniforme por capa** (θ_t igual para toda
neurona en la task t). El sanity (b) —el producto interno complejo entre las
activaciones promedio de las tasks, que debía dar `cos(120°) = −0.5`— lo refutó
de inmediato: con fase uniforme, `e^(iθ_t)` **factoriza** en la suma de la capa
siguiente:

```
W @ (r·e^(iθ_t)) = e^(iθ_t) · (W @ r)
```

y si el readout lee magnitud `|·|`, la fase **desaparece** del readout. El 5a
uniforme colapsaba a la escalar de fase 4: concluiríamos falsamente que "la fase
no sirve" cuando nunca entró en juego. El instrumento detectó el error de diseño
antes de que contaminara la conclusión.

### 2b. El rediseño 5a': fase como estrategia ALTERNATIVA a la allocation

La corrección reveló algo más profundo que un bug de readout. La fase 4 ya
separa tasks **espacialmente**: el recruitment bias produce núcleos disjuntos
(Jaccard T1-T2 = 0.00). El sanity (b) mostró que donde los núcleos son
disjuntos, las activaciones ya son ortogonales **por su soporte** —no queda
solapamiento para que la fase atenúe. El `−0.5` solo podría aparecer en neuronas
**compartidas** entre tasks, que el recruitment elimina por construcción.

Esto reorientó el experimento: la fase y la allocation de fase 4 no son ejes
ortogonales que se suman (como asumía PHASE5_DESIGN §3), sino **estrategias
alternativas para el mismo problema**. El 5a' testea la fase *en su propio
terreno*, dándole neuronas compartidas para separar:

- **Recruitment OFF** (`recruit_beta = 0`): sin reserva de neuronas frescas, los
  núcleos de las tasks se **solapan** (deseado, no efecto secundario).
- **Fase en un acumulador complejo por neurona** (pesos REALES, §2 del diseño
  intacto): `c_i += (masa de uso de la task t en i) · e^(iθ_t)`, acumulado en
  cada update. La neurona emite
  `z_i = r_i · (|c_i|/max|c|) · e^(i·arg(c_i))`.
  La masa de uso es la activación `r` de la neurona en el top-k (la señal que
  define su pertenencia a la task), no `|plast·err|` —en este cuerpo PC el error
  Hebbiano es ~0 sobre las activas (el aprendizaje vive en homeostasis +
  contrastivo), así que `|err|` no mide cuánto entrenó la task a la neurona.
- **Readout proyectado a fase de task**: `Re(z·e^(−iθ_t))`. La magnitud,
  top-k, commitment y protección siguen siendo los de fase 4, sobre `r`.

El mecanismo previsto: una neurona compartida T1+T2 acumula contribuciones en 0°
(de T1) y 120° (de T2); `arg(c_i)` se mueve y `|c_i|` cancela parcialmente
(interferencia destructiva). Ese costo de magnitud por compartir sería la
tensión que al escalar le faltaba —y, crucialmente, la aritmética sola
recompensaría fases distintas por task, **sin loss agregado**.

## 3. El instrumento decisivo: por qué el juez es el coseno, no el BWT

Un BWT que mejora puede engañar: si `|c_i|` reduce la magnitud de neuronas
compartidas, eso toca lo que leen el head y el top-k → se mide
**interferencia-por-fase como si fuera protección-escalar** (el mismo error que
el sanity (b) expuso en 2a). Por eso el criterio de éxito/parada de 5a' se ancla
en el **coseno complejo entre tasks en neuronas compartidas**, no en el BWT:

- **Coseno ≈ −0.5** → la fase separa neuronas que tasks distintas comparten.
  Ese es el mecanismo entero; si no ocurre, nada más importa.
- El BWT del barrido de γ mide el eje **conocido** de fase 4 (protección de
  magnitud), no la pregunta nueva.

## 4. Resultado: el coseno no llega a −0.5, demostrado desde los dos extremos

La tabla de los dos extremos es la pieza argumentativa central. Ambos miden el
mismo objetivo —separación por fase entre tasks en la top layer, γ=0, escala
completa— variando solo cuánto se solapan los núcleos:

| Régimen | Núcleos (Jaccard L2) | Coseno entre tasks (Re) | Por qué ≈ 0 |
|---------|----------------------|--------------------------|-------------|
| **Recruitment ON** (fase por-núcleo) | disjuntos (0.00) | +0.017 / +0.130 / +0.052 | núcleos disjuntos → **nada compartido que separar** |
| **Recruitment OFF** (acumulador `c_i`) | full overlap | +0.135 / +0.003 / +0.125 (en compartidas, n=47-64) | toda neurona promedia las 3 tasks → **la fase no puede separar** |

Requerido: **−0.5**. Obtenido: **≈ 0** en ambos extremos.

*Nota de honestidad metodológica:* los dos regímenes usan mecanismos de fase
distintos —ON asigna fase por-núcleo (y de hecho factoriza en L1, donde la
distribución casi-uniforme no forma núcleo); OFF usa el acumulador per-neurona
`c_i`, que **no** factoriza (53 fases distintas en L1, 41 en L2). No es una
comparación A/B del mismo mecanismo bajo dos β. Es el mismo *objetivo* alcanzado
por los dos caminos posibles de poblar la fase, y ninguno separa. El argumento
no depende de que los mecanismos sean idénticos, sino de que **agotan las dos
maneras de repartir la fase entre neuronas**: o las tasks no comparten neuronas,
o las comparten todas.

*La objeción obvia y por qué no aplica.* Un lector escéptico preguntaría: ¿no
podría haber un **punto intermedio** —solapamiento parcial, con un mecanismo
intermedio— donde la fase sí separe? No: los dos extremos no son dos muestras de
un continuo con un posible óptimo en el medio. Son las dos únicas **topologías**
posibles de reparto de fase: cero solapamiento (recruitment ON) o solapamiento
total (OFF). Cualquier punto intermedio —donde algunas neuronas se comparten y
otras no— requiere un mecanismo que decida **qué neuronas se comprometen con qué
task dominante**, y ese mecanismo es precisamente la allocation espacial. El
intermedio no rescata la fase: **colapsa en allocation por construcción**. No
hay vía de escape entre los dos extremos que no sea, ella misma, allocation.

Detalle del extremo OFF (el decisivo), γ=0:

```
BWT -0.388   Plast 0.289   Reten 0.166
coseno (compartidas):  T1-T2 +0.135 (n=64)   T1-T3 +0.003 (n=47)   T2-T3 +0.125 (n=47)
no-factorización:      L1 53 fases distintas, L2 41 → NO factoriza (la fase ES per-neurona)
interferencia destr.:  n_mono=0, n_multi=64  (toda neurona top-layer usada por las 3 tasks)
```

Dos lecturas importan:

1. **No es el bug de factorización.** El acumulador `c_i` es genuino y
   per-neurona (53/41 fases distintas). La fase **varía**; lo que no hace es
   **separar tasks**.
2. **El BWT −0.388 no es retención real.** Plasticity colapsó a 0.289 —las tasks
   apenas se aprenden (peak ~0.29)—, así que ese BWT es "no hay nada que
   olvidar", no protección. Confirma por qué el coseno, no el BWT, es el juez.

## 5. El hallazgo estructural

`n_mono = 0` es la clave: con recruitment apagado, **cada neurona de la top
layer es usada por las 3 tasks**. Entonces `arg(c_i)` es el resultante de
T1@0° + T2@120° + T3@240° superpuestos → **promedia las tres** en vez de
ubicarlas en bandas distintas. El acumulador `c_i` **superpone, pero no
compite**: nada decide qué task *domina* cada neurona.

Para que la fase separe, cada neurona tendría que **comprometerse con una fase
dominante** —caer en la banda de una task. Pero ese mecanismo de competencia
por-neurona **es la allocation espacial de fase 4 por otro nombre**: "cada
neurona dominada por una task" es exactamente lo que el recruitment bias produce.

Por eso fase y allocation colapsan en el mismo grado de libertad, y se ve desde
los dos extremos: con recruitment ON la competencia ya ocurrió (núcleos
disjuntos) y la fase es redundante; con recruitment OFF no hay competencia y la
fase promedia (mush). No existe configuración intermedia donde la fase aporte
una separación que la allocation no haya dado ya.

**Conclusión precisa (no la amplia):**

> La superposición en fase requiere **competencia por-neurona** para separar
> tasks, y esa competencia es funcionalmente equivalente a la **allocation
> espacial**. En este sustrato —acumulador lineal de fase + readout
> proyectado— la fase **no aporta un grado de libertad nuevo** sobre la
> partición espacial de la fase 4.

Lo que este finding **no** dice: no afirma que "la superposición en fase no
sirve para continual learning". El experimento testeó un sustrato de **fase
lineal** (un acumulador complejo y un readout proyectado). No testeó sustratos
donde la fase **compita de forma nativa** —osciladores acoplados, dinámica tipo
Kuramoto, donde la sincronización fuerza a cada unidad a comprometerse con una
fase sin un mecanismo de allocation explícito. Esa puerta queda abierta (§6); no
afirmamos que funcionaría, solo que no fue testeada.

## 6. Qué queda

El mejor resultado del proyecto sigue siendo el **aporte positivo de la fase 4**
(FINDINGS_PHASE4 §4): allocation ortogonal **emergente** entre tasks a partir de
reglas estrictamente locales, sin backprop, sin controlador central ni máscaras
explícitas —a diferencia de PackNet/HAT, que logran poblaciones disjuntas por
asignación explícita. La fase 5 **no** levantó el techo de Pareto: la
representación compleja, en este sustrato, no compra el grado de libertad que el
escalar no tenía, porque la única forma de hacer que la fase separe reintroduce
la partición espacial que ya teníamos.

**Trabajo futuro (no testeado):** un sustrato con competencia de fase nativa
podría romper la equivalencia fase↔allocation. Si la sincronización entre
unidades forzara el compromiso de fase por dinámica (no por un selector de
allocation), la fase podría separar tasks sin reducirse a la partición espacial.
Candidatos: osciladores acoplados, redes de Kuramoto, readout no-lineal sobre la
fase. Es la continuación natural —y el finding actual delimita exactamente qué
le faltó al sustrato lineal para no afirmar de más.

## Restricciones respetadas

- **Sin backprop global** en ningún punto. Magnitud, top-k, commitment,
  protección y contrastivo de magnitud son los de fase 4; el acumulador de fase
  es local (se llena con la actividad top-k de cada task).
- **No se ajustó hacia un número objetivo.** El barrido completo de γ **no se
  corrió**, y la razón importa más que el hecho: con la fase asignada (el upper
  bound) ya fallando el coseno, la plasticidad colapsa (Plast=0.289 en γ=0). Una
  "figura de tres curvas" trazada sobre una red que apenas aprende las tasks
  **no sería comparable con la frontera escalar de fase 4** —no es que falte
  correr el barrido, es que el barrido mediría protección de magnitud sobre una
  red cuya plasticidad ya está rota, produciendo una curva colapsada que
  invitaría a una comparación inválida. El criterio de parada del diseño es
  explícito para este caso.
- **Un cambio a la vez**, checkpoint y reporte antes de avanzar. El proyecto se
  detuvo en 5a por el criterio de parada; 5b/5c no se ejecutaron.

## Archivos

- `continual_learning_phase5.py` — implementación 5a' (ComplexPCLayer con
  acumulador `c_i`, recruitment OFF, readout proyectado, diagnósticos de fase).
  No toca los scripts de fase 4.
- `output/continual/phase5a_prime_gamma0.json` — el sanity decisivo (γ=0, escala
  completa): coseno en compartidas, no-factorización, interferencia destructiva.
- `output/continual/cl_phase5a_sweep.json` — el primer 5a (recruitment ON, fase
  por-núcleo); su sanity (b) es el extremo "núcleos disjuntos" de la tabla §4.
- `FINDINGS_PHASE4.md` — el techo escalar que la fase 5 intentó (y no logró)
  romper, y el aporte positivo que sigue siendo el mejor resultado del proyecto.
