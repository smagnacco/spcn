# Phase 5 Design: Activación Compleja — ¿La fase rompe el techo escalar?

## Hipótesis central

El techo de la fase 4 es de **representación**, no de mecanismo: una activación
escalar no puede preservar el rol de una neurona en una task y adaptarlo a otra
simultáneamente. Una **activación compleja** (`z = r·e^(iθ)`) separa esos dos
roles en dos grados de libertad:

- **magnitud `r`** = intensidad de respuesta (lo que el escalar ya codificaba)
- **fase `θ`** = contexto / task

Predicción falsable: con fase, la protección deja de ser "congelar la magnitud"
(que mata plasticidad) y pasa a ser "respetar la fase" (que deja la magnitud
libre). Eso debería **romper la monotonía del trade-off** de la fase 4 y mover
la frontera de Pareto hacia afuera — hacia (o más allá de) el punto de EWC.

## Criterio de éxito GLOBAL de la fase 5

> ¿La frontera de Pareto del SPCN complejo toca o cruza el punto de EWC
> (BWT ≈ −0.345, Plast ≈ 0.79) en el mismo protocolo 3-task class-incremental?

Resultado soñado: retención alta SIN sacrificar plasticidad (la frontera se
mueve hacia afuera, no a lo largo de la misma curva escalar).

Resultado negativo válido: si la frontera compleja queda donde la escalar, la
fase no compró nada → la representación compleja no era el cuello de botella.
Ambos desenlaces son publicables.

---

## Decisiones técnicas fijadas (válidas para las 3 sub-fases)

1. **Activación compleja:** cada neurona emite `z = r·e^(iθ)`.
   - `r = sigmoid(W_mag · x + b_mag)` — magnitud, igual rol que el escalar actual.
   - `θ` = la fase (asignada en 5a; emergente vía `atan2(W_phase · x)` en 5b/5c).

2. **Pesos REALES, no complejos** (al inicio). Las activaciones llevan
   magnitud/fase; los pesos siguen siendo reales. Complejizar pesos duplica
   parámetros y mete dinámica propia — solo si 5b/5c lo justifican.

3. **Top-k y commitment sobre la MAGNITUD.** La fase no dice "qué tan activa"
   sino "en qué contexto". Toda la maquinaria de allocation de la fase 4
   (top-k, recruitment, commitment, protección por consistencia) se mantiene
   IDÉNTICA, operando sobre `r`. Esto hace la fase un eje ORTOGONAL al
   allocation: allocation-por-magnitud (fase 4) + separación-por-fase (fase 5)
   son complementarios, no competidores.

4. **Contrastive update con dos roles separados:**
   - Magnitud: misma regla que fase 4 (lo que ya funciona).
   - Fase: regla propia, empuja `θ` de la activación hacia `θ` del prototipo de
     su clase. CUIDADO: la fase es circular (0° = 360°); todas las diferencias
     de fase se toman módulo 2π (usar `atan2(sin(Δθ), cos(Δθ))`).
   - ⚠️ PUNTO DE MAYOR RIESGO DE BUG. Testear aislado antes de integrar.

5. **Head lee la MAGNITUD, no la fase.** La clase se decide por qué neuronas
   están activas y cuánto (magnitud), igual que fase 4. La fase es mecanismo
   interno de anti-interferencia, NO feature de clasificación. Head idéntico al
   de fase 4 → aísla el efecto de la fase al eje de retención.

6. **Interferencia compleja (el corazón del mecanismo):** el forward de la capa
   siguiente suma complejos `z_out = Σ w·z_k`. Dos activaciones en fases
   opuestas (θ y θ+π) se cancelan; en misma fase se refuerzan. Esto es lo que
   hace que tasks en fases distintas "no se pisen". VERIFICAR explícitamente que
   ocurre — es el mecanismo entero, no un detalle.

---

## ETAPA 5a — Validar la maquinaria compleja con FASE ASIGNADA

**Objetivo:** confirmar que la red compleja funciona cuando la fase es correcta,
ANTES de pedirle que la fase emerja. Da el techo teórico (upper bound) y es el
sanity check de toda la aritmética compleja.

**Qué se implementa:**
- Activación compleja con magnitud emergente + fase ASIGNADA por task:
  T1 → 0°, T2 → 120°, T3 → 240° (equiespaciadas en el círculo).
- Forward complejo con suma de complejos y pesos reales.
- Top-k/commitment/protección sobre magnitud (idéntico a fase 4).
- Contrastive: magnitud como fase 4; fase fija (no se aprende en 5a).
- Head sobre magnitud.

**Sanity checks (deben pasar o hay bug):**
- Con fase asignada perfecta y protección OFF (γ=0), el forgetting debería ser
  MUCHO menor que el escalar de fase 4 (la fase ya separa las tasks).
- Verificar interferencia: medir el producto interno complejo entre activaciones
  promedio de T1 y T2. Debería ser ~0 (fases a 120° → coseno = −0.5, ya separa).

**Criterio de éxito 5a:**
- BWT_avg claramente mejor que el γ=0 escalar (−0.685) SIN protección.
- Plasticity_avg comparable al escalar (~0.85) — la fase no debe romper el
  aprendizaje de cada task.
- Si BWT mejora con Plast intacta → la maquinaria compleja funciona y la fase
  asignada da el upper bound. Anotá ese BWT como techo teórico.

**Criterio de PARADA 5a:**
- Si con fase asignada PERFECTA el forgetting NO mejora vs escalar → hay un bug
  en la aritmética compleja (la interferencia no está separando tasks). Debuggear
  ANTES de seguir. No tiene sentido probar fase emergente si la asignada no anda.

---

## ETAPA 5b — Fase EMERGENTE con presión de separación

**Objetivo:** liberar la fase (la red elige qué fase usar por task) pero con un
término que la empuja a separarse. Medir cuánto del techo de 5a recupera.

**Qué se agrega sobre 5a:**
- `θ = atan2(W_phase · x)` — la fase ahora se computa de los inputs, con pesos
  `W_phase` que se aprenden localmente.
- **Phase decorrelation loss (local):** penaliza que activaciones de contextos
  distintos compartan fase. Implementación local sugerida: mantener un
  prototipo de fase por clase (EMA del θ medio de cada clase) y un término que
  empuje el θ de cada clase a separarse de los prototipos de otras clases
  (repulsión de fase en el círculo). NO es backprop global — es el mismo patrón
  de prototipos contrastivos de la fase 4, aplicado al eje de fase.
- La magnitud y todo el allocation siguen igual.

**Criterio de éxito 5b:**
- La frontera de Pareto (barrido γ) del SPCN-5b se mueve HACIA AFUERA respecto
  a la escalar de fase 4 — idealmente hacia el punto de EWC.
- Medir overlap de fase entre tasks: ¿las tasks efectivamente ocupan bandas de
  fase distintas? (debería emerger separación si la presión funciona).
- Comparar BWT/Plast contra el techo de 5a: ¿cuánto recupera la fase emergente
  de lo que la fase asignada lograba?

**Criterio de PARADA 5b:**
- Si la fase emergente colapsa (todas las tasks a la misma fase pese a la
  presión) → la presión es insuficiente o mal calibrada. Barrer la fuerza del
  decorrelation loss antes de concluir.

---

## ETAPA 5c — Fase PURAMENTE emergente (sin presión)

**Objetivo:** la versión más pura. Sin decorrelation loss. ¿La fase se separa
sola, solo por la dinámica del aprendizaje?

**Qué se cambia sobre 5b:**
- Quitar el phase decorrelation loss. La fase emerge solo de `W_phase` aprendido
  por el contrastive update, sin ninguna presión explícita de separación.

**Lectura del resultado (los dos desenlaces son informativos):**
- Si 5b funcionó y 5c NO → aprendiste que la separación de fase REQUIERE presión
  explícita. Resultado real y preciso sobre qué necesita la representación
  compleja para ser útil.
- Si 5c funciona solo → resultado fuerte e inesperado: la fase se auto-organiza
  sin supervisión. Sería el mejor resultado posible del proyecto.

---

## Protocolo (idéntico fase 4, para comparación directa)

- 3 tasks class-incremental: T1=0-3, T2=4-6, T3=7-9. Single head 10-way,
  masked CE en train, eval unmasked. 5000 train / 1000 test por task. seed=0.
- Baselines de referencia (ya medidos en fase 4, MISMO protocolo): MLP naive,
  MLP+EWC. NO re-correr — usar los números de FINDINGS_PHASE4.md.
- Métricas: BWT_avg, Plasticity_avg, Retention_avg, + overlap de fase entre tasks
  (nuevo, específico de fase 5).
- Barrido protect_gamma ∈ {0, 1, 2, 5, 10} en cada sub-fase para trazar la
  frontera y compararla con la escalar.

## Restricciones (heredadas de fase 4, NO cambian)

- SIN backprop global en ningún punto. Todo update es local (Hebbian/contrastivo,
  prototipos, decorrelation por prototipos).
- NO ajustar hacia un número objetivo. El barrido es la curva que produce el
  código; el resultado es la frontera medida.
- Un cambio a la vez. Checkpoint y reporte al final de cada sub-fase ANTES de
  pasar a la siguiente.

## Figura central de la fase 5

Plot comparativo: frontera de Pareto BWT-vs-Plasticity con TRES curvas
superpuestas — escalar (fase 4), compleja-5a (fase asignada = techo),
compleja-5b/5c (fase emergente) — más el punto de EWC. La historia entera en
una imagen: ¿la fase mueve la frontera hacia EWC?
