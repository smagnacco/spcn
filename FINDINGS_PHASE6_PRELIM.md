# Phase 6 Preliminary Finding: El Contexto Adentro de la Activación No Separa Tasks (Resultado Analítico)

> **Estado:** preliminar. Este documento registra un **resultado analítico** que
> obtuvimos *antes* de implementar nada, y que disuelve la versión barata de la
> hipótesis de la fase 6. No hay código nuevo asociado: el test que se iba a
> correr se demostró degenerado, y esa demostración **es** el hallazgo. El
> proyecto espera una decisión (§5) antes de construir.

## 1. De dónde viene la fase 6

La fase 5 (FINDINGS_PHASE5) cerró con un techo mecánico: una activación
**compleja** `z = r·e^(iθ)` no rompe el techo escalar de la fase 4, porque la
fase vive en `d=1` (un ángulo) y un acumulador `c_i = Σ_t m_t·e^(iθ_t)` **promedia**
las contribuciones de tasks distintas en vez de guardarlas separadas. Separar por
fase requería competencia por-neurona, que es allocation espacial por otro nombre.

La fase 6 preguntaba si **subir la dimensión** rescata la idea: una **capsule**,
cada neurona emitiendo un vector `v_i ∈ R^d` (norma = intensidad, como el escalar
de fase 4; dirección = contexto). La intuición: en `R^d` alto, dos vectores
aleatorios son casi ortogonales (concentración de medida), así que una neurona
compartida podría sumar T1 y T2 en direcciones casi-ortogonales **sin promediarlas
destructivamente** y sin que nadie asigne direcciones. La separación la daría la
**dimensión**, no el compromiso por-unidad — el escape del teorema de la fase 5.

## 2. El test decisivo que se iba a correr (y por qué)

El plan era un test barato, análogo del coseno de la fase 5, que pudiera matar la
idea en minutos. Tras descartar varias formulaciones que medían teoremas
preexistentes (proyección lineal del input → Johnson-Lindenstrauss; random
features no-lineales → Rahimi-Recht), llegamos al test fiel al teorema de la
fase 5, de **memoria asociativa distribuida** (Hopfield; Holographic Reduced
Representations, Plate 1995; hyperdimensional computing): superponer las
contribuciones de T1 y T2 de una neurona compartida en `R^d` y medir cuánto
interfiere T2 en la **lectura** de T1. La predicción teórica: interferencia
`~1/√d`, luego la razón señal/interferencia (SIR) `~√d`, con `d=1` reproduciendo
el colapso de la fase 5.

## 3. El resultado analítico: la SIR es PLANA en d (la capsule lineal no escapa)

Al derivar la SIR con cuidado **antes** de correrla, el test se reveló degenerado
—y la razón por la que es degenerado es un teorema sobre la hipótesis, no sobre
el test.

Con una proyección lineal fija `R` (la capsule lee su contexto por proyección),
la interferencia de T2 en la lectura de T1 es

```
interferencia = ⟨R·G2, â⟩,   con  â = R·G1 / ‖R·G1‖
              = ⟨R·G2, R·G1⟩ / ‖R·G1‖
              ≈ ⟨G2, G1⟩ / ‖G1‖          (Johnson-Lindenstrauss: R lineal preserva ⟨·,·⟩)
```

donde `G1 = Σ_{x∈T1} g(x)`, `G2 = Σ_{x∈T2} g(x)` son las contribuciones acumuladas
de cada task. **El lado derecho no depende de `d`.** La SIR queda

```
SIR = ‖R·G1‖ / |⟨R·G2, â⟩| ≈ ‖G1‖ / (‖G2‖·|cos(G1,G2)|)
```

— igual al origen, **para todo `d`**.

La esperanza era que la suma **por-muestra** diera un término incoherente que
cancelara como `√(n)/√d`. No existe como cantidad separada: por **linealidad**,

```
Σ_x ⟨R·g(x), â⟩ = ⟨R·(Σ_x g(x)), â⟩ = ⟨R·G, â⟩
```

la suma por-muestra **colapsa** en la proyección de la suma acumulada. No hay
cross-terms incoherentes que la dimensión pueda cancelar, porque la operación es
lineal de punta a punta. El `√d` requiere incoherencia, y la linealidad la prohíbe.

**Lo que esto prueba (no es un bug, es un resultado):** superponer gradientes en
`R^d` y leerlos por **proyección lineal** da la misma interferencia que en el
origen, para todo `d`. **La dimensión no compra nada con operaciones lineales.**
La capsule —norma + dirección, lectura por proyección— es lineal en ese sentido,
así que **colapsa por la misma razón que la fase, sin importar `d`**. Demostrado
sin correr nada.

## 4. El hallazgo unificador

Las fases 5 y 6 colapsan por **un solo motivo**, ahora visible:

> **El contexto codificado DENTRO de la activación de la neurona no separa
> tasks** — ni en `d=1` (fase: ángulo; demostrado empíricamente en
> `FINDINGS_PHASE5.md`), ni en `d` alto con lectura lineal (capsule: dirección;
> demostrado analíticamente acá vía JL). Escapar requiere **binding NO-lineal**
> (HRR/VSA: clave aleatoria por-ítem, bind por convolución, unbind por la clave),
> que es matemáticamente equivalente a un **espacio de claves EXTERNO a la
> neurona**. Es decir: el contexto que sí separa vive **afuera** de la neurona
> (neuromodulación, key-value), **no adentro** (fase, dirección).

Por qué el binding no-lineal **no** es el test decisivo barato: HRR con clave
por-muestra sí da `√(n/d)`, pero por el teorema de Plate (1995) —conocido y
garantizado por diseño. Medirlo confirmaría que HRR funciona, no que el SPCN lo
aproveche. Es el mismo vicio recurrente del proyecto: un mecanismo que *parece*
ejercitar la idea pero mide un teorema preexistente. El binding no-lineal es el
**mecanismo** correcto; no es un **test**.

## 5. Reorientación — decisión pendiente

El catch disolvió la versión geométrica y barata de la hipótesis. La pregunta
dejó de ser **geométrica** (¿las direcciones emergen ortogonales?) y pasó a ser
**funcional** (¿un binding no-lineal integrado al aprendizaje local del SPCN
reduce el forgetting?). Eso ya **no** tiene un test más barato que construir el
mecanismo en el CL real: no hay atajo geométrico que lo prediga.

Dos caminos, y la decisión es del usuario:

- **Pivotar la fase 6 a contexto EXTERNO** (neuromodulación / key-value): el
  contexto separador vive afuera de la neurona, como una clave que modula la
  lectura. Requiere implementar el mecanismo en el CL completo (protocolo de
  fase 4/5) y medir BWT/Plasticity — no hay test barato previo. **Riesgo
  conceptual a vigilar:** que la "clave externa" sea un selector de task
  asignado = allocation/HAT disfrazado. Sólo es un aporte si la clave **emerge**
  o es **neuromoduladora genérica**, no un one-hot de task.

- **Cerrar el arco acá** con el resultado analítico: el proyecto quedaría con
  **tres** hallazgos mecánicos encadenados —(fase 2) techo de capacidad de la
  regla local; (fase 4) allocation ortogonal emergente + techo de la
  representación escalar; (fase 5+6) el contexto-adentro-de-la-activación no
  separa, ni en fase ni en dirección, y por qué (linealidad + JL)— más el aporte
  positivo de la fase 4 como mejor resultado.

**No se implementó nada de la fase 6.** El test geométrico se descartó por el
argumento de §3. Esperamos la decisión antes de construir.

## Restricciones respetadas

- **Sin backprop global**, sin ajustar hacia un número, un cambio a la vez. El
  resultado de §3 es analítico (no una corrida ajustada).
- **No se midió un teorema preexistente**: precisamente por evitarlo se descartó
  el binding HRR como "test" (mediría a Plate, no al SPCN) y la proyección lineal
  (mediría a JL). Lo que queda es el resultado analítico, que es sobre la
  hipótesis de la fase 6, no sobre un teorema de librería.

## Archivos

- `FINDINGS_PHASE5.md` — el techo de la fase (`d=1`), del que la fase 6 era la
  generalización a `d` alto.
- `FINDINGS_PHASE4.md` — la allocation ortogonal emergente, que sigue siendo el
  mejor resultado del proyecto.
