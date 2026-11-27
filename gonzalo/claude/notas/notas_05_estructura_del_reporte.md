# Notas del TP — 05. Estructura del reporte

## Pregunta

> Cómo se podría armar el reporte, qué secciones debería tener para incluir lo solicitado. Considerar que el ítem indicado como "Incluir una comparación razonada entre DT e IQL" sea mínimo.

## Respuesta

La consigna pide un reporte de **4–6 páginas** (o un Markdown equivalente) que cubra: problema, protocolo, decisiones, evidencia y límites. También tiene que incluir una comparación razonada entre DT e IQL. La bitácora, la reproducción y el uso de IA pueden ir como anexos fuera del cuerpo principal.

La propuesta sigue el orden de la rúbrica, así cada criterio tiene una sección que lo respalda. La comparación DT/IQL queda reducida a un párrafo y una tabla chica. Igual tiene que estar, porque la rúbrica la evalúa dentro de "Reporte y comprensión".

### Estructura propuesta

| # | Sección | Extensión aprox. | Criterio de la rúbrica |
|---|---|---:|---|
| 0 | Resumen | 5–8 líneas | — |
| 1 | Datos y formulación del problema | ¾ pág. | Datos y formulación (15) |
| 2 | Protocolo de evaluación y líneas base | ½ pág. | Baselines y evaluación (20) |
| 3 | Decision Transformer | ¾ pág. | Decision Transformer (15) |
| 4 | IQL | ¾ pág. | IQL (15) |
| 5 | Resultados principales | 1 pág. | Baselines y evaluación (20) |
| 6 | Return conditioning | ¾ pág. | Return conditioning (15) |
| 7 | Calidad de los logs | ¾ pág. | Calidad de los logs (10) |
| 8 | Comparación DT vs. IQL (mínima) | ¼ pág. | Reporte y comprensión (10) |
| 9 | Límites y conclusiones | ¼ pág. | Reporte y comprensión (10) |
| A | Anexos: reproducción, bitácora, uso de IA | fuera del límite | 2.3 y 6 de la consigna |

---

### 0. Resumen

En pocas líneas:
- qué se hizo: DT e IQL sobre Netflix o Goodreads, evaluados con sesiones simuladas;
- el resultado principal: el normalizado de cada modelo frente a `item_mean` y `group_posterior` en las condiciones clave;
- la conclusión más importante de cada experimento.

### 1. Datos y formulación del problema

- **Dataset elegido** y qué representa cada campo: usuario, grupo, ítems, ratings.
- Diferencias entre train y test: test tiene ratings de todo el catálogo, y su orden de IDs no es un orden temporal.
- **Conclusiones de la exploración** que se usan después, en 3 o 4 viñetas:
  - la política de registro es uniforme;
  - el rating no depende de la posición;
  - los grupos son distintos entre sí.
- **Particiones:** 90/10 por usuario, estratificado por grupo. Test reservado.
- **Trayectorias y RTG:** episodios de 20 pasos consecutivos. R̂ se calcula dentro del episodio, con escala 20–100.
- **MDP en una tabla:** estado, acción, recompensa, transición, horizonte, γ.
- **Qué necesita cada modelo en el estado:** el DT, la secuencia de tokens (R̂, s, a); IQL, un codificador del historial que da T+1 estados.

### 2. Protocolo de evaluación y líneas base

- Reglas de la sesión: 20 pasos, sin repetición, el rating se revela recién al mostrar el ítem. Qué observa la política en cada paso.
- Las tres condiciones: visible, oculto (cold-start estricto) y warm-up 5 (no es cold-start).
- Métricas:
  - principal: retorno normalizado;
  - secundarias: hit rate, NDCG@10 y precision@10 del primer paso.
- Separación de las dos fuentes de variabilidad: el IC por usuario y la dispersión entre semillas. Selección en validación, test usado una sola vez.
- **Justificación de las líneas base**, en una línea cada una. Como mínimo `item_mean` y una personalizada: `group_mean` y `group_posterior`.

### 3. Decision Transformer

- **Representación:** tokens R̂, s y a. Qué va en s_t y por qué. Embeddings de ítem, paso y tipo de token. Escalado de R̂.
- **Máscara causal, padding y alineación:** a_t se predice desde la salida de s_t. Ítems ya vistos enmascarados con −∞.
- **Evidencia de no fuga:** mencionar el test y que pasa.
- **Grupo oculto:** qué fracción de episodios se entrena con grupo "desconocido".
- **Entrenamiento:** arquitectura (d_model, capas, cabezas), optimizador, scheduler, épocas, criterio de checkpoint.
- **Curva** de cross-entropy en train y validación, comparada contra log(n_items).
- **De modelo a política:** cómo se actualiza R̂ durante la sesión, incluido el warm-up.

### 4. IQL

- **Representación del estado:** el codificador, y si Q, V y π lo comparten.
- **Las tres pérdidas:** expectil para V, TD para Q con r + γ·V(s'), y AWR para π. Escribir las fórmulas una vez.
- **Decisiones:**
  - recompensa cruda o centrada;
  - el último paso es terminal, sin bootstrapping;
  - red objetivo con Polyak (λ);
  - truncado de pesos (w_max);
  - qué se calcula sin gradiente.
- **Barrido de hiperparámetros** en validación: τ ∈ {0,7; 0,9}, β ∈ {3; 10}, γ. Una tabla chica con el normalizado de validación y la configuración elegida.
- **Curvas:** las pérdidas y el peso medio de AWR. Aclarar que las pérdidas no se usan para elegir.
- **γ = 0 frente a γ > 0:** resultado e interpretación. ¿Qué parte del valor viene de la secuencia? En este generador, solo la información que se gana sobre el grupo.
- **π frente al máximo de Q** en las tres condiciones. Explicarlo con el error de extrapolación.

> γ y π/Q también pueden ir en la sección 5. Si quedan acá, la sección de IQL cubre todo lo que pide su criterio de la rúbrica.

### 5. Resultados principales

- **Tabla central en test**, una por tipo de logs, o una con ambos bloques:

  | Política | Visible | Oculto | Warm-up 5 visible | Warm-up 5 oculto |
  |---|---:|---:|---:|---:|
  | `random`, `popularity`, `item_mean`, `group_mean`, `group_posterior` | | | | |
  | DT (R̂ elegido en val.) | | | | |
  | IQL π | | | | |
  | IQL máx. Q | | | | |

  Los modelos se reportan como **media ± desvío entre semillas**. El IC por usuario se indica aparte.
- Métricas secundarias (hit rate, NDCG@10) en una tabla compacta o en el anexo.
- **Resultados por grupo:** un gráfico o tabla de `resumen_por_grupo.csv` para la condición más informativa.
- **Lectura:** 4 a 6 viñetas que expliquen por qué cada modelo queda donde queda, usando las conclusiones de la sección 1. Por ejemplo, si supera a `item_mean` (personaliza) y si se acerca a `group_posterior` con grupo oculto (se adapta). Una diferencia menor que la dispersión entre semillas no es una mejora.

### 6. Return conditioning

- **Diseño:** soporte de train (distribución de retornos), retornos pedidos (percentiles 10–99 y valores por encima del máximo), selección en validación.
- **Gráfico** de retorno obtenido frente a pedido, agregado y por grupo, con la diagonal y el máximo de train marcados. Indicar si con warm-up se usa `full_return` o `return`.
- **Interpretación:**
  - cuánto sigue el obtenido al pedido;
  - la meseta;
  - qué pasa fuera del soporte;
  - las diferencias entre grupos, que tienen escalas distintas;
  - si el mejor pedido es el más alto;
  - la incertidumbre.

### 7. Calidad de los logs

- **Setup:** DT e IQL reentrenados con `mixto` sin otros cambios. Las líneas base se ajustan con los mismos logs, y se agregan `popularity` y el sistema anterior como referencia.
- **Tabla:** `random` frente a `mixto`, solo las filas relevantes.
- **Interpretación:**
  - qué aprende cada modelo con cada tipo de logs;
  - si alguno supera al sistema que generó los datos, y por qué;
  - por qué cambian `popularity` e `item_mean` entre los dos datasets y los dos tipos de logs (sesgo de selección).

### 8. Comparación DT vs. IQL (mínima)

Un párrafo de 4 o 5 oraciones y, si entra, una tabla de tres filas:

| | DT | IQL |
|---|---|---|
| Mecanismo | imita trayectorias condicionado por el retorno | elige acciones con ventaja positiva dentro del soporte |
| Dependencia de los logs | lo que distingue a las trayectorias de retorno alto | la cobertura de acciones por estado |
| Resultado observado | (resumen de 1 línea) | (resumen de 1 línea) |

El párrafo dice cuál rindió mejor en cada condición, con un **por qué** respaldado por la evidencia de las secciones 5–7. No hace falta repetir los detalles de cada modelo.

### 9. Límites y conclusiones

- **Límites:**
  - los datos son sintéticos, sin efectos de orden;
  - el simulador usa usuarios de test con ratings completos;
  - el catálogo de validación es distinto al de test;
  - hay pocas semillas;
  - la extrapolación del DT.
- **Conclusiones:** dos o tres frases que respondan el objetivo del TP.

### Anexos (fuera de las 4–6 páginas)

- **A. Reproducción:**
  - versiones de Python y de las librerías;
  - comandos para entrenar y evaluar;
  - ruta de configuraciones, semillas y checkpoints;
  - ubicación de las curvas sin recortes.

  Si se usa una librería, qué resuelve y cómo se verificó su contrato de entradas y salidas.
- **B. Bitácora:** contribuciones de cada integrante y decisiones verificadas, con fecha.
- **C. Uso de IA (opcional):** si se usó, para qué y qué se comprobó. Sin conversaciones completas.
- **D. Referencias.**

### Consejos para entrar en 4–6 páginas

- Escribir cada fórmula una sola vez, en las secciones 3 y 4.
- Llevar a anexos o notebooks las tablas de barridos largas y las métricas secundarias.
- Preferir gráficos que respondan una pregunta concreta, como el de retorno pedido frente a obtenido o el de resultados por grupo, en vez de muchas curvas.
- Cada sección de experimento sigue el mismo esquema: **diseño → resultado → interpretación → límite**.

### Fuentes

- [CONSIGNA.md](CONSIGNA.md), secciones 1–6
- [RUBRICA.md](RUBRICA.md)
- [GUIA.md](GUIA.md), secciones 7 y 9
