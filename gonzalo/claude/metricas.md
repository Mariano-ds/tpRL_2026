# Métricas de evaluación — TP Decision Transformer e IQL

Este documento explica qué mide cada métrica del protocolo común (`protocolo/metricas.py` y `protocolo/evaluar.py`), cómo se agregan y cómo leerlas en `reporte.md` y `resultados.md`.

---

## 1. Contexto: qué se evalúa

Cada modelo se evalúa con **sesiones simuladas** de 20 recomendaciones sin repetición sobre usuarios de test:

1. En cada paso, la política da un puntaje a cada ítem.
2. El simulador muestra el ítem de mayor puntaje entre los que todavía no se mostraron.
3. Recién entonces revela su rating, que la política puede usar en los pasos siguientes.

En la condición de **warm-up**, los primeros pasos los elige la política de registro al azar (ver `reporte.md`, §2). Todas las métricas se calculan solo sobre los **pasos controlados**: los que decide la política, es decir 20 − warm-up. Con warm-up 0 son 20 pasos; con warm-up 5 son 15.

Hay dos tipos de métricas:
- **De sesión:** miden cuánto rating acumuló la política a lo largo de los pasos controlados. Incluyen el efecto de adaptarse con el feedback.
- **De ranking:** miden la calidad del primer ranking, antes de recibir ningún feedback.

---

## 2. Métrica principal: retorno normalizado (`normalized`)

### Definición

```
normalized = (return − aleatorio) / (oráculo − aleatorio)
```

- **`return`:** suma de los ratings obtenidos en los pasos controlados.
- **`aleatorio`** (`random_expected`): retorno *esperado* de recomendar al azar. Es el rating medio de los ítems disponibles al empezar la parte controlada, multiplicado por la cantidad de pasos controlados.
- **`oráculo`** (`oracle`): retorno de elegir los mejores ítems del usuario. Es la suma de los ratings más altos entre los disponibles, tantos como pasos controlados. El oráculo conoce todos los ratings ocultos, así que es una cota superior que ninguna política real puede superar.

Los dos extremos se calculan **por usuario**, sobre los ítems que quedan disponibles después del warm-up.

### Cómo se interpreta

| Valor | Significado |
|---|---|
| 1 | igual que el oráculo: eligió los mejores ítems posibles |
| 0 | igual que el azar |
| entre 0 y 1 | fracción de la brecha entre el azar y el óptimo que la política logró cubrir |
| negativo | peor que el azar: eligió ítems peores que el promedio del usuario |

Un valor negativo no es un error. Por ejemplo, `popularity` con logs `random` y el DT cuando se le pide un retorno bajo dan valores negativos.

**Ejemplo** (una sesión real de test, DT con R̂ pedido = 46, usuario del grupo 0, warm-up 0):
- aleatorio = 75,8; oráculo = 100 (20 ítems con rating 5); `return` = 52;
- normalized = (52 − 75,8) / (100 − 75,8) ≈ −0,98.

El DT eligió ítems malos a propósito, porque se le pidió un retorno bajo.

### Por qué es la métrica principal

El retorno crudo depende mucho del usuario. Un usuario del grupo 4 (rating medio 4,16) da retornos altos con casi cualquier política, y uno del grupo 7 (media 2,12) da retornos bajos aun con la mejor. El normalizado descuenta esa escala: compara a cada usuario con su propio azar y su propio óptimo. Por eso se puede promediar entre usuarios, grupos y condiciones (con o sin warm-up).

---

## 3. Métricas de sesión complementarias

| Columna | Definición | Uso |
|---|---|---|
| `return` | suma de ratings en los pasos controlados | base del normalizado; no comparable entre grupos |
| `full_return` | suma de ratings de los **20** pasos, incluido el warm-up | return conditioning del DT: el retorno pedido R̂ se refiere a la sesión completa |
| `mean_rating` | `return` / pasos controlados | lectura intuitiva del rating promedio obtenido |
| `hit_rate` | fracción de recomendaciones controladas con rating ≥ 4 | cuántas recomendaciones "gustaron" |

`hit_rate` usa un umbral binario: un 4 y un 5 cuentan igual, y un 1 y un 3 también. Por eso puede ordenar las políticas distinto que el normalizado. Una política con más 5 y menos 1, pero menos ítems ≥ 4, puede tener mejor normalizado y peor hit rate. Esto pasa con IQL π frente a `item_mean` (`reporte.md`, §5).

---

## 4. Métricas de ranking del primer paso

### Definición

Se toman los puntajes que da la política en el **primer paso controlado**, antes de que se revele ningún rating de la parte controlada. Se ordenan los ítems disponibles y se miran los 10 primeros. Un ítem es **relevante** si su rating oculto es ≥ 4.

- **`precision@10`:** fracción de relevantes entre los 10 primeros.
- **`ndcg@10`:** como la precisión, pero premia que los relevantes estén más arriba:

  ```
  DCG@10  = Σ_{i=1..10} rel_i / log2(i + 1)
  IDCG@10 = DCG@10 del ranking ideal (todos los relevantes primero)
  NDCG@10 = DCG@10 / IDCG@10   (entre 0 y 1)
  ```

### Cómo se interpreta

Es una evaluación de **ranking estático**: mide qué tan bueno es el ranking inicial con la información que hay al empezar (el grupo si es visible y, con warm-up, los 5 ratings iniciales). No mide la capacidad de adaptarse durante la sesión, que sí captura el normalizado.

En el reporte se usa `ndcg@10` como métrica secundaria. `precision@10` se calcula, pero no se reporta.

---

## 5. Cómo se agregan: medias e incertidumbre

### Por usuario y por evaluación (lo hace el protocolo)

Cada usuario de test (1.600, 200 por grupo) se evalúa con **3 semillas de sesión**. La semilla cambia el desempate entre ítems con igual puntaje y la elección de los ítems del warm-up.

`metricas.summarize`:
1. promedia las 3 semillas de cada usuario;
2. informa la **media entre usuarios** y el **IC 95 %**:

```
IC 95 % = ± 1,96 · desvío entre usuarios / √(cantidad de usuarios)
```

El tamaño de muestra es la cantidad de **usuarios**, no de sesiones, porque las sesiones de un mismo usuario no son independientes.

### Entre entrenamientos (lo agregamos nosotros)

Cada modelo final se entrenó con **3 semillas de entrenamiento**. En las tablas, los modelos se reportan como:

```
media entre las 3 semillas de entrenamiento ± desvío entre esas semillas   [IC 95 % por usuario]
```

Las dos medidas responden preguntas distintas:

| Medida | Qué variabilidad mide | Pregunta que responde |
|---|---|---|
| Desvío entre semillas de entrenamiento | del **entrenamiento** | ¿otro entrenamiento del mismo modelo daría lo mismo? |
| IC 95 % por usuario | de la **evaluación** | ¿con otros usuarios de test daría lo mismo? |

Las líneas base no se entrenan con descenso por gradiente, así que solo tienen IC.

### Reglas de lectura

- **Una diferencia menor que el desvío entre semillas no es una mejora** (consigna, §3). Por ejemplo, IQL π y `group_mean` con logs `mixto` se tratan como empate.
- **Por grupo**, cada grupo tiene 200 usuarios y el IC es más ancho (±0,008 a ±0,017 en normalizado). Diferencias de ≈ 0,02 entre grupos no son concluyentes.
- **Validación frente a test:** en validación, el catálogo de cada sesión se restringe a los ítems que ese usuario valoró en train. Eso cambia el aleatorio y el oráculo de cada usuario, así que los normalizados de validación **no se comparan** con los de test. Por ejemplo, el DT elegido da 0,566 de promedio en validación y entre 0,68 y 0,70 en test. Validación solo sirve para elegir hiperparámetros.

---

## 6. Qué no se usa como métrica

- **"Acertar el siguiente ID":** el orden de los ítems en los JSON de test es el del catálogo, no un orden de consumo. Predecir el próximo ID no mide nada (consigna, §3).
- **Las pérdidas de entrenamiento:** una cross-entropy que baja o una L_π que sube no dicen nada sobre la calidad de recomendación (rúbrica). Los hiperparámetros se eligen con el normalizado en validación, nunca con las pérdidas.

---

## 7. Dónde aparece cada métrica

| Métrica | Dónde |
|---|---|
| `normalized` | `resultados.md`: Tablas 1, 1c, 3 y 4; fila "Normalizado" de la Tabla 2 |
| `hit_rate`, `ndcg@10` | `resultados.md`, Tabla 1b |
| `full_return` | `resultados.md`, Tablas 2 y 2b (return conditioning) |
| datos por sesión (todas las columnas) | Drive: `netflix_<logs>/final/test/seed<k>/sesiones.csv` |
| resúmenes con IC | Drive: `final/test/*/resumen.csv` y `resumen_por_grupo.csv` |
