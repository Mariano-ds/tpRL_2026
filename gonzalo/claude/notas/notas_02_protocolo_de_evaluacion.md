# Notas del TP — 02. El protocolo de evaluación

## Pregunta

> Explica en qué consiste el protocolo de evaluación.

## Respuesta

El protocolo de evaluación es un conjunto fijo de reglas, implementado en `protocolo/`, para medir **qué tan bien recomienda una política**. Todas las políticas (líneas base, DT, IQL) se evalúan igual, así que sus números son comparables.

### 1. Por qué se simulan sesiones

En RL offline no se pueden probar recomendaciones nuevas con usuarios reales. Pero cada usuario de **test** tiene rating para **todos** los ítems del catálogo, así que se puede simular una sesión interactiva:

```
reset(grupo o None)
repetir 20 veces:
    la política ve lo observado hasta ahora  →  devuelve un puntaje por ítem
    el protocolo descarta los ítems ya mostrados y elige el de mayor puntaje
    el protocolo revela el rating de ese ítem  →  recompensa
```

- Como hay rating para cualquier ítem, la política puede elegir lo que quiera y siempre hay respuesta.
- Como solo se revela lo que se muestra, la política no puede "hacer trampa" mirando ratings futuros.

Esto reemplaza la evaluación ingenua de "acertar el siguiente ID", que no mide nada: el orden de los ítems en el JSON de test es el del catálogo, no un orden de consumo.

### 2. La interfaz `Policy`

Toda política implementa dos métodos (`protocolo/simulador.py`):

```python
reset(group: int | None, horizon: int) -> None        # inicio de sesión; None = grupo oculto
scores(items_seen, ratings_seen) -> np.ndarray        # un puntaje por ítem del catálogo
```

- `scores` recibe **copias** de lo observado. En el paso t, los arrays tienen largo t.
- La política no necesita excluir los ítems ya mostrados: el protocolo los enmascara (`choose` en `simulador.py`). Elige el ítem disponible de mayor puntaje y **desempata al azar** con semilla fija.
- Para un modelo neuronal, los puntajes suelen ser las logits (DT, π de IQL) o los valores Q.

### 3. Las tres condiciones

| Condición | Qué sabe la política al empezar | Qué mide |
|---|---|---|
| Grupo visible, warm-up 0 | el grupo del usuario | si personaliza por segmento |
| Grupo oculto, warm-up 0 | nada (**cold-start estricto**) | si se adapta usando solo los ratings que va viendo |
| Warm-up 5 (visible u oculto) | 5 ítems elegidos al azar con sus ratings | si aprovecha un historial inicial (**no** es cold-start) |

- Con warm-up, los 5 primeros ítems los elige la política de registro (uniforme). Las métricas cuentan solo los **15 pasos** que decide la política.
- En todos los casos el usuario de test no aparece en train. El modelo tiene que generalizar a partir del grupo o de lo que observa en la sesión.
- La función `evaluate_grid` cruza las condiciones: warm-up {0, 5} × grupo {visible, oculto}. Resultan las cuatro columnas de la tabla final del reporte.

### 4. Métricas

Se calculan por sesión, sobre los pasos controlados (`protocolo/metricas.py`):

- **`normalized` (métrica principal):**

  ```
  normalizado = (retorno − retorno esperado al azar) / (retorno del oráculo − retorno esperado al azar)
  ```

  - *Azar:* media de los ratings disponibles del usuario × número de pasos.
  - *Oráculo:* suma de los mejores ratings disponibles, como si se conocieran de antemano.
  - Ambos se calculan sobre los ítems disponibles al empezar la parte controlada (después del warm-up).
  - 0 equivale al azar y 1 al óptimo. Puede ser negativo. Es comparable entre usuarios, porque un usuario que pone 4 o 5 a todo daría retornos crudos altos con cualquier política.
  - Ejemplo: con azar = 64, oráculo = 100 y retorno = 85, el normalizado es (85 − 64)/(100 − 64) = 0,58.
- **`return`:** suma cruda de los ratings obtenidos.
- **`hit_rate`:** fracción de recomendaciones con rating ≥ 4.
- **`ndcg@10` y `precision@10`:** miran **solo el primer paso controlado**, o sea cómo ordena la política antes de observar nada de la sesión. La relevancia es binaria (rating ≥ 4). Son métricas de **ranking estático**, complementarias. La que mide la sesión es el retorno.

### 5. Semillas e intervalos de confianza

- Cada sesión tiene azar: los desempates y los ítems del warm-up. `evaluate` corre cada usuario con varias semillas (por defecto 0, 1, 2). El generador de cada sesión se inicializa con `[semilla, índice de usuario]`, así que es reproducible.
- `summarize` **primero promedia las semillas de cada usuario** y después calcula la media y el **IC 95 %** entre usuarios. El tamaño de muestra es la cantidad de usuarios, no de sesiones.
- Este IC mide la **variabilidad de la evaluación**. Es distinto de la **dispersión entre semillas de entrenamiento** (reentrenar el modelo con ≥ 3 semillas), que mide la variabilidad del entrenamiento y que el TP exige reportar por separado. Una mejora menor que la dispersión entre semillas no cuenta como mejora.

### 6. Particiones: validación vs. test

- `split_by_user`: 90 % de los usuarios de train para ajustar y 10 % para validación, **por usuario**, estratificado por grupo y con semilla 0.
- **Validación (`--split val`):** sirve para elegir hiperparámetros (retorno pedido al DT; τ, β y γ de IQL). Como esos usuarios solo tienen ratings de los ítems de su trayectoria, el catálogo de cada sesión **se restringe a esos ítems**.
- Por eso **los números de validación no se comparan con los de test**: el catálogo es distinto. Sí sirven para comparar configuraciones entre sí.
- **Test:** se usa **una sola vez**, al final, para la tabla final.

### 7. Líneas base

| Línea base | Qué hace | Para qué sirve |
|---|---|---|
| `random` | elige al azar (es la política de registro) | define el 0 del normalizado |
| `popularity` | ordena por cantidad de apariciones en los logs | con logs uniformes rinde como `random` |
| `item_mean` | rating medio de cada ítem en train (contraído hacia la media global) | el mejor ranking **igual para todos**; ya es fuerte |
| `group_mean` | rating medio en el grupo, si es visible; si no, igual a `item_mean` | personalización por segmento |
| `group_posterior` | infiere el grupo con los ratings observados (modelo gaussiano) y recomienda el mayor rating esperado | adaptación explícita durante la sesión |

Valores de referencia en test (3 semillas, horizonte 20, media de `normalized`):

| | Netflix visible | Netflix oculto | Goodreads visible | Goodreads oculto |
|---|---:|---:|---:|---:|
| `random` | 0,00 | 0,00 | 0,00 | 0,00 |
| `popularity` | −0,06 | −0,06 | 0,04 | 0,04 |
| `item_mean` | 0,69 | 0,69 | 0,42 | 0,42 |
| `group_mean` | 0,74 | 0,69 | 0,49 | 0,42 |
| `group_posterior` | 0,74 | 0,71 | 0,49 | 0,46 |

Si un modelo no supera a `item_mean`, no está personalizando, aunque su pérdida baje. Al correr `python -m protocolo.evaluar --dataset netflix` hay que obtener estos mismos números. Si no coinciden, hay un problema de instalación o de datos.

### 8. Variante con logs de un "sistema anterior"

El protocolo de evaluación **no cambia**. Lo que cambia es el **dataset de entrenamiento**:

- `protocolo/logs.py` regenera los episodios de train como si los hubiera elegido un recomendador previo. Ese sistema conoce el grupo, muestrea con softmax(media del ítem en el grupo / τ) y explora con probabilidad ε.
- No inventa ratings: usa solo los ratings reales de cada usuario.
- Presets:

  | Preset | Retorno medio por episodio |
  |---|---:|
  | `random` (los logs originales) | 64,3 |
  | `mixto` | 74,6 |
  | `experto` | 82,5 |

- Con `--logs mixto`, las líneas base **se ajustan con esos mismos logs**, y se agrega el sistema anterior con ε = 0 como referencia.
- Con logs sesgados:
  - `popularity` mejora mucho, porque hereda lo que sabía el sistema anterior.
  - `item_mean` empeora por **sesgo de selección**: cada ítem se mostró sobre todo a los grupos que lo valoran bien.

### 9. Salidas

`write_reports` escribe en la carpeta de `--output`:

- `sesiones.csv`: una fila por sesión (usuario × semilla × política × condición).
- `resumen.csv`: media e IC 95 % por política y condición.
- `resumen_por_grupo.csv`: lo mismo, desglosado por grupo. El TP pide reportar el agregado **y** por grupo.

### Fuentes

- [protocolo/PROTOCOLO.md](protocolo/PROTOCOLO.md)
- [protocolo/simulador.py](protocolo/simulador.py), [protocolo/metricas.py](protocolo/metricas.py), [protocolo/evaluar.py](protocolo/evaluar.py)
- [GUIA.md](GUIA.md), sección 3
- [CONSIGNA.md](CONSIGNA.md), sección 3
