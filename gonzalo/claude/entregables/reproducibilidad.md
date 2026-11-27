# Reproducibilidad — TP Decision Transformer e IQL

Complemento de `reporte.md`: cómo se ejecutaron los experimentos, con qué código, semillas y versiones, y cuánto tardó cada etapa.

**Código.**
- Protocolo común (`protocolo/`), usado sin modificar.
- Código propio en `modelos/`: `comun.py`, `dt.py`, `iql.py`, `entrenar.py`, `evaluar_lotes.py`, `evaluar_modelo.py`, `experimentos.py` y `reporte.py`.
- Tests en `modelos/tests` (no fuga del DT, alineación entre política y entrenamiento, expectil, gradientes separados, paso terminal y equivalencia del evaluador por lotes). Junto con `protocolo/tests` suman 53 tests, todos aprobados en Colab.

**Qué se entrega.**
- La notebook ejecutada, con la salida de cada celda.
- `modelos.zip`, el mismo archivo con el que se generaron los resultados. Se genera con `python -m modelos.empaquetar`, su raíz es `modelos/` y no incluye datos ni checkpoints.

**Cómo reproducir.**
1. Abrir la notebook en Google Colab con un entorno de ejecución con GPU.
2. Importar `modelos.zip` en los archivos de la notebook.
3. Definir la carpeta de Google Drive para los resultados.
4. Ejecutar las celdas en orden.

La celda de preparación hace tres cosas:
- Clona el repositorio del TP.
- Descomprime el ZIP en `2026/tp`, junto a `protocolo/`.
- Corre los tests.

Los resultados se escriben en Drive (variable `TP_OUTPUTS`). Si la sesión se corta, volver a ejecutar la misma celda retoma desde los checkpoints y evaluaciones ya guardados. Por eso, si se cambia el ZIP después de las corridas, los resultados de Drive no se recalculan solos; cualquier cambio de código posterior a una corrida se registra en la bitácora.

**Celdas y etapas.**

| Celda | Etapa | Partición | Secciones del reporte |
|---|---|---|---|
| Preparación | repositorio, ZIP, Drive, GPU, tests | — | este documento |
| 2 | barridos del DT y de IQL, logs `random` | val | §4.2, §6 |
| 3 | barrido del DT, logs `mixto` | val | §7 |
| 4 | hiperparámetros elegidos | val | §4.2, §6, §7 |
| 5 | corridas finales, logs `random`, 3 semillas | test | §5, §6 |
| 6 | corridas finales, logs `mixto`, 3 semillas | test | §7 |
| 7 | tablas, return conditioning, comparación de logs, curvas | — | §5–§7 |

Cada celda ejecuta los módulos de `modelos/` por línea de comandos (`python -m modelos.experimentos …`, `python -m modelos.reporte …`). Su salida queda en la notebook y en `ejecucion.log`, en Drive. Para reproducir sin Colab, los mismos comandos se ejecutan en el mismo orden desde una terminal en `2026/tp`, como se indica en `modelos/README.md`, incluido en el ZIP.

**Evaluación por lotes.** DT e IQL se evalúan con un evaluador por lotes (`evaluar_lotes.py`) que reproduce `protocolo.evaluar.evaluate`. Usa las mismas sesiones, los mismos generadores aleatorios por sesión y las mismas funciones de métricas, pero calcula los puntajes de todas las sesiones en una sola llamada al modelo por paso. Los tests verifican la equivalencia:
- con las líneas base, el resultado es exacto;
- con DT e IQL coinciden las recomendaciones y los retornos, y el NDCG es igual salvo redondeo de float32.

**Semillas.**

| Uso | Semilla |
|---|---|
| Partición train/val (`split_by_user`) | 0 |
| Generación de logs `mixto`: train / val | 0 / 1 |
| Entrenamiento (finales) | 0, 1, 2 (la 0 se reutiliza del barrido) |
| Sesiones de evaluación en test | 0, 1, 2 |
| Sesiones de evaluación en val (selección) | 0 |

**Artefactos** (en `TP_OUTPUTS/netflix_<logs>/`):
- por corrida, `config.json` (argumentos, versiones, soporte de retornos de train), `curvas.csv`, `curvas_pasos.csv`, `checkpoint.pt` y `ultimo.pt`;
- `barridos/` con la selección en val;
- `final/` con las evaluaciones en test y las tablas.

**Versiones** (registradas en el `config.json` de cada corrida): Python 3.13.16, torch 2.14.1+cu130 (CUDA 13.0), numpy 2.4.6 y pandas 2.2.3, en Google Colab con GPU NVIDIA Tesla T4 (15 GB, driver 580.82.07).

**Tiempos** (de `ejecucion.log` y las marcas de tiempo de los archivos; corrida del 24/11/2026):

| Etapa | Duración aprox. |
|---|---:|
| Barrido del DT, logs `random` (1 entrenamiento + 36 evaluaciones en val) | 3 min |
| Barrido de IQL, logs `random` (8 entrenamientos + 64 evaluaciones en val) | 10 min |
| Barrido del DT, logs `mixto` (incluye generar los logs `mixto`) | 6 min |
| Tests | 0,5 min |
| Final `random` (6 entrenamientos nuevos, líneas base y 3 × 52 evaluaciones en test: 13 políticas × 4 condiciones) | 29 min |
| Final `mixto` (8 entrenamientos nuevos, líneas base y 3 × 20 evaluaciones en test) | 20 min |
| **Total** | **≈ 70 min** |

Cada entrenamiento tarda ≈ 2 min (DT) o ≈ 1 min (IQL). Cada evaluación de un modelo en una condición tarda ≈ 2–3 s con el evaluador por lotes.

**Uso de test.** Test se evaluó una sola vez por política, en las corridas `final`, sin `--force-eval`. La corrida `final` con logs `mixto` se lanzó dos veces; la segunda encontró las evaluaciones ya hechas ("Ya evaluado") y solo regeneró la tabla.
