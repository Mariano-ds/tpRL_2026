# Decision Transformer e IQL para recomendaciones — Reporte del TP

Diplomatura en Ciencia de Datos 2026 · Optativa Aprendizaje por Refuerzo
Autor: Gonzalo Perez Paina · Dataset: **Netflix** (752 ítems, 8 grupos)

> **Nota de edición.** Este borrador se armó a partir de tres fuentes:
> - las exportaciones de los notebooks `gonzalo_1_exploracion.md` (exploración) y `gonzalo_3_claude_agent.md` (entrenamiento y evaluación);
> - el código de `modelos/dt.py` y `modelos/iql.py`.
>
> Los datos que dependen de archivos no disponibles (`modelos/entrenar.py`, `modelos/comun.py` y las figuras) quedan marcados como **[COMPLETAR]**. Lo que se infiere sin confirmación queda como **[VERIFICAR]**.

---

## 0. Resumen

Se entrenaron un Decision Transformer (DT) y un Implicit Q-Learning (IQL) con los logs de Netflix y se evaluaron con el protocolo común de sesiones simuladas de 20 pasos, en cuatro condiciones (grupo visible u oculto, con warm-up 0 o 5). Cada modelo final se entrenó con 3 semillas.

- **Logs originales (aleatorios).** Ningún modelo supera a `group_mean` ni a `group_posterior` (0,742 de normalizado con grupo visible).
  - El DT queda a la altura de `item_mean` (0,683 frente a 0,689) y casi no distingue entre grupo visible y oculto: no personaliza.
  - IQL sí personaliza. Su mejor variante es el **máximo de Q con γ = 0** (0,728 visible), que supera a π (0,703).
- **Return conditioning.** El DT sigue al retorno pedido casi sobre la diagonal hasta unos 80. Después se satura en torno a 88, aunque se le pidan valores fuera del soporte de train.
- **Logs `mixto`.** Los tres modelos mejoran con grupo visible (DT 0,729; IQL π 0,737) y superan con amplitud al sistema que generó los datos (0,593).
  - `popularity` pasa de −0,06 a 0,69 y `item_mean` baja de 0,689 a 0,672, por sesgo de selección.
  - El máximo de Q con γ = 0,9 colapsa con grupo oculto (0,580 ± 0,030), un síntoma típico de error de extrapolación.
- **γ = 0 frente a γ = 0,9.** γ > 0 no aporta en ninguna condición: en este generador no hay valor secuencial que aprender.

---

## 1. Datos y formulación del problema

### 1.1 Datos

- **Train:** 16.000 usuarios sintéticos (2.000 por grupo). Cada uno tiene entre 25 y 200 ítems valorados (ratings enteros de 1 a 5), en el orden registrado. Los logs originales provienen de una **política de registro uniforme**.
- **Test:** 1.600 usuarios nuevos (200 por grupo), con rating para los 752 ítems. El orden de los IDs es el del catálogo, **no un orden de consumo**, así que predecir "el siguiente ID" no mide nada.
- **Grupos:** μ y σ por grupo e ítem del generador (`groups/*.csv`). Se usaron solo para análisis, nunca dentro de una política.
- **Particiones:** `split_by_user` separa 90 % de los usuarios de train para ajustar y 10 % para validación, por usuario, estratificado por grupo y con semilla 0. Test se usó una sola vez, para la tabla final.

### 1.2 Exploración (logs originales, usuarios de train después de la partición)

**La política de registro no informa sobre la calidad.**
- El rating medio es prácticamente constante a lo largo de la trayectoria: 3,226 en las posiciones 0–9, 3,215 en 10–24, 3,216 en 25–49, 3,217 en 50–99 y 3,209 en 100–199. No hay tendencia temporal ni efectos de orden.
- La correlación entre la frecuencia de un ítem y su rating medio es **−0,04**: aparecer mucho no indica ser bueno. Las dos cosas son lo esperable con una política de registro uniforme.
- Consecuencia anticipada: con estos logs, `popularity` no debería superar al azar, y la frecuencia de un ítem no le da al DT ninguna pista de calidad.

**Distribución de ratings.**
- Global: media 3,215, desvío 1,337, mediana 3. Proporciones de 1 a 5: 13,9 %, 18,0 %, 22,2 %, 24,6 % y 21,3 %.
- Por grupo, la media va de **2,12** (grupo 7) a **4,16** (grupo 4):

  | Grupo | 4 | 0 | 1 | 6 | 2 | 5 | 3 | 7 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | Media | 4,16 | 3,78 | 3,39 | 3,34 | 3,12 | 3,01 | 2,84 | 2,12 |
  | Desvío | 0,99 | 1,16 | 1,27 | 1,27 | 1,22 | 1,29 | 1,28 | 1,15 |
  | % de ratings ≥ 4 | 78 % | 65 % | 52 % | 49 % | 40 % | 38 % | 33 % | 14 % |

  Los grupos 4 y 0 son "generosos", el grupo 7 es muy exigente (el 38 % de sus ratings son 1), y los grupos 1, 6, 2 y 5 quedan cerca de la distribución global. Los desvíos por grupo (≈ 1–1,3) hacen que las distribuciones se solapen: los grupos se distinguen por tendencia, no de forma tajante.
- Consecuencia: el retorno crudo de una sesión depende mucho del grupo, lo que justifica la métrica normalizada. Además, el retorno alcanzable difiere entre grupos, algo importante para el return conditioning (sección 6).

**Ítems por usuario.**
- Media 112, mediana 112, rango 25–200 (5 % de los usuarios con 33 o menos y 5 % con 192 o más).
- La distribución es casi uniforme e igual en todos los grupos (medianas de 109 a 114).
- No hay longitudes tope, y el 10 % de usuarios con más ítems aporta solo el 17,1 % de las interacciones: ningún subconjunto de usuarios domina los datos.
- Con episodios de 20 pasos, un usuario aporta entre 1 y 10 episodios (≈ 5 en promedio), y los grupos quedan representados de forma pareja.

**Diferencias de gustos entre grupos.** Se calculó la correlación entre los ratings medios por ítem de cada par de grupos, usando solo las medias con al menos 20 ratings, y la diferencia media de escala entre pares. [COMPLETAR con los valores de la figura `gonzalo_1_exploracion_28_0.png`: el rango de correlaciones y los bloques de grupos parecidos.] Los dos mapas separan dos efectos:
- **Gustos:** qué ítems prefiere cada grupo, que es lo que vuelve útil personalizar.
- **Escala:** cuán generoso es cada grupo, que no cambia el orden de los ítems pero sí el retorno crudo.

**Conclusiones que se usan después:**
1. Con logs uniformes, la frecuencia de un ítem no informa sobre su calidad, y las acciones registradas no dependen del grupo.
2. El rating depende solo del grupo y del ítem, no del orden ni del historial. El único valor secuencial es la información que se gana sobre el grupo.
3. Los grupos difieren en escala (de 2,1 a 4,2 de media) y en gustos. Conocer el grupo es saber casi todo lo que se puede saber del usuario.

### 1.3 Episodios y retorno pendiente

Las trayectorias de cada usuario se cortan en **episodios de 20 pasos consecutivos** (`make_episodes`). Los ítems sobrantes se descartan. El retorno pendiente R̂_t = r_t + … + r_19 se calcula **dentro del episodio**, así que su escala (20–100) coincide con la de una sesión evaluada.

El retorno medio de un episodio de train es 64,3 con logs `random` y 74,6 con `mixto` (valores del protocolo).

### 1.4 MDP

| Elemento | Definición |
|---|---|
| Estado s_t | grupo (o "desconocido"), ítems mostrados y ratings revelados hasta t−1, paso t |
| Acción a_t | ítem del catálogo todavía no mostrado en la sesión |
| Recompensa r_t | rating de a_t, revelado después de mostrarlo |
| Transición | se agrega (a_t, r_t) al historial |
| Horizonte | 20 pasos; el último es terminal |
| Descuento γ | DT: no usa γ. IQL: γ ∈ {0; 0,9}. El retorno evaluado es la suma sin descontar |

El estado no es el usuario, que no se observa. Es lo que la política sabe de él en ese momento.

Qué necesita cada modelo:
- **DT:** la secuencia (R̂, s, a) de los pasos previos.
- **IQL:** un codificador del historial que devuelve T+1 estados, porque el objetivo del paso t usa V(s_{t+1}).

---

## 2. Protocolo de evaluación y líneas base

Se usó sin cambios el protocolo de `protocolo/`. En cada uno de los 20 pasos, la política devuelve un puntaje por ítem. El protocolo descarta lo ya mostrado, elige el ítem de mayor puntaje (con desempate al azar y semilla fija) y revela recién entonces su rating. La política nunca accede a los ratings ocultos.

**Condiciones:**

| Condición | Qué sabe la política al empezar | Qué mide |
|---|---|---|
| Grupo visible, warm-up 0 | el grupo | personalización por segmento |
| Grupo oculto, warm-up 0 | nada (cold-start estricto) | adaptación solo con el feedback de la sesión |
| Warm-up 5 (visible u oculto) | 5 ítems al azar con sus ratings | uso de un historial inicial; no es cold-start |

Con warm-up, las métricas cuentan solo los 15 pasos que decide la política.

**Métricas:**
- **Principal:** retorno normalizado entre el azar (0) y el oráculo (1).
- **Secundarias:** hit rate (fracción de ratings ≥ 4) y NDCG@10 del primer paso (ranking estático).

**Incertidumbre.** Cada usuario se evalúa con 3 semillas de sesión, y el IC 95 % se calcula con n = usuarios. Aparte se reporta la **dispersión entre 3 semillas de entrenamiento**. Las dos se muestran por separado.

**Validación.** Sirvió para elegir el retorno pedido al DT y τ, β y γ de IQL. El catálogo de cada sesión de validación se restringe a los ítems de ese usuario, así que sus números no se comparan con los de test.

**Líneas base:**

| Línea base | Qué hace | Por qué se incluye |
|---|---|---|
| `random` | elige al azar; es la política de registro | define el 0 del normalizado |
| `popularity` | ordena por frecuencia en los logs | mide si alcanza con imitar lo que más se mostró |
| `item_mean` | ordena por rating medio del ítem | el mejor ranking igual para todos |
| `group_mean` | rating medio en el grupo, si es visible | personalización por segmento |
| `group_posterior` | infiere el grupo con los ratings observados | adaptación explícita durante la sesión |
| `sistema_anterior_eps0` | solo con logs `mixto`: el sistema que generó los logs, sin exploración | referencia de la política de registro |

---

## 3. Decision Transformer

### 3.1 Representación y alineación

Implementación en `modelos/dt.py`: un DT clásico con tokens intercalados.

**Tokens del paso t** (de dimensión d_model = 128):

| Token | Representación |
|---|---|
| R̂_t | R̂_t / 100 → capa lineal (1 → d) |
| s_t | embedding del grupo (9 entradas: 8 grupos + "desconocido") + capa lineal sobre [rating del paso anterior centrado como (r − 3)/2, indicador de que existe rating previo] |
| a_t | embedding del ítem (n_items + 1 entradas: la última es el relleno del a_t que todavía no existe al recomendar) |

- A cada token se le suma un embedding del paso t (el mismo para los tres tokens del paso) y uno del tipo de token. Después vienen LayerNorm y dropout.
- **Qué va en s_t y por qué.** El estado es mínimo: el grupo y el rating que se acaba de revelar. El resto del historial (los ítems previos) el transformer lo ve en los tokens a_<t. s_t nunca incluye r_t. En t = 0, el rating previo y su indicador valen 0.
- **Transformer:** `nn.TransformerEncoder` con 3 capas, 4 cabezas, feed-forward de 4·d, GELU, dropout de 0,1 y pre-normalización (`norm_first=True`).
- **Máscara causal (3T × 3T):** es triangular superior. Cada token ve solo a los anteriores.
- **Alineación:** las logits de a_t se leen de la salida de los tokens s_t (posiciones 1, 4, 7, …), que vieron R̂_≤t, s_≤t y a_<t, pero no a_t.
- **Salida:** las logits son h_s · Eᵀ + b, con **pesos compartidos** con la tabla de embeddings de ítems (sin el relleno) y un sesgo por ítem.
- **Ítems ya mostrados:** `seen_before_mask` pone −∞ en las logits de los ítems a_<t, al entrenar y al recomendar.
- **Padding:** no hay. Los episodios de train tienen siempre T = 20. Al recomendar en el paso t, la secuencia tiene largo t + 1 y la máscara causal la vuelve equivalente a un prefijo.
- **Pérdida:** cross-entropy del ítem registrado en los 20 pasos (`dt_loss`).
- **No fuga:** un test altera a_t, R̂ y el estado de los pasos ≥ t y verifica que las logits del paso t no cambian. La batería completa pasa: 53 tests de `modelos/tests` y `protocolo/tests`.
- **Grupo oculto:** el token "desconocido" existe en el embedding de grupos. [COMPLETAR desde `modelos/entrenar.py`: qué fracción de episodios de train se entrena con el grupo reemplazado por "desconocido".]

### 3.2 Entrenamiento

- AdamW con scheduler y recorte de gradiente [COMPLETAR: lr, batch size, scheduler].
- 10 épocas de 289 pasos cada una.
- Checkpoint con la menor cross-entropy de validación.

La cross-entropy de validación baja de 6,613 a 6,588, apenas por debajo de la de una predicción uniforme sobre el catálogo: log(752) ≈ 6,62. Es lo esperable con logs uniformes. El ítem registrado es casi impredecible, y lo que el modelo puede aprender es la relación débil entre el ítem y el retorno que siguió. La pérdida mide imitación de los logs, no calidad de recomendación.

### 3.3 Política

`DTPolicy` arma en cada paso t una secuencia de largo t + 1:
- R̂_k = objetivo − suma de los ratings revelados antes de k, incluidos los del warm-up (el objetivo se refiere a la sesión completa de 20 pasos);
- los ítems ya mostrados, más un relleno en la posición t;
- los ratings previos como estado;
- el grupo, o "desconocido" si está oculto.

Las logits del paso t son los puntajes que recibe el protocolo. Si la sesión supera el horizonte del modelo, la política lanza un error.

---

## 4. IQL

### 4.1 Diseño

Implementación en `modelos/iql.py`.

**Codificador del historial (`HistoryEncoder`, d = 128):**
- Cada par previo (ítem, rating) se representa como embedding del ítem + capa lineal del rating centrado como (r − 3)/2.
- La secuencia arranca con un vector de inicio aprendido, que representa el historial vacío. A todas las posiciones se les suma el embedding del grupo (o "desconocido").
- Una **GRU** recorre la secuencia. A su salida se le suman otra vez el grupo y un embedding del paso, y pasa por LayerNorm → lineal → GELU.
- Con k pares, devuelve **k + 1 estados** h_0 … h_k, donde h_t resume los pasos < t. Es causal por construcción, y h_{t+1} se usa como s' en el objetivo de Q.

**Tres redes independientes.** Q, V y π tienen **cada una su propio codificador** y una cabeza MLP (lineal → GELU → lineal):
- Q y π dan una salida por ítem (n_items);
- V da una salida escalar.

Así, cada pérdida entrena solo su red, incluido su codificador. La red objetivo Q̄ es una copia de Q sin gradiente.

Las tres pérdidas, con u = Q̄(s, a) − V(s):

```
L_V = E[ |τ − 1(u < 0)| · u² ]
L_Q = E[ (r + γ · (1 − terminal) · V(s') − Q(s, a))² ]
L_π = E[ min(exp(β · (Q̄(s, a) − V(s))), w_max) · (−log π(a | s)) ]
```

- **Recompensa:** está centrada y escalada como r' = (r − 3)/2, en el rango [−1, 1]. No cambia qué acción es mejor, pero fija la escala de las ventajas y, con ella, el efecto de β. Concuerda con el Q medio ≈ 0,107 observado con γ = 0, porque (3,215 − 3)/2 ≈ 0,107.
- **Último paso:** es terminal (`not_done[-1] = 0`), con objetivo TD = r', sin bootstrapping.
- **Red objetivo:** Q̄ se actualiza con promedio de Polyak (`lerp_`) [COMPLETAR desde `entrenar.py`: valor de λ y frecuencia]. Q̄ se usa para el objetivo de V y para las ventajas de AWR.
- **Gradientes separados:**
  - Q̄(s, a) y los pesos de AWR se calculan bajo `no_grad`;
  - V(s') entra en L_Q con `detach()`;
  - como las redes no comparten parámetros, L_V solo mueve a V, L_Q solo a Q y L_π solo a π.
- **Pesos de AWR:** min(exp(β·u), w_max), con w_max = 100 por defecto [VERIFICAR que no se cambió en `entrenar.py`].
- **Diagnósticos registrados:** NLL de behavior cloning, media y desvío de la ventaja, peso medio, fracción truncada, V medio y Q medio.
- **Ítems ya mostrados:** se enmascaran en las logits de π al entrenar. Al recomendar, el protocolo los descarta tanto para π como para el máximo de Q.
- **Dos políticas (`IQLPolicy`):** `mode="pi"` ordena por las logits de π en el estado actual h_t, y `mode="q"` ordena por Q(s_t, ·) sobre todo el catálogo disponible.

**Diferencia de diseño con el DT.** IQL resume el historial completo de pares (ítem, rating) en un estado recurrente, con el grupo sumado en todas las posiciones. El DT, en cambio, ve el grupo solo en los tokens s_t y los ítems previos en los tokens a.

### 4.2 Barrido en validación (semilla 0)

Se barrieron γ ∈ {0; 0,9} × τ ∈ {0,7; 0,9} × β ∈ {3; 10}. Cada configuración se evaluó con π y con el máximo de Q. La tabla muestra el promedio de las cuatro condiciones:

| γ | τ | β | π | máx. Q |
|---:|---:|---:|---:|---:|
| 0 | 0,7 | 3 | 0,606 | 0,617 |
| **0** | **0,7** | **10** | **0,606** | 0,614 |
| 0 | 0,9 | 3 | 0,605 | 0,617 |
| 0 | 0,9 | 10 | 0,604 | 0,616 |
| 0,9 | 0,7 | 3 | 0,605 | 0,604 |
| 0,9 | 0,7 | 10 | 0,606 | 0,601 |
| 0,9 | 0,9 | 3 | 0,605 | 0,601 |
| 0,9 | 0,9 | 10 | 0,604 | 0,601 |

- **Configuración elegida:** γ = 0, τ = 0,7, β = 10, la mejor en π. Para la comparación de γ se mantuvo también la mejor con γ = 0,9 (τ = 0,7, β = 10).
- **τ y β casi no influyen:** el rango de π es de 0,604 a 0,606, menor que la dispersión entre semillas observada después (≈ 0,005). La elección entre ellos es prácticamente un empate.
- **Pesos de AWR:** con β = 3, el peso medio es ≈ 1,2 y no hay truncados, así que π es casi behavior cloning. Con β = 10, el peso medio es ≈ 9,4 y alrededor del 4 % se trunca: π es más selectiva sin quedar dominada por el truncado.
- **Las pérdidas no sirvieron para elegir:** L_π sube durante el entrenamiento (por ejemplo, de 8,6 a 48 con β = 10) porque los pesos se vuelven más selectivos. Las curvas están en el Anexo A.

---

## 5. Resultados principales (test)

**Tabla 1. Logs `random`, normalizado.** Los modelos se reportan como media ± desvío entre 3 semillas de entrenamiento. El IC 95 % por usuario es ±0,005–0,006 en todas las celdas.

| Política | Visible | Oculto | Warm-up 5, vis. | Warm-up 5, oc. |
|---|---:|---:|---:|---:|
| `random` | −0,002 | −0,002 | 0,003 | 0,003 |
| `popularity` | −0,058 | −0,058 | −0,117 | −0,117 |
| `item_mean` | 0,689 | 0,689 | 0,705 | 0,705 |
| `group_mean` | **0,742** | 0,689 | **0,749** | 0,705 |
| `group_posterior` | **0,742** | **0,707** | **0,749** | **0,729** |
| DT (R̂ = 117) | 0,683 ± 0,002 | 0,676 ± 0,011 | 0,700 ± 0,002 | 0,695 ± 0,002 |
| IQL π, γ = 0 | 0,703 ± 0,006 | 0,643 ± 0,007 | 0,708 ± 0,004 | 0,687 ± 0,003 |
| IQL máx. Q, γ = 0 | 0,728 ± 0,002 | 0,669 ± 0,010 | 0,736 ± 0,004 | 0,694 ± 0,007 |
| IQL π, γ = 0,9 | 0,702 ± 0,005 | 0,653 ± 0,009 | 0,705 ± 0,006 | 0,674 ± 0,005 |
| IQL máx. Q, γ = 0,9 | 0,700 ± 0,006 | 0,643 ± 0,018 | 0,711 ± 0,003 | 0,675 ± 0,014 |

Las métricas secundarias ordenan las políticas igual. Por ejemplo, con grupo visible, el hit rate es 0,886 para el DT, 0,914 para IQL máx. Q (γ = 0) y 0,918 para `group_mean`. Las tablas completas de hit rate y NDCG@10 están en el Anexo B.

**Lectura:**

- **Ningún modelo supera a la mejor línea base.** `group_posterior` domina las cuatro condiciones. Con logs uniformes, el rating medio por grupo e ítem es un estimador insesgado y casi exacto (los datos son sintéticos y por grupo), y esa línea base lo usa directamente.
- **El DT no personaliza con logs aleatorios.**
  - Rinde casi igual con grupo visible (0,683) y oculto (0,676), y apenas por debajo de `item_mean` (0,689).
  - Con logs uniformes, la acción registrada no depende del grupo. Lo único que relaciona acción, grupo y retorno es que los ítems buenos para ese grupo aparecen más en los episodios de retorno alto, y esa señal es débil.
  - El DT aprende algo parecido a un ranking global de ítems buenos.
- **IQL personaliza, sobre todo el máximo de Q.**
  - Con γ = 0, Q(s, a) es una regresión del rating esperado condicionada al estado, y el máximo de Q equivale a un `group_mean` aprendido: 0,728 frente a 0,742.
  - El máximo de Q supera a π porque, con logs uniformes, **todas las acciones están bien cubiertas** en todos los estados. El riesgo de extrapolación que motiva a IQL casi no aparece.
  - π, en cambio, está atada a la distribución de comportamiento por la pérdida de AWR (es una clasificación de acciones uniformes reponderadas), y eso la limita.
- **Con grupo oculto, ningún modelo supera a `item_mean`** (0,689) ni se acerca a `group_posterior` (0,707). No aprendieron a inferir el grupo a partir de pocos ratings dentro de la sesión.
  - Con warm-up 5 y grupo oculto pasa lo mismo: 0,687–0,695 frente a 0,705 de `item_mean` y 0,729 de `group_posterior`.
  - El warm-up mejora un poco a todas las políticas, pero los modelos lo aprovechan menos que la inferencia bayesiana explícita.
- **γ = 0 frente a γ = 0,9.** Con π, la diferencia es menor o igual a la dispersión entre semillas en todas las condiciones. Con el máximo de Q, γ = 0,9 es peor con grupo visible (0,700 frente a 0,728) y más inestable (desvío entre semillas de 0,018 con grupo oculto).
  - Interpretación: en este generador un ítem no cambia los ratings futuros. El único valor secuencial es la información que aporta sobre el grupo, y ninguna variante la explota.
  - γ > 0 solo agrega bootstrapping y su error.
- **π frente al máximo de Q.** Con logs uniformes, el máximo de Q es mejor o igual a π en todas las condiciones con γ = 0. La ventaja de π aparece recién con logs sesgados (sección 7).

**Resultados por grupo:** [COMPLETAR con `resumen_por_grupo.csv` o la figura por grupo, en la condición de grupo visible y warm-up 0.]

---

## 6. Return conditioning (DT, logs `random`)

**Diseño.** Se pidieron nueve retornos:
- los percentiles 10, 25, 50, 75, 90 y 99 de los retornos de los episodios de train: 46, 57, 65, 73, 81 y 89;
- el máximo de train, 97 [VERIFICAR la correspondencia exacta percentil–valor];
- dos valores fuera del soporte: 107 y 117.

El retorno pedido se eligió en validación con el promedio de las cuatro condiciones. Ganó R̂ = 117 (0,566), en empate práctico con 107 (0,566) y 97 (0,564). En test se evaluó el barrido completo con las 3 semillas. Con warm-up se compara `full_return` (los 20 pasos), porque el objetivo se refiere a la sesión completa.

**Tabla 2. Retorno obtenido frente a pedido (test, warm-up 0, media de 3 semillas).**

| Pedido | 46 | 57 | 65 | 73 | 81 | 89 | 97 (máx.) | 107 | 117 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Obtenido, visible | 47,9 | 57,5 | 65,0 | 72,8 | 80,1 | 86,0 | 88,1 | 88,2 | 88,2 |
| Obtenido, oculto | 49,4 | 58,2 | 65,2 | 72,2 | 79,0 | 84,6 | 87,4 | 87,9 | 88,0 |
| Normalizado, visible | −0,61 | −0,34 | −0,11 | 0,14 | 0,38 | 0,59 | 0,68 | 0,68 | 0,68 |

El desvío entre semillas del retorno obtenido es ≤ 0,5 puntos en todas las celdas.

Figuras: `return_conditioning.png` (agregado) y `return_conditioning_por_grupo.png`, con la diagonal y el máximo de train marcados. En la exportación aparecen como `gonzalo_3_claude_agent_files/gonzalo_3_claude_agent_23_13.png` y `…_23_14.png`.

**Interpretación:**

- **Sobre la diagonal hasta unos 80.** Entre 46 y 81, el retorno obtenido está a menos de 2 puntos del pedido. El condicionamiento funciona **en ambas direcciones**: si se pide 46, el modelo elige deliberadamente ítems malos y queda por debajo del azar (normalizado −0,61). Esto prueba que el DT sí aprendió qué ítems llevan a retornos bajos o altos, aunque no los use para personalizar por grupo.
- **Saturación desde unos 89.** Con 89 obtiene 85–86, y desde 97 se estanca en ≈ 88, que es el techo de lo que el modelo sabe hacer. Pedir 107 o 117, fuera del soporte, no empeora ni mejora: es extrapolación sin efecto. Elegir 117 en validación equivale, en la práctica, a "pedir el máximo".
- **El mejor retorno pedido no es el más alto en un sentido útil.** Por encima del máximo de train, el resultado es plano. El valor elegido está fuera del soporte, pero empata con el máximo de train.
- **Grupo oculto frente a visible.** Con retornos bajos, el modelo con grupo oculto queda algo por encima del pedido (49,4 frente a 46). Sin el grupo le cuesta encontrar ítems que sean malos para ese usuario en particular.
- **Por grupo:** [COMPLETAR con la figura `return_conditioning_por_grupo.png`.] Por la exploración (sección 1.2), un retorno pedido no significa lo mismo en todos los grupos. Con una media de 4,16, un retorno de 88 está al alcance del grupo 4. Para el grupo 7, que tiene una media de 2,12 y solo un 14 % de ratings ≥ 4, ese retorno es casi imposible: para él, pedir 88 es extrapolación aunque esté dentro del soporte global. Cabe esperar que los grupos generosos se saturen más arriba y que el grupo 7 se aparte antes de la diagonal.
- **Con logs `mixto`** (validación, Anexo C), el efecto del grupo con retornos bajos es mucho más fuerte. Por ejemplo, con R̂ = 57, el normalizado es −0,25 con grupo visible y 0,15 con grupo oculto. Con grupo visible, el modelo imita las sesiones malas del sistema anterior para ese grupo. Con grupo oculto, no puede apuntar a los ítems malos del usuario.

---

## 7. Calidad de los logs

**Setup.** DT e IQL se reentrenaron con los logs `mixto` (3 semillas), con los mismos hiperparámetros de IQL elegidos con logs `random`. El retorno pedido al DT se volvió a elegir en validación y resultó R̂ = 99, el máximo de train con estos logs. Las líneas base se ajustaron con los mismos logs `mixto`, y se agregó el sistema anterior con ε = 0.

**Tabla 3. Normalizado en test.** Los modelos se reportan como media ± desvío entre semillas, en las dos condiciones de warm-up 0.

| Política | `random` visible | `random` oculto | `mixto` visible | `mixto` oculto |
|---|---:|---:|---:|---:|
| `popularity` | −0,058 | −0,058 | 0,688 | 0,688 |
| `item_mean` | 0,689 | 0,689 | 0,672 | 0,672 |
| `group_mean` | 0,742 | 0,689 | 0,741 | 0,672 |
| `group_posterior` | 0,742 | 0,707 | 0,741 | 0,705 |
| `sistema_anterior_eps0` | — | — | 0,593 | 0,542 |
| DT | 0,683 ± 0,002 | 0,676 ± 0,011 | 0,729 ± 0,002 | 0,683 ± 0,006 |
| IQL π, γ = 0 | 0,703 ± 0,006 | 0,643 ± 0,007 | **0,737 ± 0,002** | 0,667 ± 0,010 |
| IQL máx. Q, γ = 0 | 0,728 ± 0,002 | 0,669 ± 0,010 | 0,730 ± 0,004 | 0,673 ± 0,007 |
| IQL π, γ = 0,9 | 0,702 ± 0,005 | 0,653 ± 0,009 | 0,735 ± 0,001 | 0,677 ± 0,002 |
| IQL máx. Q, γ = 0,9 | 0,700 ± 0,006 | 0,643 ± 0,018 | 0,706 ± 0,006 | **0,580 ± 0,030** |

La tabla completa, con warm-up 5 y métricas secundarias, está en el Anexo B.

**Interpretación:**

- **Las líneas base cambian por sesgo de selección.**
  - `popularity` pasa de rendir peor que el azar a 0,688: con logs sesgados, lo que más se mostró es lo que el sistema anterior consideraba bueno, y esa línea base hereda su conocimiento.
  - `item_mean` empeora (0,689 → 0,672): cada ítem se mostró sobre todo a los grupos que lo valoran bien, así que su media observada sobreestima lo que pasaría con un usuario cualquiera.
  - `group_mean` no cambia (0,741), porque condicionar por grupo elimina ese sesgo.
- **Diferencia entre datasets:** [COMPLETAR si se corre Goodreads. Según el protocolo, con logs `random`, `popularity` da −0,06 en Netflix y 0,04 en Goodreads, e `item_mean` 0,69 y 0,42. La diferencia viene de cuánto coinciden los gustos entre grupos en cada catálogo.]
- **El DT ahora personaliza.** Con grupo visible sube de 0,683 a 0,729 y se separa del caso oculto (0,683). En `mixto`, las sesiones de retorno alto son las del sistema anterior explotando, que elige según el grupo. Así, "imitar las trayectorias de retorno alto condicionado al grupo" pasa a significar "elegir lo bueno para ese grupo".
- **IQL π es el mejor modelo** (0,737, a 0,004 de `group_mean`, dentro del IC). La pérdida de AWR parte de una política de comportamiento que ya es buena y la mejora reponderando por ventaja.
- **Todos los modelos superan al sistema que generó los datos** (0,593 / 0,542). El DT y las variantes de π lo superan por 0,12 o más en ambas condiciones. La excepción es el máximo de Q con γ = 0,9 y grupo oculto, que lo supera por solo 0,04. Hay dos razones:
  1. Los modelos eligen de forma determinística el ítem de mayor puntaje, mientras que el sistema anterior **muestrea** con softmax(μ/τ) (con τ = 0,25) usando medias que excluyen al propio usuario.
  2. `mixto` mezcla sesiones de distinta calidad, incluidas aleatorias (ε = 1), que dan cobertura. Los métodos offline pueden "recombinar" lo mejor de cada sesión.
- **El máximo de Q con γ = 0,9 colapsa con grupo oculto** (0,580 ± 0,030, con la mayor dispersión entre semillas de la tabla). Con logs sesgados, cada ítem se mostró casi solo a los grupos a los que les gusta, así que la cobertura por estado es despareja. Sin el grupo, Q sobreestima ítems poco cubiertos, el máximo los elige y el bootstrapping con γ > 0 propaga el error. π, que se entrena solo con acciones registradas, no sufre este problema (0,677). Es el **error de extrapolación** que IQL evita al entrenar y que el máximo de Q reintroduce al recomendar.
- **Grupo oculto:** con logs `mixto`, ningún modelo supera a `group_posterior` (0,705). El DT (0,683) queda cerca de `popularity` (0,688).

---

## 8. Comparación DT vs. IQL

| | DT | IQL |
|---|---|---|
| Mecanismo | imita trayectorias condicionado por el retorno | estima valores y extrae una π con ventaja positiva dentro del soporte |
| Qué necesita de los logs | que las trayectorias de retorno alto difieran por sus acciones | cobertura de acciones por estado |
| Resultado observado | con logs `random` no personaliza (≈ `item_mean`); con `mixto` personaliza (0,729) | personaliza con ambos logs; su mejor variante depende de la cobertura (máx. Q con `random`, π con `mixto`) |

Con logs uniformes, IQL aprovecha mejor la información. Q aprende el rating esperado por estado y acción con todas las acciones cubiertas. En cambio, el DT solo puede explotar la correlación débil entre ítem y retorno, sin una atribución por acción.

Con logs de un sistema que ya personaliza, la diferencia se reduce (0,729 frente a 0,737). Las trayectorias de retorno alto ya contienen la decisión correcta, y el DT tiene algo concreto que imitar.

En las dos situaciones, el DT da un control explícito del retorno que ninguna variante de IQL ofrece, aunque saturado arriba. A su vez, π de IQL es la única opción robusta cuando la cobertura es despareja.

---

## 9. Límites y conclusiones

**Límites:**
- Los datos son sintéticos: ratings que dependen solo del grupo y del ítem, sin efectos de orden. Esto deja poco valor secuencial, lo que explica que γ > 0 no aporte.
- El simulador usa usuarios de test con ratings completos y no modela abandono.
- El catálogo de validación difiere del de test, así que la selección en validación es una aproximación.
- Hay solo 3 semillas de entrenamiento.
- Los hiperparámetros de IQL con `mixto` se tomaron del barrido con `random`.
- Solo se evaluó Netflix.

**Conclusiones:**
- Con logs uniformes, ninguno de los dos métodos supera a las líneas base por grupo. IQL personaliza más que el DT.
- El DT sigue al retorno pedido dentro del soporte y se satura arriba.
- Con logs de un sistema anterior, ambos superan con amplitud a ese sistema y quedan a la par de `group_mean`.
- Ninguno aprendió a inferir el grupo durante la sesión como lo hace `group_posterior`. Este es el principal margen de mejora.

---

## Anexos (fuera del límite de páginas)

### A. Reproducción

- **Entorno:** Google Colab con GPU Tesla T4, torch 2.14.1+cu130, numpy 2.4.6, pandas 2.2.3 [COMPLETAR la versión de Python].
- **Código:** `modelos/` (ZIP desde el repositorio del grupo), con `dt.py`, `iql.py`, `entrenar.py`, `experimentos.py`, `evaluar_modelo.py`, `evaluar_lotes.py`, `reporte.py` y `tests/`.
- **Tests:** `python -m pytest -q modelos/tests protocolo/tests` → 53 tests pasan. Cubren la no fuga del DT y la equivalencia del evaluador por lotes con el protocolo.
- **Comandos,** desde `2026/tp`:

  ```bash
  python -m modelos.experimentos barrido-dt  --dataset netflix --logs random
  python -m modelos.experimentos barrido-iql --dataset netflix --logs random
  python -m modelos.experimentos barrido-dt  --dataset netflix --logs mixto
  python -m modelos.experimentos final --dataset netflix --logs random
  python -m modelos.experimentos final --dataset netflix --logs mixto --hp-de random --no-rc
  python -m modelos.reporte tabla|rc|logs|curvas ...
  ```

- **Salidas:** `TP_OUTPUTS=/content/drive/MyDrive/tp_rl_2026/outputs/modelos`. Contiene los checkpoints, las curvas sin recortes (`curvas.png` en cada corrida), las evaluaciones (`sesiones.csv`, `resumen*.csv`) y las selecciones (`barridos/*/elegido.json`).
- **Semillas:** entrenamiento {0, 1, 2}; sesiones de evaluación {0, 1, 2}; partición train/validación con semilla 0.

### B. Tablas completas

[COMPLETAR: pegar las tablas de hit rate y NDCG@10 para logs `random` y `mixto`, generadas por `modelos.reporte tabla` (`final/tabla_*.md`).]

### C. Barrido del DT con logs `mixto` (validación)

| Pedido | 57 | 67 | 77 | 84 | 89 | 95 | **99** | 109 | 119 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Promedio de las 4 condiciones | −0,160 | 0,116 | 0,378 | 0,517 | 0,574 | 0,597 | **0,598** | 0,596 | 0,593 |

### D. Bitácora de decisiones y contribuciones

[COMPLETAR: integrantes, contribuciones de cada uno y decisiones verificadas, con fecha.]

### E. Uso de IA

Las celdas de acceso al repositorio y el código de los análisis de la exploración se elaboraron con Claude. [COMPLETAR: qué partes generó, qué se verificó (tests de no fuga, coincidencia de las líneas base con la tabla del protocolo, revisión de máscaras y alineación) y cómo.]

### F. Referencias

- Chen et al. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling.* NeurIPS.
- Kostrikov, Nair y Levine (2022). *Offline Reinforcement Learning with Implicit Q-Learning.* ICLR.
- Peng et al. (2019). *Advantage-Weighted Regression.*
- Levine et al. (2020). *Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems.*
