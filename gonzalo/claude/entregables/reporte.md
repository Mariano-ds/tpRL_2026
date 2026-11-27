# Decision Transformer e IQL para recomendaciones — Reporte del TP

**Diplomatura en Ciencia de Datos 2026 · Optativa Aprendizaje por Refuerzo**
**Grupo:** Juliana Lescano, Mariano Baggini y Gonzalo Perez Paina
**Profesores:** Jorge Palombarini y Juan Barsce
Dataset: Netflix (752 ítems, 8 grupos)

---

## 0. Resumen

Se entrenaron un Decision Transformer (DT) y un Implicit Q-Learning (IQL) con los logs de Netflix y se evaluaron con el protocolo común de sesiones simuladas de 20 pasos, en cuatro condiciones (grupo visible u oculto, con warm-up 0 o 5). Cada modelo final se entrenó con 3 semillas.

- **Logs originales (aleatorios).** Ningún modelo supera a `group_mean` ni a `group_posterior`. El DT empata con `item_mean` y casi no distingue entre grupo visible y oculto: no personaliza. IQL sí personaliza, y su mejor variante es el **máximo de Q con γ = 0**, que supera a π.
- **Return conditioning.** El DT sigue al retorno pedido casi sobre la diagonal hasta unos 80 y después se satura, aunque se le pidan valores fuera del soporte de train. Cada grupo tiene su propio techo y, a veces, su propio piso.
- **Logs `mixto`.** Con grupo visible mejoran el DT e IQL π, que queda empatado con `group_mean`. Todas las variantes superan al sistema que generó los datos, algo que también logra `popularity`. El sesgo de selección cambia las líneas base, y el máximo de Q con γ = 0,9 colapsa con grupo oculto, un síntoma típico de error de extrapolación.
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

**Diferencias de gustos entre grupos.** Se calculó la correlación entre los ratings medios por ítem de cada par de grupos, usando solo las medias con al menos 20 ratings, y la diferencia media de escala entre pares. Con los usuarios de train después de la partición, los 752 ítems tienen al menos 20 ratings en cada grupo:
- **Correlaciones todas positivas pero moderadas:** de 0,33 a 0,81, con mediana 0,56. Los grupos coinciden en parte sobre qué ítems son buenos, pero no del todo.
- **Bloques de grupos parecidos:** {0, 1, 3} (0,71–0,81) y {0, 2, 4, 6} (0,64–0,81; el par 2–6 llega a 0,81). El grupo 0 hace de puente entre los dos bloques.
- **Grupos más distintos:** el grupo 5 correlaciona ≤ 0,51 con todos los demás, y el grupo 7 entre 0,33 y 0,66. Son los grupos donde un ranking global debería rendir peor.
- **Escala:** la diferencia media entre pares llega a 2,03 puntos (grupo 4 frente a grupo 7) y es casi transitiva: 4 > 0 > 1 ≈ 6 > 2 > 5 ≈ 3 > 7.

Los dos mapas separan dos efectos:
- **Gustos:** qué ítems prefiere cada grupo, que es lo que vuelve útil personalizar.
- **Escala:** cuán generoso es cada grupo, que no cambia el orden de los ítems pero sí el retorno crudo.

**Conclusiones que se usan después:**
1. Con logs uniformes, la frecuencia de un ítem no informa sobre su calidad, y las acciones registradas no dependen del grupo.
2. El rating depende solo del grupo y del ítem, no del orden ni del historial. El único valor secuencial es la información que se gana sobre el grupo.
3. Los grupos difieren en escala y en gustos. Conocer el grupo es saber casi todo lo que se puede saber del usuario.

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

- A cada token se le suma un embedding del paso t (el mismo para los tres tokens del paso) y uno del tipo de token. Después viene LayerNorm (la capa de dropout existe, pero con p = 0 no actúa; ver abajo).
- **Qué va en s_t y por qué.** El estado es mínimo: el grupo y el rating que se acaba de revelar. El resto del historial (los ítems previos) el transformer lo ve en los tokens a_<t. s_t nunca incluye r_t. En t = 0, el rating previo y su indicador valen 0.
- **Transformer:** `nn.TransformerEncoder` con 3 capas, 4 cabezas, feed-forward de 4·d, GELU y pre-normalización (`norm_first=True`). **Dropout = 0** (valor por defecto de `entrenar.py`; la clase admite 0,1). Se eligió por costo: en CPU cada capa de dropout tardaba ≈ 0,1 s por lote, casi el triple del paso completo. La regularización queda a cargo del weight decay y de la selección del checkpoint por cross-entropy de validación. Los `config.json` de las corridas en Colab confirman `dropout: 0.0`.
- **Máscara causal (3T × 3T):** es triangular superior. Cada token ve solo a los anteriores.
- **Alineación:** las logits de a_t se leen de la salida de los tokens s_t (posiciones 1, 4, 7, …), que vieron R̂_≤t, s_≤t y a_<t, pero no a_t.
- **Salida:** las logits son h_s · Eᵀ + b, con **pesos compartidos** con la tabla de embeddings de ítems (sin el relleno) y un sesgo por ítem.
- **Ítems ya mostrados:** `seen_before_mask` pone −∞ en las logits de los ítems a_<t, al entrenar y al recomendar.
- **Padding:** no hay. Los episodios de train tienen siempre T = 20. Al recomendar en el paso t, la secuencia tiene largo t + 1 y la máscara causal la vuelve equivalente a un prefijo.
- **Pérdida:** cross-entropy del ítem registrado en los 20 pasos (`dt_loss`).
- **No fuga:** un test altera a_t, R̂ y el estado de los pasos ≥ t y verifica que las logits del paso t no cambian. Pasa junto con el resto de los tests (ver `reproducibilidad.md`).
- **Grupo oculto:** el token "desconocido" existe en el embedding de grupos. En cada lote de train, el **30 %** de los episodios se entrena con el grupo reemplazado por "desconocido" (*group dropout*, `group_dropout = 0,3`), elegidos al azar en cada época. Así el mismo modelo sirve con grupo visible y oculto. IQL usa el mismo mecanismo y la misma fracción.

### 3.2 Entrenamiento

- AdamW con lr máximo 1e-3 y weight decay 0,1, scheduler `OneCycleLR` con sus valores por defecto (el lr sube durante el primer 30 % de los pasos y luego baja con forma coseno) y recorte de la norma del gradiente en 1,0.
- Lotes de 256 episodios: 10 épocas de 289 pasos cada una (⌈73.827 / 256⌉).
- Checkpoint con la menor cross-entropy de validación.

Con logs `random`, la cross-entropy de validación queda apenas por debajo de la de una predicción uniforme sobre el catálogo, log(752) ≈ 6,62. Es lo esperable con logs uniformes. El ítem registrado es casi impredecible, y lo que el modelo puede aprender es la relación débil entre el ítem y el retorno que siguió. La pérdida mide imitación de los logs, no calidad de recomendación.

**Curvas del DT** (`resultados.md`, Figura 2):

- **Logs `random`:**
  - Durante las cuatro primeras épocas la CE queda pegada a log(752). Baja en las épocas 5–6, cuando el lr empieza a descender después del pico del OneCycle, y desde la época 7 la de val queda plana en 6,588.
  - La CE de train sigue bajando (6,577), así que el modelo empieza a memorizar episodios de train sin ganar en val.
  - El checkpoint elegido es el de la época 8, 9 o 10 según la semilla. Entre esas épocas la CE de val difiere en menos de 0,001.
  - La accuracy de val es ≈ 0,18 %, apenas por encima del 0,13 % de elegir al azar.
- **Logs `mixto`:**
  - La CE de val baja de 6,33 a 6,141 en las tres semillas, casi 0,5 nats por debajo de log(752). Equivale a una perplejidad de ≈ 465 ítems en lugar de 752.
  - La accuracy de val es ≈ 0,7 %.
  - El ítem registrado es ahora bastante más predecible, porque lo eligió un sistema que conocía el grupo. Por eso el DT tiene algo que imitar (sección 7).

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

- **Recompensa:** está centrada y escalada como r' = (r − 3)/2, en el rango [−1, 1]. No cambia qué acción es mejor, pero fija la escala de las ventajas y, con ella, el efecto de β.
- **Último paso:** es terminal (`not_done[-1] = 0`), con objetivo TD = r', sin bootstrapping.
- **Red objetivo:** Q̄ se actualiza con promedio de Polyak (`lerp_`), Q̄ ← (1 − λ)·Q̄ + λ·Q con λ = 0,005, **después de cada paso de optimización**. Q̄ se usa para el objetivo de V y para las ventajas de AWR.
- **Optimización:** tres Adam independientes (uno por red: V, Q y π), lr 3e-4, sin weight decay, lotes de 256 episodios, recorte de gradiente en 1,0 y 10 épocas. Se guarda el último checkpoint: las pérdidas no miden calidad, y la configuración se elige con el protocolo en validación.
- **Gradientes separados:**
  - Q̄(s, a) y los pesos de AWR se calculan bajo `no_grad`;
  - V(s') entra en L_Q con `detach()`;
  - como las redes no comparten parámetros, L_V solo mueve a V, L_Q solo a Q y L_π solo a π.
- **Pesos de AWR:** min(exp(β·u), w_max), con w_max = 100 (`max_weight` de `entrenar.py`, igual en todas las corridas).
- **Diagnósticos registrados:** NLL de behavior cloning, media y desvío de la ventaja, peso medio, fracción truncada, V medio y Q medio.
- **Ítems ya mostrados:** se enmascaran en las logits de π al entrenar. Al recomendar, el protocolo los descarta tanto para π como para el máximo de Q.
- **Dos políticas (`IQLPolicy`):** `mode="pi"` ordena por las logits de π en el estado actual h_t, y `mode="q"` ordena por Q(s_t, ·) sobre todo el catálogo disponible.

**Diferencia de diseño con el DT.** IQL resume el historial completo de pares (ítem, rating) en un estado recurrente, con el grupo sumado en todas las posiciones. El DT, en cambio, ve el grupo solo en los tokens s_t y los ítems previos en los tokens a.

### 4.2 Barrido en validación (semilla 0)

Se barrieron γ ∈ {0; 0,9} × τ ∈ {0,7; 0,9} × β ∈ {3; 10}. Cada configuración se evaluó con π y con el máximo de Q, con el promedio del normalizado en las cuatro condiciones como criterio. La grilla completa está en la **Tabla 4** de `resultados.md`.

- **Configuración elegida:** γ = 0, τ = 0,7, β = 10, la mejor en π. Para la comparación de γ se mantuvo también la mejor con γ = 0,9 (τ = 0,7, β = 10).
- **τ y β casi no influyen:** el rango de π (0,002) es menor que la dispersión entre semillas observada después (≈ 0,005). La elección entre ellos es prácticamente un empate.
- **Con γ = 0, el máximo de Q no depende de τ ni de β.** El objetivo de Q es solo r, y Q no recibe gradientes de V ni de π. Por eso las filas τ = 0,7 y τ = 0,9 con β = 3 dan exactamente el mismo valor. Las demás diferencias entre filas de máx. Q con γ = 0 no vienen de los hiperparámetros: son variación del entrenamiento, probablemente por operaciones no deterministas de la GPU. Esto da una idea del ruido que hay entre corridas que en teoría son iguales.
- **Pesos de AWR:** con β = 3, el peso medio es ≈ 1,2 y no hay truncados, así que π es casi behavior cloning. Con β = 10, el peso medio sube de 1,4 a 9,4 durante el entrenamiento y alrededor del 4 % se trunca: π es más selectiva sin quedar dominada por el truncado.
- **Las pérdidas no sirvieron para elegir:** con β = 10, L_π sube de 8,6 a 48 porque los pesos se concentran, no porque π empeore.

**Curvas de IQL** (`resultados.md`, Figura 3):

- **γ = 0:**
  - Todo se estabiliza hacia la época 4. Q medio queda en 0,107 = (3,215 − 3)/2, exactamente la recompensa media centrada de los logs `random`. Con `mixto` queda en 0,365 = (3,73 − 3)/2, también la recompensa media de esos logs.
- **γ = 0,9:**
  - **Q y V medios no se estabilizan en 10 épocas.** Con logs `random`, Q medio sube sin parar, de 0,16 a 1,22. Con `mixto` llega a 2,59 y sigue subiendo.
  - L_Q de train también crece (0,27 → 0,36) porque su objetivo, V(s'), se mueve. La L_Q de val, en cambio, baja levemente.
  - Como referencia: con la recompensa media centrada de `random`, la suma descontada con γ = 0,9 sobre los pasos que faltan da en promedio ≈ 0,65. El expectil τ = 0,7 empuja V por encima de la media, así que un valor algo mayor es esperable. Que siga creciendo en la época 10 indica que el bootstrapping todavía no convergió.
  - **Los resultados con γ = 0,9, en especial los del máximo de Q, salen de una Q que no terminó de converger.** Es un límite de la comparación de γ: con más épocas podría cambiar, aunque la selección en val (sección 4.2) ya mostraba que γ = 0,9 no mejora.

---

## 5. Resultados principales (test)

Ver la **Tabla 1** en `resultados.md`.

Ver la **Tabla 1b** en `resultados.md`.

**Las métricas secundarias no ordenan igual que el normalizado.** Coinciden en los extremos: `group_mean`/`group_posterior` arriba, el máximo de Q con γ = 0 como mejor modelo y `popularity` abajo. En el medio, el orden cambia:
- **IQL π frente a `item_mean` y el DT.** Con grupo visible, IQL π supera a `item_mean` en normalizado, pero queda por debajo en hit rate y en NDCG@10. π logra un retorno total algo mayor con menos ítems ≥ 4. Sus ratings se reparten distinto, por ejemplo con más 5 o menos 1, y eso el hit rate no lo distingue. Además, su primera recomendación es la peor de los modelos: π ordena por imitación reponderada y no por rating esperado.
- **NDCG@10 del DT con grupo oculto:** tiene la dispersión entre semillas más grande de la tabla: el ranking inicial sin información depende mucho de la semilla.

Por eso las conclusiones se apoyan en el normalizado, que es la métrica principal del protocolo.

**Lectura:**

- **`popularity` queda por debajo del azar** (IC ±0,007), aunque con logs uniformes la frecuencia no tiene información. La razón es que es determinística: recomienda siempre la misma lista fija, encabezada por los ítems que por azar aparecieron más en los logs. Su resultado depende de cuán buenos sean esos ítems en particular, y la correlación levemente negativa entre frecuencia y rating (−0,04, sección 1.2) hace que esa lista quede algo por debajo de la media. No es una señal aprendida, sino la varianza de fijar una lista arbitraria. En cambio, `random` promedia sobre todas las listas.
- **Ningún modelo supera a la mejor línea base.** `group_posterior` domina las cuatro condiciones. Con logs uniformes, el rating medio por grupo e ítem es un estimador insesgado y casi exacto (los datos son sintéticos y por grupo), y esa línea base lo usa directamente.
- **El DT no personaliza con logs aleatorios.**
  - Rinde casi igual con grupo visible y oculto, y **empata en la práctica con `item_mean`**: la diferencia, 0,006, es del orden del IC por usuario, así que no permite decir que sea peor.
  - Con logs uniformes, la acción registrada no depende del grupo. Lo único que relaciona acción, grupo y retorno es que los ítems buenos para ese grupo aparecen más en los episodios de retorno alto, y esa señal es débil.
  - El DT aprende algo parecido a un ranking global de ítems buenos.
- **IQL personaliza, sobre todo el máximo de Q.**
  - Con γ = 0, Q(s, a) es una regresión del rating esperado condicionada al estado, y el máximo de Q equivale a un `group_mean` aprendido, a 0,014 de la línea base.
  - El máximo de Q supera a π porque, con logs uniformes, **todas las acciones están bien cubiertas** en todos los estados. El riesgo de extrapolación que motiva a IQL casi no aparece.
  - π, en cambio, está atada a la distribución de comportamiento por la pérdida de AWR (es una clasificación de acciones uniformes reponderadas), y eso la limita.
- **Con grupo oculto, ningún modelo supera a `item_mean` ni se acerca a `group_posterior`,** tampoco con warm-up 5. No aprendieron a inferir el grupo a partir de pocos ratings dentro de la sesión.
  - El warm-up mejora un poco a todas las políticas, pero los modelos lo aprovechan menos que la inferencia bayesiana explícita.
  - IQL π con grupo oculto queda incluso 0,046 por debajo de `item_mean`, que no usa ninguna información del usuario. Los resultados por grupo (Tabla 1c) muestran un sesgo hacia el grupo 4. Ahí se plantean dos hipótesis y cómo ponerlas a prueba.
- **γ = 0 frente a γ = 0,9.** Con π, con grupo visible (con o sin warm-up) la diferencia es menor que la dispersión entre semillas. Con grupo oculto va en sentidos opuestos:
  - sin warm-up, γ = 0,9 rinde 0,010 más, del orden de la dispersión;
  - con warm-up 5, γ = 0 rinde 0,013 más, por encima de la dispersión.

  No hay una ventaja consistente. Con el máximo de Q, γ = 0,9 es peor con grupo visible (0,028 menos) y más inestable entre semillas.
  - Interpretación: en este generador un ítem no cambia los ratings futuros. El único valor secuencial es la información que aporta sobre el grupo, y ninguna variante la explota.
  - γ > 0 solo agrega bootstrapping y su error. Además, con γ = 0,9 la Q no había convergido al final del entrenamiento (Figura 3 de `resultados.md`).
- **π frente al máximo de Q.** Con logs uniformes, el máximo de Q es mejor o igual a π en todas las condiciones con γ = 0. La ventaja de π aparece recién con logs sesgados (sección 7).

Ver la **Tabla 1c** en `resultados.md`.

**Lectura por grupo:**
- **Cuánto vale personalizar depende del grupo.** La ganancia de `group_mean` sobre `item_mean` va de 0,02 (grupos 0 y 7) a 0,14 (grupo 4). El grupo 4 es el más generoso y tiene gustos poco compartidos con los grupos 1, 3 y 5 (sección 1.2). El grupo 0 es el "puente" entre bloques, así que el ranking global ya le sirve.
- **El DT reproduce `item_mean` grupo por grupo,** con diferencias de −0,018 a +0,008, sin acercarse a `group_mean` en ningún grupo. Esto confirma que, con logs aleatorios, el DT aprendió un ranking global.
- **El máximo de Q con γ = 0 queda a ≤ 0,015 de `group_mean` en siete de los ocho grupos.** La excepción es el grupo 4 (0,056 menos), justo donde más vale personalizar.
- **IQL π es desparejo:** es el mejor modelo en los grupos 4 y 0, y el peor en los grupos 5 (por debajo de `item_mean`) y 7.
- **Con grupo oculto, IQL π y el máximo de Q se inclinan hacia el grupo 4.**
  - En el grupo 4, π oculto supera incluso a `group_posterior`.
  - En los otros siete grupos queda por debajo de `item_mean`: entre 0,03 y 0,13 menos.
  - El máximo de Q muestra el mismo patrón, más atenuado (grupo 4: +0,05 sobre `item_mean`; grupo 5: −0,05).

  **Hipótesis (sin verificar)** para este sesgo y para el bajo rendimiento con grupo oculto en general:
  1. Las recompensas no se normalizan por grupo. En los estados sin grupo, V promedia sobre todos los grupos. Entonces las transiciones de usuarios del grupo 4, que pone ratings altos, tienen ventajas positivas casi siempre, y exp(β·u) con β = 10 las amplifica. Así, π "desconocido" termina imitando lo que le gusta al grupo 4.
  2. El group dropout (sección 3.1) puede ser insuficiente: el modelo puede tratar "desconocido" como un noveno grupo en lugar de aprender a inferirlo de los ratings de la sesión.

  Pruebas controladas posibles, siempre en validación: normalizar la recompensa por grupo, o reentrenar con una fracción de group dropout mayor (por ejemplo 0,6), y comparar la condición de grupo oculto por grupo.

---

## 6. Return conditioning (DT, logs `random`)

**Diseño.** Se pidieron nueve retornos:
- los percentiles 10, 25, 50, 75, 90 y 99 de los retornos de los episodios de train: 46, 57, 65, 73, 81 y 89;
- el máximo de train, 97;
- dos valores fuera del soporte: máximo + 10 = 107 y máximo + 20 = 117.

La lista sale de `targets_barrido` (`modelos/experimentos.py`), a partir del soporte de retornos de train (`train_return_support` en `barridos/dt/elegido.json`). Como los ratings son enteros, los percentiles del retorno también lo son, y los valores de arriba son exactos. El mínimo es 26 y el percentil 1 es 35.

El retorno pedido se eligió en validación con el promedio de las cuatro condiciones. Ganó R̂ = 117, en empate práctico con 107 y 97 (`resultados.md`, sección 4). En test se evaluó el barrido completo con las 3 semillas. Con warm-up se compara `full_return` (los 20 pasos), porque el objetivo se refiere a la sesión completa.

Ver la **Tabla 2** en `resultados.md`.

Con grupo oculto, el desvío entre semillas del retorno obtenido es mayor, hasta 0,50 puntos (R̂ = 97). El IC por usuario es similar.

**Incertidumbre.** Las dos fuentes de variación son chicas frente a las diferencias entre retornos pedidos.
- **Desvío entre semillas:** el seguimiento del retorno pedido es estable entre entrenamientos.
- **IC por usuario:** estima con precisión el retorno medio obtenido. Lo calculamos desde `sesiones.csv`, porque `return_conditioning.csv` solo trae el desvío entre semillas.

El IC por usuario es más angosto en el centro del soporte que en los extremos. Dentro del soporte, el DT no solo acierta en promedio: lleva a casi todos los usuarios al retorno pedido. Es como un controlador. Como R̂_t se descuenta con cada rating revelado, cuando un usuario puntúa más alto de lo esperado, el modelo compensa en los pasos siguientes. En los extremos el IC se ensancha porque el modelo ya no puede compensar: con R̂ alto alcanza el techo de cada usuario, y con R̂ bajo, el piso.

**Warm-up 5.** El objetivo se refiere a la sesión completa, y los 5 ítems del warm-up son al azar, así que el techo baja. Entre 46 y 73 el seguimiento es casi tan bueno como sin warm-up (a ≤ 3,4 puntos del pedido). Desde 81 se satura antes.

Los gráficos de retorno obtenido frente a pedido, agregado y por grupo, están en la Figura 1 de `resultados.md`.

**Interpretación:**

- **Sobre la diagonal hasta unos 80 (en el agregado).** Entre 46 y 81, el retorno obtenido está a menos de 2 puntos del pedido. Por grupo, el rango varía (ver más abajo). El condicionamiento funciona **en ambas direcciones**: si se pide 46, el modelo elige deliberadamente ítems malos y queda por debajo del azar. Esto prueba que el DT sí aprendió qué ítems llevan a retornos bajos o altos, aunque no los use para personalizar por grupo.
- **Saturación desde unos 89.** Desde 97 el retorno obtenido se estanca: es el techo de lo que el modelo sabe hacer. Ese techo agregado es el promedio de techos muy distintos por grupo (Tabla 2b). Pedir 107 o 117, fuera del soporte, no empeora ni mejora: es extrapolación sin efecto. Elegir 117 en validación equivale, en la práctica, a "pedir el máximo".
- **El mejor retorno pedido no es el más alto en un sentido útil.** Por encima del máximo de train, el resultado es plano. El valor elegido está fuera del soporte, pero empata con el máximo de train.
- **Grupo oculto frente a visible.** Con retornos bajos, el modelo con grupo oculto queda algo por encima del pedido. Sin el grupo le cuesta encontrar ítems que sean malos para ese usuario en particular.
- **Por grupo (Tabla 2b y Figura 1 de `resultados.md`).** Por la exploración de la sección 1.2, un mismo retorno pedido no significa lo mismo en todos los grupos. El soporte relevante es el de cada grupo, no el global, y la diagonal agregada esconde que **cada grupo tiene su propio piso y su propio techo**:
  - **Techo** (fila 117): sigue el orden de generosidad de la sección 1.2, salvo el grupo 5, que supera al 2 aunque es algo menos generoso.
  - **El grupo 7 se aparta de la diagonal ya en 81**, y desde ahí queda plano. Para él, pedir 81 o más es extrapolación, aunque esté dentro del soporte global.
  - **El grupo 4 tiene piso** (filas 46 y 57): casi no hay ítems que el grupo 4 puntúe bajo, y el modelo no puede bajar más.
  - Los demás grupos siguen la diagonal entre 46 y 81 a menos de 1 punto, y se saturan en su propio techo.
  - **En normalizado**, el techo de cada grupo queda por debajo de `group_mean` en todos los grupos (Tabla 1c). Pedir más retorno no alcanza para personalizar como la línea base.

  Con grupo oculto, el patrón es el mismo pero más suave. El grupo 4 tiene un piso todavía más alto (62,5 al pedir 46), porque sin el grupo el modelo no sabe qué ítems son malos para ese usuario. El grupo 7 se curva más gradualmente, desde 73.
- **Con logs `mixto`** (solo en validación, porque en test se evaluó únicamente el retorno elegido), el efecto del grupo con retornos bajos es mucho más fuerte. Por ejemplo, con R̂ = 57, el normalizado en val es −0,25 con grupo visible y 0,15 con grupo oculto. Con grupo visible, el modelo imita las sesiones malas del sistema anterior para ese grupo. Con grupo oculto, no puede apuntar a los ítems malos del usuario.

---

## 7. Calidad de los logs

**Setup.** DT e IQL se reentrenaron con los logs `mixto` (3 semillas), con los mismos hiperparámetros de IQL elegidos con logs `random`. El retorno pedido al DT se volvió a elegir en validación y resultó R̂ = 99, el máximo de train con estos logs. Las líneas base se ajustaron con los mismos logs `mixto`, y se agregó el sistema anterior con ε = 0.

Ver la **Tabla 3** en `resultados.md`.

**Con warm-up 5:**
- **Grupo visible:** el orden de las políticas es el mismo que sin warm-up, e IQL π sigue empatado con `group_mean`.
- **Grupo oculto: el orden cambia.** Sin warm-up, el DT era el mejor modelo; con warm-up 5, las dos π de IQL pasan adelante. La π de IQL aprovecha más los 5 ratings iniciales (+0,037) que el DT (+0,011). Aun así, π solo empata con `popularity`, y `group_posterior` sigue claramente arriba.
- **Modelos frente a `item_mean` con grupo oculto.** Con logs `mixto`, `item_mean` queda debilitada por el sesgo de selección, así que varios modelos la superan: sin warm-up, el DT y π con γ = 0,9; con warm-up, las dos π. Con logs `random`, ningún modelo la supera.
- **Los 5 ítems de warm-up atenúan el colapso del máximo de Q con γ = 0,9 (+0,066), pero no lo eliminan.** Ver el historial ayuda a Q a ubicar al usuario, pero sigue por debajo de todas las demás variantes y de `item_mean`.

**Interpretación:**

- **Las líneas base cambian por sesgo de selección.**
  - `popularity` pasa de rendir peor que el azar a superar a `item_mean`: con logs sesgados, lo que más se mostró es lo que el sistema anterior consideraba bueno, y esa línea base hereda su conocimiento.
  - `item_mean` empeora (−0,017): cada ítem se mostró sobre todo a los grupos que lo valoran bien, así que su media observada sobreestima lo que pasaría con un usuario cualquiera. El sesgo **no es uniforme**: es grande para los ítems que les gustan a pocos grupos (se mostraron casi solo a esos grupos) y chico para los que les gustan a todos. Por eso no solo cambia el nivel de las medias, sino también el **orden** de los ítems, y el ranking global pasa a favorecer ítems de nicho.
  - `group_mean` prácticamente no cambia, porque condicionar por grupo elimina ese sesgo.
- **Diferencia entre datasets** (valores de referencia de `protocolo/PROTOCOLO.md`, solo con logs `random`; no se corrió Goodreads). `item_mean` da 0,69 en Netflix y 0,42 en Goodreads. `group_mean` con grupo visible da 0,74 y 0,49.
  - `item_mean` es un único ranking para todos los usuarios. Rinde bien solo si los grupos coinciden en qué ítems son buenos. En Netflix, la correlación entre los ratings medios por ítem de distintos grupos es siempre positiva (sección 1.2), así que un ranking global ya captura buena parte (Tabla 1c). En Goodreads el ranking global capta mucho menos (0,42), lo que indica gustos más divergentes entre grupos o más varianza dentro de cada grupo. Conocer el grupo agrega en ambos casos ≈ 0,05–0,07; en Goodreads ni siquiera `group_mean` pasa de 0,49.
  - Con logs `random`, `popularity` no tiene información en ningún dataset. Su valor (−0,06 o 0,04) es el de una lista fija arbitraria, con signo distinto según el azar de cada catálogo.
  - El cambio con logs sesgados solo se midió en Netflix. Según el protocolo, el sesgo de selección se agrava al pasar de `mixto` a `experto`: `item_mean` pasa de 0,67 a 0,44 y `popularity` queda en 0,69. Para comparar `mixto` entre datasets habría que correr al menos las líneas base en Goodreads (`python -m protocolo.evaluar --dataset goodreads --logs mixto`).
- **El DT ahora personaliza.** Con grupo visible sube 0,046 respecto de logs `random` y se separa del caso oculto. En `mixto`, las sesiones de retorno alto son las del sistema anterior explotando, que elige según el grupo. Así, "imitar las trayectorias de retorno alto condicionado al grupo" pasa a significar "elegir lo bueno para ese grupo".
- **IQL π es el mejor modelo** (a 0,004 de `group_mean`, dentro del IC: empate con la línea base). Frente al DT, la diferencia de 0,008 es mayor que la dispersión entre semillas, pero chica. La pérdida de AWR parte de una política de comportamiento que ya es buena y la mejora reponderando por ventaja.
- **Todos los modelos superan al sistema que generó los datos.** El DT y las variantes de π lo superan por 0,12 o más en todas las condiciones. La excepción es el máximo de Q con γ = 0,9 y grupo oculto, que lo supera por solo 0,04. No es un mérito exclusivo de los modelos: **`popularity`, `item_mean` y `group_mean` también lo superan**. Hay dos razones:
  1. Los modelos eligen de forma determinística el ítem de mayor puntaje, mientras que el sistema anterior **muestrea** con softmax(μ/τ) (con τ = 0,25) usando medias que excluyen al propio usuario.
  2. `mixto` mezcla sesiones de distinta calidad, incluidas aleatorias (ε = 1), que dan cobertura. Los métodos offline pueden "recombinar" lo mejor de cada sesión.
- **El máximo de Q con γ = 0,9 colapsa con grupo oculto,** con la mayor dispersión entre semillas de la tabla. Con logs sesgados, cada ítem se mostró casi solo a los grupos a los que les gusta, así que la cobertura por estado es despareja. Sin el grupo, Q sobreestima ítems poco cubiertos, el máximo los elige y el bootstrapping con γ > 0 propaga el error. π, que se entrena solo con acciones registradas, no sufre este problema. Es el **error de extrapolación** que IQL evita al entrenar y que el máximo de Q reintroduce al recomendar.
- **Grupo oculto:** con logs `mixto`, ningún modelo supera a `group_posterior`, y el mejor sin warm-up (el DT) queda cerca de `popularity`.

---

## 8. Comparación DT vs. IQL

| | DT | IQL |
|---|---|---|
| Mecanismo | imita trayectorias condicionado por el retorno | estima valores y extrae una π con ventaja positiva dentro del soporte |
| Qué necesita de los logs | que las trayectorias de retorno alto difieran por sus acciones | cobertura de acciones por estado |
| Resultado observado | con logs `random` no personaliza (≈ `item_mean`); con `mixto` personaliza | personaliza con ambos logs; su mejor variante depende de la cobertura (máx. Q con `random`, π con `mixto`) |

Con logs uniformes, IQL aprovecha mejor la información. Q aprende el rating esperado por estado y acción con todas las acciones cubiertas. En cambio, el DT solo puede explotar la correlación débil entre ítem y retorno, sin una atribución por acción.

Con logs de un sistema que ya personaliza, la diferencia se reduce a 0,008. Las trayectorias de retorno alto ya contienen la decisión correcta, y el DT tiene algo concreto que imitar.

En las dos situaciones, el DT da un control explícito del retorno que ninguna variante de IQL ofrece, aunque saturado arriba. A su vez, π de IQL es la única opción robusta cuando la cobertura es despareja.

---

## 9. Conclusiones y límites

- Con logs uniformes, ninguno de los dos métodos supera a las líneas base por grupo. IQL personaliza más que el DT.
- El DT sigue al retorno pedido dentro del soporte y se satura arriba. Cada grupo tiene su propio techo y algunos tienen piso, así que el soporte relevante es el de cada grupo, no el global.
- Con logs de un sistema anterior, ambos superan con amplitud a ese sistema y quedan a la par de `group_mean`.
- Ninguno aprendió a inferir el grupo durante la sesión como lo hace `group_posterior`. Este es el principal margen de mejora.

**Límites.**
- **Datos y simulador:** los datos son sintéticos y el rating depende solo del grupo y del ítem, sin efectos de orden. Por eso hay poco valor secuencial y γ > 0 no puede aportar. Además, el simulador usa usuarios de test con ratings completos y no modela abandono, así que los resultados miden retorno simulado y no comportamiento real.
- **Selección e incertidumbre:**
  - La selección en validación es aproximada, porque su catálogo difiere del de test.
  - Los hiperparámetros de IQL para `mixto` se tomaron del barrido con `random`.
  - Con 3 semillas y 200 usuarios por grupo, las diferencias chicas por grupo no son concluyentes.
- **Pendiente de verificar:**
  - con γ = 0,9, la Q no convergió en 10 épocas;
  - las hipótesis sobre el grupo oculto no se pusieron a prueba;
  - solo se evaluó Netflix.

---

## 10. Uso de IA

[COMPLETAR por el grupo; el detalle va en la bitácora.] Indicar qué partes se hicieron con asistencia de IA (por ejemplo, la implementación inicial de `modelos/`, el evaluador por lotes o la redacción del reporte) y, sobre todo, qué se verificó:
- los tests de no fuga y de alineación;
- la equivalencia del evaluador por lotes con el protocolo;
- la coincidencia de las líneas base con los valores de referencia del protocolo;
- la revisión de cada cifra del reporte contra los CSV de resultados;
- las decisiones de modelado que el grupo revisó y puede justificar.

