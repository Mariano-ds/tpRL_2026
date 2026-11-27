# Notas del TP — 01. Modelos a entrenar y parámetros

## Pregunta

> El directorio "2026/tp" tiene el enunciado y archivos auxiliares de un trabajo práctico (TP) de un curso sobre aprendizaje por refuerzo (RL). Te iré haciendo preguntas con el objetivo de entender el objetivo del TP y de las técnicas de machine learning que se debe implementar. Primeramente explica cuáles son los modelos que hay que entrenar, sus nociones básicas y cuáles son los parámetros a establecer (según el enunciado del TP).

## Respuesta

En el TP se entrenan **dos modelos de RL offline** para recomendar ítems (películas de Netflix o libros de Goodreads). Ninguno interactúa con usuarios mientras aprende: los dos aprenden solo de los logs (trayectorias ya registradas). Después se comparan contra líneas base con un protocolo común de sesiones simuladas.

### Contexto común: el problema como MDP

- **Episodio:** una sesión de 20 recomendaciones a un usuario, sin repetir ítems.
- **Estado s_t:** lo que la política sabe del usuario hasta ese momento. Incluye el grupo (si es visible), los ítems ya mostrados con sus ratings y el paso t. El usuario en sí no es observable.
- **Acción a_t:** el ítem a mostrar, entre los que todavía no se mostraron.
- **Recompensa r_t:** el rating (1–5) que el usuario le da a a_t. Se revela recién después de mostrarlo.
- **Retorno pendiente (return-to-go):** R̂_t = r_t + … + r_19. Se calcula **dentro de cada episodio de 20 pasos**, así que queda entre 20 y 100. No se calcula sobre toda la historia del usuario.
- Los datos son sintéticos: hay 8 grupos de usuarios con gustos distintos. El rating depende solo del grupo y del ítem, y el orden en que se ven los ítems no influye.

---

### 1. Decision Transformer (DT)

#### Idea básica

El DT convierte el RL en **aprendizaje supervisado sobre secuencias**. No estima valores: aprende a responder "¿qué acción aparecía en los datos cuando faltaba obtener R̂_t y la situación era s_t?", es decir P(a_t | R̂_t, s_t, historia).

- **Entrenamiento:** es una clasificación (cross-entropy sobre los n_items) del ítem registrado. La entrada incluye el retorno que **realmente** se obtuvo desde ese paso.
- **Recomendación:** se le pide un **retorno objetivo** (por ejemplo 90). En cada paso se le pasa R̂_t = objetivo − ratings ya obtenidos, y el modelo propone acciones "como las de las trayectorias que lograron eso". Ese retorno es una condición de entrada, **no una garantía**.

#### Elementos técnicos clave

- **Tokens:** cada paso aporta tres, en el orden `R̂_0, s_0, a_0, R̂_1, s_1, a_1, …`.
  - R̂ se escala (por ejemplo ÷100) y pasa por una capa lineal.
  - Cada ítem tiene un embedding.
  - Se suman además un embedding del paso t y uno del tipo de token.
- **Máscara causal:** cada token solo ve a los anteriores.
- **Alineación del target:** a_t se predice **desde la salida del token s_t**, nunca desde la de a_t. Si no, hay fuga y el TP no es válido. Hay que incluir un test obligatorio de no fuga.
- **Estado s_t:** su contenido es una decisión de diseño que hay que justificar. Por ejemplo, el grupo (o un token "desconocido") más el rating anterior. Nunca puede incluir r_t.
- **Ítems ya mostrados:** sus logits se enmascaran con −∞, tanto al entrenar como al recomendar.

#### Parámetros a establecer

| Parámetro | Qué indica el enunciado |
|---|---|
| `d_model`, `n_layers`, `n_heads` | el esqueleto propone 128 / 3 / 4 |
| Horizonte / largo de contexto | 20 |
| Escalado de R̂ | por ejemplo ÷100 |
| Contenido del estado s_t | decisión de diseño a documentar |
| Fracción de episodios con grupo oculto en train | por ejemplo 30 %, para que funcione con grupo oculto (hay que declararlo) |
| Optimizador | AdamW, scheduler (por ejemplo OneCycle) y recorte de gradiente |
| Selección de checkpoint | el de menor cross-entropy en validación |
| **Retorno objetivo** | el hiperparámetro central. Se barren percentiles de los retornos de train (10, 25, 50, 75, 90, 99) más uno o dos por encima del máximo, y se elige el mejor **en validación** |

---

### 2. Implicit Q-Learning (IQL)

#### Idea básica

Es un método basado en **funciones de valor**:

- **Q(s, a):** retorno esperado si se muestra el ítem a en el estado s.
- **V(s):** retorno esperado desde s.
- **Ventaja:** A = Q − V.

El Q-learning clásico usa `max_a' Q(s', a')`. En el caso offline, ese máximo recorre ítems que casi no aparecen en los logs y cuyo Q está mal estimado. Esos errores optimistas se propagan (**error de extrapolación**). IQL **nunca evalúa Q en acciones fuera de los datos**.

#### Sus tres componentes

1. **V(s) por regresión de expectil:** se ajusta sobre Q̄(s, a) de las acciones registradas, con la pérdida asimétrica L_τ(u) = |τ − 1(u<0)|·u². Con τ > 0,5, V se acerca al "mejor valor visto en los datos" desde s, sin consultar ítems no observados.
2. **Q(s, a) por diferencias temporales:** el objetivo es r + γ·V(s'). En el último paso es solo r, sin bootstrapping.
3. **Política π(a|s) por AWR (regresión ponderada por ventaja):** es una clasificación de la acción registrada, como behavior cloning, pero cada ejemplo pesa exp(β·(Q̄ − V)), truncado en w_max.
   - Con β → 0 se reduce a behavior cloning.
   - Con β grande casi todo el peso va a la mejor acción **de los datos**.

Otros detalles:

- **Red objetivo Q̄:** es una copia de Q que se actualiza con promedio de Polyak, lo que evita que Q y V se persigan mutuamente.
- **Gradientes separados:** cada pérdida actualiza solo su red. V(s') y los pesos de AWR se calculan sin gradiente.
- **Codificador del estado:** recorre el historial (ítem, rating), por ejemplo con una GRU o un transformer causal, y le suma el grupo y el paso. Devuelve T+1 estados, porque hace falta V(s_{t+1}).

#### Parámetros a establecer

| Parámetro | Qué indica el enunciado |
|---|---|
| **τ (expectil)** | barrido en {0,7; 0,9} |
| **β (temperatura AWR)** | barrido en {3; 10} |
| **γ (descuento)** | comparar obligatoriamente **γ = 0** (bandit contextual) con **γ > 0** (por ejemplo 0,9) |
| λ de Polyak | ≈ 0,005 por paso |
| w_max (truncado de pesos) | por ejemplo 100 |
| Recompensa cruda o centrada | por ejemplo (r−3)/2. Cambia la escala de las ventajas y, con eso, el efecto de β |
| Arquitectura del codificador | si Q, V y π lo comparten o no, y qué pérdida lo entrena |
| Optimizadores | uno por red |

τ, β y γ se eligen **con el protocolo en validación, no mirando las pérdidas**. Las pérdidas de Q y π pueden subir mientras la política mejora.

Del mismo modelo IQL se evalúan **dos políticas**:

- **`pi`:** ordena los ítems según π.
- **`q`:** elige el máximo de Q entre todos los ítems disponibles.

Hay que explicar en qué se diferencian. La clave es que `q` reintroduce, al recomendar, el máximo que IQL evita al entrenar.

---

### Requisitos comunes a ambos modelos

- Usan los **mismos episodios** de train (`make_episodes(train, 20)`) y la misma partición de validación. Train y validación se separan **por usuario**.
- Cada modelo final se entrena con **al menos 3 semillas**. Se reporta la media y la dispersión entre semillas, por separado del IC por usuario.
- Se evalúan en tres condiciones: **grupo visible**, **grupo oculto** (cold-start) y **warm-up de 5 ítems**.
- Cada modelo se entrena con dos tipos de logs:
  - **`random`:** los originales, de política uniforme.
  - **`mixto`:** generados por un "sistema anterior". Las líneas base se ajustan con los mismos logs.
- El conjunto de test se usa **una sola vez**, al final.

### Fuentes

- [CONSIGNA.md](CONSIGNA.md), secciones 1–5
- [GUIA.md](GUIA.md), secciones 2, 4 y 5
- [RUBRICA.md](RUBRICA.md)
