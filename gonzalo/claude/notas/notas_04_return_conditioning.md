# Notas del TP — 04. Return conditioning

## Pregunta

> Qué es el "return conditioning" y qué experimento se debe realizar para su evaluación?

## Respuesta

### 1. Qué es

El **return conditioning** (condicionamiento por retorno) es el mecanismo central del Decision Transformer. El modelo aprende una política **condicionada a un retorno deseado**:

```
P(a_t | R̂_t, s_t, historia previa)
```

Es decir: "¿qué acción aparecía en los datos cuando todavía faltaba obtener R̂_t y la situación era s_t?".

**Al entrenar**, R̂_t es el retorno pendiente que **efectivamente** se obtuvo en el episodio registrado, desde el paso t hasta el final (dentro de los 20 pasos):

```
R̂_t = r_t + r_{t+1} + … + r_19
```

El modelo ve muchas trayectorias, de retorno alto y bajo, y aprende qué acciones se asocian a cada nivel de retorno.

**Al recomendar**, el retorno lo elige el usuario del modelo. Se fija un **retorno objetivo** (por ejemplo 90) y en cada paso se le pasa al modelo lo que falta:

```
R̂_t = objetivo − (suma de los ratings ya revelados)
```

Ejemplo: con objetivo 90, si los dos primeros ratings fueron 5 y 4, en el paso 2 se pide R̂_2 = 90 − 9 = 81 para los 18 pasos restantes.

- Con warm-up, también se restan los ratings del warm-up. El objetivo se refiere a los **20 pasos de la sesión completa**.
- La idea es que, al pedir un retorno alto, el modelo elija acciones "como las de las trayectorias que lograron ese retorno".

#### La advertencia clave

El retorno pedido es una **condición de entrada, no una garantía**. El modelo no optimiza nada: imita lo que había en los datos para ese nivel de retorno. Pueden pasar tres cosas:

- Si en los datos no hay trayectorias con ese retorno, el modelo **extrapola**: recibe valores que nunca vio.
- Si las trayectorias de retorno alto no se distinguen por sus acciones, el modelo no tiene qué imitar.
- Con logs **uniformes al azar** (los originales), una trayectoria de retorno alto se debe en parte al grupo del usuario (hay grupos que valoran todo más alto) y en parte a la suerte. Las acciones no fueron elegidas por ser buenas. Lo que el DT puede aprender es "qué ítems tienden a aparecer en sesiones de retorno alto", que se parece a "qué ítems son buenos, para ese grupo".

### 2. El experimento

La consigna (§4) y la guía (§4.6) piden una **intervención controlada**: se varía **solo** el retorno pedido, se mantienen fijas las condiciones de evaluación y se mide el retorno obtenido.

#### Paso 1: definir el soporte de train

Se calcula la distribución de retornos de los episodios de entrenamiento:

```python
ep = make_episodes(train, length=20)
retornos = ep.returns_to_go()[:, 0]      # retorno de cada episodio, entre 20 y 100
```

Con los logs `random`, el retorno medio es de unos 64. Con `mixto`, de unos 75: el soporte cambia según los logs.

Se eligen los retornos pedidos:

- **dentro del soporte:** los percentiles 10, 25, 50, 75, 90 y 99;
- **fuera del soporte:** uno o dos valores **por encima del máximo** de train, para observar la extrapolación.

#### Paso 2: elegir el retorno pedido en validación

Se evalúa el DT con cada retorno pedido sobre los usuarios de **validación**, con el protocolo. El que da **mejor retorno normalizado** es el que va a la tabla final de resultados.

Nunca se elige mirando test.

#### Paso 3: el barrido en test

En test se evalúa el mismo barrido de retornos pedidos y se grafica **retorno obtenido frente a retorno pedido**. El gráfico marca:

- la **diagonal** (obtenido = pedido);
- el **máximo de train** (el límite del soporte).

Se hace para el **agregado** y **para cada uno de los 8 grupos**. Las condiciones del protocolo (grupo visible, oculto y warm-up) se mantienen iguales en todo el barrido.

Detalle práctico: el objetivo se refiere a la sesión completa de 20 pasos. Con warm-up conviene comparar contra el retorno de los 20 pasos (columna `full_return` de `sesiones.csv`), no solo el de los 15 pasos controlados (`return`). También hay que declarar qué se graficó.

#### Paso 4: incertidumbre

- **IC 95 % por usuario**, que da `summarize`.
- **Dispersión entre semillas de entrenamiento:** el DT final se entrena con ≥ 3 semillas.

Las diferencias entre retornos pedidos cercanos pueden ser menores que esa dispersión. En ese caso no hay efecto demostrado.

### 3. Cómo leer el gráfico

| Forma de la curva | Interpretación |
|---|---|
| Sobre la diagonal | el modelo obtiene lo que se le pide |
| Crece pero por debajo de la diagonal | responde al pedido, pero no alcanza lo solicitado |
| Plana | el modelo no distingue entre pedir más o menos; el condicionamiento no tiene efecto |
| Comportamiento raro más allá del máximo de train | extrapolación: el modelo nunca vio esos valores |

### 4. Preguntas a responder en el reporte

- ¿Hasta dónde el retorno obtenido sigue al pedido?
- ¿Qué pasa fuera del soporte?
- ¿Los grupos se comportan igual? Los grupos tienen escalas de rating distintas, así que su oráculo es distinto. Un mismo retorno pedido (por ejemplo 90) puede ser alcanzable para un grupo y estar fuera de su soporte para otro. Además, con grupo oculto, el retorno pedido puede actuar como una pista implícita sobre "qué tipo de usuario" se está esperando.
- ¿El mejor retorno pedido es el más alto? No necesariamente: pedir demasiado puede llevar al modelo a una zona que no conoce.
- ¿Cómo cambia todo al entrenar con logs `mixto`, que tienen trayectorias de retorno alto con acciones elegidas por ser buenas?

Un resultado **sin mejora es válido** si el experimento permite interpretarlo. La rúbrica asigna **15 puntos** a este experimento. Los criterios son: intervención controlada, resultados por grupo, incertidumbre y límites.

### Fuentes

- [CONSIGNA.md](CONSIGNA.md), sección 4
- [GUIA.md](GUIA.md), secciones 2.3, 4.1, 4.5 y 4.6
- [protocolo/PROTOCOLO.md](protocolo/PROTOCOLO.md), "Reglas para Decision Transformer"
- [RUBRICA.md](RUBRICA.md)
