# Notas del TP — 03. Otros aspectos clave del práctico

## Pregunta

> Además de los modelos a entrenar y del protocolo, qué otra consideración o aspecto clave está involucrado en el práctico (excluye la exploración de los datos)?

## Respuesta

Además de DT, IQL y el protocolo, el TP gira alrededor de seis ejes. Algunos son experimentos obligatorios con puntaje propio en la rúbrica. Otros son condiciones de validez: si no se cumplen, los resultados no cuentan.

### 1. El problema de fondo del RL offline: cobertura de los datos

Es la idea que une todo el TP: **solo se puede aprender lo que los logs cubren**. No se puede preguntar "¿qué hubiera pasado si…?".

- La **política de registro** (la que generó los logs) determina qué se puede aprender. Con logs uniformes, que un ítem aparezca mucho no dice nada. Con logs de un buen recomendador, sí.
- Cada método resuelve esto a su manera:
  - **DT:** imita las trayectorias de los datos, condicionado por el retorno.
  - **IQL:** evita evaluar Q fuera de los datos, con el expectil y AWR.
  - **Máximo de Q:** reintroduce el **error de extrapolación** al recomendar.
- La comparación razonada entre DT e IQL que pide el reporte pasa por acá. "Imitar las sesiones que salieron bien" (DT) no es lo mismo que "elegir acciones mejores que el promedio en cada estado" (IQL).

### 2. Return conditioning (15 puntos en la rúbrica)

Es una **intervención controlada** sobre el DT: se varía el retorno pedido y se mide el obtenido.

- Los retornos pedidos salen del soporte de train (percentiles 10, 25, 50, 75, 90, 99), más uno o dos **por encima del máximo**.
- El mejor retorno pedido se elige en validación. En test se grafica el obtenido frente al pedido, con la diagonal y el máximo de train marcados. El gráfico va **agregado y por grupo**.
- Hay que interpretar el gráfico:
  - si el retorno obtenido sigue al pedido o se queda plano;
  - qué pasa fuera del soporte (**extrapolación**);
  - si los grupos se comportan igual;
  - si el mejor retorno pedido es el más alto.
- El retorno pedido **no es una garantía**. Un resultado sin mejora es válido si se puede interpretar.

### 3. Calidad de los logs (10 puntos)

Se entrenan DT e IQL **dos veces**: con los logs originales (`random`) y con los `mixto` del "sistema anterior". Nada más cambia entre las dos corridas.

- Las líneas base se ajustan **con los mismos logs** que los modelos. `popularity` y el sistema anterior se incluyen como referencias.
- Preguntas a responder:
  - qué aprende cada modelo en cada caso;
  - si alguno **supera a la política que generó sus datos**, y por qué podría hacerlo;
  - por qué cambian `popularity` e `item_mean` de un dataset a otro.
- Concepto central: el **sesgo de selección**. Cuando una política decide qué se muestra, lo observado deja de ser una muestra al azar. `popularity` hereda el conocimiento del sistema anterior. `item_mean` empeora porque cada ítem se mostró sobre todo a los grupos a los que les gusta.

### 4. ¿Importa la secuencia? Cold-start y adaptación

Los datos tienen una propiedad fuerte: **el rating depende solo del grupo y del ítem**, no de lo visto antes. No hay saturación, novedad ni efectos de orden. De ahí salen dos experimentos de IQL y una pregunta de fondo:

- **γ = 0 frente a γ > 0:** con γ = 0 el problema es un **bandit contextual**. La consigna pregunta "¿qué parte del valor viene de la secuencia?". Si γ > 0 no mejora, hay que explicar qué valor futuro había para aprender. Una recomendación no cambia los ratings futuros. Solo puede **aportar información** sobre el grupo del usuario.
- **π frente al máximo de Q:** relacionarlo con el error de extrapolación.
- **Grupo oculto:** el modelo tiene que inferir el grupo con los ratings que va viendo, como hace explícitamente `group_posterior`. Para el DT conviene entrenar con el grupo reemplazado por "desconocido" en una fracción de los episodios (por ejemplo 30 %), y declararlo.

### 5. Rigor experimental y validez

Son condiciones transversales. Varias aparecen como "errores frecuentes" en la guía.

**Sin fugas de información:**
- El target del DT no puede aparecer en el contexto que lo predice, y hay un **test obligatorio de no fuga**.
- El estado no incluye r_t.
- R̂ se calcula dentro del episodio de 20 pasos.
- En IQL no se hace bootstrapping en el último paso.

**Gradientes correctos en IQL:** los pesos de AWR y V(s') se calculan sin gradiente.

**Separación de datos:**
- train y validación se separan **por usuario**;
- los hiperparámetros se eligen en validación;
- test se usa una sola vez;
- las líneas base nunca se ajustan con usuarios de validación o test.

**Las pérdidas no miden calidad.** En IQL, las de Q y π pueden subir mientras la política mejora. Las conclusiones se sacan con el protocolo.

**Dos fuentes de variabilidad:**
- dispersión entre ≥ 3 semillas de entrenamiento;
- IC 95 % por usuario de la evaluación.

Una diferencia menor que la dispersión entre semillas no es una mejora. No vale reportar la mejor semilla.

**Comparabilidad:** no se mezclan números de validación con números de test, ni de protocolos distintos.

### 6. Reproducibilidad y entregables

Para cada modelo hay que guardar:
- configuración y semillas;
- **curvas sin recortes**;
- checkpoints e instrucciones de reproducción;
- versiones de las librerías.

Si se usa una librería, hay que explicar qué resuelve y verificar su contrato de entradas y salidas.

La entrega incluye:
- notebooks o scripts ejecutados: exploración, entrenamiento, evaluación, return conditioning y logs;
- la tabla de resultados por condición y tipo de logs;
- un **reporte de 4–6 páginas** que cubra problema, protocolo, decisiones, evidencia y límites;
- una **bitácora** de contribuciones y decisiones;
- una declaración del uso de IA (opcional), indicando qué se verificó.

Además hay una **explicación individual** sobre una decisión de modelado y un resultado de evaluación.

La rúbrica valora el **diseño experimental y la interpretación**, no superar un puntaje. Los resultados negativos bien verificados son válidos.

### Resumen

| Aspecto | Dónde está | Peso en la rúbrica |
|---|---|---|
| Formulación como MDP y trayectorias/RTG | Consigna §1, Guía §2 | 15 (junto con datos) |
| Return conditioning | Consigna §4, Guía §4.6 | 15 |
| Calidad de los logs y sesgo de selección | Consigna §5, Guía §6 | 10 |
| γ = 0 vs γ > 0; π vs máximo de Q | Consigna §2.2, Guía §5.8 | dentro de IQL (15) |
| Rigor: sin fugas, val/test, semillas | Consigna §3, Guía §7–8 | dentro de Baselines y evaluación (20) |
| Reporte, comparación DT/IQL, bitácora | Consigna §6 | 10 |

### Fuentes

- [CONSIGNA.md](CONSIGNA.md), secciones 2–6
- [GUIA.md](GUIA.md), secciones 2, 4.6, 5.8, 6, 7, 8 y 9
- [RUBRICA.md](RUBRICA.md)
- [protocolo/PROTOCOLO.md](protocolo/PROTOCOLO.md)
