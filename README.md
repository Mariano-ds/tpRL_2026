# Sistema de Recomendación con Offline RL: Decision Transformer e IQL

## 📋 Información Administrativa y Fechas Clave
* **Materia:** Aprendizaje por Refuerzos — DiploDatos 2026
* **Docentes:** [Juan Barsce] (`jbarsce@gmail.com`) | [Jorge Palombarini] (`palombarini@gmail.com`)

| Hito | Fecha Límite | Entregable | Estado |
| :--- | :--- | :--- | :--- |
| Definición de baselines y entorno | DD/MM/2026 | Ejecución protocolo base | 🟡 En curso |
| Entrenamiento DT e IQL | DD/MM/2026 | Checkpoints + métricas | ⚪ Pendiente |
| Entrega Final y Defensa | DD/MM/2026 | Repo consolidado + Informe | ⚪ Pendiente |

## 🎯 Entrega y explicación

- Notebooks ejecutados o scripts equivalentes para exploración, entrenamiento de ambos modelos, evaluación, return conditioning y calidad de los logs.
- Código, versiones, semillas, configuración y checkpoints; no duplicar los datasets proporcionados.
- Tabla de resultados de DT, IQL y líneas base, y curvas relevantes.
- Reporte de 4–6 páginas o Markdown equivalente: problema, protocolo, decisiones, evidencia y límites. Incluir una comparación razonada entre DT e IQL.
- Bitácora breve de contribuciones y decisiones verificadas. IA opcional: indicar si se utilizó y qué se comprobó; no entregar conversaciones completas.

## 🗂️ Organización del repositorio

El repositorio se organiza principalmente por integrante. Cada integrante trabajará sobre sus propios archivos y subirá sus notebooks y materiales al repositorio.

La estructura general será:

    /
    ├── README.md
    │
    ├── gonzalo/
    │   ├── fernando_p1.ipynb
    │   └── ...
    │
    ├── juliana/
    │   └── ...
    │
    ├── mariano/
    │   └── ...
    │
    └── trabajo_comun/
        └── entregas/
            ├── tp_final.ipynb
            └── ...

### 📁 Carpetas de cada integrante

Cada integrante tendrá una carpeta propia para almacenar sus trabajos.

Los notebooks se identificarán indicando el nombre y el trabajo correspondiente. Por ejemplo:

    integrante_1_p1.ipynb
    integrante_1_p2.ipynb
    integrante_1_p3.ipynb

Además de los archivos `.ipynb`, podrán incorporarse otros formatos necesarios para el proyecto, como `.pdf` para infografías, presentaciones u otros materiales.

### 📁 Carpeta `trabajo_comun`

La carpeta `trabajo_comun` contendrá los archivos que sean producto de la integración de los trabajos individuales, incluyendo el notebook común/final del proyecto.

La idea es que cada integrante trabaje principalmente **sobre sus propios archivos** y que los aportes sean posteriormente integrados en los archivos comunes.

Se incluye un subdirectorio para guardar las `entregas` y otro para mantener un registro de las `correcciones` hechas por la mentora.

## 📌 Forma de trabajo

El flujo de trabajo será sencillo:

1. Cada integrante trabaja principalmente en los archivos de su propia carpeta.
2. Los notebooks pueden desarrollarse utilizando **Google Colab** y **Google Drive**.
3. Una vez realizado un aporte, se sube la versión correspondiente al repositorio de GitHub.
4. GitHub conserva el historial de versiones de los archivos, por lo que no es necesario crear múltiples copias como `final_v2`, `final_v3`, etc.
5. Los aportes individuales se utilizarán posteriormente para construir y actualizar los archivos comunes.
6. Se procurará evitar que varias personas modifiquen simultáneamente el mismo archivo común para reducir conflictos y pérdida de cambios.

### 📎 Versionado

GitHub permite conservar el historial de modificaciones de los archivos. Por este motivo, se recomienda mantener un mismo archivo y actualizarlo en lugar de crear diferentes versiones manuales.

Por ejemplo, se utilizará:

    integrante_1_p1.ipynb

y no:

    integrante_1_p1_v2.ipynb
    integrante_1_p1_final.ipynb
    integrante_1_p1_final2.ipynb

Cada actualización quedará registrada en el historial del repositorio.
