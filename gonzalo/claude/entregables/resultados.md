# Resultados — TP Decision Transformer e IQL

Tablas y curvas de resultados que acompañan a `reporte.md`. Dataset Netflix, evaluación con el protocolo común en test (salvo la Tabla 4, en validación).

- **Formato:** los modelos se reportan como media ± desvío entre 3 semillas de entrenamiento. El IC 95 % por usuario se indica en cada tabla, por separado. Las líneas base son deterministas o no dependen del entrenamiento, así que no tienen desvío entre semillas.
- **Políticas:** DT con el retorno pedido elegido en val (R̂ = 117 con logs `random`, R̂ = 99 con `mixto`). IQL con τ = 0,7 y β = 10, γ ∈ {0; 0,9}, evaluado con π y con el máximo de Q.
- **Fuente:** los archivos de `TP_OUTPUTS/netflix_<logs>/` que genera la notebook (celda 7): `final/tabla_*.md`, `final/tabla_por_grupo.csv`, `final/return_conditioning*.csv` y `final/test/seed<k>/sesiones.csv`.

---

## 1. Logs `random` (originales)

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

**Tabla 1b. Métricas secundarias, logs `random`, warm-up 0** (media ± desvío entre semillas; IC por usuario ±0,006–0,009). El NDCG@10 se mide solo en el primer paso: es un ranking estático y no depende del feedback de la sesión.

| Política | Hit rate, vis. | Hit rate, oc. | NDCG@10, vis. | NDCG@10, oc. |
|---|---:|---:|---:|---:|
| `random` | 0,459 | 0,459 | 0,459 | 0,459 |
| `popularity` | 0,429 | 0,429 | 0,412 | 0,412 |
| `item_mean` | 0,894 | 0,894 | 0,924 | 0,924 |
| `group_mean` | **0,918** | 0,894 | **0,936** | 0,924 |
| `group_posterior` | **0,918** | **0,896** | **0,936** | **0,924** |
| DT | 0,886 ± 0,002 | 0,882 ± 0,006 | 0,915 ± 0,004 | 0,880 ± 0,039 |
| IQL π, γ = 0 | 0,887 ± 0,002 | 0,847 ± 0,007 | 0,890 ± 0,001 | 0,894 ± 0,004 |
| IQL máx. Q, γ = 0 | 0,914 ± 0,001 | 0,872 ± 0,007 | 0,931 ± 0,001 | 0,918 ± 0,001 |
| IQL π, γ = 0,9 | 0,891 ± 0,005 | 0,859 ± 0,009 | 0,887 ± 0,005 | 0,837 ± 0,009 |
| IQL máx. Q, γ = 0,9 | 0,902 ± 0,005 | 0,858 ± 0,010 | 0,916 ± 0,003 | 0,911 ± 0,012 |

**Tabla 1c. Normalizado por grupo, logs `random`, warm-up 0** (media de 3 semillas). Cada grupo tiene 200 usuarios, así que el IC por usuario es más ancho que en el agregado: de ±0,008 a ±0,017. El desvío entre semillas de los modelos es ≤ 0,016 con grupo visible y ≤ 0,026 con grupo oculto. Diferencias menores a ≈ 0,02 entre celdas no son concluyentes. Fuente: `final/tabla_por_grupo.csv`.

| Política | g0 | g1 | g2 | g3 | g4 | g5 | g6 | g7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *Grupo visible* | | | | | | | | |
| `item_mean` | 0,772 | 0,714 | 0,637 | 0,647 | 0,714 | 0,735 | 0,697 | 0,593 |
| `group_mean` | **0,793** | **0,767** | **0,679** | **0,683** | **0,854** | **0,780** | **0,767** | **0,616** |
| DT | 0,771 | 0,722 | 0,635 | 0,632 | 0,700 | 0,725 | 0,701 | 0,575 |
| IQL π, γ = 0 | 0,780 | 0,760 | 0,662 | 0,630 | 0,833 | 0,688 | 0,741 | 0,534 |
| IQL máx. Q, γ = 0 | 0,778 | 0,762 | 0,673 | 0,677 | 0,798 | 0,772 | 0,754 | 0,615 |
| *Grupo oculto* | | | | | | | | |
| `item_mean` | **0,772** | 0,714 | 0,637 | 0,647 | 0,714 | **0,735** | 0,697 | 0,593 |
| `group_posterior` | 0,769 | **0,728** | **0,638** | **0,651** | 0,801 | 0,734 | **0,727** | **0,605** |
| DT | 0,757 | 0,707 | 0,636 | 0,625 | 0,697 | 0,720 | 0,694 | 0,574 |
| IQL π, γ = 0 | 0,714 | 0,677 | 0,608 | 0,565 | **0,807** | 0,607 | 0,644 | 0,524 |
| IQL máx. Q, γ = 0 | 0,719 | 0,683 | 0,634 | 0,626 | 0,763 | 0,684 | 0,677 | 0,567 |

---

## 2. Return conditioning del DT (logs `random`)

**Tabla 2. Retorno obtenido frente a pedido (test, warm-up 0, media de 3 semillas).**

| Pedido | 46 | 57 | 65 | 73 | 81 | 89 | 97 (máx.) | 107 | 117 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Obtenido, visible | 47,9 | 57,5 | 65,0 | 72,8 | 80,1 | 86,0 | 88,1 | 88,2 | 88,2 |
| Obtenido, oculto | 49,4 | 58,2 | 65,2 | 72,2 | 79,0 | 84,6 | 87,4 | 87,9 | 88,0 |
| IC 95 % por usuario, visible | ±0,24 | ±0,10 | ±0,06 | ±0,07 | ±0,13 | ±0,24 | ±0,32 | ±0,32 | ±0,33 |
| Desvío entre semillas, visible | 0,13 | 0,08 | 0,05 | 0,09 | 0,11 | 0,10 | 0,11 | 0,09 | 0,08 |
| Normalizado, visible | −0,61 | −0,34 | −0,11 | 0,14 | 0,38 | 0,59 | 0,68 | 0,68 | 0,68 |
| Normalizado, oculto | −0,54 | −0,30 | −0,09 | 0,13 | 0,36 | 0,55 | 0,66 | 0,67 | 0,68 |
| Obtenido con warm-up 5, visible (`full_return`) | 49,4 | 58,1 | 65,1 | 72,2 | 78,5 | 81,9 | 82,7 | 82,7 | 82,7 |

**Tabla 2b. Retorno obtenido por grupo, grupo visible, warm-up 0** (media de 3 semillas; fuente: `return_conditioning_por_grupo.csv`).

| Pedido | g0 | g1 | g2 | g3 | g4 | g5 | g6 | g7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 46 | 46,9 | 45,7 | 46,0 | 46,2 | **59,6** | 46,1 | 45,9 | 46,5 |
| 57 | 56,9 | 56,9 | 57,0 | 57,1 | **60,8** | 57,0 | 56,9 | 57,4 |
| 65 | 64,9 | 64,7 | 64,9 | 65,0 | 65,6 | 65,0 | 64,8 | 65,3 |
| 73 | 72,8 | 72,6 | 72,9 | 73,0 | 73,2 | 72,9 | 72,7 | 72,3 |
| 81 | 80,7 | 80,7 | 81,0 | 80,4 | 81,0 | 80,8 | 80,7 | **75,4** |
| 89 | 88,7 | 88,7 | 86,4 | 83,9 | 88,8 | 87,5 | 88,2 | 75,6 |
| 97 | 94,2 | 91,2 | 86,6 | 84,1 | 94,5 | 89,0 | 90,0 | 75,6 |
| 117 | 94,4 | 91,0 | 86,3 | 84,1 | 94,9 | 89,1 | 90,1 | 75,5 |

**Figura 1. Retorno obtenido frente a pedido, agregado y por grupo** (diagonal, banda del soporte de train entre p1 y el máximo, y retorno elegido en val).

![Return conditioning agregado](figuras/rc_random.png)
![Return conditioning por grupo](figuras/rc_random_por_grupo.png)

---

## 3. Logs `mixto`

**Tabla 3. Logs `mixto`, normalizado en test** (mismo formato que la Tabla 1, que tiene los valores con logs `random`). Media ± desvío entre 3 semillas de entrenamiento; IC por usuario ±0,004–0,006 en todas las celdas.

| Política | Visible | Oculto | Warm-up 5, vis. | Warm-up 5, oc. |
|---|---:|---:|---:|---:|
| `popularity` | 0,688 | 0,688 | 0,704 | 0,704 |
| `item_mean` | 0,672 | 0,672 | 0,695 | 0,695 |
| `group_mean` | **0,741** | 0,672 | **0,750** | 0,695 |
| `group_posterior` | **0,741** | **0,705** | **0,750** | **0,727** |
| `sistema_anterior_eps0` | 0,593 | 0,542 | 0,595 | 0,546 |
| DT (R̂ = 99) | 0,729 ± 0,002 | 0,683 ± 0,006 | 0,740 ± 0,003 | 0,694 ± 0,006 |
| IQL π, γ = 0 | 0,737 ± 0,002 | 0,667 ± 0,010 | 0,747 ± 0,002 | 0,704 ± 0,005 |
| IQL máx. Q, γ = 0 | 0,730 ± 0,004 | 0,673 ± 0,007 | 0,741 ± 0,004 | 0,690 ± 0,008 |
| IQL π, γ = 0,9 | 0,735 ± 0,001 | 0,677 ± 0,002 | 0,745 ± 0,002 | 0,702 ± 0,007 |
| IQL máx. Q, γ = 0,9 | 0,706 ± 0,006 | **0,580 ± 0,030** | 0,719 ± 0,004 | 0,646 ± 0,016 |

---

## 4. Selección en validación

**Tabla 4. Barrido de IQL en validación** (semilla 0; promedio del normalizado en las cuatro condiciones, con π y con el máximo de Q). En negrita, la configuración elegida.

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

DT: el retorno pedido se eligió con el mismo criterio. Con logs `random` ganó R̂ = 117 (0,566), en empate práctico con 107 (0,566) y 97 (0,564). Con logs `mixto` ganó R̂ = 99, el máximo de train con esos logs.

---

## 5. Curvas de entrenamiento

Semilla 0, sin recortes. Las semillas 1 y 2 están en Drive (`netflix_<logs>/<modelo>/<config>/seed<k>/curvas.png`) y son casi idénticas.

**Figura 2. DT, logs `random` y `mixto`** (cross-entropy y accuracy de train y val por época).

![Curvas del DT, logs random](figuras/curvas_dt_random_seed0.png)
![Curvas del DT, logs mixto](figuras/curvas_dt_mixto_seed0.png)

**Figura 3. IQL** (de izquierda a derecha: L_V, L_Q, L_π, NLL de behavior cloning, media y desvío de la ventaja, peso medio de AWR, fracción truncada, V medio y Q medio). γ = 0 y γ = 0,9 con logs `random`, y γ = 0,9 con logs `mixto`.

![Curvas de IQL, γ = 0, logs random](figuras/curvas_iql_g0_random_seed0.png)
![Curvas de IQL, γ = 0,9, logs random](figuras/curvas_iql_g09_random_seed0.png)
![Curvas de IQL, γ = 0,9, logs mixto](figuras/curvas_iql_g09_mixto_seed0.png)
