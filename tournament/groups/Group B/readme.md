# BrandonAgent V3 — Agente heurístico para Connect-4

## 1. Descripción general

Este repositorio contiene la entrega individual del agente **BrandonAgent** para el reto de Connect-4 del curso **Fundamentos de Inteligencia Artificial**.

El agente final está implementado en `policy.py` y corresponde a una política determinista basada en reglas tácticas y evaluación heurística de posiciones. La idea central es decidir la columna a jugar mediante una jerarquía de prioridad:

1. **Ganar inmediatamente** si existe una columna que complete cuatro fichas.
2. **Bloquear al rival** si el oponente puede ganar en su siguiente turno.
3. **Evaluar posiciones futuras de forma heurística** usando ventanas de cuatro casillas.
4. **Evitar regalar una victoria inmediata** al rival después de la jugada propia.
5. **Usar preferencia por columnas centrales** como criterio de desempate.

La versión final no realiza entrenamiento offline ni búsqueda profunda. Su decisión es online, rápida y explicable: simula cada acción legal, evalúa el tablero resultante y selecciona la columna con mejor puntaje.

---

## 2. Estructura de archivos

La carpeta individual debe tener una estructura similar a la siguiente:

```text
Group B/
├── policy.py                # Agente final: BrandonAgent V3 con heurística de ventanas
├── policy_V1_brandon.py      # Versión base: ganar, bloquear y preferir centro
├── entrega.ipynb             # Notebook de análisis empírico y gráficas
└── readme.md                 # Guía de uso y descripción técnica del agente
```

> Nota: `policy.py` debe conservar ese nombre porque normalmente es el archivo que carga el torneo. Las versiones adicionales se usan para análisis comparativo en `entrega.ipynb`.

---

## 3. Versiones evaluadas

| Versión | Archivo | Descripción |
|---|---|---|
| V1 | `policy_V1_brandon.py` | Política táctica básica: gana si puede, bloquea si debe y juega por preferencia de centro. |
| V3 / Final | `policy.py` | Política táctica + heurística posicional por ventanas de cuatro casillas. También penaliza jugadas que permiten victoria inmediata del rival. |

---

## 4. Diseño técnico del agente final

El agente representa el tablero como una matriz `6 x 7`, donde:

```text
0  = celda vacía
-1 = ficha roja
1  = ficha amarilla
```

La función principal es:

```python
def act(self, s: np.ndarray) -> int:
```

Esta función recibe el estado actual del tablero y retorna una columna legal entre `0` y `6`.

### 4.1. Reglas tácticas inmediatas

Primero se evalúan jugadas terminales inmediatas:

```text
Si mi agente puede ganar ahora, juega esa columna.
Si el rival puede ganar en su próximo turno, bloquea esa columna.
```

Esto evita errores tácticos básicos y garantiza que el agente aproveche victorias directas.

### 4.2. Heurística por ventanas de cuatro

Cuando no existe victoria ni bloqueo inmediato, el agente evalúa cada columna legal. Para cada acción:

1. Simula poner su ficha en esa columna.
2. Calcula un puntaje del tablero resultante.
3. Revisa si esa jugada permite que el rival gane inmediatamente.
4. Selecciona la columna con mayor puntaje.

La evaluación del tablero se basa en ventanas de cuatro casillas:

```text
Horizontales
Verticales
Diagonales descendentes (\)
Diagonales ascendentes (/)
```

Cada ventana se puntúa así:

```text
4 fichas propias              → +100000
3 fichas propias + 1 vacío    → +80
2 fichas propias + 2 vacíos   → +20
1 ficha propia + 3 vacíos     → +1

4 fichas rivales              → -100000
3 fichas rivales + 1 vacío    → -90
2 fichas rivales + 2 vacíos   → -25
```

Además, el agente suma una bonificación por controlar la columna central, porque desde el centro existen más combinaciones potenciales de cuatro en línea.

---

## 5. Cómo ejecutar el torneo

Desde la raíz del proyecto, entrar a la carpeta `tournament` y ejecutar:

```bash
cd tournament
python main.py
```

Si estás usando un Python específico, asegúrate de que ese mismo entorno tenga instalado `numpy`:

```bash
python -m pip install numpy pandas matplotlib
```

---

## 6. Cómo ejecutar el notebook de análisis

Desde la carpeta del grupo:

```bash
cd "tournament/groups/Group B"
jupyter notebook entrega.ipynb
```

También puede abrirse desde VS Code o JupyterLab.

El notebook genera:

1. Comparación de **V1 vs V3** contra un agente aleatorio.
2. Evaluación en ambos colores: rojo y amarillo.
3. Auto-desempeño: agente contra sí mismo.
4. Tiempo promedio por jugada.
5. Análisis opcional de una variable numérica de configuración: peso de control del centro.

---

## 7. Métricas usadas

El análisis empírico usa las siguientes métricas:

| Métrica | Descripción |
|---|---|
| `win_rate` | Proporción de partidas ganadas por el agente evaluado. |
| `loss_rate` | Proporción de partidas perdidas. |
| `draw_rate` | Proporción de empates. |
| `avg_moves` | Número promedio de movimientos por partida. |
| `avg_time_per_move_ms` | Tiempo promedio de decisión por jugada en milisegundos. |

---

## 8. Conclusión esperada

La hipótesis experimental es que `policy.py` debería superar a `policy_V1_brandon.py` porque la V1 solo toma decisiones tácticas inmediatas, mientras que la V3 evalúa la calidad posicional del tablero cuando no hay una jugada obligatoria.

En particular, se espera que la V3:

- Mantenga buen desempeño contra el agente aleatorio.
- Reduzca decisiones miopes en posiciones no terminales.
- Genere mejores amenazas ofensivas y defensivas.
- Conserve un tiempo de respuesta bajo al no usar búsqueda profunda.

---

## 9. Limitaciones y mejoras futuras

La principal limitación es que el agente final solo analiza una jugada propia hacia adelante, con una revisión adicional para evitar victoria inmediata del rival. No simula secuencias largas como:

```text
yo juego → rival responde → yo respondo → rival responde
```

Por eso, una mejora futura sería incorporar:

1. **Minimax con poda Alpha-Beta** para búsqueda a profundidad limitada.
2. **MCTS con UCT** para estimar mejores acciones mediante simulaciones.
3. **Ajuste automático de pesos heurísticos** para optimizar los valores de la función de evaluación.
4. **Detección explícita de amenazas dobles**, donde una jugada crea dos formas simultáneas de ganar.
