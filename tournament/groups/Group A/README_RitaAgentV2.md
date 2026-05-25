# RitaAgent V2 — Agente MCTS + UCB para Connect-4

## 1. Descripción general

`RitaAgent V2` es un agente autónomo para jugar **Connect-4** dentro del entorno del torneo del curso **Fundamentos de Inteligencia Artificial**. La solución está implementada para la interfaz esperada por el torneo:

```python
mount(timeout)
act(board) -> int
```

El agente recibe un tablero `numpy.ndarray` de tamaño `6 x 7` y retorna una columna legal entre `0` y `6`.

La estrategia combina:

1. Reglas tácticas inmediatas antes de iniciar búsqueda.
2. Monte Carlo Tree Search, MCTS, como mecanismo principal de decisión.
3. UCB como política de selección dentro del árbol.
4. Rollout táctico simple como default policy.
5. Backpropagation con recompensa terminal simple.

El objetivo de diseño fue construir un agente explicable, coherente con los conceptos del curso y mejorable experimentalmente.

---

## 2. Estructura esperada de archivos

```text
tournament/
└── groups/
    └── Group A/
        ├── policy.py
        ├── rita_v1.py
        ├── rita_v2.py
        ├── entrega.ipynb
        └── README.md
```

### `policy.py`

Archivo usado por el torneo. Debe contener la versión final del agente.

### `rita_v1.py`

Primera versión funcional del agente. Usa reglas tácticas inmediatas y MCTS + UCB con un presupuesto menor de simulación.

### `rita_v2.py`

Versión mejorada del agente. Aumenta el presupuesto de búsqueda y mejora la política de simulación.

### `entrega.ipynb`

Notebook de análisis experimental. Contiene las gráficas y comparaciones usadas para validar rendimiento, configuración, debilidades y mejoras futuras.

### `README.md`

Guía de uso, descripción del agente y explicación técnica breve.

---

## 3. Requisitos

El agente no requiere datasets externos. Solo necesita el entorno base del proyecto y dependencias estándar de Python.

Dependencias principales:

```text
python >= 3.10
numpy
pandas
matplotlib
```

Instalación mínima:

```bash
pip install numpy pandas matplotlib
```

Con entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
```

En macOS o Linux, si se usa `python3`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
```

---

## 4. Ejecución

Desde la raíz del proyecto:

```bash
cd AI-Final-Project/tournament
python main.py
```

Validación sintáctica:

```bash
python -m py_compile "groups/Group A/policy.py"
python -m py_compile "groups/Group A/rita_v1.py"
python -m py_compile "groups/Group A/rita_v2.py"
```

Ejecución del notebook de análisis:

```bash
jupyter notebook "groups/Group A/entrega.ipynb"
```

---

## 5. Interfaz del agente

El agente hereda de `connect4.policy.Policy` e implementa dos métodos principales.

### `mount(timeout)`

Inicializa los parámetros del agente:

- `rng`: generador aleatorio.
- `rollout_depth`: profundidad máxima de simulación.
- `exploration_c`: constante de exploración de UCB.
- `num_iterations`: presupuesto máximo de iteraciones MCTS.
- `time_limit`: límite temporal por decisión.

### `act(board)`

Recibe el tablero actual y retorna una columna legal. La decisión se toma en tres niveles:

1. Verificar victoria inmediata.
2. Bloquear victoria inmediata del rival.
3. Ejecutar MCTS si no hay una jugada táctica urgente.

---

## 6. Convención del tablero

El entorno usa la siguiente codificación:

```text
0  -> casilla vacía
-1 -> jugador 1
 1 -> jugador 2
```

El turno se detecta contando fichas:

```python
player_1_count = np.sum(board == -1)
player_2_count = np.sum(board == 1)
```

Si ambos jugadores tienen la misma cantidad de fichas, juega `-1`. Si `-1` tiene una ficha más, juega `1`.

---

## 7. Diseño técnico del agente

### 7.1 Reglas tácticas previas a MCTS

Antes de ejecutar búsqueda, el agente evalúa dos condiciones:

1. Si puede ganar en una jugada, juega esa columna.
2. Si el rival puede ganar en una jugada, bloquea esa columna.

Estas reglas reducen errores tácticos básicos y evitan gastar MCTS en posiciones donde existe una respuesta inmediata obligatoria.

### 7.2 Monte Carlo Tree Search

Cuando no existe victoria ni bloqueo inmediato, el agente construye un árbol MCTS. Cada nodo almacena:

```python
board
player_to_move
player_just_moved
parent
action
children
untried_actions
visits
total_reward
```

El ciclo principal es:

```text
selection -> expansion -> simulation -> backpropagation
```

### 7.3 Selection con UCB

La selección usa UCB:

```text
UCB = average_reward + c * sqrt(log(parent.visits + 1) / child.visits)
```

donde:

```text
average_reward = child.total_reward / child.visits
```

UCB balancea:

- **Explotación:** elegir acciones con buena recompensa promedio.
- **Exploración:** probar acciones poco visitadas.

### 7.4 Expansion

La expansión toma una acción no explorada, simula esa jugada y crea un nodo hijo.

### 7.5 Simulation / Default policy

En la V1, el rollout era más aleatorio. En la V2, la simulación usa una default policy táctica:

1. Si el jugador puede ganar, gana.
2. Si el rival puede ganar, bloquea.
3. Si no hay amenaza inmediata, prefiere columnas centrales.

Esta mejora es válida porque las reglas antes de MCTS solo revisan el estado real actual, mientras que la default policy actúa sobre estados futuros simulados.

### 7.6 Backpropagation

Después de cada simulación, el resultado se propaga desde el nodo expandido hasta la raíz:

```text
winner == 0                  -> reward = 0
winner == player_just_moved  -> reward = 1
winner != player_just_moved  -> reward = -1
```

En cada nodo:

```python
node.visits += 1
node.total_reward += reward
```

Esto permite que UCB use una estimación empírica del valor de cada acción.

---

## 8. Versiones

### Rita V1

Primera versión funcional:

- Reglas tácticas inmediatas.
- MCTS + UCB.
- Backpropagation con recompensa `1`, `0`, `-1`.
- Menor presupuesto de búsqueda.
- Rollouts más simples.

### Rita V2

Versión optimizada:

- Aumento de iteraciones MCTS de `350` a `1000`.
- Límite de tiempo cercano a `1.4` segundos por decisión.
- Rollout táctico simple.
- Preferencia por columnas centrales.
- Mejor estabilidad de Q-values por mayor número de simulaciones.

---

## 9. Justificación de hiperparámetros

### `num_iterations = 1000`

MCTS estima el valor de las acciones mediante muestreo. Cada iteración adicional incrementa visitas y actualiza recompensas acumuladas.

El valor estimado de una acción puede interpretarse como:

```text
Q(s, a) ≈ total_reward / visits
```

Pasar de 350 a 1000 iteraciones aumenta la cantidad de muestras disponibles y reduce el ruido de las estimaciones.

### `rollout_depth = 42`

Connect-4 tiene máximo 42 jugadas. Por tanto, una profundidad de 42 permite que un rollout llegue potencialmente hasta un estado terminal o empate.

### `exploration_c = sqrt(2)`

Se usa como valor clásico para balancear exploración y explotación en UCB.

### `time_limit ≈ 1.4`

Si el entorno asigna cerca de 60 segundos para una partida completa, una cota razonable por jugada es:

```text
60 / 42 ≈ 1.43 segundos
```

---

## 10. Análisis experimental

El análisis experimental se encuentra en:

```text
entrega.ipynb
```

Incluye:

1. Comparación Rita V1 vs Rita V2.
2. Comparación contra jugador aleatorio.
3. Self-play.
4. Recursos: `num_iterations` vs win rate.
5. Costo computacional: `num_iterations` vs tiempo promedio.
6. Estimación de Q-values con distintos trials.
7. Análisis de debilidades.
8. Comparación grupal si aplica.

Métricas principales:

```text
win_rate
loss_rate
draw_rate
tiempo promedio por jugada
número promedio de movimientos
intervalo de confianza
```

---

## 11. Debilidades identificadas

### 11.1 No garantiza optimalidad

MCTS aproxima la decisión mediante simulaciones. No resuelve exhaustivamente todo el árbol de juego.

### 11.2 Costo computacional

Más rollouts pueden mejorar la estimación, pero aumentan el tiempo promedio por jugada.

### 11.3 Recompensa simple

Backpropagation usa recompensa terminal simple. No diferencia estados intermedios estratégicamente buenos o malos si el rollout no llega a victoria.

### 11.4 No reutiliza el árbol

Cada llamada a `act()` crea un nuevo árbol MCTS. Esto genera recomputación entre turnos.

### 11.5 Sin memoria de estados

No hay tabla de transposición ni cache de estados evaluados.

---

## 12. Mejoras futuras

1. Reutilización del árbol entre turnos.
2. Memoria de estados o tabla de transposición.
3. Evaluación heurística de estados no terminales.
4. Ajuste dinámico de `exploration_c`.
5. Filtro de acciones seguras antes de MCTS.
6. Búsqueda minimax/alpha-beta corta como verificación táctica previa.
7. Ablación más extensa de la default policy.

---

## 13. Commits importantes

Reemplazar los siguientes enlaces por los commits reales del repositorio:

```text
Código final del agente:
https://github.com/<usuario-o-organizacion>/<repo>/commit/<hash-final-policy>

Rita V1:
https://github.com/<usuario-o-organizacion>/<repo>/commit/<hash-rita-v1>

Rita V2:
https://github.com/<usuario-o-organizacion>/<repo>/commit/<hash-rita-v2>

Notebook de entrega:
https://github.com/<usuario-o-organizacion>/<repo>/commit/<hash-entrega-ipynb>
```

---

## 14. Resumen técnico corto

`RitaAgent V2` es un agente basado en MCTS + UCB. Antes de buscar, revisa victorias y bloqueos inmediatos. Si no hay jugadas urgentes, ejecuta MCTS con 1000 iteraciones, selección UCB, rollout táctico simple y backpropagation con recompensa `1`, `0`, `-1`. La mejora principal frente a V1 es que V2 usa más simulaciones y una default policy menos aleatoria, lo que permite estimar Q-values con menor ruido y tomar decisiones más estables.

---

## 15. Autoría

Autor: Rita  
Grupo: Group A  
Curso: Fundamentos de Inteligencia Artificial  
Proyecto: Agente Connect-4
