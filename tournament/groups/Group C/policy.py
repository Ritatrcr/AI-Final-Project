import time
import numpy as np
from connect4.policy import Policy

# Override solo existe en Python 3.12+. Este try/except lo hace compatible
# con cualquier version de Python que use Gradescope.
try:
    from typing import override
except ImportError:
    def override(f):
        return f

# ─────────────────────────────────────────────────────────────────
#  CONCEPTO DEL CURSO: OPI + Minimax
#
#  Connect-4 es un OPI (dos agentes, suma cero):
#    S  = tablero numpy 6x7          (estado del MDP)
#    A  = columnas 0-6 disponibles   (acciones del MDP)
#    T  = colocar ficha y cambiar turno (transicion determinista)
#    R  = +1 gano, -1 pierdo         (recompensa terminal)
#
#  La solucion optima en un OPI es MINIMAX:
#    - Yo maximizo mi recompensa esperada.
#    - El oponente minimiza mi recompensa esperada.
#    - Alternamos hasta llegar al estado terminal o al limite depth.
# ─────────────────────────────────────────────────────────────────

class MinimaxAgent(Policy):

    TIME_BUDGET = 60.0                     # Segundos totales por partida
    _COL_WEIGHTS = [1, 1, 2, 2, 2, 1, 1]   # peso de cada columna (centro = mas valioso)

    @override
    def mount(self, timeout=None) -> None:
        # Gradescope pasa el timeout como argumento: policy.mount(POLICY_ACTION_TIMEOUT)
        # Si lo recibe, lo usamos como presupuesto de tiempo.
        if timeout is not None:
            self.TIME_BUDGET = float(timeout)
        self._start_time = time.time()

    # ── Gestion de tiempo ───────────────────────────────────────
    def _remaining_time(self) -> float:
        # Fallback: si act() es llamado sin mount() previo, inicializamos el reloj aqui.
        if not hasattr(self, '_start_time'):
            self._start_time = time.time()
        return max(0.0, self.TIME_BUDGET - (time.time() - self._start_time))

    def _compute_depth(self, board: np.ndarray) -> int:
        # Profundiad dinamica segun tiempo y jugadas restantes.
        remaining = self._remaining_time()
        empty     = int(np.sum(board == 0))

        if remaining >= 60:
            depth = 7
        elif remaining >= 10:
            depth = 5
        else:
            depth = 3

        if empty <= 5:
            depth = empty
        elif empty <= 10:
            depth = max(depth, 8)

        return min(depth, empty)

    # ── Helpers del MDP (S, A, T) ───────────────────────────────
    def _valid_cols(self, board: np.ndarray) -> list:
        # Se ordenan las columnas disponibles segun su peso (centrales primero).
        return sorted(
            [c for c in range(7) if board[0, c] == 0],
            key=lambda c: abs(c - 3)
        )

    def _place(self, board: np.ndarray, col: int, color: int) -> np.ndarray:
        # T(s, a): simula colocar 'color' en 'col' y retorna el nuevo estado s'.
        b = board.copy()
        for r in range(5, -1, -1):
            if b[r, col] == 0:
                b[r, col] = color
                break
        return b

    def _winner(self, board: np.ndarray) -> int:
        # Detecta si hay ganador
        rows, cols = 6, 7
        for r in range(rows):
            for c in range(cols):
                p = board[r, c]
                if p == 0:
                    continue
                if c+3 < cols and all(board[r, c+i] == p for i in range(4)):
                    return p
                if r+3 < rows and all(board[r+i, c] == p for i in range(4)):
                    return p
                if r+3 < rows and c+3 < cols and all(board[r+i, c+i] == p for i in range(4)):
                    return p
                if r+3 < rows and c-3 >= 0 and all(board[r+i, c-i] == p for i in range(4)):
                    return p
        return 0

    # ── Value Function V(s) ─────────────────────────────────────
    def _window_score(self, window: np.ndarray, me: int) -> int:
        """Puntua una ventana de 4 celdas segun cuantas fichas mias/rival hay."""
        opp    = -me
        mine   = int(np.sum(window == me))
        theirs = int(np.sum(window == opp))
        empty  = int(np.sum(window == 0))
        if mine > 0 and theirs > 0: return 0
        if mine == 3 and empty == 1: return 50
        if mine == 2 and empty == 2: return 10
        if mine == 1 and empty == 3: return 2
        if theirs == 3 and empty == 1: return -80
        if theirs == 2 and empty == 2: return -10
        return 0

    def _evaluate(self, board: np.ndarray, me: int) -> float:
        # Valora el tablero usando ventanas de 4. +1 gano, -1 pierdo, score en intermedios.
        w = self._winner(board)
        if w == me:  return 1.0
        if w == -me: return -1.0
        score = 0
        center = board[:, 3]
        score += int(np.sum(center == me))  * 6
        score -= int(np.sum(center == -me)) * 6
        for r in range(6):
            for c in range(4):
                score += self._window_score(board[r, c:c+4], me)
        for c in range(7):
            for r in range(3):
                score += self._window_score(board[r:r+4, c], me)
        for r in range(3):
            for c in range(4):
                score += self._window_score(np.array([board[r+i, c+i] for i in range(4)]), me)
        for r in range(3):
            for c in range(3, 7):
                score += self._window_score(np.array([board[r+i, c-i] for i in range(4)]), me)
        return score / 1000.0

    # ── Minimax con Alpha-Beta Pruning ──────────────────────────
    def _minimax(self, board: np.ndarray, depth: int, is_max: bool, me: int,
                 alpha: float = -float("inf"), beta: float = float("inf")) -> float:
        # Maximiza mi recompensa esperada con poda alpha-beta para explorar mas profundo.
        winner = self._winner(board)
        valid  = self._valid_cols(board)

        if winner != 0 or len(valid) == 0 or depth <= 0:
            return self._evaluate(board, me)

        if is_max:
            best = -float("inf")
            for col in valid:
                val  = self._minimax(self._place(board, col, me), depth - 1, False, me, alpha, beta)
                best = max(best, val)
                alpha = max(alpha, best)
                if alpha >= beta: break
            return best
        else:
            best = float("inf")
            for col in valid:
                val  = self._minimax(self._place(board, col, -me), depth - 1, True, me, alpha, beta)
                best = min(best, val)
                beta = min(beta, best)
                if alpha >= beta: break
            return best

    # ── Politica pi(s) → a ──────────────────────────────────────
    @override
    def act(self, s: np.ndarray) -> int:
        # Dado el estado actual del tablero, retorna la mejor columna a jugar segun minimax.
        # Optimizacion: primer movimiento siempre al centro
        if np.all(s == 0):
            return 3

        # Detectar color: Rojo (-1) mueve primero.
        reds    = int(np.sum(s == -1))
        yellows = int(np.sum(s == 1))
        me      = -1 if reds == yellows else 1

        depth      = self._compute_depth(s)
        valid      = self._valid_cols(s)
        best_score = -float("inf")
        best_col   = valid[0]

        for col in valid:
            score = self._minimax(self._place(s, col, me), depth - 1, False, me, -float("inf"), float("inf"))
            if score > best_score:
                best_score = score
                best_col   = col

        return best_col