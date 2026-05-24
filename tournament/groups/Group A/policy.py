import math
import time

import numpy as np
from connect4.policy import Policy


class Node:
    """
    Nodo para representar un estado dentro del arbol MCTS.
    """

    def __init__(
        self,
        board: np.ndarray,
        player_to_move: int,
        player_just_moved: int,
        parent: "Node | None" = None,
        action: int | None = None,
        untried_actions: list[int] | None = None,
    ) -> None:
        self.board = board
        self.player_to_move = player_to_move
        self.player_just_moved = player_just_moved
        self.parent = parent
        self.action = action

        self.children: list[Node] = []
        self.untried_actions = untried_actions if untried_actions is not None else []

        self.visits = 0
        self.total_reward = 0.0


class RitaVersion2(Policy):
    """
    RitaVersion2: MCTS + UCB mejorado.

    Mejoras frente a RitaVersion1:
    - Mas rollouts.
    - Default policy tactica en simulation().
    - Expansion ordenada por heuristica.
    - Evita jugadas suicidas.
    - Detecta amenazas dobles.
    - Evalua tableros no terminales con heuristica de ventanas.
    """

    ROWS = 6
    COLS = 7
    EMPTY = 0

    # En este entorno:
    # jugador 1 = -1
    # jugador 2 = 1
    PLAYER_1_VALUE = -1
    PLAYER_2_VALUE = 1

    def mount(self, timeout: float | None = None) -> None:
        self.rng = np.random.default_rng(42)

        # Profundidad maxima de los rollouts.
        self.rollout_depth = 42

        # Constante de exploracion de UCB.
        self.exploration_c = math.sqrt(2)

        # Mas rollouts que la V1.
        self.num_iterations = 1000

        # Respeta el timeout del autocalificador si lo envia.
        if timeout is None:
            self.time_limit = 0.9
        else:
            self.time_limit = max(0.05, min(0.9, float(timeout) * 0.80))

    def act(self, s: np.ndarray) -> int:
        if not hasattr(self, "num_iterations"):
            self.mount()

        board = np.array(s, copy=True)
        available_cols = self.get_available_cols(board)

        if not available_cols:
            return 0

        if len(available_cols) == 1:
            return int(available_cols[0])

        current_player = self.get_current_player(board)
        opponent = self.get_opponent(current_player)

        # Apertura: el centro suele ser la mejor primera jugada.
        if np.count_nonzero(board) == 0 and 3 in available_cols:
            return 3

        # 1. Si puedo ganar inmediatamente, gano.
        winning_move = self.find_winning_move(board, current_player)
        if winning_move is not None:
            return int(winning_move)

        # 2. Si el rival puede ganar inmediatamente, bloqueo.
        blocking_move = self.find_winning_move(board, opponent)
        if blocking_move is not None:
            return int(blocking_move)

        # 3. Evitar jugadas que regalen victoria inmediata al rival.
        safe_actions = self.get_safe_actions(board, current_player, available_cols)
        if not safe_actions:
            safe_actions = available_cols.copy()

        # 4. Si puedo crear amenaza doble, la juego.
        double_threat_move = self.find_double_threat_move(
            board=board,
            player=current_player,
            candidate_actions=safe_actions,
        )

        if double_threat_move is not None:
            return int(double_threat_move)

        # 5. Si el rival puede crear amenaza doble, intento bloquear esa columna.
        opponent_double_threat = self.find_double_threat_move(
            board=board,
            player=opponent,
            candidate_actions=available_cols,
        )

        if opponent_double_threat is not None and opponent_double_threat in safe_actions:
            return int(opponent_double_threat)

        # 6. MCTS con acciones iniciales ordenadas por heuristica.
        root_actions = self.order_actions(board, current_player, safe_actions)

        root = Node(
            board=board,
            player_to_move=current_player,
            player_just_moved=opponent,
            parent=None,
            action=None,
            untried_actions=root_actions,
        )

        start_time = time.time()
        iterations = 0

        while (
            iterations < self.num_iterations
            and time.time() - start_time < self.time_limit
        ):
            # Tree policy: selection + expansion.
            node = self.selection(root)
            node = self.expansion(node)

            # Default policy: rollout tactico.
            final_board = self.simulation(node.board, node.player_to_move)

            # Backpropagation con resultado terminal o heuristico.
            self.backpropagation(node, final_board)

            iterations += 1

        if not root.children:
            return int(self.choose_center_preferred_move(available_cols))

        # Escoge el hijo mas visitado.
        # En empate, usa reward promedio y heuristica de la jugada.
        best_child = max(
            root.children,
            key=lambda child: (
                child.visits,
                self.average_reward(child),
                self.score_action(board, child.action, current_player),
            ),
        )

        return int(best_child.action)

    # ------------------------------------------------------------
    # MCTS
    # ------------------------------------------------------------

    def selection(self, node: Node) -> Node:
        """
        Tree policy, parte 1.

        Baja por el arbol usando UCB hasta llegar a un border state:
        un nodo no terminal con acciones sin expandir.
        """

        while not self.is_terminal(node.board) and len(node.untried_actions) == 0:
            if not node.children:
                return node

            node = self.best_ucb_child(node)

        return node

    def expansion(self, node: Node) -> Node:
        """
        Tree policy, parte 2.

        Expande una accion no probada desde el border state.
        En esta V2, las acciones ya vienen ordenadas por heuristica.
        """

        if self.is_terminal(node.board) or not node.untried_actions:
            return node

        # Tomar la accion mas prometedora pendiente.
        action = node.untried_actions.pop(0)

        next_board = self.play_move(
            board=node.board,
            col=action,
            player=node.player_to_move,
        )

        next_player = self.get_opponent(node.player_to_move)

        next_available = self.get_available_cols(next_board)
        next_safe = self.get_safe_actions(next_board, next_player, next_available)

        if not next_safe:
            next_safe = next_available

        ordered_next_actions = self.order_actions(
            board=next_board,
            player=next_player,
            actions=next_safe,
        )

        child = Node(
            board=next_board,
            player_to_move=next_player,
            player_just_moved=node.player_to_move,
            parent=node,
            action=action,
            untried_actions=ordered_next_actions,
        )

        node.children.append(child)

        return child

    def simulation(self, board: np.ndarray, player_to_move: int) -> np.ndarray:
        """
        Default policy mejorada.

        En la V1, la simulacion elegia columnas aleatorias.
        En esta V2, el rollout usa una politica tactica:
        - ganar si puede,
        - bloquear si debe,
        - evitar jugadas suicidas,
        - crear amenazas dobles,
        - preferir centro,
        - y solo usar aleatoriedad si hay empate entre buenas acciones.
        """

        rollout_board = board.copy()
        current_player = player_to_move
        depth = 0

        while not self.is_terminal(rollout_board) and depth < self.rollout_depth:
            available_cols = self.get_available_cols(rollout_board)

            if not available_cols:
                break

            action = self.choose_rollout_action(
                board=rollout_board,
                player=current_player,
                available_cols=available_cols,
            )

            rollout_board = self.play_move(
                board=rollout_board,
                col=action,
                player=current_player,
            )

            current_player = self.get_opponent(current_player)
            depth += 1

        return rollout_board

    def backpropagation(self, node: Node, final_board: np.ndarray) -> None:
        """
        Actualiza visitas y recompensa acumulada desde el nodo hasta la raiz.

        Si el rollout termino con ganador, usa recompensa terminal.
        Si no termino con ganador, usa una heuristica de tablero.
        """

        winner = self.get_winner(final_board)

        while node is not None:
            node.visits += 1

            if winner == 0:
                # Recompensa heuristica si no hubo ganador.
                raw_score = self.evaluate_board(final_board, node.player_just_moved)

                # Normalizacion para que UCB no se distorsione demasiado.
                reward = max(-1.0, min(1.0, raw_score / 1000.0))

            elif winner == node.player_just_moved:
                reward = 1.0
            else:
                reward = -1.0

            node.total_reward += reward
            node = node.parent

    # ------------------------------------------------------------
    # UCB
    # ------------------------------------------------------------

    def best_ucb_child(self, node: Node) -> Node:
        return max(node.children, key=lambda child: self.ucb_score(node, child))

    def ucb_score(self, parent: Node, child: Node) -> float:
        """
        UCB = Q_promedio + c * sqrt(log(N_padre + 1) / N_hijo)
        """

        if child.visits == 0:
            return math.inf

        average_reward = child.total_reward / child.visits

        exploration = self.exploration_c * math.sqrt(
            math.log(parent.visits + 1) / child.visits
        )

        return average_reward + exploration

    def average_reward(self, node: Node) -> float:
        if node.visits == 0:
            return 0.0

        return node.total_reward / node.visits

    # ------------------------------------------------------------
    # Politica tactica para rollouts y expansion
    # ------------------------------------------------------------

    def choose_rollout_action(
        self,
        board: np.ndarray,
        player: int,
        available_cols: list[int],
    ) -> int:
        """
        Politica de simulacion.
        Esta funcion reemplaza el rollout aleatorio puro de la V1.
        """

        opponent = self.get_opponent(player)

        # 1. Ganar si puedo.
        winning_move = self.find_winning_move(board, player)
        if winning_move is not None:
            return int(winning_move)

        # 2. Bloquear si el rival gana.
        blocking_move = self.find_winning_move(board, opponent)
        if blocking_move is not None:
            return int(blocking_move)

        # 3. Evitar jugadas suicidas.
        safe_actions = self.get_safe_actions(board, player, available_cols)
        if not safe_actions:
            safe_actions = available_cols.copy()

        # 4. Crear amenaza doble si se puede.
        double_threat_move = self.find_double_threat_move(
            board=board,
            player=player,
            candidate_actions=safe_actions,
        )

        if double_threat_move is not None:
            return int(double_threat_move)

        # 5. Ordenar por heuristica.
        ordered_actions = self.order_actions(board, player, safe_actions)

        # Mantener algo de variabilidad Monte Carlo:
        # elegir entre las dos mejores si existen.
        top_k = min(2, len(ordered_actions))

        return int(self.rng.choice(ordered_actions[:top_k]))

    def order_actions(
        self,
        board: np.ndarray,
        player: int,
        actions: list[int],
    ) -> list[int]:
        """
        Ordena acciones desde la mas prometedora a la menos prometedora.
        """

        return sorted(
            actions,
            key=lambda col: self.score_action(board, col, player),
            reverse=True,
        )

    def score_action(self, board: np.ndarray, col: int, player: int) -> float:
        """
        Puntua una accion candidata.
        Sirve para ordenar expansion y mejorar rollouts.
        """

        if col not in self.get_available_cols(board):
            return -math.inf

        opponent = self.get_opponent(player)
        next_board = self.play_move(board, col, player)

        # Ganar inmediatamente es maxima prioridad.
        if self.get_winner(next_board) == player:
            return 1_000_000

        score = 0.0

        # Bloquear victoria inmediata del rival.
        opponent_winning_move = self.find_winning_move(board, opponent)

        if opponent_winning_move is not None and col == opponent_winning_move:
            score += 500_000

        # Evitar jugadas suicidas.
        if self.is_losing_move(board, col, player):
            score -= 800_000

        # Crear doble amenaza.
        if self.count_winning_moves(next_board, player) >= 2:
            score += 200_000

        # Evitar que el rival quede con doble amenaza.
        if self.count_winning_moves(next_board, opponent) >= 2:
            score -= 250_000

        # Preferencia por centro.
        center_bonus = {
            3: 50,
            2: 30,
            4: 30,
            1: 10,
            5: 10,
            0: 0,
            6: 0,
        }

        score += center_bonus.get(col, 0)

        # Evaluacion general del tablero.
        score += self.evaluate_board(next_board, player)

        return score

    def get_safe_actions(
        self,
        board: np.ndarray,
        player: int,
        actions: list[int],
    ) -> list[int]:
        """
        Retorna acciones que no regalan victoria inmediata al rival.
        """

        safe = []

        for col in actions:
            if not self.is_losing_move(board, col, player):
                safe.append(col)

        return safe

    def is_losing_move(self, board: np.ndarray, col: int, player: int) -> bool:
        """
        Una jugada es suicida si, despues de jugarla,
        el rival puede ganar inmediatamente.
        """

        opponent = self.get_opponent(player)
        next_board = self.play_move(board, col, player)

        # Si yo gano con esta jugada, no es suicida.
        if self.get_winner(next_board) == player:
            return False

        return self.find_winning_move(next_board, opponent) is not None

    def count_winning_moves(self, board: np.ndarray, player: int) -> int:
        """
        Cuenta cuantas columnas permiten ganar inmediatamente.
        Sirve para detectar amenazas dobles.
        """

        count = 0

        for col in self.get_available_cols(board):
            next_board = self.play_move(board, col, player)

            if self.get_winner(next_board) == player:
                count += 1

        return count

    def find_double_threat_move(
        self,
        board: np.ndarray,
        player: int,
        candidate_actions: list[int],
    ) -> int | None:
        """
        Busca una jugada que deje al jugador con dos formas de ganar
        en el siguiente turno.
        """

        best_col = None
        best_score = -math.inf

        for col in candidate_actions:
            if col not in self.get_available_cols(board):
                continue

            next_board = self.play_move(board, col, player)

            if self.get_winner(next_board) == player:
                return int(col)

            winning_moves = self.count_winning_moves(next_board, player)

            if winning_moves >= 2:
                score = self.score_action(board, col, player)

                if score > best_score:
                    best_score = score
                    best_col = col

        if best_col is None:
            return None

        return int(best_col)

    # ------------------------------------------------------------
    # Heuristica de tablero
    # ------------------------------------------------------------

    def evaluate_board(self, board: np.ndarray, player: int) -> float:
        """
        Evalua el tablero desde la perspectiva de player.
        """

        opponent = self.get_opponent(player)
        winner = self.get_winner(board)

        if winner == player:
            return 100_000

        if winner == opponent:
            return -100_000

        score = 0.0

        # Control del centro.
        center_col = board[:, self.COLS // 2]
        score += 8 * int(np.sum(center_col == player))
        score -= 8 * int(np.sum(center_col == opponent))

        # Ventanas de 4.
        for window in self.get_all_windows(board):
            score += self.evaluate_window(window, player)

        return score

    def evaluate_window(self, window: list[int], player: int) -> float:
        """
        Evalua una ventana de 4 casillas.
        """

        opponent = self.get_opponent(player)

        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(self.EMPTY)

        score = 0.0

        # Patrones propios.
        if player_count == 4:
            score += 100_000
        elif player_count == 3 and empty_count == 1:
            score += 120
        elif player_count == 2 and empty_count == 2:
            score += 20
        elif player_count == 1 and empty_count == 3:
            score += 2

        # Patrones del rival.
        if opponent_count == 4:
            score -= 100_000
        elif opponent_count == 3 and empty_count == 1:
            score -= 150
        elif opponent_count == 2 and empty_count == 2:
            score -= 25
        elif opponent_count == 1 and empty_count == 3:
            score -= 2

        return score

    def get_all_windows(self, board: np.ndarray) -> list[list[int]]:
        """
        Retorna todas las ventanas posibles de 4 casillas.
        """

        windows = []

        # Horizontales.
        for row in range(self.ROWS):
            for col in range(self.COLS - 3):
                windows.append([
                    int(board[row, col]),
                    int(board[row, col + 1]),
                    int(board[row, col + 2]),
                    int(board[row, col + 3]),
                ])

        # Verticales.
        for row in range(self.ROWS - 3):
            for col in range(self.COLS):
                windows.append([
                    int(board[row, col]),
                    int(board[row + 1, col]),
                    int(board[row + 2, col]),
                    int(board[row + 3, col]),
                ])

        # Diagonales descendentes.
        for row in range(self.ROWS - 3):
            for col in range(self.COLS - 3):
                windows.append([
                    int(board[row, col]),
                    int(board[row + 1, col + 1]),
                    int(board[row + 2, col + 2]),
                    int(board[row + 3, col + 3]),
                ])

        # Diagonales ascendentes.
        for row in range(3, self.ROWS):
            for col in range(self.COLS - 3):
                windows.append([
                    int(board[row, col]),
                    int(board[row - 1, col + 1]),
                    int(board[row - 2, col + 2]),
                    int(board[row - 3, col + 3]),
                ])

        return windows

    # ------------------------------------------------------------
    # Helpers de Connect-4
    # ------------------------------------------------------------

    def get_available_cols(self, board: np.ndarray) -> list[int]:
        return [col for col in range(self.COLS) if board[0, col] == self.EMPTY]

    def get_current_player(self, board: np.ndarray) -> int:
        player_1_count = int(np.sum(board == self.PLAYER_1_VALUE))
        player_2_count = int(np.sum(board == self.PLAYER_2_VALUE))

        if player_1_count <= player_2_count:
            return self.PLAYER_1_VALUE

        return self.PLAYER_2_VALUE

    def get_opponent(self, player: int) -> int:
        if player == self.PLAYER_1_VALUE:
            return self.PLAYER_2_VALUE

        return self.PLAYER_1_VALUE

    def play_move(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        new_board = board.copy()

        for row in range(self.ROWS - 1, -1, -1):
            if new_board[row, col] == self.EMPTY:
                new_board[row, col] = player
                break

        return new_board

    def get_winner(self, board: np.ndarray) -> int:
        directions = [
            (0, 1),    # Horizontal
            (1, 0),    # Vertical
            (1, 1),    # Diagonal descendente
            (1, -1),   # Diagonal ascendente
        ]

        for row in range(self.ROWS):
            for col in range(self.COLS):
                player = board[row, col]

                if player == self.EMPTY:
                    continue

                for d_row, d_col in directions:
                    count = 0

                    for step in range(4):
                        r = row + d_row * step
                        c = col + d_col * step

                        if (
                            0 <= r < self.ROWS
                            and 0 <= c < self.COLS
                            and board[r, c] == player
                        ):
                            count += 1
                        else:
                            break

                    if count == 4:
                        return int(player)

        return 0

    def is_terminal(self, board: np.ndarray) -> bool:
        return (
            self.get_winner(board) != 0
            or len(self.get_available_cols(board)) == 0
        )

    def find_winning_move(self, board: np.ndarray, player: int) -> int | None:
        for col in self.get_available_cols(board):
            next_board = self.play_move(board, col, player)

            if self.get_winner(next_board) == player:
                return int(col)

        return None

    def choose_center_preferred_move(self, available_cols: list[int]) -> int:
        preferred_order = [3, 2, 4, 1, 5, 0, 6]

        for col in preferred_order:
            if col in available_cols:
                return int(col)

        return int(self.rng.choice(available_cols))


