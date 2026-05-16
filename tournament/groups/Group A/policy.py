import math
import time

import numpy as np
from connect4.policy import Policy



class Node:
    """
    Nodo simple para representar un estado dentro del arbol MCTS.

    Estructura.
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



class RitaVersion1(Policy):

    # board
    ROWS = 6
    COLS = 7
    EMPTY = 0

    # En este entorno el jugador 1 usa -1 y el jugador 2 usa 1.
    PLAYER_1_VALUE = -1
    PLAYER_2_VALUE = 1



    def mount(self, timeout: float | None = None) -> None:
        # Generador aleatorio
        self.rng = np.random.default_rng(42)

        # Parametro para limitar la profundidad de las simulaciones en MCTS.
        self.rollout_depth = 42

        # Constante de exploracion de UCB.
        # sqrt(2) es un valor clasico para balancear exploracion y explotacion.
        self.exploration_c = math.sqrt(2)

        # Presupuesto de MCTS.
        # el agente intenta hacer 350 veces este proceso antes de decidir una columna.
        self.num_iterations = 350
        self.time_limit = 0.7



    def act(self, s: np.ndarray) -> int:

        if not hasattr(self, "num_iterations"):
            self.mount()

        # Copia de Tablero
        board = np.array(s, copy=True)

        # Obtener las columnas disponibles para jugar.
        available_cols = self.get_available_cols(board)

        # si no hay columnas disponibles, retornamos 0.
        # En una partida normal esto casi no deberia pasar porque seria empate
        if not available_cols:
            return 0
        
        current_player = self.get_current_player(board)

        # Regla tactica 1:
        # Si puedo ganar inmediatamente, juego esa columna.
        winning_move = self.find_winning_move(board, current_player)

        if winning_move is not None:
            return int(winning_move)
        
        # Regla tactica 2:
        # Si el rival puede ganar en una jugada, bloqueo esa columna.
        blocking_move = self.find_winning_move(board, self.get_opponent(current_player))

        if blocking_move is not None:
            return int(blocking_move)



        # Si no hay jugada ganadora ni bloqueo urgente, usamos MCTS.
        root = Node(
            board=board,
            player_to_move=current_player,
            player_just_moved=self.get_opponent(current_player),
            parent=None,
            action=None,
            untried_actions=available_cols.copy(),
        )

        start_time = time.time()
        iterations = 0

        while iterations < self.num_iterations and time.time() - start_time < self.time_limit:
            node = self.selection(root)
            node = self.expansion(node)
            winner = self.simulation(node.board, node.player_to_move)
            self.backpropagation(node, winner)
            iterations += 1

        # Si por alguna razon MCTS no alcanzo a expandir ningun hijo,
        # usamos la regla simple de preferir el centro.
        if not root.children:
            return int(self.choose_center_preferred_move(available_cols))

        # La accion final se elige por el hijo mas visitado.
        # Esto suele ser mas estable que elegir solo por mayor recompensa promedio.
        best_child = max(root.children, key=lambda child: child.visits)

        return int(best_child.action)
    



# MCTS---------------------------------------------------------

    def expansion(self, node: Node) -> Node:
        """
        Fase de expansion de MCTS.

        Si el nodo tiene acciones no probadas, se escoge una,
        se simula esa jugada y se crea un nuevo nodo hijo.

        Si el nodo ya es terminal o no tiene acciones pendientes,
        se retorna el mismo nodo.
        """
        # revisa si puede expandir
        if self.is_terminal(node.board) or not node.untried_actions:
            return node

        # node.untried_actions es una lista de columnas que todavía no se han expandido desde ese nodo.
        """
        antes:   [0, 1, 2, 3, 4, 5, 6]
        acción:  3
        después: [0, 1, 2, 4, 5, 6]
        """
        action_index = int(self.rng.integers(len(node.untried_actions))) # escoge una posición aleatoria dentro de la lista.
        action = node.untried_actions.pop(action_index) # saca esa accion de la lista 

        # Simulamos la jugada en el tablero. play_move() no modifica el tablero original. Crea una copia y pone la ficha ahí.
        next_board = self.play_move(node.board, action, node.player_to_move)
        """ 
        node.board es el tablero viejo.
        next_board es el tablero nuevo después de la jugada.
        """

        # Cambiamos el turno al rival.
        next_player = self.get_opponent(node.player_to_move)

        # Creamos el hijo.
        child = Node(
            board=next_board,
            player_to_move=next_player,
            player_just_moved=node.player_to_move,
            parent=node,
            action=action,
            untried_actions=self.get_available_cols(next_board),
        )

        # agrega hijo al nodo padre
        node.children.append(child)

        return child
    
  

    def simulation(self, board: np.ndarray, player_to_move: int) -> int:
        """
        Fase de simulacion de MCTS.

        Desde un tablero dado, completa una partida usando jugadas aleatorias.
        Retorna el ganador:
        -1 si gana el jugador -1
         1 si gana el jugador 1
         0 si hay empate o no se alcanza un ganador
        """


        rollout_board = board.copy()
        current_player = player_to_move
        depth = 0


        while not self.is_terminal(rollout_board) and depth < self.rollout_depth:
            available_cols = self.get_available_cols(rollout_board)

            if not available_cols:
                break

            action = int(self.rng.choice(available_cols)) # escoge accion aleatoria
            rollout_board = self.play_move(rollout_board, action, current_player) # juega esa accion en el tablero de simulacion

            current_player = self.get_opponent(current_player)
            depth += 1

        return self.get_winner(rollout_board)
    
    def backpropagation(self, node: Node, winner: int) -> None:
        """
        Fase de backpropagation de MCTS.

        Despues de una simulacion, subimos desde el nodo actual hasta la raiz
        actualizando:
        - visits: cuantas veces se visito el nodo
        - total_reward: recompensa acumulada desde la perspectiva del jugador
          que hizo la jugada que llevo a ese nodo
        """
        while node is not None:
            node.visits += 1

            if winner == 0:
                reward = 0.0
            elif winner == node.player_just_moved:
                reward = 1.0
            else:
                reward = -1.0

            node.total_reward += reward
            node = node.parent


    def selection(self, node: Node) -> Node:
        """
        Fase de seleccion de MCTS.

        Baja por el arbol mientras:
        - el nodo no sea terminal
        - el nodo ya no tenga acciones pendientes por probar

        Si el nodo tiene acciones sin probar, se detiene ahi para que expansion()
        pueda crear un nuevo hijo.
        """
        while not self.is_terminal(node.board) and len(node.untried_actions) == 0:
            if not node.children:
                return node

            node = self.best_ucb_child(node)

        return node


    # MCTS: UCB ------------------------------------------------------------
    

    def best_ucb_child(self, node: Node) -> Node:
        """
        Escoge el hijo con mejor puntaje UCB.

        UCB balancea:
        - explotacion: que tan bueno ha sido el hijo
        - exploracion: que tan poco se ha visitado
        """
        return max(node.children, key=lambda child: self.ucb_score(node, child))

    def ucb_score(self, parent: Node, child: Node) -> float:
        """
        Calcula el puntaje UCB de un hijo.

        Formula:
        UCB = promedio_recompensa + c * sqrt(log(visitas_padre) / visitas_hijo)
        """
        if child.visits == 0:
            return math.inf

        average_reward = child.total_reward / child.visits

        exploration = self.exploration_c * math.sqrt(
            math.log(parent.visits + 1) / child.visits
        )

        return average_reward + exploration


# Helpers----------------------------------------------------------
    def get_available_cols(self, board: np.ndarray) -> list[int]:
        """
        Retorna las columnas donde todavia se puede jugar (que no esten llenas).
        """
        return [col for col in range(self.COLS) if board[0, col] == self.EMPTY]
    
    def get_current_player(self, board: np.ndarray) -> int:
        """
        Detecta que jugador debe mover contando fichas.

        Reglas del entorno:
        - El jugador 1 usa -1.
        - El jugador 2 usa 1.
        - Si ambos tienen la misma cantidad de fichas, juega -1.
        - Si -1 tiene una ficha mas, juega 1.
        """
        player_1_count = int(np.sum(board == self.PLAYER_1_VALUE))
        player_2_count = int(np.sum(board == self.PLAYER_2_VALUE))

        if player_1_count <= player_2_count:
            return self.PLAYER_1_VALUE

        return self.PLAYER_2_VALUE

    def get_opponent(self, player: int) -> int:
        """
        Retorna el jugador contrario.
        """
        if player == self.PLAYER_1_VALUE:
            return self.PLAYER_2_VALUE

        return self.PLAYER_1_VALUE

    def play_move(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        """
        Simula poner una ficha en una columna.
        No modifica el tablero original.
        Retorna una copia del tablero con la nueva ficha.
        """
        new_board = board.copy()

        #5, 4, 3, 2, 1, 0
        for row in range(self.ROWS - 1, -1, -1):
            if new_board[row, col] == self.EMPTY:
                new_board[row, col] = player
                break

        return new_board
    
    def get_winner(self, board: np.ndarray) -> int:
        """
        Revisa si hay cuatro fichas consecutivas de algun jugador.

        Retorna:
        -1 si gana el jugador -1
         1 si gana el jugador 1
         0 si nadie ha ganado
        """

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
        """
        Un tablero es terminal si:
        - alguien gano
        - o ya no hay columnas disponibles
        """
        winner = self.get_winner(board)

        if winner != 0:
            return True

        return len(self.get_available_cols(board)) == 0
    
    def find_winning_move(self, board: np.ndarray, player: int) -> int | None:
        """
        Busca si el jugador tiene una jugada ganadora inmediata.

        Prueba cada columna disponible:
        - simula poner la ficha del jugador
        - revisa si con esa jugada gana
        - si gana, retorna esa columna
        """
        for col in self.get_available_cols(board):
            next_board = self.play_move(board, col, player)

            if self.get_winner(next_board) == player:
                return int(col)

        return None
    
    def choose_center_preferred_move(self, available_cols: list[int]) -> int:
        """
        Escoge una columna disponible dando prioridad al centro.

        En Connect-4 las columnas centrales suelen ser mejores porque permiten
        formar mas lineas horizontales y diagonales.
        """
        preferred_order = [3, 2, 4, 1, 5, 0, 6]

        for col in preferred_order:
            if col in available_cols:
                return int(col)

        return int(self.rng.choice(available_cols))

