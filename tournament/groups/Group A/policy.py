import numpy as np
from connect4.policy import Policy
from typing import override

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

    @override
    def mount(self) -> None:
        # Generador aleatorio
        self.rng = np.random.default_rng(42)



    @override
    def act(self, s: np.ndarray) -> int:

        #Copia de Tablero
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

        # Regla simple 3:
        # Si no hay una urgencia tactica, prefiero jugar cerca del centro.
        return int(self.choose_center_preferred_move(available_cols))
    


#Helpers
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



#MCTS---------------------------------------------------------

    def expansion(self, node: Node) -> Node:
        """
        Fase de expansion de MCTS.

        Si el nodo tiene acciones no probadas, se escoge una,
        se simula esa jugada y se crea un nuevo nodo hijo.

        Si el nodo ya es terminal o no tiene acciones pendientes,
        se retorna el mismo nodo.
        """
        if self.is_terminal(node.board) or not node.untried_actions:
            return node

        # Escogemos una accion no probada.
        action_index = int(self.rng.integers(len(node.untried_actions)))
        action = node.untried_actions.pop(action_index)

        # Simulamos la jugada en el tablero.
        next_board = self.play_move(node.board, action, node.player_to_move)

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

        node.children.append(child)

        return child