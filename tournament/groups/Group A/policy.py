import numpy as np
from connect4.policy import Policy
from typing import override


class RitaVersion1(Policy):

    # board
    ROWS = 6
    COLS = 7
    EMPTY = 0

    @override
    def mount(self) -> None:
        # Generador aleatorio
        self.rng = np.random.default_rng(42)

    @override
    def act(self, s: np.ndarray) -> int:
        board = np.array(s, copy=True)

        # Obtener las columnas disponibles para jugar.
        available_cols = self.get_available_cols(board)

        # si no hay columnas disponibles, retornamos 0.
        # En una partida normal esto casi no deberia pasar porque seria empate
        if not available_cols:
            return 0

        # Primera version: elegir una columna valida al azar.
        return int(self.rng.choice(available_cols))

    def get_available_cols(self, board: np.ndarray) -> list[int]:
        """
        Retorna las columnas donde todavia se puede jugar (que no esten llenas).
        """
        return [col for col in range(self.COLS) if board[0, col] == self.EMPTY]