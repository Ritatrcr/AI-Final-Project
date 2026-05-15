import numpy as np
from connect4.policy import Policy


class BrandonAgent(Policy):
    ROWS = 6
    COLS = 7
    EMPTY = 0
    RED = -1
    YELLOW = 1

    def mount(self) -> None:
        self.col_order = [3, 2, 4, 1, 5, 0, 6]

    def act(self, s: np.ndarray) -> int:
        board = np.array(s, copy=True)
        legal = self.legal_actions(board)

        if not legal:
            return 0

        me = self.current_player(board)
        opp = -me

        for col in self.ordered_columns(legal):
            if self.is_winning_move(board, col, me):
                return int(col)

        for col in self.ordered_columns(legal):
            if self.is_winning_move(board, col, opp):
                return int(col)

        return int(self.ordered_columns(legal)[0])

    def legal_actions(self, board: np.ndarray):
        return [c for c in range(self.COLS) if board[0, c] == self.EMPTY]

    def ordered_columns(self, legal):
        legal_set = set(legal)
        return [c for c in self.col_order if c in legal_set]

    def current_player(self, board: np.ndarray) -> int:
        red_count = int(np.sum(board == self.RED))
        yellow_count = int(np.sum(board == self.YELLOW))

        if red_count == yellow_count:
            return self.RED

        return self.YELLOW

    def drop_piece(self, board: np.ndarray, col: int, piece: int) -> np.ndarray:
        next_board = board.copy()

        for row in range(self.ROWS - 1, -1, -1):
            if next_board[row, col] == self.EMPTY:
                next_board[row, col] = piece
                break

        return next_board

    def is_winning_move(self, board: np.ndarray, col: int, piece: int) -> bool:
        if col not in self.legal_actions(board):
            return False

        next_board = self.drop_piece(board, col, piece)
        return self.has_four(next_board, piece)

    def has_four(self, board: np.ndarray, piece: int) -> bool:
        directions = [
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1),
        ]

        for row in range(self.ROWS):
            for col in range(self.COLS):
                if board[row, col] != piece:
                    continue

                for dr, dc in directions:
                    count = 0

                    for k in range(4):
                        r = row + dr * k
                        c = col + dc * k

                        if 0 <= r < self.ROWS and 0 <= c < self.COLS and board[r, c] == piece:
                            count += 1
                        else:
                            break

                    if count == 4:
                        return True

        return False