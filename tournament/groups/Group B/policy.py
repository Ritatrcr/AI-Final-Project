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

        safe = []

        for col in self.ordered_columns(legal):
            next_board = self.drop_piece(board, col, me)
            opp_legal = self.legal_actions(next_board)

            gives_win = False

            for opp_col in opp_legal:
                if self.is_winning_move(next_board, opp_col, opp):
                    gives_win = True
                    break

            if not gives_win:
                safe.append(col)

        candidates = safe if safe else self.ordered_columns(legal)

        best_col = candidates[0]
        best_score = -10**12

        for col in candidates:
            next_board = self.drop_piece(board, col, me)
            score = self.score_board(next_board, me)

            if score > best_score:
                best_score = score
                best_col = col

        return int(best_col)

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

    def score_board(self, board: np.ndarray, me: int) -> int:
        opp = -me
        score = 0

        center = self.COLS // 2
        score += int(np.sum(board[:, center] == me)) * 8
        score -= int(np.sum(board[:, center] == opp)) * 8

        for window in self.get_windows(board):
            score += self.score_window(window, me, opp)

        return score

    def get_windows(self, board: np.ndarray):
        windows = []

        for row in range(self.ROWS):
            for col in range(self.COLS - 3):
                windows.append(board[row, col:col + 4])

        for col in range(self.COLS):
            for row in range(self.ROWS - 3):
                windows.append(board[row:row + 4, col])

        for row in range(self.ROWS - 3):
            for col in range(self.COLS - 3):
                windows.append(np.array([board[row + i, col + i] for i in range(4)]))

        for row in range(3, self.ROWS):
            for col in range(self.COLS - 3):
                windows.append(np.array([board[row - i, col + i] for i in range(4)]))

        return windows

    def score_window(self, window: np.ndarray, me: int, opp: int) -> int:
        mine = int(np.sum(window == me))
        theirs = int(np.sum(window == opp))
        empty = int(np.sum(window == self.EMPTY))

        if mine > 0 and theirs > 0:
            return 0

        if mine == 4:
            return 100000
        if mine == 3 and empty == 1:
            return 500
        if mine == 2 and empty == 2:
            return 50
        if mine == 1 and empty == 3:
            return 3

        if theirs == 4:
            return -100000
        if theirs == 3 and empty == 1:
            return -700
        if theirs == 2 and empty == 2:
            return -60
        if theirs == 1 and empty == 3:
            return -2

        return 0