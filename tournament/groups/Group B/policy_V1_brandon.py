
 
import numpy as np
from connect4.policy import Policy #importa la clase base Policy ,el agente hereda para que el entorno de el torneo pueda llamarlo correctamente


class BrandonAgent(Policy):
    ROWS = 6
    COLS = 7
    EMPTY = 0
    RED = -1
    YELLOW = 1
    col_order = [3, 2, 4, 1, 5, 0, 6]  # define el orden de preferencia de columnas.

    def mount(self, timeout=None) -> None:       # configura la variable ante de iniciar
        pass
    
    def act(self, s: np.ndarray) -> int: #act recibe el estado actual del tablero s y retorna una columna entre 0 y 6
        board = np.array(s, copy=True)
        legal = self.legal_actions(board) #obtiene las columnas legales 
       
        me = self.current_player(board) #obtiene con cual ficha esta jugando el agente
        opp = -me
        
        for col in self.ordered_columns(legal): # para las legales
            if self.is_winning_move(board, col, me): # mira si alguna es una jugada ganadora
                return int(col)                    
        
        for col in self.ordered_columns(legal):
            if self.is_winning_move(board, col, opp):  # si el oponende tiene jugada ganadora
                return int(col)                   # devuelve esa columna 

        return int(self.ordered_columns(legal)[0]) #juega por defecto en las columnas preferidas


    def legal_actions(self, board: np.ndarray): #devuelve una lista de las columnas legales
        return [c for c in range(self.COLS) if board[0, c] == self.EMPTY] # si la columa de arriba esta vacia, es legal

    def ordered_columns(self, legal):         #ordena las columnas legales con el orden de preferencia
        legal_set = set(legal)
        return [c for c in self.col_order if c in legal_set]

    def current_player(self, board: np.ndarray) -> int: #determina quien debe jugar contando las fichas
        red_count = int(np.sum(board == self.RED))      #Asumiendo que inicia el rojo
        yellow_count = int(np.sum(board == self.YELLOW))

        if red_count == yellow_count:
            return self.RED
        return self.YELLOW

    def drop_piece(self, board: np.ndarray, col: int, piece: int) -> np.ndarray: #simula colocar la ficha
        next_board = board.copy() 

        for row in range(self.ROWS - 1, -1, -1):
            if next_board[row, col] == self.EMPTY:
                next_board[row, col] = piece
                break

        return next_board

    def is_winning_move(self, board: np.ndarray, col: int, piece: int) -> bool:
        if col not in self.legal_actions(board): #si la columna no es legal, no es una jugada ganadora
            return False

        next_board = self.drop_piece(board, col, piece) #simulación 
        return self.has_four(next_board, piece)     #revisa si con esa simulación se consigue 4 en linea

    def has_four(self, board: np.ndarray, piece: int) -> bool:
        directions = [
            (0, 1),  #horizontal quieto y a la derecha
            (1, 0),  #vertical
            (1, 1),  #diagonal \ baja y despues a la derecha
            (1, -1), #diagonal / baja y despues a la izquierda
        ]

        for row in range(self.ROWS):
            for col in range(self.COLS): #recorre el tablero como si cada punto fuera el inicio de una linea de 4
                if board[row, col] != piece: #si la ficha no es de el jugador coninua
                    continue

                for dr, dc in directions: #revisa las 4 direcciones posibles
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