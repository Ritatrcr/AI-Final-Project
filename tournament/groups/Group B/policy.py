import numpy as np
from connect4.policy import Policy #importa la clase base Policy ,el agente hereda para que el entorno de el torneo pueda llamarlo correctamente


class BrandonAgent(Policy):
    ROWS = 6
    COLS = 7
    EMPTY = 0
    RED = -1
    YELLOW = 1
    col_order = [3, 2, 4, 1, 5, 0, 6]  #define el orden de preferencia de columnas.

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

        
        # si ningina de las condiciones se cumple, evalua cada columna legal con una funcion de ventanas de 4
        return int(self.best_heuristic_move(board, legal, me, opp))


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

    def best_heuristic_move(self, board: np.ndarray, legal, me: int, opp: int) -> int:
        #esta funcion prueba todas las columnas legales y escoge la que deje mejor posicion
        best_score = -float("inf")
        best_col = self.ordered_columns(legal)[0] #por defecto mantiene la preferencia por el centro

        for col in self.ordered_columns(legal):
            next_board = self.drop_piece(board, col, me) #simula mi jugada en esa columna

            score = self.score_position(next_board, me, opp) #evalua que tan bueno queda el tablero

            #seguridad tactica: evita jugadas que le regalen una victoria inmediata al rival
            for opp_col in self.legal_actions(next_board):
                if self.is_winning_move(next_board, opp_col, opp):
                    score -= 100000
                    break

            if score > best_score:
                best_score = score
                best_col = col

        return best_col

    def score_position(self, board: np.ndarray, me: int, opp: int) -> int:
        #esta funcion calcula un puntaje general del tablero usando ventanas de 4
        score = 0

        #mantiene preferencia por controlar el centro porque desde ahi hay mas combinaciones posibles
        center_col = board[:, self.COLS // 2]
        center_count = int(np.sum(center_col == me))
        score += center_count * 6

        #ventanas horizontales
        for row in range(self.ROWS):
            for col in range(self.COLS - 3):
                window = list(board[row, col:col + 4])
                score += self.evaluate_window(window, me, opp)

        #ventanas verticales
        for col in range(self.COLS):
            for row in range(self.ROWS - 3):
                window = list(board[row:row + 4, col])
                score += self.evaluate_window(window, me, opp)

        #ventanas diagonales \ 
        for row in range(self.ROWS - 3):
            for col in range(self.COLS - 3):
                window = [board[row + i, col + i] for i in range(4)]
                score += self.evaluate_window(window, me, opp)

        #ventanas diagonales /
        for row in range(3, self.ROWS):
            for col in range(self.COLS - 3):
                window = [board[row - i, col + i] for i in range(4)]
                score += self.evaluate_window(window, me, opp)

        return score

    def evaluate_window(self, window, me: int, opp: int) -> int:
        #esta funcion evalua una ventana de 4 casillas
        #premia oportunidades propias y penaliza oportunidades del rival
        me_count = window.count(me)
        opp_count = window.count(opp)
        empty_count = window.count(self.EMPTY)

        score = 0

        #ataque: ventanas que favorecen a mi agente
        if me_count == 4:
            score += 100000
        elif me_count == 3 and empty_count == 1:
            score += 80
        elif me_count == 2 and empty_count == 2:
            score += 20
        elif me_count == 1 and empty_count == 3:
            score += 1

        #defensa: ventanas que favorecen al rival
        if opp_count == 4:
            score -= 100000
        elif opp_count == 3 and empty_count == 1:
            score -= 90
        elif opp_count == 2 and empty_count == 2:
            score -= 25

        return score