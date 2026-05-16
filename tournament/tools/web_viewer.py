import os
import sys
import time
import inspect
import importlib.util
from pathlib import Path

# Esto ayuda a que Streamlit encuentre la carpeta connect4/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st


ROWS = 6
COLS = 7
EMPTY = 0

# Formato consistente con el torneo:
# primer jugador = -1
# segundo jugador = 1
P1 = -1
P2 = 1


# ------------------------------------------------------------
# Agentes internos de prueba
# ------------------------------------------------------------

class RandomAgent:
    def __init__(self, seed=123):
        self.rng = np.random.default_rng(seed)

    def mount(self):
        pass

    def act(self, s: np.ndarray) -> int:
        legal = legal_actions(s)
        return int(self.rng.choice(legal))


class CenterAgent:
    def mount(self):
        pass

    def act(self, s: np.ndarray) -> int:
        legal = legal_actions(s)
        order = [3, 2, 4, 1, 5, 0, 6]

        for c in order:
            if c in legal:
                return c

        return int(legal[0])


# ------------------------------------------------------------
# Lógica básica de Connect-4 para visualización local
# ------------------------------------------------------------

def new_board():
    return np.zeros((ROWS, COLS), dtype=int)


def legal_actions(board: np.ndarray):
    return [c for c in range(COLS) if board[0, c] == EMPTY]


def apply_move(board: np.ndarray, col: int, player: int):
    new = board.copy()

    for r in range(ROWS - 1, -1, -1):
        if new[r, col] == EMPTY:
            new[r, col] = player
            return new

    raise ValueError(f"Columna llena: {col}")


def check_winner(board: np.ndarray):
    # Retorna -1 si gana P1, 1 si gana P2, 0 si no hay ganador.
    directions = [
        (0, 1),    # horizontal
        (1, 0),    # vertical
        (1, 1),    # diagonal \
        (1, -1),   # diagonal /
    ]

    for r in range(ROWS):
        for c in range(COLS):
            player = board[r, c]

            if player == EMPTY:
                continue

            for dr, dc in directions:
                count = 0

                for k in range(4):
                    rr = r + dr * k
                    cc = c + dc * k

                    if (
                        0 <= rr < ROWS
                        and 0 <= cc < COLS
                        and board[rr, cc] == player
                    ):
                        count += 1
                    else:
                        break

                if count == 4:
                    return int(player)

    return 0


def is_draw(board: np.ndarray):
    return len(legal_actions(board)) == 0 and check_winner(board) == 0


# ------------------------------------------------------------
# Carga dinámica de agentes desde policy.py
# ------------------------------------------------------------

def load_module_from_path(path: str):
    path = str(Path(path).resolve())

    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")

    module_name = "agent_" + str(abs(hash(path)))
    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el módulo desde: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def find_policy_class(module):
    candidates = []

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue

        if hasattr(obj, "mount") and hasattr(obj, "act"):
            candidates.append(obj)

    if not candidates:
        raise ValueError(
            "No encontré una clase de agente con métodos mount() y act(s). "
            "Revisa el policy.py."
        )

    # Preferimos clases que no se llamen ABC ni base abstracta.
    candidates = sorted(candidates, key=lambda cls: cls.__name__)
    return candidates[0]


def call_mount(agent):
    """
    Algunos agentes tienen mount().
    Otros tienen mount(timeout).
    Este helper intenta soportar ambos casos.
    """
    if not hasattr(agent, "mount"):
        return

    try:
        agent.mount()
    except TypeError:
        agent.mount(1.0)


def load_agent(agent_source: str):
    agent_source = agent_source.strip()

    if agent_source.upper() == "RANDOM":
        agent = RandomAgent()
        call_mount(agent)
        return agent

    if agent_source.upper() == "CENTER":
        agent = CenterAgent()
        call_mount(agent)
        return agent

    if agent_source.upper() == "HUMAN":
        raise ValueError("HUMAN no se carga como agente automático.")

    module = load_module_from_path(agent_source)
    cls = find_policy_class(module)
    agent = cls()
    call_mount(agent)

    return agent


# ------------------------------------------------------------
# Adaptación de estado y acciones
# ------------------------------------------------------------

def board_for_agent(board: np.ndarray, player: int, state_mode: str):
    if state_mode == "raw_board":
        return board.copy()

    if state_mode == "current_player_as_1":
        return (board * player).copy()

    return board.copy()


def normalize_action(action, indexing_mode: str):
    action = int(action)

    if indexing_mode == "0-6":
        return action

    if indexing_mode == "1-7":
        return action - 1

    return action


def get_agent_action(agent, board, player, state_mode, indexing_mode):
    state = board_for_agent(board, player, state_mode)
    raw_action = agent.act(state)
    action = normalize_action(raw_action, indexing_mode)

    return int(action), raw_action


# ------------------------------------------------------------
# Visualización
# ------------------------------------------------------------

def board_to_html(board: np.ndarray):
    html = """
    <style>
    .board {
        display: inline-block;
        background-color: #1f5fd1;
        padding: 12px;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .row {
        display: flex;
    }
    .cell {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        background-color: white;
        margin: 5px;
        border: 2px solid #164aa3;
    }
    .p1 {
        background-color: #e53935;
    }
    .p2 {
        background-color: #fdd835;
    }
    .empty {
        background-color: white;
    }
    .labels {
        display: flex;
        margin-left: 12px;
        margin-bottom: 4px;
    }
    .label {
        width: 68px;
        text-align: center;
        font-weight: 600;
        color: #333;
    }
    </style>
    """

    html += "<div class='labels'>"
    for c in range(COLS):
        html += f"<div class='label'>{c}</div>"
    html += "</div>"

    html += "<div class='board'>"

    for r in range(ROWS):
        html += "<div class='row'>"

        for c in range(COLS):
            val = board[r, c]

            if val == P1:
                cls = "cell p1"
            elif val == P2:
                cls = "cell p2"
            else:
                cls = "cell empty"

            html += f"<div class='{cls}'></div>"

        html += "</div>"

    html += "</div>"

    return html


# ------------------------------------------------------------
# Ejecución automática de partida agente vs agente
# ------------------------------------------------------------

def play_match(agent1, agent2, state_mode, indexing_mode, delay):
    board = new_board()
    player = P1
    history = []

    placeholder = st.empty()
    status = st.empty()

    for turn in range(42):
        agent = agent1 if player == P1 else agent2
        agent_name = "Agente 1 / Rojo" if player == P1 else "Agente 2 / Amarillo"

        legal = legal_actions(board)

        try:
            action, raw_action = get_agent_action(
                agent=agent,
                board=board,
                player=player,
                state_mode=state_mode,
                indexing_mode=indexing_mode,
            )
        except Exception as e:
            winner = P2 if player == P1 else P1
            return board, history, winner, f"Error en {agent_name}: {e}"

        if action not in legal:
            winner = P2 if player == P1 else P1
            msg = (
                f"{agent_name} hizo una acción inválida. "
                f"Acción original: {raw_action}, acción interpretada: {action}, legales: {legal}"
            )
            return board, history, winner, msg

        board = apply_move(board, action, player)

        history.append({
            "turn": turn + 1,
            "player": "P1 / Rojo" if player == P1 else "P2 / Amarillo",
            "raw_action": raw_action,
            "interpreted_action": action,
        })

        placeholder.markdown(board_to_html(board), unsafe_allow_html=True)
        status.info(f"Turno {turn + 1}: {agent_name} jugó columna {action}")
        time.sleep(delay)

        winner = check_winner(board)

        if winner != 0:
            return board, history, winner, "Victoria normal"

        if is_draw(board):
            return board, history, 0, "Empate"

        player *= -1

    return board, history, 0, "Empate por límite de turnos"


# ------------------------------------------------------------
# Modo interactivo: jugar contra una persona real
# ------------------------------------------------------------

def is_human_source(agent_source: str) -> bool:
    return agent_source.strip().upper() == "HUMAN"


def rerun_app():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def init_interactive_match(agent1_source, agent2_source, state_mode, indexing_mode):
    agent1_is_human = is_human_source(agent1_source)
    agent2_is_human = is_human_source(agent2_source)

    agent1 = None if agent1_is_human else load_agent(agent1_source)
    agent2 = None if agent2_is_human else load_agent(agent2_source)

    st.session_state.interactive_match = {
        "board": new_board(),
        "player": P1,
        "history": [],
        "turn": 0,
        "winner": 0,
        "msg": "En juego",
        "game_over": False,
        "agent1": agent1,
        "agent2": agent2,
        "agent1_is_human": agent1_is_human,
        "agent2_is_human": agent2_is_human,
        "state_mode": state_mode,
        "indexing_mode": indexing_mode,
    }


def current_player_is_human(match):
    if match["player"] == P1:
        return match["agent1_is_human"]

    return match["agent2_is_human"]


def get_current_agent(match):
    if match["player"] == P1:
        return match["agent1"]

    return match["agent2"]


def get_current_player_name(match):
    if match["player"] == P1:
        return "Agente 1 / Rojo"

    return "Agente 2 / Amarillo"


def apply_interactive_action(action, raw_action):
    match = st.session_state.interactive_match

    if match["game_over"]:
        return

    board = match["board"]
    player = match["player"]
    legal = legal_actions(board)
    player_name = get_current_player_name(match)

    if action not in legal:
        match["game_over"] = True
        match["winner"] = P2 if player == P1 else P1
        match["msg"] = (
            f"{player_name} hizo una acción inválida. "
            f"Acción original: {raw_action}, acción interpretada: {action}, legales: {legal}"
        )
        return

    board = apply_move(board, action, player)

    match["turn"] += 1
    match["board"] = board

    match["history"].append({
        "turn": match["turn"],
        "player": "P1 / Rojo" if player == P1 else "P2 / Amarillo",
        "raw_action": raw_action,
        "interpreted_action": action,
    })

    winner = check_winner(board)

    if winner != 0:
        match["game_over"] = True
        match["winner"] = winner
        match["msg"] = "Victoria normal"
        return

    if is_draw(board):
        match["game_over"] = True
        match["winner"] = 0
        match["msg"] = "Empate"
        return

    match["player"] *= -1


def run_agent_turns_until_human():
    if "interactive_match" not in st.session_state:
        return

    match = st.session_state.interactive_match

    while not match["game_over"] and not current_player_is_human(match):
        agent = get_current_agent(match)
        player = match["player"]

        try:
            action, raw_action = get_agent_action(
                agent=agent,
                board=match["board"],
                player=player,
                state_mode=match["state_mode"],
                indexing_mode=match["indexing_mode"],
            )
        except Exception as e:
            match["game_over"] = True
            match["winner"] = P2 if player == P1 else P1
            match["msg"] = f"Error en {get_current_player_name(match)}: {e}"
            return

        apply_interactive_action(action, raw_action)


def render_interactive_match():
    if "interactive_match" not in st.session_state:
        st.info("Presiona 'Iniciar/Reiniciar partida' para comenzar.")
        return

    run_agent_turns_until_human()

    match = st.session_state.interactive_match
    board = match["board"]

    st.subheader("Partida interactiva")
    st.markdown(board_to_html(board), unsafe_allow_html=True)

    if match["game_over"]:
        st.subheader("Resultado")

        if match["winner"] == P1:
            st.success(f"Gana Agente 1 / Rojo. Motivo: {match['msg']}")
        elif match["winner"] == P2:
            st.success(f"Gana Agente 2 / Amarillo. Motivo: {match['msg']}")
        else:
            st.warning(f"Empate. Motivo: {match['msg']}")

        st.subheader("Historial de movimientos")
        st.dataframe(match["history"], use_container_width=True)
        return

    player_name = get_current_player_name(match)
    legal = legal_actions(board)

    st.info(f"Turno de {player_name}")

    if current_player_is_human(match):
        st.markdown("### Elige una columna")

        button_cols = st.columns(COLS)

        for col in range(COLS):
            disabled = col not in legal

            if button_cols[col].button(
                f"Col {col}",
                disabled=disabled,
                key=f"human_move_{match['turn']}_{col}",
            ):
                apply_interactive_action(col, col)
                run_agent_turns_until_human()
                rerun_app()

    else:
        st.info("Turno del agente. Actualizando...")
        run_agent_turns_until_human()
        rerun_app()

    st.subheader("Historial de movimientos")
    st.dataframe(match["history"], use_container_width=True)


# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------

st.set_page_config(
    page_title="Connect-4 Agent Viewer",
    page_icon="🎮",
    layout="wide",
)

st.title("Connect-4 Agent Viewer")
st.caption("Interfaz local para enfrentar agentes o jugar contra una persona real")

with st.sidebar:
    st.header("Configuración")

    st.markdown("### Agente 1 / Rojo")
    agent1_path = st.text_input(
        "Ruta del Agente 1",
        value="groups/Group B/policy.py",
        help="También puedes escribir RANDOM, CENTER o HUMAN.",
    )

    st.markdown("### Agente 2 / Amarillo")
    agent2_path = st.text_input(
        "Ruta del Agente 2",
        value="RANDOM",
        help="También puedes escribir RANDOM, CENTER o HUMAN.",
    )

    st.markdown("### Compatibilidad")
    indexing_mode = st.selectbox(
        "Índice de columnas que retorna el agente",
        options=["0-6", "1-7"],
        index=0,
    )

    state_mode = st.selectbox(
        "Estado enviado al agente",
        options=["raw_board", "current_player_as_1"],
        index=0,
        help=(
            "raw_board envía el tablero con -1 para rojo y 1 para amarillo. "
            "current_player_as_1 multiplica el tablero por el jugador actual."
        ),
    )

    delay = st.slider(
        "Velocidad de visualización",
        min_value=0.0,
        max_value=1.5,
        value=0.25,
        step=0.05,
    )

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Tablero inicial")
    st.markdown(board_to_html(new_board()), unsafe_allow_html=True)

with col2:
    st.subheader("Uso rápido")
    st.code(
        "PYTHONPATH=. streamlit run tools/web_viewer.py\n\n"
        "Ejemplos de agentes:\n"
        "HUMAN\n"
        "RANDOM\n"
        "CENTER\n"
        "groups/Group A/policy.py\n"
        "groups/Group B/policy.py\n"
        "groups/Group C/policy.py",
        language="bash",
    )

st.divider()

human_mode = is_human_source(agent1_path) or is_human_source(agent2_path)

if human_mode:
    st.info("Modo interactivo activado. Usa HUMAN como Agente 1 o Agente 2.")

    if st.button("Iniciar/Reiniciar partida", type="primary"):
        try:
            init_interactive_match(
                agent1_source=agent1_path,
                agent2_source=agent2_path,
                state_mode=state_mode,
                indexing_mode=indexing_mode,
            )
            run_agent_turns_until_human()
            rerun_app()
        except Exception as e:
            st.error(f"No se pudo iniciar la partida interactiva: {e}")

    render_interactive_match()

else:
    if st.button("Jugar partida", type="primary"):
        try:
            agent1 = load_agent(agent1_path)
            agent2 = load_agent(agent2_path)

            final_board, history, winner, msg = play_match(
                agent1=agent1,
                agent2=agent2,
                state_mode=state_mode,
                indexing_mode=indexing_mode,
                delay=delay,
            )

            st.subheader("Resultado")

            if winner == P1:
                st.success(f"Gana Agente 1 / Rojo. Motivo: {msg}")
            elif winner == P2:
                st.success(f"Gana Agente 2 / Amarillo. Motivo: {msg}")
            else:
                st.warning(f"Empate. Motivo: {msg}")

            st.subheader("Historial de movimientos")
            st.dataframe(history, use_container_width=True)

        except Exception as e:
            st.error(f"No se pudo ejecutar la partida: {e}")