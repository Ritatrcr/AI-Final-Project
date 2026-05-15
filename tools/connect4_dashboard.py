from __future__ import annotations

import importlib.util
import inspect
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Project paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
TOURNAMENT_DIR = ROOT_DIR / "tournament"

if str(TOURNAMENT_DIR) not in sys.path:
    sys.path.insert(0, str(TOURNAMENT_DIR))


try:
    from connect4.policy import Policy
except Exception as exc:
    st.error(
        "No se pudo importar connect4.policy. "
        "Revisa que exista AI-Final-Project/tournament/connect4/policy.py"
    )
    raise exc


# ============================================================
# Game constants
# ============================================================

ROWS = 6
COLS = 7
EMPTY = 0
RED = -1
YELLOW = 1


# ============================================================
# Local random agent
# ============================================================

class RandomPlayer(Policy):
    def mount(self) -> None:
        self.rng = np.random.default_rng()

    def act(self, s: np.ndarray) -> int:
        legal = [c for c in range(COLS) if s[0, c] == EMPTY]
        return int(self.rng.choice(legal))


# ============================================================
# Agent loading
# ============================================================

def load_module_from_file(file_path: Path):
    module_name = "loaded_agent_" + str(abs(hash(str(file_path))))

    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el archivo: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def find_policy_class(module: Any):
    candidates = []

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue

        try:
            if issubclass(obj, Policy) and obj is not Policy:
                candidates.append(obj)
        except TypeError:
            pass

    if not candidates:
        raise ValueError(
            "No se encontró ninguna clase que herede de connect4.policy.Policy."
        )

    return candidates[0]


def load_agent_class(agent_option: str):
    if agent_option == "Random":
        return RandomPlayer

    file_path = TOURNAMENT_DIR / "groups" / agent_option / "policy.py"

    if not file_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {file_path}")

    module = load_module_from_file(file_path)
    return find_policy_class(module)


# ============================================================
# Connect-4 engine
# ============================================================

def new_board() -> np.ndarray:
    return np.zeros((ROWS, COLS), dtype=int)


def legal_actions(board: np.ndarray) -> list[int]:
    return [c for c in range(COLS) if board[0, c] == EMPTY]


def drop_piece(board: np.ndarray, col: int, player: int) -> np.ndarray:
    next_board = board.copy()

    for row in range(ROWS - 1, -1, -1):
        if next_board[row, col] == EMPTY:
            next_board[row, col] = player
            return next_board

    return next_board


def has_four(board: np.ndarray, player: int) -> bool:
    directions = [
        (0, 1),    # horizontal
        (1, 0),    # vertical
        (1, 1),    # diagonal \
        (1, -1),   # diagonal /
    ]

    for row in range(ROWS):
        for col in range(COLS):
            if board[row, col] != player:
                continue

            for dr, dc in directions:
                count = 0

                for k in range(4):
                    r = row + dr * k
                    c = col + dc * k

                    if (
                        0 <= r < ROWS
                        and 0 <= c < COLS
                        and board[r, c] == player
                    ):
                        count += 1
                    else:
                        break

                if count == 4:
                    return True

    return False


# ============================================================
# Board rendering
# ============================================================

def board_to_html(board: np.ndarray) -> str:
    html = """
    <style>
    .board-wrap {
        margin-top: 8px;
    }
    .label-row {
        display: flex;
        margin-left: 12px;
        margin-bottom: 4px;
    }
    .label {
        width: 64px;
        text-align: center;
        font-weight: 700;
        color: #334155;
    }
    .board {
        display: inline-block;
        background: #1d4ed8;
        padding: 12px;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.22);
    }
    .row {
        display: flex;
    }
    .cell {
        width: 54px;
        height: 54px;
        border-radius: 50%;
        margin: 5px;
        border: 2px solid #0f2f75;
        background: white;
    }
    .red {
        background: #ef4444;
    }
    .yellow {
        background: #facc15;
    }
    </style>
    """

    html += "<div class='board-wrap'>"

    html += "<div class='label-row'>"
    for c in range(COLS):
        html += f"<div class='label'>{c}</div>"
    html += "</div>"

    html += "<div class='board'>"

    for row in range(ROWS):
        html += "<div class='row'>"

        for col in range(COLS):
            value = board[row, col]

            if value == RED:
                cls = "cell red"
            elif value == YELLOW:
                cls = "cell yellow"
            else:
                cls = "cell"

            html += f"<div class='{cls}'></div>"

        html += "</div>"

    html += "</div></div>"

    return html


# ============================================================
# Game execution
# ============================================================

def play_one_game(
    red_cls,
    yellow_cls,
    red_name: str,
    yellow_name: str,
):
    board = new_board()

    red_agent = red_cls()
    yellow_agent = yellow_cls()

    red_agent.mount()
    yellow_agent.mount()

    current_player = RED
    history = []

    for turn in range(1, ROWS * COLS + 1):
        agent = red_agent if current_player == RED else yellow_agent
        agent_name = red_name if current_player == RED else yellow_name

        legal = legal_actions(board)

        if not legal:
            return {
                "winner": "Draw",
                "winner_color": 0,
                "turns": turn - 1,
                "reason": "Board full",
                "board": board,
                "history": history,
            }

        try:
            action = int(agent.act(board.copy()))
        except Exception as exc:
            winner = yellow_name if current_player == RED else red_name
            winner_color = YELLOW if current_player == RED else RED

            return {
                "winner": winner,
                "winner_color": winner_color,
                "turns": turn,
                "reason": f"{agent_name} error: {exc}",
                "board": board,
                "history": history,
            }

        if action not in legal:
            winner = yellow_name if current_player == RED else red_name
            winner_color = YELLOW if current_player == RED else RED

            return {
                "winner": winner,
                "winner_color": winner_color,
                "turns": turn,
                "reason": f"{agent_name} invalid action {action}. Legal actions: {legal}",
                "board": board,
                "history": history,
            }

        board = drop_piece(board, action, current_player)

        history.append(
            {
                "turn": turn,
                "player": agent_name,
                "color": "Red" if current_player == RED else "Yellow",
                "action": action,
            }
        )

        if has_four(board, current_player):
            return {
                "winner": agent_name,
                "winner_color": current_player,
                "turns": turn,
                "reason": "Four in a row",
                "board": board,
                "history": history,
            }

        current_player = -current_player

    return {
        "winner": "Draw",
        "winner_color": 0,
        "turns": ROWS * COLS,
        "reason": "Move limit",
        "board": board,
        "history": history,
    }


def play_visual_game(
    red_cls,
    yellow_cls,
    red_name: str,
    yellow_name: str,
    delay: float,
):
    board = new_board()

    red_agent = red_cls()
    yellow_agent = yellow_cls()

    red_agent.mount()
    yellow_agent.mount()

    current_player = RED
    history = []

    board_box = st.empty()
    status_box = st.empty()
    table_box = st.empty()

    board_box.markdown(board_to_html(board), unsafe_allow_html=True)
    status_box.info(f"Inicia {red_name} como rojo.")
    time.sleep(delay)

    for turn in range(1, ROWS * COLS + 1):
        agent = red_agent if current_player == RED else yellow_agent
        agent_name = red_name if current_player == RED else yellow_name
        color_name = "Red" if current_player == RED else "Yellow"

        legal = legal_actions(board)

        if not legal:
            status_box.warning("Empate: tablero lleno.")
            return {
                "winner": "Draw",
                "winner_color": 0,
                "turns": turn - 1,
                "reason": "Board full",
                "board": board,
                "history": history,
            }

        try:
            action = int(agent.act(board.copy()))
        except Exception as exc:
            winner = yellow_name if current_player == RED else red_name
            winner_color = YELLOW if current_player == RED else RED

            status_box.error(
                f"Error de {agent_name}. Gana {winner}. Detalle: {exc}"
            )

            return {
                "winner": winner,
                "winner_color": winner_color,
                "turns": turn,
                "reason": f"{agent_name} error: {exc}",
                "board": board,
                "history": history,
            }

        if action not in legal:
            winner = yellow_name if current_player == RED else red_name
            winner_color = YELLOW if current_player == RED else RED

            status_box.error(
                f"{agent_name} hizo jugada inválida {action}. "
                f"Legales: {legal}. Gana {winner}."
            )

            return {
                "winner": winner,
                "winner_color": winner_color,
                "turns": turn,
                "reason": f"{agent_name} invalid action {action}. Legal actions: {legal}",
                "board": board,
                "history": history,
            }

        board = drop_piece(board, action, current_player)

        history.append(
            {
                "turn": turn,
                "player": agent_name,
                "color": color_name,
                "action": action,
            }
        )

        board_box.markdown(board_to_html(board), unsafe_allow_html=True)
        status_box.info(
            f"Turno {turn}: {agent_name} ({color_name}) jugó columna {action}."
        )
        table_box.dataframe(pd.DataFrame(history), use_container_width=True)

        time.sleep(delay)

        if has_four(board, current_player):
            status_box.success(
                f"Ganó {agent_name} ({color_name}) en {turn} turnos."
            )

            return {
                "winner": agent_name,
                "winner_color": current_player,
                "turns": turn,
                "reason": "Four in a row",
                "board": board,
                "history": history,
            }

        current_player = -current_player

    status_box.warning("Empate por límite de movimientos.")

    return {
        "winner": "Draw",
        "winner_color": 0,
        "turns": ROWS * COLS,
        "reason": "Move limit",
        "board": board,
        "history": history,
    }


# ============================================================
# Series and statistics
# ============================================================

def get_starting_order(
    game_id: int,
    start_mode: str,
    agent_a_name: str,
    agent_a_cls,
    agent_b_name: str,
    agent_b_cls,
):
    if start_mode == "A siempre inicia":
        return agent_a_name, agent_a_cls, agent_b_name, agent_b_cls

    if start_mode == "B siempre inicia":
        return agent_b_name, agent_b_cls, agent_a_name, agent_a_cls

    if start_mode == "Alternar":
        if game_id % 2 == 1:
            return agent_a_name, agent_a_cls, agent_b_name, agent_b_cls
        return agent_b_name, agent_b_cls, agent_a_name, agent_a_cls

    if random.random() < 0.5:
        return agent_a_name, agent_a_cls, agent_b_name, agent_b_cls

    return agent_b_name, agent_b_cls, agent_a_name, agent_a_cls


def run_series(
    agent_a_name: str,
    agent_a_cls,
    agent_b_name: str,
    agent_b_cls,
    games: int,
    start_mode: str,
):
    rows = []

    for game_id in range(1, games + 1):
        red_name, red_cls, yellow_name, yellow_cls = get_starting_order(
            game_id=game_id,
            start_mode=start_mode,
            agent_a_name=agent_a_name,
            agent_a_cls=agent_a_cls,
            agent_b_name=agent_b_name,
            agent_b_cls=agent_b_cls,
        )

        result = play_one_game(
            red_cls=red_cls,
            yellow_cls=yellow_cls,
            red_name=red_name,
            yellow_name=yellow_name,
        )

        rows.append(
            {
                "game": game_id,
                "red": red_name,
                "yellow": yellow_name,
                "winner": result["winner"],
                "winner_color": (
                    "Red"
                    if result["winner_color"] == RED
                    else "Yellow"
                    if result["winner_color"] == YELLOW
                    else "Draw"
                ),
                "turns": result["turns"],
                "reason": result["reason"],
            }
        )

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame, agent_a: str, agent_b: str):
    total = len(df)

    a_wins = int((df["winner"] == agent_a).sum())
    b_wins = int((df["winner"] == agent_b).sum())
    draws = int((df["winner"] == "Draw").sum())

    avg_turns = float(df["turns"].mean()) if total > 0 else 0.0

    summary = pd.DataFrame(
        [
            {
                "agent": agent_a,
                "wins": a_wins,
                "losses": b_wins,
                "draws": draws,
                "win_rate": a_wins / total if total else 0.0,
            },
            {
                "agent": agent_b,
                "wins": b_wins,
                "losses": a_wins,
                "draws": draws,
                "win_rate": b_wins / total if total else 0.0,
            },
        ]
    )

    return summary, avg_turns


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Connect-4 Agent Dashboard",
    page_icon="🎮",
    layout="wide",
)

st.title("Connect-4 Agent Dashboard")
st.caption("Visualización paso a paso y estadísticas para agentes de Connect-4.")

available_agents = ["Group A", "Group B", "Group C", "Random"]

with st.sidebar:
    st.header("Configuración")

    agent_a = st.selectbox(
        "Agente A",
        available_agents,
        index=1,
    )

    agent_b = st.selectbox(
        "Agente B",
        available_agents,
        index=0,
    )

    games = st.number_input(
        "Cantidad de partidas para estadísticas",
        min_value=1,
        max_value=5000,
        value=100,
        step=10,
    )

    start_mode = st.selectbox(
        "Quién inicia en estadísticas",
        [
            "Alternar",
            "A siempre inicia",
            "B siempre inicia",
            "Aleatorio",
        ],
        index=0,
    )

    visual_start = st.selectbox(
        "Quién inicia en la partida visual",
        [
            "A inicia",
            "B inicia",
            "Aleatorio",
        ],
        index=0,
    )

    delay = st.slider(
        "Velocidad de visualización",
        min_value=0.0,
        max_value=2.0,
        value=0.35,
        step=0.05,
    )

    st.divider()

    st.markdown("### Rutas")
    st.code(
        f"ROOT_DIR = {ROOT_DIR}\nTOURNAMENT_DIR = {TOURNAMENT_DIR}",
        language="text",
    )


tab_visual, tab_stats, tab_round_robin = st.tabs(
    [
        "Partida paso a paso",
        "Estadísticas",
        "Todos contra todos",
    ]
)


with tab_visual:
    st.subheader("Visualizar una sola partida")

    st.write(
        "Esta opción muestra una partida completa movimiento por movimiento, "
        "con la velocidad que configures en la barra lateral."
    )

    if st.button("Jugar partida paso a paso", type="primary"):
        try:
            agent_a_cls = load_agent_class(agent_a)
            agent_b_cls = load_agent_class(agent_b)

            if visual_start == "A inicia":
                red_name, red_cls = agent_a, agent_a_cls
                yellow_name, yellow_cls = agent_b, agent_b_cls
            elif visual_start == "B inicia":
                red_name, red_cls = agent_b, agent_b_cls
                yellow_name, yellow_cls = agent_a, agent_a_cls
            else:
                if random.random() < 0.5:
                    red_name, red_cls = agent_a, agent_a_cls
                    yellow_name, yellow_cls = agent_b, agent_b_cls
                else:
                    red_name, red_cls = agent_b, agent_b_cls
                    yellow_name, yellow_cls = agent_a, agent_a_cls

            st.markdown(
                f"**Rojo:** {red_name}  \n"
                f"**Amarillo:** {yellow_name}"
            )

            result = play_visual_game(
                red_cls=red_cls,
                yellow_cls=yellow_cls,
                red_name=red_name,
                yellow_name=yellow_name,
                delay=float(delay),
            )

            st.divider()

            if result["winner"] == "Draw":
                st.warning(
                    f"Resultado final: empate en {result['turns']} turnos. "
                    f"Razón: {result['reason']}"
                )
            else:
                st.success(
                    f"Resultado final: ganó {result['winner']} "
                    f"en {result['turns']} turnos. Razón: {result['reason']}"
                )

        except Exception as exc:
            st.error(f"Error en la partida visual: {exc}")


with tab_stats:
    st.subheader("Correr muchas partidas")

    st.write(
        "Esta opción corre varias partidas entre los dos agentes seleccionados "
        "y puede alternar quién inicia."
    )

    if st.button("Correr estadísticas", type="primary"):
        try:
            agent_a_cls = load_agent_class(agent_a)
            agent_b_cls = load_agent_class(agent_b)

            df = run_series(
                agent_a_name=agent_a,
                agent_a_cls=agent_a_cls,
                agent_b_name=agent_b,
                agent_b_cls=agent_b_cls,
                games=int(games),
                start_mode=start_mode,
            )

            summary, avg_turns = summarize_results(df, agent_a, agent_b)

            st.markdown("### Resumen")
            st.dataframe(
                summary.style.format({"win_rate": "{:.2%}"}),
                use_container_width=True,
            )

            st.metric("Promedio de turnos", f"{avg_turns:.2f}")

            st.markdown("### Detalle")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name=f"results_{agent_a}_vs_{agent_b}.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Error al correr estadísticas: {exc}")


with tab_round_robin:
    st.subheader("Todos contra todos")

    st.write(
        "Corre enfrentamientos ordenados entre Group A, Group B y Group C. "
        "Cada par se evalúa en ambos colores."
    )

    if st.button("Correr mezcla Group A / Group B / Group C", type="primary"):
        try:
            names = ["Group A", "Group B", "Group C"]
            rows = []

            for red_name in names:
                for yellow_name in names:
                    if red_name == yellow_name:
                        continue

                    red_cls = load_agent_class(red_name)
                    yellow_cls = load_agent_class(yellow_name)

                    df_pair = run_series(
                        agent_a_name=red_name,
                        agent_a_cls=red_cls,
                        agent_b_name=yellow_name,
                        agent_b_cls=yellow_cls,
                        games=int(games),
                        start_mode="A siempre inicia",
                    )

                    red_wins = int((df_pair["winner"] == red_name).sum())
                    yellow_wins = int((df_pair["winner"] == yellow_name).sum())
                    draws = int((df_pair["winner"] == "Draw").sum())
                    avg_turns = float(df_pair["turns"].mean())

                    rows.append(
                        {
                            "red": red_name,
                            "yellow": yellow_name,
                            "games": int(games),
                            "red_wins": red_wins,
                            "yellow_wins": yellow_wins,
                            "draws": draws,
                            "red_win_rate": red_wins / int(games),
                            "yellow_win_rate": yellow_wins / int(games),
                            "avg_turns": avg_turns,
                        }
                    )

            mix_df = pd.DataFrame(rows)

            st.dataframe(
                mix_df.style.format(
                    {
                        "red_win_rate": "{:.2%}",
                        "yellow_win_rate": "{:.2%}",
                        "avg_turns": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

            csv = mix_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Descargar todos contra todos CSV",
                data=csv,
                file_name="round_robin_results.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Error en todos contra todos: {exc}")