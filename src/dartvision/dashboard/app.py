"""Entrypoint della dashboard Streamlit.

Legge solo dal database (modulo ``data``): non importa nulla dai
livelli di visione/detection. Questo e' l'unico modo in cui questo
file dovrebbe cambiare quando si tocca la pipeline CV: mai.

Uso:
    uv run streamlit run src/dartvision/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from dartvision.analytics import compute_stats
from dartvision.dashboard import components, data, theme
from dartvision.dashboard.board_figure import build_board_figure

REFRESH_SECONDS = 2

st.set_page_config(page_title="DartVision", page_icon="🎯", layout="wide")
st.markdown(theme.CUSTOM_CSS, unsafe_allow_html=True)

st.title("🎯 DartVision")
st.caption("Live scoring &amp; analytics per il gioco delle freccette, da una singola webcam.")

config = data.get_config()

games = data.list_games()
if not games:
    st.warning(
        "Nessuna partita nel database. Popolalo con "
        "`uv run python scripts/process_video.py --video <file.mp4> --player1 <nome> --player2 <nome>` "
        "oppure `uv run python scripts/seed_demo_data.py` per dati dimostrativi."
    )
    st.stop()

if "selected_game_id" not in st.session_state:
    st.session_state.selected_game_id = games[-1].id

selected_id = components.render_games_sidebar(games, st.session_state.selected_game_id)
st.session_state.selected_game_id = selected_id

show_all_games_heatmap = st.sidebar.checkbox(
    "Heatmap su tutte le partite", value=False,
    help="Mostra sul bersaglio gli impatti di TUTTE le partite salvate, non solo di quella selezionata.",
)
st.sidebar.caption(f"Aggiornamento live ogni {REFRESH_SECONDS}s.")


@st.fragment(run_every=REFRESH_SECONDS)
def live_game_view(game_id: int) -> None:
    game = data.get_repository().get_game(game_id)
    player_names = {1: game.player1_name, 2: game.player2_name}
    throws = data.list_throws_for_game(game_id)

    throws_by_player = {
        1: [t for t in throws if t.player_number == 1],
        2: [t for t in throws if t.player_number == 2],
    }
    stats_by_player = {p: compute_stats(throws_by_player[p]) for p in (1, 2)}
    remaining_by_player = {
        p: (throws_by_player[p][-1].remaining_after if throws_by_player[p] else game.starting_score)
        for p in (1, 2)
    }

    is_finished = game.finished_at is not None
    # di chi e' il prossimo turno: alterna rispetto a chi ha tirato per ultimo.
    next_player = 1
    if throws:
        next_player = 2 if throws[-1].player_number == 1 else 1

    col1, col2 = st.columns(2)
    for col, player_number in zip((col1, col2), (1, 2)):
        with col:
            components.render_player_card(
                player_number=player_number,
                player_name=player_names[player_number],
                remaining=remaining_by_player[player_number],
                starting_score=game.starting_score,
                stats=stats_by_player[player_number],
                is_current_turn=(not is_finished and player_number == next_player),
                is_winner=(game.winner_player_number == player_number),
            )

    components.render_last_turn_banner(throws, player_names)

    heatmap_throws = data.list_all_throws() if show_all_games_heatmap else throws
    heatmap_players = player_names if not show_all_games_heatmap else {1: "Giocatore 1", 2: "Giocatore 2"}
    fig = build_board_figure(config, heatmap_throws, heatmap_players)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Storico tiri")
    components.render_history_table(throws, player_names)

    if is_finished:
        st.divider()
        components.render_end_of_game_panel(player_names, stats_by_player, game.winner_player_number)


live_game_view(selected_id)
