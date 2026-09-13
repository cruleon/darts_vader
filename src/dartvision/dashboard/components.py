"""Componenti UI riutilizzabili della dashboard (KPI, tabella storico, ecc.)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dartvision.analytics import GameStats
from dartvision.persistence.models import GameRecord, ThrowRecord


def render_status_pill(is_bust: bool, is_checkout: bool) -> str:
    if is_bust:
        return '<span class="dv-status-pill dv-status-bust">BUST</span>'
    if is_checkout:
        return '<span class="dv-status-pill dv-status-checkout">CHECKOUT</span>'
    return '<span class="dv-status-pill dv-status-ok">ok</span>'


def render_player_card(
    player_number: int,
    player_name: str,
    remaining: int,
    starting_score: int,
    stats: GameStats,
    is_current_turn: bool,
    is_winner: bool,
) -> None:
    if is_winner:
        badge = '<span class="dv-status-pill dv-status-checkout">VINCITORE</span>'
    elif is_current_turn:
        badge = '<span class="dv-status-pill dv-status-turn">turno corrente</span>'
    else:
        badge = ""

    progress = 1.0 - (remaining / starting_score if starting_score else 0)

    st.markdown(
        f"""
        <div class="dv-card dv-player-card dv-player-{player_number}">
            <div class="dv-card-title">{player_name} {badge}</div>
            <div class="dv-hero">{remaining}</div>
            <div class="dv-hero-sub">su {starting_score} &middot; {progress:.0%} completata</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    row1 = st.columns(2)
    row1[0].metric("Media / freccetta", f"{stats.average_per_dart:.1f}")
    row1[1].metric("Media / turno", f"{stats.average_per_turn:.1f}")
    row2 = st.columns(2)
    row2[0].metric("Miglior turno", stats.best_turn_points)
    row2[1].metric("Bust", stats.bust_count)


def render_last_turn_banner(throws: list[ThrowRecord], player_names: dict[int, str]) -> None:
    if not throws:
        return
    last = throws[-1]
    last_turn_throws = [
        t for t in throws if t.player_number == last.player_number and t.turn_number == last.turn_number
    ]
    is_bust = last_turn_throws[0].is_bust
    is_checkout = any(t.is_checkout for t in last_turn_throws)
    turn_points = sum(t.points for t in last_turn_throws) if not is_bust else 0
    darts_desc = " &nbsp;&middot;&nbsp; ".join(
        f"{t.ring.replace('_', ' ')} ({t.points})" for t in last_turn_throws
    )
    player_name = player_names[last.player_number]

    st.markdown(
        f"""
        <div class="dv-card">
            <div class="dv-card-title">Ultimo turno - {player_name} {render_status_pill(is_bust, is_checkout)}</div>
            <div style="color:#ffffff;font-size:1.1rem;font-weight:600;">{turn_points} punti</div>
            <div class="dv-hero-sub">{darts_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_table(throws: list[ThrowRecord], player_names: dict[int, str]) -> None:
    if not throws:
        st.info("Nessun tiro ancora registrato per questa partita.")
        return

    rows = [
        {
            "Giocatore": player_names[t.player_number],
            "Turno": t.turn_number,
            "Tiro": t.throw_number,
            "Ora": t.timestamp.strftime("%H:%M:%S"),
            "Settore": t.sector if t.sector is not None else "-",
            "Anello": t.ring.replace("_", " "),
            "Punti": t.points,
            "Residuo": t.remaining_after,
            "Bust": "si" if t.is_bust else "",
            "Chiusura": "si" if t.is_checkout else "",
            "Dist. chiusura (mm)": f"{t.closing_distance_mm:.0f}" if t.closing_distance_mm is not None else "-",
        }
        for t in reversed(throws)
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height=360)


def render_end_of_game_panel(
    player_names: dict[int, str],
    stats_by_player: dict[int, GameStats],
    winner_player_number: int,
) -> None:
    st.markdown(
        f'<div class="dv-hero-sub" style="font-size:1.05rem;">'
        f"🏆 Vince <b>{player_names[winner_player_number]}</b>!</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="dv-card-title">Statistiche di fine partita</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for col, player_number in zip((col1, col2), (1, 2)):
        stats = stats_by_player[player_number]
        with col:
            st.markdown(f"**{player_names[player_number]}**")
            metric_cols = st.columns(2)
            metric_cols[0].metric("Media / freccetta", f"{stats.average_per_dart:.1f}")
            metric_cols[1].metric("Media / turno", f"{stats.average_per_turn:.1f}")
            metric_cols2 = st.columns(2)
            metric_cols2[0].metric("Turni giocati", stats.num_turns)
            metric_cols2[1].metric("Bust", stats.bust_count)
            if stats.average_closing_distance_mm is not None:
                st.metric("Dist. media chiusura", f"{stats.average_closing_distance_mm:.0f} mm")


def render_games_sidebar(games: list[GameRecord], selected_id: int | None) -> int | None:
    if not games:
        st.sidebar.info("Nessuna partita ancora salvata nel database.")
        return None

    def _label(g: GameRecord) -> str:
        matchup = f"{g.player1_name} vs {g.player2_name}"
        if g.finished_at is not None:
            status = f"vince {g.player_name(g.winner_player_number)}"
        else:
            status = "in corso"
        return f"#{g.id} - {matchup} - {status}"

    options = {g.id: _label(g) for g in reversed(games)}
    default_index = list(options.keys()).index(selected_id) if selected_id in options else 0
    chosen = st.sidebar.selectbox(
        "Partita", options=list(options.keys()), format_func=lambda k: options[k], index=default_index
    )
    return chosen
