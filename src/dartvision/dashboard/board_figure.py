"""Figura Plotly del bersaglio: sfondo "cromato" + impatti dei due giocatori.

Principio guida (color-formula del design system): il bersaglio e' solo
cornice (toni scuri quasi monocromi) cosi' l'unico elemento a cui l'occhio
va sono i dati veri. Con due giocatori il colore ora porta un'identita'
reale (categorica, 2 serie, ordine fisso: slot1/slot2 dalla palette) — non
piu' solo "storico vs ultimo": chi ha tirato cosa. L'ultimo tiro in
assoluto resta evidenziato, ma via dimensione/alone nel colore DEL SUO
giocatore, non con una terza tinta che competerebbe con l'identita'.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from dartvision.config import AppConfig
from dartvision.dashboard import theme
from dartvision.persistence.models import ThrowRecord
from dartvision.scoring.board_geometry import polar_to_cartesian


def _screen_xy(angle_deg: float, radius_mm: float) -> tuple[float, float]:
    """Coordinate (mm, mm) -> coordinate di plot (y invertita: in alto = in alto)."""
    x, y = polar_to_cartesian(angle_deg, radius_mm)
    return x, -y


def _pie_wedge_path(r_outer: float, a0_deg: float, a1_deg: float, n_arc: int = 5) -> str:
    angles = np.linspace(a0_deg, a1_deg, n_arc)
    pts = [_screen_xy(a, r_outer) for a in angles]
    commands = ["M0,0"] + [f"L{x:.2f},{y:.2f}" for x, y in pts] + ["Z"]
    return " ".join(commands)


def _board_shapes(config: AppConfig) -> list[dict]:
    board = config.board
    sector_width = 360.0 / board.sector_count
    shapes: list[dict] = []

    for i in range(board.sector_count):
        center_angle = board.sector0_offset_deg + i * sector_width
        a0, a1 = center_angle - sector_width / 2, center_angle + sector_width / 2
        fill = theme.SURFACE if i % 2 == 0 else theme.SURFACE_RAISED
        shapes.append(
            dict(
                type="path",
                path=_pie_wedge_path(board.double_outer_radius_mm, a0, a1),
                fillcolor=fill,
                line=dict(width=0),
                layer="below",
            )
        )

    ring_radii = [
        board.triple_inner_radius_mm,
        board.triple_outer_radius_mm,
        board.double_inner_radius_mm,
        board.double_outer_radius_mm,
        board.outer_bull_radius_mm,
    ]
    for r in ring_radii:
        shapes.append(
            dict(
                type="circle",
                x0=-r, y0=-r, x1=r, y1=r,
                line=dict(color=theme.BASELINE, width=1),
                layer="below",
            )
        )

    r_bull = board.inner_bull_radius_mm
    shapes.append(
        dict(
            type="circle",
            x0=-r_bull, y0=-r_bull, x1=r_bull, y1=r_bull,
            fillcolor="rgba(255,255,255,0.07)",
            line=dict(color=theme.BASELINE, width=1),
            layer="below",
        )
    )
    return shapes


def _hover_text(t: ThrowRecord, player_name: str) -> str:
    ring_label = t.ring.replace("_", " ")
    sector_label = f"settore {t.sector}" if t.sector is not None else "centro"
    return (
        f"<b>{t.points} punti</b> - {player_name}<br>{sector_label} - {ring_label}"
        f"<br>{t.timestamp:%H:%M:%S}<extra></extra>"
    )


def build_board_figure(
    config: AppConfig,
    throws: list[ThrowRecord],
    player_names: dict[int, str],
    show_density: bool = True,
) -> go.Figure:
    board_edge = config.board.double_outer_radius_mm
    view_extent = board_edge * 1.08  # margine sottile attorno al bersaglio

    fig = go.Figure()

    if show_density and len(throws) >= 8:
        fig.add_trace(
            go.Histogram2dContour(
                x=[t.x_mm for t in throws],
                y=[-t.y_mm for t in throws],
                colorscale=[
                    [0.0, theme.SEQUENTIAL_BLUE_LOW],
                    [1.0, theme.SEQUENTIAL_BLUE_HIGH],
                ],
                showscale=False,
                ncontours=8,
                line=dict(width=0),
                contours=dict(coloring="fill"),
                opacity=0.45,
                hoverinfo="skip",
            )
        )

    last = throws[-1] if throws else None

    for player_number in (1, 2):
        player_throws = [t for t in throws if t.player_number == player_number]
        history = [t for t in player_throws if t is not last]
        if not history:
            continue
        color = theme.PLAYER_COLORS[player_number]
        fig.add_trace(
            go.Scatter(
                x=[t.x_mm for t in history],
                y=[-t.y_mm for t in history],
                mode="markers",
                marker=dict(size=10, color=color, opacity=0.75, line=dict(width=2, color=theme.SURFACE)),
                name=player_names[player_number],
                text=[_hover_text(t, player_names[player_number]) for t in history],
                hovertemplate="%{text}",
                hoverlabel=dict(bgcolor=theme.SURFACE_RAISED, font_color=theme.INK_PRIMARY),
            )
        )

    if last is not None:
        color = theme.PLAYER_COLORS[last.player_number]
        halo = theme.PLAYER_HALO_COLORS[last.player_number]
        fig.add_trace(
            go.Scatter(
                x=[last.x_mm], y=[-last.y_mm],
                mode="markers",
                marker=dict(size=34, color=halo, line=dict(width=0)),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[last.x_mm], y=[-last.y_mm],
                mode="markers",
                marker=dict(size=16, color=color, line=dict(width=3, color=theme.INK_PRIMARY)),
                name=f"Ultimo tiro ({player_names[last.player_number]})",
                text=[_hover_text(last, player_names[last.player_number])],
                hovertemplate="%{text}",
                hoverlabel=dict(bgcolor=theme.SURFACE_RAISED, font_color=theme.INK_PRIMARY),
            )
        )

    fig.update_layout(
        shapes=_board_shapes(config),
        showlegend=bool(throws),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, x=0.0,
            font=dict(color=theme.INK_SECONDARY, size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-view_extent, view_extent], visible=False, fixedrange=True),
        yaxis=dict(
            range=[-view_extent, view_extent], visible=False, fixedrange=True,
            scaleanchor="x", scaleratio=1,
        ),
        font=dict(family=theme.FONT_FAMILY, color=theme.INK_PRIMARY),
        height=560,
    )
    return fig
