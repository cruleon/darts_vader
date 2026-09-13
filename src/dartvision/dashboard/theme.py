"""Token di colore/tipografia della dashboard.

Palette validata (contrasto + separazione CVD) per superficie scura,
tenuta come costanti Python cosi' sia il CSS iniettato sia le figure
Plotly leggono dagli stessi valori.
"""

from __future__ import annotations

PAGE_BG = "#0d0d0d"
SURFACE = "#1a1a19"
SURFACE_RAISED = "#232320"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"
BORDER = "rgba(255,255,255,0.10)"

ACCENT = "#d95926"  # slot categorico 2 (orange, step scuro)
ACCENT_SOFT = "rgba(217,89,38,0.28)"
BLUE = "#3987e5"  # slot categorico 1
BLUE_SOFT = "rgba(57,135,229,0.28)"
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#e66767"

# Identita' dei due giocatori: ordine fisso (slot 1, slot 2), mai ciclato.
PLAYER_COLORS = {1: BLUE, 2: ACCENT}
PLAYER_HALO_COLORS = {1: BLUE_SOFT, 2: ACCENT_SOFT}

SEQUENTIAL_BLUE_LOW = "rgba(57,135,229,0.05)"
SEQUENTIAL_BLUE_HIGH = "rgba(134,182,239,0.55)"  # step 250, pop contro sfondo scuro

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

CUSTOM_CSS = f"""
<style>
.stApp {{
    background-color: {PAGE_BG};
}}
.block-container {{
    padding-top: 1.5rem;
    max-width: 1400px;
}}
[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem 1.2rem;
}}
[data-testid="stMetricLabel"] {{
    color: {INK_MUTED};
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
[data-testid="stMetricValue"] {{
    color: {INK_PRIMARY};
    font-weight: 600;
}}
.dv-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.6rem;
}}
.dv-card-title {{
    color: {INK_MUTED};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}}
.dv-hero {{
    font-size: 2.4rem;
    font-weight: 700;
    color: {INK_PRIMARY};
    line-height: 1.1;
}}
.dv-hero-sub {{
    color: {INK_SECONDARY};
    font-size: 0.85rem;
    margin-top: 0.2rem;
}}
.dv-status-pill {{
    display: inline-block;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}}
.dv-status-bust {{ background: rgba(230,103,103,0.18); color: {CRITICAL}; }}
.dv-status-checkout {{ background: rgba(12,163,12,0.18); color: {GOOD}; }}
.dv-status-ok {{ background: rgba(195,194,183,0.12); color: {INK_SECONDARY}; }}
.dv-status-turn {{ background: rgba(57,135,229,0.18); color: {BLUE}; }}

.dv-player-card {{ border-left: 3px solid transparent; }}
.dv-player-card.dv-player-1 {{ border-left-color: {BLUE}; }}
.dv-player-card.dv-player-2 {{ border-left-color: {ACCENT}; }}

section[data-testid="stSidebar"] {{
    background-color: {SURFACE};
}}
</style>
"""
