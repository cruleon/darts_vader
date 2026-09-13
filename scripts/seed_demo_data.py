"""Popola il database con una partita dimostrativa realistica.

A differenza di ``process_video.py``, questo script NON passa dalla
visione artificiale: costruisce i tiri direttamente con i Livelli 3-4
(scoring + persistenza), gia' validati dai rispettivi test. Serve solo
a dare alla dashboard dati vari e realistici (turni normali, un bust,
una chiusura) senza dipendere dal rumore di un video sintetico, che e'
gia' stato verificato separatamente (vedi ``process_video.py`` sul
video sintetico di calibrazione/detection).

Uso:
    uv run python scripts/seed_demo_data.py
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta

from dartvision.config import AppConfig, load_config
from dartvision.persistence.db import connect
from dartvision.persistence.repository import DartRepository
from dartvision.scoring.board_geometry import polar_to_cartesian, sector_center_angle
from dartvision.scoring.game_501 import Game501
from dartvision.scoring.match_501 import Match501
from dartvision.scoring.score import ScoredThrow, score_point


def _radii(config: AppConfig) -> dict[str, float]:
    board = config.board
    return {
        "single_outer": (board.triple_outer_radius_mm + board.double_inner_radius_mm) / 2,
        "single_inner": (board.outer_bull_radius_mm + board.triple_inner_radius_mm) / 2,
        "triple": (board.triple_inner_radius_mm + board.triple_outer_radius_mm) / 2,
        "double": (board.double_inner_radius_mm + board.double_outer_radius_mm) / 2,
    }


def _build_demo_plan(config: AppConfig, seed: int) -> list[list[ScoredThrow]]:
    """Genera una partita completa scegliendo, ad ogni tiro, un punto del
    bersaglio MAI usato prima (cosi' il "giocatore" e' vario) tramite una
    strategia greedy: aggressiva finche' non capita un bust, poi prudente
    fino alla chiusura esatta. Ritorna la lista dei turni (liste di
    ``ScoredThrow``) pronta per ``Game501.play_turn``.

    ``seed`` introduce una variazione deterministica (scelta a caso tra i
    candidati migliori/peggiori, non il migliore/peggiore in assoluto):
    senza, due chiamate produrrebbero la stessa identica sequenza di
    punteggi, perche' l'algoritmo e' altrimenti puramente deterministico.
    """
    rng = random.Random(seed)
    board = config.board
    radii = _radii(config)

    def coord(sector: int | None, ring: str) -> tuple[float, float]:
        if ring == "bullseye":
            return (0.0, 0.0)
        if ring == "bull":
            return polar_to_cartesian(0.0, (board.inner_bull_radius_mm + board.outer_bull_radius_mm) / 2)
        return polar_to_cartesian(sector_center_angle(sector, board), radii[ring])

    used: set[tuple] = set()

    def all_candidates():
        out = []
        for s in board.sector_order:
            for ring in ("single_outer", "single_inner", "triple", "double"):
                if (s, ring) not in used:
                    x, y = coord(s, ring)
                    out.append((s, ring, x, y, score_point(x, y, board)))
        if ("bull", "bull") not in used:
            x, y = coord(None, "bull")
            out.append(("bull", "bull", x, y, score_point(x, y, board)))
        if ("bullseye", "bullseye") not in used:
            x, y = coord(None, "bullseye")
            out.append(("bullseye", "bullseye", x, y, score_point(x, y, board)))
        return out

    def take(entry):
        used.add((entry[0], entry[1]))
        return entry

    game = Game501(board)
    turns: list[list[ScoredThrow]] = []

    top_k, bottom_k = 4, 4

    for _ in range(4):  # fase aggressiva: mira tra i valori piu' alti disponibili
        darts = []
        for _ in range(3):
            ranked = sorted(all_candidates(), key=lambda e: -e[4].points)
            chosen = rng.choice(ranked[:top_k])
            darts.append(take(chosen))
        turns.append([d[4] for d in darts])
        game.play_turn(turns[-1])

    for _ in range(10):  # fase prudente: chiude appena possibile, altrimenti tiri piccoli
        if game.finished:
            break
        resid = game.remaining
        closer = next(
            (e for e in all_candidates() if e[4].points == resid and e[4].ring in ("double", "bullseye")),
            None,
        )
        if closer is not None:
            turns.append([take(closer)[4]])
            game.play_turn(turns[-1])
            continue

        darts = []
        for _ in range(3):
            ranked = sorted(all_candidates(), key=lambda e: e[4].points)
            chosen = rng.choice(ranked[:bottom_k])
            darts.append(take(chosen))
        turns.append([d[4] for d in darts])
        game.play_turn(turns[-1])

    return turns


def _interleave(plan_p1: list, plan_p2: list) -> list:
    """Alterna i turni dei due piani (P1, P2, P1, P2, ...), accodando
    l'eventuale resto se le due liste hanno lunghezze diverse."""
    interleaved = []
    for t1, t2 in zip(plan_p1, plan_p2):
        interleaved.append(t1)
        interleaved.append(t2)
    interleaved.extend(plan_p1[len(plan_p2):])
    interleaved.extend(plan_p2[len(plan_p1):])
    return interleaved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/board_config.yaml")
    parser.add_argument("--db", default="data/dartvision.db")
    parser.add_argument("--starting-score", type=int, default=501)
    parser.add_argument("--player1", default="Giocatore 1")
    parser.add_argument("--player2", default="Giocatore 2")
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="Tronca la partita dopo N turni totali (per generare una partita 'in corso', non conclusa)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    conn = connect(args.db)
    repository = DartRepository(conn)

    # Ogni chiamata a _build_demo_plan ha il proprio set di posizioni "usate"
    # e un seed diverso: i due giocatori simulati non condividono nulla e
    # non giocano turni identici.
    plan_p1 = _build_demo_plan(config, seed=1)
    plan_p2 = _build_demo_plan(config, seed=2)
    interleaved = _interleave(plan_p1, plan_p2)
    if args.max_turns is not None:
        interleaved = interleaved[: args.max_turns]

    match = Match501(config.board, args.player1, args.player2, starting_score=args.starting_score)
    total_turns = len(interleaved)
    started_at = datetime.now() - timedelta(minutes=total_turns)
    game_id = repository.create_game(
        starting_score=args.starting_score,
        player1_name=args.player1,
        player2_name=args.player2,
        started_at=started_at,
    )

    turn_numbers = {1: 0, 2: 0}
    for i, throws in enumerate(interleaved, start=1):
        if match.finished:
            break
        result = match.play_turn(throws)
        turn_numbers[result.player_number] += 1
        timestamp = datetime.now() - timedelta(minutes=(total_turns - i))
        repository.add_turn_outcome(
            game_id, result.player_number, turn_numbers[result.player_number],
            result.outcome, timestamp=timestamp,
        )
        status = "BUST" if result.outcome.is_bust else ("CHECKOUT!" if result.outcome.is_checkout else "ok")
        player_name = match.player_name(result.player_number)
        print(
            f"{player_name} - turno {turn_numbers[result.player_number]}: "
            f"{result.outcome.turn_points} punti, residuo {result.outcome.remaining_after} [{status}]"
        )

    print(f"\nPartita demo #{game_id} salvata in {args.db}: {sum(turn_numbers.values())} turni totali.")
    if match.winner is not None:
        print(f"Vince {match.player_name(match.winner)}!")
    conn.close()


if __name__ == "__main__":
    main()
