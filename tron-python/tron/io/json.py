import json
from pathlib import Path

import numpy as np

# Adjust these imports to your project
from tron.game import (
    PovGameState,
    GameState2D,
    Player2D,
    Direction,
    from_bitboard,
    from_2d_game_state,
)

from tron.ai.benchmarks import Tactic


def tactic_to_dict(tactic: Tactic) -> dict:
    """Convert a Tactic into a JSON-serializable dictionary."""
    game_state_2d = from_bitboard(tactic.pov_game_state.game_state)

    return {
        "pov_game_state": {
            "game_state": {
                "grid": game_state_2d.grid.astype(int).tolist(),
                "players": [
                    {
                        "row": int(player.row),
                        "col": int(player.col),
                        "can_move": player.can_move,
                    }
                    for player in game_state_2d.players
                ],
            },
            "hero_index": tactic.pov_game_state.hero_index,
            "opponent_index": tactic.pov_game_state.opponent_index,
        },
        "opposing_dirs": [
            direction.name for direction in tactic.opposing_dirs
        ],
        "expected_hero_dirs": [
            direction.name for direction in tactic.expected_hero_dirs
        ],
    }


def tactic_from_dict(data: dict) -> Tactic:
    """Convert a JSON dictionary back into a Tactic."""
    pov_data = data["pov_game_state"]
    game_state_data = pov_data["game_state"]

    game_state_2d = GameState2D(
        grid=np.array(game_state_data["grid"], dtype=bool),
        players=tuple(
            Player2D(
                player["row"],
                player["col"],
                player["can_move"],
            )
            for player in game_state_data["players"]
        ),
    )

    return Tactic(
        pov_game_state=PovGameState(
            game_state=from_2d_game_state(game_state_2d),
            hero_index=pov_data["hero_index"],
            opponent_index=pov_data["opponent_index"],
        ),
        opposing_dirs=[
            Direction[name]
            for name in data["opposing_dirs"]
        ],
        expected_hero_dirs=[
            Direction[name]
            for name in data["expected_hero_dirs"]
        ],
    )


def write_tactics_json(
    tactics: tuple[Tactic, ...] | list[Tactic],
    path: str | Path,
) -> None:
    """Write tactics to a JSON file."""
    data = [tactic_to_dict(tactic) for tactic in tactics]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def read_tactics_json(path: str | Path) -> tuple[Tactic, ...]:
    """Read tactics from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return tuple(tactic_from_dict(tactic) for tactic in data)