import random

from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import tron

from tron.game import (
    GameState,
    GameStatus,
    StatusInfo,
    Direction,
    from_2d_game_state,
    GameState2D,
    Player2D,
)

from tron.ai.minimax import basic_minimax, MinimaxContext, MinimaxResult

from tron.ai.tron_model import RandomTronModel


def main():

    model = RandomTronModel()

    tron_dir = Path(tron.__file__).resolve().parent.parent

    datasets_dir = tron_dir / "datasets"

    out_dir = datasets_dir / f"20260511_5x5_perfect_play_all_starts_no_obstacles"
    out_dir.mkdir(exist_ok=False)

    DEPTH = 20

    games = []

    for i in tqdm(range(25)):

        for j in tqdm(range(i + 1, 25)):

            assert j != i

            i_row, i_col = i // 5, i % 5
            j_row, j_col = j // 5, j % 5

            grid = [[0] * 5 for _ in range(5)]

            grid[i_row][i_col] = 1
            grid[j_row][j_col] = 1

            game = from_2d_game_state(
                GameState2D(
                    grid=np.array(
                        grid,
                        dtype=bool,
                    ),
                    players=(Player2D(i_row, i_col, True), Player2D(j_row, j_col, True)),
                )
            )

            game_status: StatusInfo = tron.get_status(game)

            curr_game = [deepcopy(game)]

            while game_status.status == GameStatus.IN_PROGRESS:

                p1_mm_result: MinimaxResult = basic_minimax(
                    game,
                    depth=DEPTH,
                    is_hero=True,
                    context=MinimaxContext(
                        model, hero_index=0, opponent_index=1, win_magnitude=10
                    ),
                )

                p2_mm_result: MinimaxResult = basic_minimax(
                    game,
                    depth=DEPTH,
                    is_hero=True,
                    context=MinimaxContext(
                        model, hero_index=1, opponent_index=0, win_magnitude=10
                    ),
                )

                p1_dir = (
                    Direction.UP
                    if p1_mm_result.principal_variation is None
                    else p1_mm_result.principal_variation
                )
                p2_dir = (
                    Direction.UP
                    if p2_mm_result.principal_variation is None
                    else p2_mm_result.principal_variation
                )

                game = tron.next(game, directions=(p1_dir, p2_dir))

                curr_game.append(game)

                game_status = tron.get_status(game)

            games.append(curr_game)

    print(f"{len(games)}")

    for full_game in games:

        for k, gstate in enumerate(full_game):

            if k == len(full_game) - 1:
                assert tron.get_status(gstate).status != GameStatus.IN_PROGRESS
            else:
                assert tron.get_status(gstate).status == GameStatus.IN_PROGRESS

    # Serialize game data
    serialized_data = tron.to_proto(games)

    # Save the serialized data to a file.
    with open(out_dir / f"perfect{len(games)}.bin", "wb") as f:
        f.write(serialized_data)


if __name__ == "__main__":
    main()
