import random

from pathlib import Path
from tqdm import tqdm
from copy import deepcopy


import tron

from tron.game import GameState, GameStatus, StatusInfo, Direction

from tron.ai.minimax import basic_minimax, MinimaxContext, MinimaxResult

from tron.ai.tron_model import RandomTronModel



def get_start_position(
    n_rows: int,
    n_cols: int,
    p_neutral: float,
    p_obstacles: float,
    obstacle_density_range: tuple,
) -> tron.GameState:

    is_neutral_start = p_neutral > random.random()
    are_obstacles = p_obstacles > random.random()

    min_d, max_d = obstacle_density_range
    obstacle_density = random.uniform(min_d, max_d) if are_obstacles else 0.0

    return tron.GameState.new_game(
            num_players=2,
            num_rows=n_rows,
            num_cols=n_cols,
            random_starts=True,
            neutral_starts=is_neutral_start,
            obstacle_density=obstacle_density,
        )


def main():

    NUM_ROWS = NUM_COLS = 5
    P_NEUTRAL_START = 0.75
    P_OBSTACLES = 0.5
    OBSTACLE_DENSITY_RANGE = (0.0, 0.3)

    DEPTH = 3


    N_GAMES = 10_000

    model = RandomTronModel()

    for i in tqdm(range(100)):


        games = []

        for j in tqdm(range(N_GAMES)):


            game = get_start_position(
                NUM_ROWS, NUM_COLS, P_NEUTRAL_START, P_OBSTACLES, OBSTACLE_DENSITY_RANGE
            )
            game_status: StatusInfo = tron.get_status(game)

            curr_game = [deepcopy(game)]


            while game_status.status == GameStatus.IN_PROGRESS:


                p1_mm_result: MinimaxResult = basic_minimax(
                    game,
                    depth=DEPTH,
                    is_hero=True,
                    context=MinimaxContext(model, hero_index=0, opponent_index=1, win_magnitude=10)
                )

                p2_mm_result: MinimaxResult = basic_minimax(
                    game,
                    depth=DEPTH,
                    is_hero=True,
                    context=MinimaxContext(model, hero_index=1, opponent_index=0, win_magnitude=10)
                )

                p1_dir = Direction.UP if p1_mm_result.principal_variation is None else p1_mm_result.principal_variation
                p2_dir = Direction.UP if p2_mm_result.principal_variation is None else p2_mm_result.principal_variation

                
                game = tron.next(
                    game, directions=(p1_dir, p2_dir)
                )

                curr_game.append(game)

                game_status = tron.get_status(game)

            games.append(curr_game)


        assert len(games) == N_GAMES

        for full_game in games:

            for k, gstate in enumerate(full_game):

                if k == len(full_game) - 1:
                    assert tron.get_status(gstate).status != GameStatus.IN_PROGRESS
                else:
                    assert tron.get_status(gstate).status == GameStatus.IN_PROGRESS


        # Serialize game data
        serialized_data = tron.to_proto(games)

        tron_dir = Path(tron.__file__).resolve().parent.parent

        datasets_dir = tron_dir / "datasets"

        # Save the serialized data to a file.
        with open(
            datasets_dir / f"20260507_5x5_random_depth{DEPTH}" / f"{i:04d}_ngames{N_GAMES}.bin", "wb"
        ) as f:
            f.write(serialized_data)

if __name__=="__main__":
    main()