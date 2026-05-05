
import random
import tron

from pathlib import Path




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
    P_NEUTRAL_START = 1.0
    P_OBSTACLES = 0.5
    OBSTACLE_DENSITY_RANGE = (0.0, 0.3)


    games = []
    for i in range(100):

        curr_game = [
            get_start_position(
                NUM_ROWS, NUM_COLS, P_NEUTRAL_START, P_OBSTACLES, OBSTACLE_DENSITY_RANGE
            )
        ]

        games.append(curr_game)

    # Serialize game data
    serialized_data = tron.to_proto(games)

    tron_dir = Path(tron.__file__).resolve().parent.parent

    datasets_dir = tron_dir / "datasets"

    # Save the serialized data to a file.
    with open(
        datasets_dir / "20260505_5x5_100_starts.bin", "wb"
    ) as f:
        f.write(serialized_data)

if __name__=="__main__":
    main()