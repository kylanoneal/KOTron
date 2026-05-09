import random

from tron.game import (
    GameState,

)

def get_start_position(
    n_rows: int,
    n_cols: int,
    p_neutral: float,
    p_obstacles: float,
    obstacle_density_range: tuple,
) -> GameState:

    is_neutral_start = p_neutral > random.random()
    are_obstacles = p_obstacles > random.random()

    min_d, max_d = obstacle_density_range
    obstacle_density = random.uniform(min_d, max_d) if are_obstacles else 0.0

    return GameState.new_game(
        num_players=2,
        num_rows=n_rows,
        num_cols=n_cols,
        random_starts=True,
        neutral_starts=is_neutral_start,
        obstacle_density=obstacle_density,
    )
