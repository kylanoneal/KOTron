from functools import lru_cache

from game.ko_tron import KOTron, DirectionUpdate, Direction,Player, GameStatus

import cProfile
import pstats
from io import StringIO

@lru_cache(maxsize=1000)
def cached_next(game: KOTron, direction_updates: DirectionUpdate):
    return KOTron.next(game, direction_updates)



def normal_next(game: KOTron, direction_updates: DirectionUpdate):
    return KOTron.next(game, direction_updates)


if __name__=="__main__":


        # Profile the method
    profiler = cProfile.Profile()
    profiler.enable()


    game = KOTron(num_players=2, num_rows=10, num_cols=10, random_starts=False)
    dir_updates = tuple([DirectionUpdate(Direction.UP, i) for i in range(2)])

    dir_updates_2 = tuple([DirectionUpdate(Direction.UP, i) for i in range(2)])

    print(f" Equality: {dir_updates == dir_updates_2}")

    print(f" Equality: {DirectionUpdate(Direction.UP, 0) == DirectionUpdate(Direction.UP, 0)}")

    print(f"Hash (1, 2): {hash(dir_updates)}, {hash(dir_updates_2)}")



    for i in range(1000):

        equivalent_game = KOTron(num_players=2, num_rows=10, num_cols=10, random_starts=False)
        equivalent_dir_updates = tuple([DirectionUpdate(Direction.UP, i) for i in range(2)])
        cached_next(equivalent_game, equivalent_dir_updates)
        normal_next(equivalent_game, equivalent_dir_updates)

    
    profiler.disable()

    # Print profiling results
    s = StringIO()
    sortby = 'cumulative'  # Change to 'time' to sort by execution time
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
