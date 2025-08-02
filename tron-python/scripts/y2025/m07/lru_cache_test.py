
from cachetools import LRUCache, cached
# TODO: This should probably be in different spot
# Initialize an LRU cache with 20 mil max size
cache = LRUCache(maxsize=20000000)

@cached(cache)
def lru_eval(model: TronModel, game, player_index):
    return model.run_inference([HeroGameState(game, player_index)])[0]
