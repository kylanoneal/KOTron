import tron

from tron.game import GameState



game = GameState.new_game(random_starts=True)


print(f"Tron path: {tron.__file__}")

print(tron.get_status(game))




