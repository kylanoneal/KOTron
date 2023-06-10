from game.BMTron import *

if __name__ == '__main__':
    game = BMTron(num_players=2, dimension=20)
    input = get_model_input_from_game_state(game_state=game, player_num=0)
    print(input.shape)