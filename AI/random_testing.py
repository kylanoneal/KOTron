import os
import re

from game.BMTron import *
from pytorch_models import *
from model_architectures import *
from reinforcement_learning import *
def test_attn_net():
    game = BMTron()

    model_input = get_model_input_from_raw_info(game.collision_table, game.get_heads(), 0, model_type=EvaluationAttentionConvNet)

    print(type(model_input))
    print("model input grid shape:", model_input[0].shape, "heads shape:", model_input[1].shape)
    model = EvaluationAttentionConvNet().to(device)


    output = model(model_input)

def test_mask():
    shape = (1, 1, 20, 20)
    test_output = torch.ones(shape)
    test_head = torch.tensor([20, 20])
    test_head = test_head.unsqueeze(0)

    mask = get_attn_mask(shape, test_head, sigma=10)
    print(mask)


def test_collision_table():
    dimension = 40
    collision_table = []

    for i in range(dimension):
        collision_table.append([])

        for j in range(dimension):
            collision_table[i].append(0)


    collision_table[38][1] = 100

    for row in collision_table:
        print(row)

def print_2d_grid(grid):
    for row in grid:
        print(row)

def test_data_processing():
    game = BMTron(dimension=10)
    for i in range(3):
        game.move_racers()

    grid, heads = get_relevant_info_from_game_state(game)
    print_2d_grid(grid)

    for i in range(len(game.players)):
        model_input = get_model_input_from_raw_info(grid, heads, i, is_part_of_batch=True)

        print(model_input.shape)
        print_2d_grid(model_input.squeeze(0).tolist())

def test_simulate_game():
    game_data, winner_player_num = simulate_game(2, 5, 10, True, get_random_next_action)

    print("Winner player num:", winner_player_num)
    for grid, heads in game_data:
        print("Heads:", heads)
        print_2d_grid(grid)

    return game_data, winner_player_num


def test_process_game_data(game_data, winner_player_num):
    processed_game_data = process_game_data(game_data, winner_player_num)

    print("Processed game data:")
    for grid, heads, player_num, game_progress, wlt in processed_game_data:

        print("\n-------------------------------------------")
        print("Game progress:", game_progress)
        print("Player num:", player_num)
        print("Won or lost or tied:", wlt)
        print("Heads:", heads)
        print_2d_grid(grid)


    with open("test-game.json", 'w') as file:
        json.dump(processed_game_data, file)


def test_process_json_data(filepath, decay_fn, model_type):

    processed_data = process_json_data(filepath, decay_fn, model_type)

    for model_input, eval in processed_data:
        print("\n-------------------------------------------")
        print("Eval:", eval)
        print("shape:", model_input.shape)
        print_2d_grid(model_input.squeeze(0).tolist())




if __name__ == '__main__':
    init_utility_gui(5)
    game_data, winner_player_num = test_simulate_game()
    test_process_game_data(game_data, winner_player_num)
    test_process_json_data("test-game.json", lambda x: x**5, EvaluationNetConv2)