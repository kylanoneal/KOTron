from pytorch_models import *
from model_architectures import *
from MCTS import *
from game.BMTron import *
from unit_testing.UtilityGUI import *
from copy import deepcopy

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

import os
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE: ", device)

# mcts = MCTS(model, n_simulations=4, exploration_factor=0.5)

# Clean this up
current_game_iteration = 0
cumulative_loss = 0
cumulative_game_lengths = 0
cumulative_diff_from_zero = 0
player_zero_wins = games_tied = games_not_tied = 0
game_dataset = []


def game_loop():
    global current_game_iteration, cumulative_game_lengths, game_dataset
    global games_tied, games_not_tied, player_zero_wins

    if SHOW_GAMES:
        init_utility_gui(GAME_DIMENSION)

    for i in range(MAX_GAME_ITERATIONS):
        current_game_iteration = i
        # print("MCTS model weights")
        # print_weights_sum_of_squares(mcts.model)

        game_data, winner_player_num = simulate_game(NUM_PLAYERS, GAME_DIMENSION, RANDOM_START_MOVES, SHOW_GAMES,
                                                     get_next_action)

        print("Game iteration:", i)
        print("Winner:", winner_player_num, end="\n\n")

        cumulative_game_lengths += len(game_data)

        if winner_player_num == -1:
            games_tied += 1
        else:
            games_not_tied += 1
            if winner_player_num == 0:
                player_zero_wins += 1

        game_dataset.extend(process_game_data(game_data, winner_player_num, GAME_LENGTH_THRESHOLD))

        if (current_game_iteration + 1) % ITERATIONS_BEFORE_SAVING_DATASET == 0:
            save_dataset()
        if (current_game_iteration + 1) % ITERATIONS_BEFORE_LOGGING == 0:
            update_game_data_logs()


def simulate_game(num_players, game_dimension, random_start_moves, show_game, action_fn):
    game_data = []

    game = BMTron(num_players=num_players, dimension=game_dimension)
    random_move_count = 0
    while not game.winner_found:

        for player_num in range(num_players - 1, -1, -1):

            if random_move_count < random_start_moves:
                action = get_random_next_action(game, player_num)
            elif DO_MCTS:
                # Monte Carlo:
                raise NotImplementedError
                # root_node = Node(game, player_num)
                # available_actions, action_probs = MCTS_OBJ.search(root_node)
                # if len(available_actions) > 0:
                #     action = np.random.choice(available_actions, size=1, p=action_probs).item()
                # else:
                #     action = Directions.up
            else:
                action = action_fn(game, player_num)
            game.update_direction(player_num, action)

        random_move_count += 1

        game_data.append(get_relevant_info_from_game_state(game))
        game.move_racers()
        game.check_for_winner()

        if show_game:
            show_game_state(game)

    return game_data, game.winner_player_num


def process_game_data(game_data, winner_player_num, game_length_threshold=0):
    processed_game_data = []
    if len(game_data) > game_length_threshold:
        for turn_num, (game_grid, heads) in enumerate(game_data):
            game_progress = turn_num / (len(game_data) - 1)

            for player_num, head in enumerate(heads):
                won_or_lost_or_tied = (1 if player_num == winner_player_num else -1) \
                    if winner_player_num != -1 else 0

                processed_game_data.append((game_grid, heads, player_num, game_progress, won_or_lost_or_tied))

    return processed_game_data


def update_game_data_logs():
    global cumulative_loss, cumulative_game_lengths, cumulative_diff_from_zero
    global games_tied, games_not_tied, player_zero_wins
    writer.add_scalar('Avg length of game', cumulative_game_lengths / ITERATIONS_BEFORE_LOGGING,
                      current_game_iteration)
    writer.add_scalar("Percent player zero wins:", player_zero_wins / games_not_tied, current_game_iteration)
    writer.add_scalar("Percent games tied:", games_tied / ITERATIONS_BEFORE_LOGGING, current_game_iteration)
    # writer.add_scalar('Training loss', cumulative_loss / ITERATIONS_BEFORE_LOGGING, current_game_iteration)
    # writer.add_scalar('Predictions diff from zero', cumulative_diff_from_zero / ITERATIONS_BEFORE_LOGGING,
    #                   current_game_iteration)
    cumulative_game_lengths = cumulative_loss = cumulative_diff_from_zero = 0
    games_tied = games_not_tied = player_zero_wins = 0


def update_training_logs(training_loss, weights_sum_of_squares, train_iter):
    writer.add_scalar('Training loss:', training_loss, train_iter)
    writer.add_scalar('Sum of squares of model weights:', weights_sum_of_squares, train_iter)


def save_dataset():
    global game_dataset

    # Use current time to distinguish datasets
    formatted_datetime = datetime.now().strftime("%m-%d-%H-%M-%S")

    with open(curr_sims_dir + "game-sims-" + formatted_datetime + ".json", 'w') as file:
        json.dump(game_dataset, file)

    # Clear list after outputting to file
    game_dataset = []


# Move this somewhere else
def print_head_locations(model_input):
    np_input = model_input.numpy()
    reshaped_array = np_input.reshape(2, 42, 42)
    head_grid = reshaped_array[1]

    for i in range(len(head_grid)):
        for j in range(len(head_grid)):
            if head_grid[i][j] == 1:
                print("1 at ", i, j)
            if head_grid[i][j] == -1:
                print("-1 at: ", i, j)


def get_next_action(game, player_num):
    """Try all available actions and choose action based on softmax of evaluations"""
    evaluations = []

    available_actions = game.get_possible_directions(player_num)

    # print("I: ", i, "action: ", action)
    for action in available_actions:
        next_game_state = deepcopy(game)
        next_game_state.update_direction(player_num, action)
        next_game_state.move_racers()

        grid, heads = get_relevant_info_from_game_state(next_game_state)
        curr_eval = model(get_model_input_from_raw_info(grid, heads, player_num, model_type=MODEL_TYPE))
        # print("curr eval: ", curr_eval)
        # print("Raw model output: ", model(get_model_input_from_game_state(game, player_num)))
        evaluations.append(curr_eval.item())

    # print("EVALUATIONS: ", evaluations, "\n\n\n")
    if len(available_actions):

        temperature = INITIAL_TEMP * (TEMP_DECAY_FACTOR ** current_game_iteration)
        # change this value to control randomness

        exp_values = np.exp(np.array(evaluations) / temperature)
        # print("exp values: ", exp_values)
        action_probs = exp_values / np.sum(exp_values)
        chosen_action = np.random.choice(available_actions, size=1, p=action_probs).item()

        # Calculate the softmax function

        # print("Softmax evaluations: ", action_probs)
        # print("Chosen action:", chosen_action)
        # print("type of chosen action: ", type(chosen_action))
        # print("after softmax: ", evaluations)

        # print("Chose action:", chosen_action)
    else:
        # No possible actions, just return up
        chosen_action = Directions.up

    return chosen_action


def get_random_next_action(game, player_num):
    """Try all available actions and choose action based on softmax of evaluations"""
    available_actions = game.get_possible_directions(player_num)

    # if len(available_actions) > 0:
    return available_actions[np.random.randint(0, len(available_actions))] if len(
        available_actions) > 0 else Directions.up


def train_on_single_game(data):
    # Define the optimizer
    # optimizer = optim.SGD(model.parameters(), lr = 0.1)
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.MSELoss()

    # Number of epochs
    n_epochs = 1  # Modify as needed

    global cumulative_loss
    global cumulative_diff_from_zero

    print("TRAINING...")
    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_diff_from_zero = 0

        # random.shuffle(data)
        for state, state_value in data:
            # Convert outputs to tensors and move them to the device

            state_value = torch.tensor(state_value, dtype=torch.float32).to(device)

            # Backward pass and optimization
            optimizer.zero_grad()

            # Forward pass
            predicted_value_estimate = model(state)

            epoch_diff_from_zero += np.abs(0 - predicted_value_estimate.item())
            # Define the loss function (CrossEntropy for action probs, MSE for value estimates)

            # Ignoring Action Probs for now
            # action_probs_loss = F.cross_entropy(predicted_action_probs, action_probs)

            # print("Preicted vs actual state values: ", predicted_value_estimate, state_value)

            value_loss = loss_function(predicted_value_estimate, state_value)
            value_loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += value_loss.item()

        cumulative_loss += epoch_loss / len(data)
        cumulative_diff_from_zero += epoch_diff_from_zero / len(data)

        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / len(data)}')


def train_on_past_simulations(model_to_trn, trn_loader, trn_optimizer, trn_criterion, epochs=1, logging=False):
    # Move this to own function
    # with open(validation_file, 'r') as file:
    #     # Load the JSON data into a Python object
    #     val_game_sims = json.load(file)
    #
    # validation_data = []
    # for binary_grid, heads, p_num, game_progress, won_lost_or_tied in val_game_sims:
    #     target = get_model_evaluation(decay_function, game_progress, won_lost_or_tied)
    #
    #     model_input = get_model_input_from_raw_info(binary_grid, heads, p_num,
    #                                                 model_type=MODEL_TYPE,
    #                                                 is_part_of_batch=True)
    #
    #     validation_data.append((model_input, target))
    #
    # valid_loader = DataLoader(validation_data, batch_size=32, shuffle=True)
    #
    # print("Valid loader created")

    # Get all filenames in the directory


    print("dataloaders created")

    # best_valid_loss = float('inf')

    total_train_loss = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in trn_loader:
            # Forward pass
            outputs = model_to_trn(inputs)
            loss = trn_criterion(outputs, targets)

            # Backward pass and optimization
            trn_optimizer.zero_grad()
            loss.backward()
            trn_optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        total_train_loss += epoch_loss

        # model.eval()
        # valid_loss = 0
        # with torch.no_grad():
        #     for inputs, targets in valid_loader:
        #         outputs = model(inputs)
        #         loss = criterion(outputs, targets)
        #
        #         valid_loss += loss.item()
        #
        # valid_loss /= len(valid_loader)
        #
        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     best_model = deepcopy(model.state_dict())

        # print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        print(f"Epoch: {epoch + 1}, Train Loss: {epoch_loss:.4f}")

    print("Training completed")
    average_train_loss = total_train_loss / epochs
    return average_train_loss

def process_json_data(filepath, model_type, decay_fn, tie_is_neutral):
    with open(filepath, 'r') as file:
        # Load the JSON data into a Python object
        game_sims = json.load(file)

    processed_data = []
    for binary_grid, heads, p_num, game_progress, won_lost_or_tied in game_sims:
        target = get_model_evaluation(decay_fn, game_progress, won_lost_or_tied, tie_is_neutral)

        model_input = get_model_input_from_raw_info(binary_grid, heads, p_num,
                                                    model_type=model_type,
                                                    is_part_of_batch=True)

        processed_data.append((model_input, target))

    return processed_data


def get_weights_sum_of_squares(model_to_sum):
    total_sum_of_squares = torch.tensor(0, dtype=torch.float32).to(device)
    for name, param in model_to_sum.named_parameters():
        if param.requires_grad and param.data is not None:
            total_sum_of_squares += torch.sum(param.data.pow(2))
    return total_sum_of_squares.item()


# maybe this function is unecessaryily complicated
def sorted_dir(directory):
    # Regular expression to match numbers in a string
    num_re = re.compile(r'(\d+)')

    # Helper function to convert a string to integer if it is a number
    def atoi(text):
        return int(text) if text.isdigit() else text

    # Helper function to split and parse the filenames
    def natural_keys(text):
        return [atoi(c) for c in re.split(num_re, text)]

    return sorted(os.listdir(directory), key=natural_keys)

def process_sims_and_train_loop(json_files, logging=True):

    train_iter = 0
    for json_file in json_files:
        processed_data = process_json_data(json_file, MODEL_TYPE, DECAY_FN, TIE_IS_NEUTRAL)
        train_loader = DataLoader(processed_data, batch_size=BATCH_SIZE, shuffle=True)
        print("Data loaded from json file:", json_file)
        avg_train_loss = train_on_past_simulations(model, train_loader, optimizer, criterion)

        if logging:
            update_training_logs(avg_train_loss, get_weights_sum_of_squares(model), train_iter)
            train_iter += 1


NUM_PLAYERS = 2
RANDOM_START_MOVES = 8
GAME_LENGTH_THRESHOLD = 8
TEMP_DECAY_FACTOR = 1
INITIAL_TEMP = 0.05
DECAY_FN = lambda x: x ** 5
LR = 0.0001
OPTIMIZER_TYPE = optim.Adam
BATCH_SIZE = 32
TIE_IS_NEUTRAL = False

GAME_DIMENSION = 20
MAX_GAME_ITERATIONS = 50
ITERATIONS_BEFORE_LOGGING = 50
ITERATIONS_BEFORE_SAVING_DATASET = 50

CURR_MODEL_ITER = 2
SIMULATE_TRAIN_CYCLES = 100

MODEL_TYPE = EvaluationNetConv2
MODEL_DESCRIPTION = "conv-net-v2-20x20-1e-4-LR"
CHECKPOINT_FILE = "iteration-1-past-data.pt"
MAIN_SIMULATION_DIR = "K://tron-simulation-data/"
INNER_SIMULATION_DIR = "conv-net-v2-20x20/"

TRAIN_ONLY = True
SHOW_GAMES = True
LOAD_MODEL = False
DO_MCTS = False
MONTE_ITERATIONS = 10

checkpoints_dir = "checkpoints/" + MODEL_DESCRIPTION + "/"
outer_sims_dir = MAIN_SIMULATION_DIR + INNER_SIMULATION_DIR

if LOAD_MODEL:
    model = torch.load(checkpoints_dir + CHECKPOINT_FILE).to(device)
else:
    model = MODEL_TYPE(GAME_DIMENSION).to(device)

criterion = nn.MSELoss()
optimizer = OPTIMIZER_TYPE(model.parameters(), lr=LR)

MCTS_OBJ = MCTS(model, n_simulations=MONTE_ITERATIONS) if DO_MCTS else None

if __name__ == '__main__':
    print("Cuda available?", torch.cuda.is_available())
    # cProfile.run('game_loop()')

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    if TRAIN_ONLY:
        training_logs_dir = "training-logs-" + str(GAME_DIMENSION) + "x" + str(GAME_DIMENSION) + "/"
        writer = SummaryWriter(training_logs_dir + MODEL_DESCRIPTION + "/iteration-" + str(CURR_MODEL_ITER))

        curr_sims_dirs = sorted_dir(outer_sims_dir)

        json_files = []
        for sim_dir in curr_sims_dirs:
            for sim_file in os.listdir(outer_sims_dir + sim_dir):
                json_files.append(outer_sims_dir + sim_dir + "/" + sim_file)
        # sim_dirs = os.listdir(MAIN_sims_dir)

        process_sims_and_train_loop(json_files, logging=True)

        formatted_datetime = datetime.now().strftime("%m-%d-%H-%M-%S")

        checkpoint_out_file = checkpoints_dir + "iteration-" + str(
            CURR_MODEL_ITER) + "-past-data" + formatted_datetime + ".pt"
        torch.save(model, checkpoint_out_file)

    else:

        if not os.path.exists(outer_sims_dir):
            os.mkdir(outer_sims_dir)

        for i in range(SIMULATE_TRAIN_CYCLES):
            runs_dir = "runs-" + str(GAME_DIMENSION) + "x" + str(GAME_DIMENSION) + "/"
            writer = SummaryWriter(runs_dir + MODEL_DESCRIPTION + "/iteration-" + str(CURR_MODEL_ITER))

            curr_sims_dir = outer_sims_dir + "iteration-" + str(CURR_MODEL_ITER) + "/"
            print("curr sims dir:", curr_sims_dir)

            if not os.path.exists(curr_sims_dir):
                os.mkdir(curr_sims_dir)

            game_loop()

            json_files = []
            for sim_file in os.listdir(curr_sims_dir):
                json_files.append(curr_sims_dir + "/" + sim_file)

            process_sims_and_train_loop(json_files, logging=False)

            formatted_datetime = datetime.now().strftime("%m-%d-%H-%M-%S")
            checkpoint_out_file = checkpoints_dir + "iteration-" + str(CURR_MODEL_ITER) + ".pt"
            torch.save(model, checkpoint_out_file)

            CURR_MODEL_ITER += 1
