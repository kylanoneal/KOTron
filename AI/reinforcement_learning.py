from pytorch_models import *
from AI.model_architectures import *
from AI.MCTS import *
# from UtilityGUI import show_game_state

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

import os

TEMP_DECAY_FACTOR = 1
INITIAL_TEMP = 0.2
MAX_GAME_ITERATIONS = 5000

ITERATIONS_BEFORE_LOGGING = 100
ITERATIONS_BEFORE_SAVING_DATASET = 1000

# Default `log_dir` is "runs" - we'll be more specific here


device = "cpu" if torch.cuda.is_available() else "cpu"
print("DEVICE: ", device)

# mcts = MCTS(model, n_simulations=4, exploration_factor=0.5)

current_game_iteration = 0
cumulative_loss = 0
cumulative_game_lengths = 0
cumulative_diff_from_zero = 0
game_dataset = []


def game_loop():
    global current_game_iteration, cumulative_game_lengths, game_dataset
    num_players = 2

    for i in range(MAX_GAME_ITERATIONS):
        current_game_iteration = i
        # print("MCTS model weights")
        # print_weights_sum_of_squares(mcts.model)
        print("GAME ITERATION:", i, "\n\n\n")

        game_data = []

        game = BMTron(num_players=num_players, dimension=40)
        while not game.winner_found:

            for player_num in range(num_players):
                # Monte Carlo:
                # root_node = Node(game, player_num)
                # actions, action_probs = mcts.search(root_node)

                action = get_next_action(game, player_num)
                game.update_direction(player_num, action)

            game_data.append(get_relevant_info_from_game_state(game))
            game.move_racers()
            game.check_for_winner()

            # show_game_state(game)

        winner_player_num = game.winner_player_num
        print("Winner player num:", winner_player_num)

        labeled_game_data = []

        # Start with the ending moves and decay towards the start
        # game_data.reverse()

        cumulative_game_lengths += len(game_data)

        for turn_num, (game_grid, heads) in enumerate(game_data):
            game_progress = turn_num / (len(game_data) - 1)

            for player_num, head in enumerate(heads):
                won_or_lost_or_tied = (
                    1 if player_num == winner_player_num else -1) if winner_player_num != -1 else 0
                game_dataset.append((game_grid, heads, player_num, game_progress, won_or_lost_or_tied))

        if (current_game_iteration + 1) % ITERATIONS_BEFORE_SAVING_DATASET == 0:
            save_dataset()
        if (current_game_iteration + 1) % ITERATIONS_BEFORE_LOGGING == 0:
            update_logs()


def decay_function(game_progress: float) -> float:
    """Give game progress from 0.0 - 1.0, decay the importance by cubing the input"""
    return game_progress ** 3


def update_logs():
    global cumulative_loss, cumulative_game_lengths, cumulative_diff_from_zero
    writer.add_scalar('Avg length of game', cumulative_game_lengths / ITERATIONS_BEFORE_LOGGING,
                      current_game_iteration)
    writer.add_scalar('Training loss', cumulative_loss / ITERATIONS_BEFORE_LOGGING, current_game_iteration)
    writer.add_scalar('Predictions diff from zero', cumulative_diff_from_zero / ITERATIONS_BEFORE_LOGGING,
                      current_game_iteration)
    cumulative_game_lengths = cumulative_loss = cumulative_diff_from_zero = 0


def save_dataset():
    global game_dataset

    # Use current time to distinguish datasets
    formatted_datetime = datetime.now().strftime("%m-%d-%H-%M")

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
        curr_eval = model(get_model_input_from_raw_info(grid, heads, player_num, model_type=model_type))
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
        # print("action probs after softmax:", action_probs, "\n\n")
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


def get_next_action_random(game, player_num):
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


def train_on_past_simulations(epochs=10):
    # Get all filenames in the directory
    filenames = os.listdir(curr_sims_dir)

    for filename in filenames:

        with open(curr_sims_dir + filename, 'r') as file:
            # Load the JSON data into a Python object
            game_sims = json.load(file)

        processed_data = []
        for binary_grid, heads, p_num, game_progress, won_lost_or_tied in game_sims:
            target = get_model_evaluation(decay_function, game_progress, won_lost_or_tied)

            model_input = get_model_input_from_raw_info(binary_grid, heads, p_num,
                                                        model_type=model_type,
                                                        is_part_of_batch=True)

            processed_data.append((model_input, target))

        print("JSON file extended")

        # Shuffle and split into train and validation
        train_data, valid_data = train_test_split(processed_data, test_size=0.2, random_state=36)

        print("Splits created")

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)

        print("dataloaders created")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        best_valid_loss = float('inf')

        print_weights_sum_of_squares()

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    valid_loss += loss.item()

            valid_loss /= len(valid_loader)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = deepcopy(model.state_dict())

            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        model.load_state_dict(best_model)

        print('Training completed.')


def print_weights_sum_of_squares():
    total_sum_of_squares = torch.tensor(0, dtype=torch.float32)
    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None:
            total_sum_of_squares += torch.sum(param.data.pow(2))
    print("Sum of squares of model weights:", total_sum_of_squares.item())


if __name__ == '__main__':
    # print("CUDA AVAILABLE?", torch.cuda.is_available())
    # cProfile.run('game_loop()')
    model_type = EvaluationAttentionConvNet
    curr_model_iteration = 1
    simulate_train_cycles = 5

    # model = torch.load("checkpoints/trained-on-v3-data-06-07-22-05.pt").to(device)
    model = model_type().to(device)
    model_description = "attn-net-v1"
    
    training_only = True
    
    if training_only:
        outer_sims_dir = "game-simulation-data/conv-net-v1/"
        sim_dirs = os.listdir(outer_sims_dir)
        
        for sim_dir in sim_dirs:
            curr_sims_dir = outer_sims_dir + sim_dir + "/"
            train_on_past_simulations()
            formatted_datetime = datetime.now().strftime("%m-%d-%H-%M")
            checkpoint_out_dir = "checkpoints/" + model_description + "/iteration-" + str(curr_model_iteration) + ".pt"
            torch.save(model, checkpoint_out_dir)
        
    else:
    
        os.mkdir("game-simulation-data/" + model_description + "/")
    
        for i in range(simulate_train_cycles):
            writer = SummaryWriter("runs/" + model_description + "/iteration-" + str(curr_model_iteration))
    
            curr_sims_dir = "game-simulation-data/" + model_description + "/iteration-" + str(curr_model_iteration)
            os.mkdir(curr_sims_dir)
            game_loop()
    
            train_on_past_simulations()
            formatted_datetime = datetime.now().strftime("%m-%d-%H-%M")
            checkpoint_out_dir = "checkpoints/" + model_description + "/iteration-" + str(curr_model_iteration) + ".pt"
            torch.save(model, checkpoint_out_dir)

            curr_model_iteration += 1
