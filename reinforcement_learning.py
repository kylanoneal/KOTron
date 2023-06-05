import torch

from pytorch_models import *
from model_architectures import *
from MCTS import *
from UtilityGUI import show_game_state

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

import cProfile

TEMP_REDUCTION_FACTOR = 0.999
INITIAL_TEMP = 0.2
MAX_GAME_ITERATIONS = 1000

ITERATIONS_BEFORE_LOGGING = 50

# Default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/converging')

device = "cpu" if torch.cuda.is_available() else "cpu"
print("DEVICE: ", device)
# model = torch.load("reinforcement_model_5.pth", map_location=torch.device(device))
# model = torch.load("checkpoints/20x20_model06-02-17-30", map_location=torch.device(device))
model = DirectionValueNetConv().to(device)

mcts = MCTS(model, n_simulations=4, exploration_factor=0.5)

current_game_iteration = 0
cumulative_loss = 0
cumulative_game_lengths = 0


def game_loop():
    num_players = 2

    game_dataset = []
    for i in range(MAX_GAME_ITERATIONS):

        global current_game_iteration
        current_game_iteration = i
        # print("MCTS model weights")
        # print_weights_sum_of_squares(mcts.model)
        print("GAME ITERATION:", i, "\n\n\n")

        game_data = []

        game = BMTron(num_players=num_players, dimension=40)
        while not game.winner_found:

            turn_data = []

            for player_num in range(num_players):
                # root_node = Node(game, player_num)
                # actions, action_probs = mcts.search(root_node)

                # print("LENGTH OF ACTIONS:", len(actions))
                # action = Directions(np.random.choice(range(4), size=1, p=action_probs))

                action = get_next_action(game, player_num)
                game.update_direction(player_num, action)

                model_input = get_model_input_from_game_state(game, player_num)
                # GEt rid of action probs?
                # action_probs = None
                turn_data.append(model_input)

            game_data.append(turn_data)
            game.move_racers()
            game.check_for_winner()

            show_game_state(game)

        winner_player_num = game.winner_player_num
        print("Winner player num:", winner_player_num)

        labeled_game_data = []

        decay_factor = 0.95
        curr_decay = 1
        # Start with the ending moves and decay towards the start
        # game_data.reverse()

        global cumulative_game_lengths
        cumulative_game_lengths += len(game_data)

        for turn_num, turn in enumerate(game_data):
            value_magnitude = turn_num / len(game_data)
            for player_num, model_input in enumerate(turn):
                # print("printing head locs:")
                # print_head_locations(model_input)
                if winner_player_num == -1:
                    evaluation = 0
                else:
                    evaluation = value_magnitude if player_num == winner_player_num else -value_magnitude
                labeled_game_data.append((model_input, evaluation))

            curr_decay *= decay_factor

        # game_dataset.extend(labeled_game_data)
        training(labeled_game_data)
        update_tensorboard()

        if (i + 1) % 300 == 0:
            current_datetime = datetime.now()

            # Format the date and time
            formatted_datetime = current_datetime.strftime("%m-%d-%H-%M")
            torch.save(model, "checkpoints/20x20_model" + formatted_datetime)

            non_tensor_data = [(model_input.tolist(), evaluation) for model_input, evaluation in game_dataset]
            with open("game_simulation_data/game_sims" + formatted_datetime + ".json", 'w') as file:
                json.dump(non_tensor_data, file)

            # Clear list after outputting to file
            game_dataset = []

        print("after training:")
        print_weights_sum_of_squares(model)


def update_tensorboard():
    global cumulative_loss, cumulative_game_lengths

    if (current_game_iteration + 1) % ITERATIONS_BEFORE_LOGGING == 0:
        writer.add_scalar('Avg length of game', cumulative_game_lengths / ITERATIONS_BEFORE_LOGGING, current_game_iteration)
        writer.add_scalar('training loss', cumulative_loss / ITERATIONS_BEFORE_LOGGING, current_game_iteration)
        cumulative_game_lengths = 0
        cumulative_loss = 0


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
    available_actions = []
    evaluations = []

    for i, action in enumerate(list(Directions)):
        # print("I: ", i, "action: ", action)
        if not BMTron.are_opposite_directions(game.players[player_num].direction, action):
            next_game_state = deepcopy(game)
            next_game_state.update_direction(player_num, action)
            next_game_state.move_racers()

            if next_game_state.players[player_num].can_move:
                available_actions.append(action)
                curr_eval = model(get_model_input_from_game_state(next_game_state, player_num))
                # print("curr eval: ", curr_eval)
                # print("Raw model output: ", model(get_model_input_from_game_state(game, player_num)))
                evaluations.append(curr_eval.item())

    # print("EVALUATIONS: ", evaluations, "\n\n\n")
    if len(available_actions):
        # print("Evaluations before softmax: ", evaluations)
        # print("Move evaluations:", evaluations)
        # print("Available actions:", available_actions)
        # evaluations = np.array(evaluations)
        # print("evaluations:", evaluations)

        temperature = INITIAL_TEMP * (TEMP_REDUCTION_FACTOR ** current_game_iteration)
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


def training(data):
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    # Number of epochs
    n_epochs = 1  # Modify as needed

    global cumulative_loss

    print("TRAINING...")
    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0

        random.shuffle(data)
        for state, state_value in data:
            # Convert outputs to tensors and move them to the device

            state_value = torch.tensor(state_value, dtype=torch.float32).reshape((1, 1)).to(device)

            # Backward pass and optimization
            optimizer.zero_grad()

            # Forward pass
            predicted_value_estimate = model(state)

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

        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / len(data)}')


def print_weights_sum_of_squares(model):
    total_sum_of_squares = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None:
            total_sum_of_squares += torch.sum(param.data.pow(2))
    print("Sum of squares of weights:", total_sum_of_squares.item())


if __name__ == '__main__':
    print("CUDA AVAILABLE?", torch.cuda.is_available())
    cProfile.run('game_loop()')
    # game_loop()
