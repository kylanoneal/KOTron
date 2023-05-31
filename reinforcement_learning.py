import torch

from BMTron import *
from models import *
from model_architectures import *

from MCTS import *

from UtilityGUI import show_game_state

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DirectionValueNetConv().to(device)
mcts = MCTS(model, n_simulations=20, exploration_factor=0.000001)

def game_loop():

    num_players = 2
    game_iterations = 100

    for i in range(game_iterations):

        #print("MCTS model weights")
        #print_weights_sum_of_squares(mcts.model)
        print("GAME ITERATION:", i, "\n\n\n")

        game_data = []

        game = BMTron(num_players=num_players)
        while not game.winner_found:

            turn_data = []

            for player_num in range(num_players):
                root_node = Node(game, player_num)
                action_probs = mcts.search(root_node)
                print("MCTS action probs:", action_probs)

                action = Directions(np.random.choice(range(4), size=1, p=action_probs))
                game.update_direction(player_num, action)

                model_input = get_model_input_from_game_state(game, player_num)
                turn_data.append((model_input, action_probs))

            game_data.append(turn_data)
            game.move_racers()
            game.check_for_winner()

            show_game_state(game)

        winner_player_num = game.winner_player_num
        print("Winner player num:", winner_player_num)

        labeled_game_data = []
        for turn in game_data:
            for player_num, (model_input, action_probs) in enumerate(turn):
                if winner_player_num == -1:
                    evaluation = 0
                else:
                    evaluation = 1 if player_num == winner_player_num else -1
                labeled_game_data.append((model_input, action_probs, evaluation))

        training(labeled_game_data)
        #print("after training:")
        #print_weights_sum_of_squares(model)

    torch.save(model, "reinforcement_model.pth")

def training(data):

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs
    n_epochs = 10  # Modify as needed

    print("TRAINING DIS SHIT")
    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0

        for state, action_probs, value_estimate in data:

            # Convert outputs to tensors and move them to the device
            action_probs = torch.tensor(action_probs).reshape((1, 4)).to(device)

            value_estimate = torch.tensor(value_estimate, dtype=torch.float32).reshape((1, 1)).to(device)


            # Forward pass
            predicted_action_probs, predicted_value_estimate = model(state)


            # Define the loss function (CrossEntropy for action probs, MSE for value estimates)
            loss = F.cross_entropy(predicted_action_probs, action_probs) + \
                   F.mse_loss(predicted_value_estimate, value_estimate)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / len(data)}')


def print_weights_sum_of_squares(model):
    total_sum_of_squares = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None:
            total_sum_of_squares += torch.sum(param.data.pow(2))
    print("Sum of squares of weights:", total_sum_of_squares.item())
if __name__ == '__main__':
    game_loop()
