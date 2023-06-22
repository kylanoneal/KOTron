import os, re, importlib
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from AI.pytorch_game_utils import *
from unit_testing.UtilityGUI import show_game_state


def game_loop(model, cfg, dataset_save_path):
    cumulative_game_lengths = games_tied = games_not_tied = player_zero_wins = 0
    game_dataset = []

    for game_iter in range(cfg["max_game_iterations"]):

        game_data, winner_player_num = simulate_game(model, cfg)

        print("Game iteration:", game_iter)
        print("Winner:", winner_player_num, end="\n\n")

        cumulative_game_lengths += len(game_data)

        if winner_player_num == -1:
            games_tied += 1
        else:
            games_not_tied += 1
            if winner_player_num == 0:
                player_zero_wins += 1

        game_dataset.extend(process_game_data(game_data, winner_player_num, cfg["game_length_threshold"]))

        if (game_iter + 1) % cfg["iterations_before_saving_dataset"] == 0:
            save_dataset(game_dataset, dataset_save_path)
            game_dataset = []
        if (game_iter + 1) % cfg["iterations_before_logging"] == 0:
            update_game_data_logs(cumulative_game_lengths, games_tied, games_not_tied, player_zero_wins,
                                  cfg["iterations_before_logging"], game_iter)
            cumulative_game_lengths = games_tied = games_not_tied = player_zero_wins = 0


# Maybe don't use the cfg here, or parse into some local variables to increase speed
def simulate_game(model, cfg):
    game_data = []

    game = BMTron(num_players=cfg["num_players"], dimension=cfg["game_dimension"],
                  random_starts=cfg["is_random_starts"])

    random_move_count = 0
    while not game.winner_found:

        for player_num in range(cfg["num_players"]):

            if random_move_count < cfg["random_start_moves"]:
                action = get_random_next_action(game, player_num)
            else:
                action = cfg["action_fn"](model, game, player_num, cfg["head_val"], cfg["temperature"])
            game.update_direction(player_num, action)

        random_move_count += 1

        game_data.append(get_relevant_info_from_game_state(game))
        game.move_racers()
        game.check_for_winner()

        if cfg["show_games"]:
            show_game_state(game)

    return game_data, game.winner_player_num


def update_game_data_logs(cum_game_lengths, games_tied, games_not_tied, player_zero_wins, n_games, current_game_iter):
    writer.add_scalar('Avg length of game', cum_game_lengths / n_games,
                      current_game_iter)
    writer.add_scalar("Percent player zero wins:", player_zero_wins / games_not_tied, current_game_iter)
    writer.add_scalar("Percent games tied:", games_tied / n_games, current_game_iter)


def update_training_logs(training_loss, avg_output_magnitude, weights_sum_of_squares, train_iter):
    writer.add_scalar('Training loss:', training_loss, train_iter)
    writer.add_scalar('Sum of squares of model weights:', weights_sum_of_squares, train_iter)
    writer.add_scalar('Average prediction magnitude:', avg_output_magnitude, train_iter)


def save_dataset(game_dataset, save_dir):
    # Use current time to distinguish datasets
    formatted_datetime = datetime.now().strftime("%m-%d-%H-%M-%S")
    with open(save_dir + "game-sims-" + formatted_datetime + ".json", 'w') as file:
        json.dump(game_dataset, file)



def train_on_past_simulations(model, trn_loader, trn_optimizer, trn_criterion, epochs=1, logging=False):
    total_trn_loss = 0.0
    total_output_magnitude = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_output_magnitude = 0.0
        for inputs, targets in trn_loader:
            # Forward pass
            outputs = model(inputs)
            loss = trn_criterion(outputs, targets)

            epoch_output_magnitude += (torch.sum(torch.abs(outputs)).item() / len(outputs))

            # Backward pass and optimization
            trn_optimizer.zero_grad()
            loss.backward()
            trn_optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(trn_loader)
        epoch_output_magnitude /= len(trn_loader)
        total_trn_loss += epoch_loss
        total_output_magnitude += epoch_output_magnitude
        print(f"Epoch: {epoch + 1}, Train Loss: {epoch_loss:.4f}")

    print("Training completed")
    average_trn_loss = total_trn_loss / epochs
    average_output_magnitude = total_output_magnitude / epochs
    return average_trn_loss, average_output_magnitude


def get_weights_sum_of_squares(model):
    total_sum_of_squares = torch.tensor(0, dtype=torch.float32).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None:
            total_sum_of_squares += torch.sum(param.data.pow(2))
    return total_sum_of_squares.item()


# maybe this function is unecessaryily complicated
def get_sorted_dir(directory):
    # Regular expression to match numbers in a string
    num_re = re.compile(r'(\d+)')

    # Helper function to convert a string to integer if it is a number
    def atoi(text):
        return int(text) if text.isdigit() else text

    # Helper function to split and parse the filenames
    def natural_keys(text):
        return [atoi(c) for c in re.split(num_re, text)]

    return sorted(os.listdir(directory), key=natural_keys)


def get_dataloader_from_json(json_file, cfg):
    processed_data = process_json_data(json_file, cfg["model_type"], cfg["decay_fn"], cfg["tie_is_neutral"],
                                       cfg["head_val"])
    return DataLoader(processed_data, batch_size=cfg["batch_size"], shuffle=True)

def process_sims_and_train_loop(model, cfg, json_files, logging):
    train_iter = 0
    for json_file in json_files:
        train_loader = get_dataloader_from_json(json_file, cfg)
        print("Data loaded from json file:", json_file)
        
        avg_train_loss, avg_out_magnitude = train_on_past_simulations(model, train_loader, optimizer, criterion)

        if logging:
            update_training_logs(avg_train_loss, avg_out_magnitude, get_weights_sum_of_squares(model), train_iter)
            train_iter += 1


def parse_config(cfg_filepath):
    with open(cfg_filepath) as f:
        config_dict = json.load(f)

    # Handle type values                    
    for key in ("model_type", "optimizer_type", "loss_type", "action_fn"):
        value = config_dict[key]
        module_name, class_name = value.rsplit(".", 1)
        module = importlib.import_module(module_name)
        config_dict[key] = getattr(module, class_name)

    # Handle decay lambda function
    config_dict["decay_fn"] = eval(config_dict["decay_fn"])

    config_dict["model"] = config_dict["model_type"](config_dict["game_dimension"]).to(device)

    return config_dict


def init_model(cfg):
    if cfg["load_model"]:
        print("checkpoints_dir", checkpoints_dir, "file", cfg["checkpoint_file"])
        model = torch.load(checkpoints_dir + cfg["checkpoint_file"]).to(device)
    else:
        model = script_cfg["model_type"](script_cfg["game_dimension"]).to(device)

    return model


def init_loss_and_optim(cfg, model):
    return cfg["loss_type"](), cfg["optimizer_type"](model.parameters(), cfg["lr"])


CONFIG_FILEPATH = "configs/test-config.json"

if __name__ == '__main__':
    script_cfg = parse_config(CONFIG_FILEPATH)
    print("Cuda available?", torch.cuda.is_available())
    # cProfile.run('game_loop()')

    checkpoints_dir = "checkpoints/" + script_cfg["model_description"] + "/"
    outer_sims_dir = script_cfg["main_simulation_dir"] + script_cfg["inner_simulation_dir"]

    script_model = init_model(script_cfg)
    criterion, optimizer = init_loss_and_optim(script_cfg, script_model)

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    if script_cfg["train_only"]:
        training_logs_dir = "training-logs-" + str(script_cfg["game_dimension"]) + "x" + str(
            script_cfg["game_dimension"]) + "/"
        writer = SummaryWriter(
            training_logs_dir + script_cfg["model_description"] + "/iteration-" + str(script_cfg["curr_model_iter"]))

        curr_sims_dirs = get_sorted_dir(outer_sims_dir)

        json_files = []
        for sim_dir in curr_sims_dirs:
            for sim_file in os.listdir(outer_sims_dir + sim_dir):
                json_files.append(outer_sims_dir + sim_dir + "/" + sim_file)

        process_sims_and_train_loop(json_files, script_cfg, logging=True)

        formatted_datetime = datetime.now().strftime("%m-%d-%H-%M-%S")

        checkpoint_out_file = checkpoints_dir + "iteration-" + str(
            script_cfg["curr_model_iter"]) + "-past-data" + formatted_datetime + ".pt"
        torch.save(script_model, checkpoint_out_file)

    else:

        if not os.path.exists(outer_sims_dir):
            os.mkdir(outer_sims_dir)

        for i in range(script_cfg["simulate_train_cycles"]):
            runs_dir = "runs-" + str(script_cfg["game_dimension"]) + "x" + str(script_cfg["game_dimension"]) + "/"
            if script_cfg["temperature"] == 0.0:
                runs_dir = "evaluation-" + runs_dir

            writer = SummaryWriter(
                runs_dir + script_cfg["model_description"] + "/iteration-" + str(script_cfg["curr_model_iter"]))

            curr_sims_dir = outer_sims_dir + "iteration-" + str(script_cfg["curr_model_iter"]) + "/"
            print("curr sims dir:", curr_sims_dir)

            if not os.path.exists(curr_sims_dir):
                os.mkdir(curr_sims_dir)

            game_loop(script_model, script_cfg, curr_sims_dir)

            json_files = []
            for sim_file in os.listdir(curr_sims_dir):
                json_files.append(curr_sims_dir + "/" + sim_file)

            process_sims_and_train_loop(script_model, script_cfg, json_files, logging=False)

            formatted_datetime = datetime.now().strftime("%m-%d-%H-%M-%S")
            checkpoint_out_file = checkpoints_dir + "iteration-" + str(script_cfg["curr_model_iter"]) + ".pt"
            torch.save(script_model, checkpoint_out_file)

            script_cfg["curr_model_iter"] += 1
