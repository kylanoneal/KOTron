import torch
import json
import os, re, importlib
import warnings
from datetime import datetime
from pathlib import Path, WindowsPath
from tqdm import tqdm
from typing import Union, Sequence, Optional

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from AI.pytorch_game_utils import device, get_next_action, get_random_next_action, process_game_data, get_relevant_info_from_game_state, process_json_data
from game.utility_gui import show_game_state
from game.ko_tron import KOTron


def game_loop(model, cfg, game_data_out_file, loop_iter):
    cumulative_game_lengths = games_tied = games_not_tied = player_zero_wins = 0
    game_dataset = []

    for game_iter in range(cfg["max_game_iterations"]):

        game_data, winner_player_num = simulate_game(model, cfg)

        cumulative_game_lengths += len(game_data)

        if winner_player_num == -1:
            games_tied += 1
        else:
            games_not_tied += 1
            if winner_player_num == 0:
                player_zero_wins += 1

        game_dataset.extend(process_game_data(game_data, winner_player_num, cfg["game_length_threshold"]))

        if game_iter % 250 == 0:
            print(f"{game_iter} out of {cfg['max_game_iterations']} games played. Loop iter: {loop_iter}")

    # Update logs
    update_game_data_logs(cumulative_game_lengths, games_tied, games_not_tied, player_zero_wins,
                          cfg["max_game_iterations"], loop_iter)

    # Save games to json
    save_dataset(game_dataset, game_data_out_file)


# Maybe don't use the cfg here, or parse into some local variables to increase speed
def simulate_game(model, cfg):
    game_data = []

    game = KOTron(num_players=cfg["num_players"], dimension=cfg["game_dimension"],
                  random_starts=cfg["is_random_starts"])

    random_move_count = 0
    while not game.winner_found:

        for player_num in range(cfg["num_players"]):

            if random_move_count < cfg["random_start_moves"]:
                random_move_count += 1
                action = get_random_next_action(game, player_num)
            else:
                action = cfg["action_fn"](model, game, player_num, cfg["head_val"], cfg["temperature"])
            game.update_direction(player_num, action)

        game_data.append(get_relevant_info_from_game_state(game))
        game.move_racers()
        game.check_for_winner()

        if cfg["show_games"]:
            show_game_state(game)

    return game_data, game.winner_player_num


def update_game_data_logs(cum_game_lengths, games_tied, games_not_tied, player_zero_wins, n_games, loop_iter):
    writer.add_scalar('Avg length of game', cum_game_lengths / n_games,
                      loop_iter)
    writer.add_scalar("Percent player zero wins:", player_zero_wins / games_not_tied, loop_iter)
    writer.add_scalar("Percent games tied:", games_tied / n_games, loop_iter)


def update_training_logs(training_loss, avg_output_magnitude, weights_sum_of_squares, train_iter):
    writer.add_scalar('Training loss:', training_loss, train_iter)
    writer.add_scalar('Sum of squares of model weights:', weights_sum_of_squares, train_iter)
    writer.add_scalar('Average prediction magnitude:', avg_output_magnitude, train_iter)


def save_dataset(game_dataset, out_file):
    assert not out_file.exists(), "File already exists!"
    with open(out_file, 'w') as f:
        json.dump(game_dataset, f, indent=2)


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



def get_dataloader_from_json(json_file, cfg):
    processed_data = process_json_data(json_file, cfg["model_type"], cfg["decay_fn"], cfg["tie_is_neutral"],
                                       cfg["head_val"])
    return DataLoader(processed_data, batch_size=cfg["batch_size"], shuffle=True)


def process_sims_and_train_loop(model, cfg, json_file: Path, logging, train_iter):

    train_loader = get_dataloader_from_json(json_file, cfg)
    print("Data loaded from json file:", json_file)

    avg_train_loss, avg_out_magnitude = train_on_past_simulations(model, train_loader, optimizer, criterion)

    if logging:
        update_training_logs(avg_train_loss, avg_out_magnitude, get_weights_sum_of_squares(model), train_iter)


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
        print(f"Checkpoint file: {cfg['checkpoint_file']}")
        model = torch.load(cfg["checkpoint_file"]).to(device)
    else:
        model = script_cfg["model_type"](script_cfg["game_dimension"]).to(device)

    return model


def init_loss_and_optim(cfg, model):
    return cfg["loss_type"](), cfg["optimizer_type"](model.parameters(), cfg["lr"])


CONFIG_FILEPATH = "configs/20240725_10x10.config.json"

if __name__ == '__main__':
    script_cfg = parse_config(CONFIG_FILEPATH)
    print("Cuda available?", torch.cuda.is_available())

    checkpoints_folder = Path("./model_checkpoints") / script_cfg["model_description"]
    checkpoints_folder.mkdir(exist_ok=True, parents=True)

    warnings.warn("DUPLICATE RUN ID")

    script_model = init_model(script_cfg)
    criterion, optimizer = init_loss_and_optim(script_cfg, script_model)

    if script_cfg["train_only"]:

        game_data_path = Path(script_cfg["game_data_path"])
        assert game_data_path.exists(), f"Game data path doesn't exist: {game_data_path}"

        training_logdir = Path(f"./train_tb/{script_cfg["model_description"]}")

        writer = SummaryWriter(str(training_logdir))


        json_files = list(game_data_path.glob('*.json'))


        for i, json_file in tqdm(enumerate(json_files)):
            process_sims_and_train_loop(script_model, script_cfg, json_file, logging=True, train_iter=i)

            if i % 100 == 0:
                checkpoint_out_file = checkpoints_folder / f"{script_cfg['model_description']}_{i}.pt"
                torch.save(script_model, checkpoint_out_file)

    else:
        game_data_path = Path(script_cfg["game_data_path"]) / script_cfg["model_description"]
        game_data_path.mkdir(exist_ok=False)

        tb_folder = Path(f"./tb") / script_cfg['model_description']
        tb_folder.mkdir(exist_ok=False, parents=True)

        writer = SummaryWriter(str(tb_folder))

        for i in range(script_cfg["simulate_train_cycles"]):
            game_data_out_file = game_data_path / f"game_data_{i:04}.json"

            game_loop(script_model, script_cfg, game_data_out_file, i)

            process_sims_and_train_loop(script_model, script_cfg, game_data_out_file, logging=True, train_iter=i)

            formatted_datetime = datetime.now().strftime("%m-%d-%H-%M-%S")
            checkpoint_out_file = checkpoints_folder / str(i)
            torch.save(script_model, checkpoint_out_file)

            script_cfg["curr_model_iter"] += 1
