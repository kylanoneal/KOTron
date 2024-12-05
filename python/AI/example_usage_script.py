import json
from rl_utils import simulate_game, get_dataloader, train
from pytorch_game_utils import save_model



# Doing the simulate and train cycle
if __name__=="__main__":

    config_path = Path("./configs/ex.config.json")
    rl_config = RLConfig(config_path)

    for i in range(rl_config.train_cycles):

        game_collection = GameCollection()

        for j in range(rl_config.game_iterations):

            game_container = simulate_game()
            game_collection.add_game(game_container)


        game_collection.save_to_json()

        train_loader = get_dataloader(game_collection)

        train(model, optimizer, criterion, train_loader)

        save_model(model)


# Training on past data
if __name__=="__main__":

    data_path = Path("./path_to_data")

    for json_file in data_path:

        game_collection = GameCollection.load_from_json()

        train_loader = get_dataloader(game_collection)

        train(model, optimizer, criterion, train_loader)

        save_model(model)

