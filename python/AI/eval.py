import json
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from AI.Bot import ReinforcementBot, RandomBot, TronBot
from game.ko_tron import KOTron, Directions
from AI.pytorch_game_utils import get_model_input_from_raw_info, get_model

game_state_dict = {
    "grid": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 0, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 0, 1],
        [2, 2, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "heads": [(0, 5), (1, 4)],
    "directions": [Directions.down, Directions.right]
}


def transpose_grid(xy_grid: list[list[int]]) -> list[list[int]]:
    grid_t = deepcopy(xy_grid)

    for x in range(len(xy_grid)):
        for y in range(len(xy_grid[0])):
            grid_t[y][x] = xy_grid[x][y]

    return grid_t

def eval_preset_positions(model):
    grid = transpose_grid(game_state_dict['grid'])

    heads = game_state_dict['heads']
    transposed_heads = [(heads[0][1], heads[0][0]), (heads[1][1], heads[1][0])]

    evals = []

    for player_num in range(2):
        model_input = get_model_input_from_raw_info(
            grid, game_state_dict['heads'], player_num=player_num
        )

        evals.append(model(model_input).item())

    for player_num in range(2):
        model_input = get_model_input_from_raw_info(
            game_state_dict['grid'], transposed_heads, player_num=player_num
        )

        evals.append(model(model_input).item())
    return evals



def bot_vs_random(
    game: KOTron,
    bot1: TronBot,
    bot2: TronBot,
    bot1_description,
    bot2_description,
    num_games: int,
    results_file: Path
):

    assert bot1.player_num != bot2.player_num

    if 0 > bot1.player_num > 1 or 0 > bot2.player_num > 1:
        raise ValueError("NO sir")

    if results_file.exists():
        with open(results_file, "r") as f:
            results_json = json.load(f)
    else:
        results_json = {}

    bot1_wins = 0
    bot2_wins = 0
    games_tied = 0

    random_bot = RandomBot(game, 1)

    for i in tqdm(range(num_games)):

        game.new_game_state()

        while not game.winner_found:
            bot1.bot_move()
            bot2.bot_move()
            game.move_racers()
            game.check_for_winner()

        if game.winner_player_num == 0:
            bot1_wins += 1
        elif game.winner_player_num == 1:
            bot2_wins += 1
        elif game.winner_player_num == -1:
            games_tied += 1
        else:
            ValueError(f"Unexpected winner player num: {game.winner_player_num}")

    # Get the current date and time
    now = datetime.now()

    # Format the datetime string
    formatted_datetime = now.strftime("%Y-%m-%d %H:%M:%S")

    results_json[formatted_datetime] = {}

    r = results_json[formatted_datetime]

    print(f"Total games played: {num_games}")
    print(f"Tie rate: {games_tied / num_games}")
    print(f"Bot 1 ({type(bot1)}) won {bot1_wins}, winrate: {bot1_wins / num_games}")
    print(f"Bot 2 ({type(bot2)}) won {bot2_wins}, winrate: {bot2_wins / num_games}")

    r["games_played"] = num_games
    r["tie_rate"] = round(games_tied / num_games, 3)
    r[f"{bot1_description}_winrate"] = round(bot1_wins / num_games, 3)
    r[f"{bot2_description}_winrate"] = round(bot2_wins / num_games, 3)

    with open(results_file, "w") as f:
        json.dump(results_json, f, indent=2)


def bot_v_bot_main():
    game = KOTron(2, dimension=10, random_starts=True)

    model_path_100 = Path(
        "./AI/model_checkpoints/20240726_train_only_v1/20240726_train_only_v1_100.pt"
    )
    model_path_900 = Path(
        "./AI/model_checkpoints/20240726_train_only_v1/20240726_train_only_v1_900.pt"
    )

    one_stride_200k_path = Path(
        "./model_checkpoints/0728_random_train_one_stride/0728_random_train_one_stride_19.pt"
    )

    bot1_description = "random_bot"
    bot1 = RandomBot(game, player_num=0)

    bot2_description = "one_stride_model_200k_games"
    bot2 = ReinforcementBot(game, 1, one_stride_200k_path)

    results_file = Path("./AI/random_200k_results.json")


    bot_vs_random(
        game=game,
        bot1=bot1,
        bot2=bot2,
        bot1_description=bot1_description,
        bot2_description=bot2_description,
        num_games=10000,
        results_file=results_file
    )


if __name__=="__main__":

    bot_v_bot_main()

    # model_path_100 = Path(
    #     "./AI/model_checkpoints/20240726_train_only_v1/20240726_train_only_v1_100.pt"
    # )
    # model_path_900 = Path(
    #     "./AI/model_checkpoints/20240726_train_only_v1/20240726_train_only_v1_900.pt"
    # )
    # model_path_57 = Path("./AI/model_checkpoints/20240726_legit_run_v2/57")

    # model_path_132 = Path("./AI/model_checkpoints/20240727_legit_run_v4/132")
    # # model_path_0 = Path("./AI/model_checkpoints/20240726_legit_run_v2/0")

    # model_path_176 = Path("./model_checkpoints/20240727_continuation_v3/176")

    # model_900 = get_model(model_path_900)
    # model_176 = get_model(model_path_176)
    # model_132 = get_model(model_path_132)

    # print("Model seen 900k games: ")
    # print(eval_preset_positions(model_900))

    # print("Model continued from that 900k:")
    # print(eval_preset_positions(model_176))

    # print("Model from scratch, 132k games:")
    # print(eval_preset_positions(model_132))