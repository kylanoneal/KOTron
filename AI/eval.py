import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from AI.Bot import ReinforcementBot, RandomBot, TronBot
from game.ko_tron import KOTron


def bot_vs_random(game: KOTron, bot1: TronBot, bot2: TronBot, bot1_description, bot2_description, num_games: int):

    assert bot1.player_num != bot2.player_num

    if 0 > bot1.player_num > 1 or 0 > bot2.player_num > 1:
        raise ValueError("NO sir")

    with open("./AI/results.json", 'r') as f:
        results_json = json.load(f)

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

    r['games_played'] = num_games
    r['tie_rate'] = round(games_tied / num_games, 3)
    r[f"{bot1_description}_winrate"] = round(bot1_wins / num_games, 3)
    r[f"{bot2_description}_winrate"] = round(bot2_wins / num_games, 3)

    with open("./AI/results.json", "w") as f:
        json.dump(results_json, f, indent=2)


if __name__ == '__main__':
    game = KOTron(2, dimension=10, random_starts=True)

    # Allows us to load any PyTorch model from 'model_architectures.py'
    # import sys
    # sys.path.append('../AI')

    model_path_100 = Path("./AI/model_checkpoints/20240726_train_only_v1/20240726_train_only_v1_100.pt")
    model_path_900 = Path("./AI/model_checkpoints/20240726_train_only_v1/20240726_train_only_v1_900.pt")
    model_path_57 = Path("./AI/model_checkpoints/20240726_legit_run_v2/57")

    model_path_132 = Path("./AI/model_checkpoints/20240727_legit_run_v4/132")
    #model_path_0 = Path("./AI/model_checkpoints/20240726_legit_run_v2/0")


    # bot1 = RandomBot(game, player_num=0)

    # bot1_description = "57_epoch_rl_bot"
    # bot1 = ReinforcementBot(game, 0, model_path_57)
    #
    # bot1_description = "100_epoch_rl_bot"
    # bot1 = ReinforcementBot(game, 0, model_path_100)

    bot1_description = "57_epoch_rl_bot"
    bot1 = ReinforcementBot(game, 0, model_path_57)


    bot2_description = "132_epoch_rl_bot"
    bot2 = ReinforcementBot(game, 1, model_path_132)
    #
    # #
    # bot2_description = "random_clown"
    # bot2 = RandomBot(game, player_num=1)

    bot_vs_random(game=game, bot1=bot1, bot2=bot2, bot1_description=bot1_description,
                  bot2_description=bot2_description, num_games=2000)
