import pickle
import json
import imageio
import datetime

from pathlib import Path
from tqdm import tqdm


from tron.ai.minimax_oracle_pessimistic import OracleGameState

from tron.utils.data_utils import (
    make_oracle_data,
    swap_two_player_game,
    find_special_cases,
    make_tactics_from_oracle,
    subsample_and_augment

)

from tron.game import GameState, PovGameState
from tron.ai.minimax_oracle_pessimistic import GameResult

from tron.utils.viz_utils import render_game_state_image

from tron.io.proto import labeled_game_states_to_proto
from tron.io.json import tactic_to_dict

def main():


    script_dir = Path(__file__).resolve().parent

    out_dir = script_dir / "8x8_validation_data_v1"
    out_dir.mkdir(exist_ok=True)

    # out_dir = (
    #     outer_run_dir
    #     / f"{datetime.datetime.now().strftime("%Y%m%d-%H%M")}"
    # )
    # out_dir.mkdir(exist_ok=True)

    # game_data_out_dir = out_dir / "game_data"
    # game_data_out_dir.mkdir(exist_ok=True)

    # viz_out_dir = out_dir / "viz"
    # viz_out_dir.mkdir(exist_ok=True)

    grid_dim = 8

    tactics_json = []

    game_data = []

    for i in range(1000):



        game = GameState.new_game(num_players=2, num_rows=grid_dim, num_cols=grid_dim, random_starts=True, neutral_starts=True, obstacle_density=0.35)

        # viz = render_game_state_image(
        #     PovGameState(game, 0, 1)
        # )

        # imageio.imwrite(
        #     viz_out_dir / f"{i:04d}_start_position.png", viz
        # )

        oracle_examples, _special_cases = make_oracle_data(game)

        

        print(f"Game #{i}, {len(oracle_examples)} examples created")

        tactics = make_tactics_from_oracle(oracle_examples, _special_cases, 5)

        # print(f"\t-{len(tactics)} tactics created.\n\n")
        # if len(tactics) > 0:

        #     tactic_viz = render_game_state_image(
        #         tactics[0].pov_game_state
        #     )

        #     imageio.imwrite(
        #         viz_out_dir / f"{i:04d}_tactic.png", tactic_viz
        #     )


        tactics_json.extend([tactic_to_dict(t) for t in tactics])


        game_data.extend(subsample_and_augment(oracle_examples, keep_rate=0.01))
        

    print(f"{len(tactics_json)=}, {len(game_data)=}")

    # Serialize game data
    serialized_data: bytes = labeled_game_states_to_proto(game_data)

    # Save the serialized data to a file.
    with open(
        out_dir / f"{grid_dim}x{grid_dim}_validation_{i+1}games.bin", "wb"
    ) as f:
        f.write(serialized_data)

    with open(out_dir / "tactics.json", 'w') as f:
        json.dump(tactics_json, f, indent=2)





if __name__ == "__main__":
    main()
