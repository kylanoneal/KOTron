import pickle
import imageio
from pathlib import Path

from tron.utils.data_utils import (
    label_every_gamestate,
)

from tron.game import GameState
from tron.ai.minimax_oracle_pessimistic import GameResult


def main():

    script_dir = Path(__file__).resolve().parent

    out_dir = script_dir / "perfect_data_oracle"
    out_dir.mkdir(exist_ok=True)

    grid_dim = 3

    oracle_table = label_every_gamestate(grid_dim=grid_dim)

    to_remove = set()

    for gs in oracle_table:

        symmetric_gs = GameState(
            gs.num_rows, gs.num_cols, gs.board, (gs.players[1], gs.players[0])
        )

        oracle_info = oracle_table[gs]
        symmetric_oracle_info = oracle_table[symmetric_gs]

        if oracle_info.result == GameResult.TIE:

            if symmetric_oracle_info.result != GameResult.TIE:
                print("bad! one tie one win")

                to_remove.update([gs, symmetric_gs])
        else:

            _diff1 = (
                oracle_info.result == GameResult.HERO_WIN
                and symmetric_oracle_info.result != GameResult.OPPO_WIN
            )
            _diff2 = (
                oracle_info.result == GameResult.OPPO_WIN
                and symmetric_oracle_info.result != GameResult.HERO_WIN
            )

            is_diff_result = _diff1 or _diff2

            if is_diff_result:

                to_remove.update([gs, symmetric_gs])
                print("Different result!")

            else:

                if oracle_info.steps_to_result != symmetric_oracle_info.steps_to_result:

                    _rem = (
                        gs
                        if oracle_info.steps_to_result
                        < symmetric_oracle_info.steps_to_result
                        else symmetric_gs
                    )
                    to_remove.add(_rem)
                    print(f"Diff steps to same winning result!")

    print(f"Removing {len(to_remove)} game states")

    for remove in to_remove:

        del oracle_table[remove]

    # viz_out_dir = script_dir / "viz_4x4_disambiguities"
    # viz_out_dir.mkdir(exist_ok=True)

    # for special_case_type, special_cases in special_case_dict.items():

    #     if len(special_cases) > 30:
    #         special_cases = special_cases[:30]

    #     special_model_examples = []
    #     for h0_gs, h0_oi, h1_oi in special_cases:

    #         h0_label = h0_oi.steps_to_result * (
    #             1 if h0_oi.result == GameResult.HERO_WIN else -1
    #         )

    #         h1_label = h1_oi.steps_to_result * (
    #             1 if h1_oi.result == GameResult.HERO_WIN else -1
    #         )

    #         h0_label = (
    #             abs(h0_oi.steps_to_result * 1000)
    #             if h0_oi.result == GameResult.TIE
    #             else h0_label
    #         )
    #         h1_label = (
    #             abs(h1_oi.steps_to_result * 1000)
    #             if h1_oi.result == GameResult.TIE
    #             else h1_label
    #         )

    #         special_model_examples.append(
    #             ModelExample(
    #                 LabeledExample(
    #                     PovGameState(h0_gs, 0, 1),
    #                     h0_label,
    #                 ),
    #                 prediction=h1_label,
    #             )
    #         )

    #     viz_grid_dim = int(len(special_model_examples) ** 0.5) + 1
    #     # random.shuffle(diff_examples)
    #     viz = render_model_example_image(
    #         special_model_examples, num_rows=viz_grid_dim, num_cols=viz_grid_dim
    #     )

    #     imageio.imwrite(
    #         viz_out_dir / f"{special_case_type.name}_{grid_dim}x{grid_dim}.png", viz
    #     )


if __name__ == "__main__":
    main()
