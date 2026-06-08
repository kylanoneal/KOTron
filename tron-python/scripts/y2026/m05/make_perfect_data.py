import pickle
import imageio
from pathlib import Path

from tron.utils.data_utils import (
    label_every_gamestate,
    disambiguate_oracle_tables,
    SpecialCase,
    render_model_example_image,
    GameResult,
    ModelExample,
    LabeledExample,
    PovGameState,
)


def main():

    script_dir = Path(__file__).resolve().parent

    out_dir = script_dir / "perfect_data_oracle"
    out_dir.mkdir(exist_ok=True)

    grid_dim = 4

    # h0_oracle_table = label_every_gamestate(grid_dim, 0)
    # h1_oracle_table = label_every_gamestate(grid_dim, 1)

    # with (out_dir / f"{grid_dim}x{grid_dim}_h0.pkl").open("wb") as f:
    #     pickle.dump(h0_oracle_table, f)

    # with (out_dir / f"{grid_dim}x{grid_dim}_h1.pkl").open("wb") as f:
    #     pickle.dump(h1_oracle_table, f)

    # with (out_dir / f"{grid_dim}x{grid_dim}_h0.pkl").open("rb") as f:
    #     h0_oracle_table = pickle.load(f)

    # with (out_dir / f"{grid_dim}x{grid_dim}_h1.pkl").open("rb") as f:
    #     h1_oracle_table = pickle.load(f)

    # h0_disambiguated_table, special_case_dict = disambiguate_oracle_tables(
    #     h0_oracle_table, h1_oracle_table
    # )

    # with (out_dir / f"{grid_dim}x{grid_dim}_disambig.pkl").open("wb") as f:
    #     pickle.dump(h0_disambiguated_table, f)

    # with (out_dir / f"{grid_dim}x{grid_dim}_special_cases.pkl").open("wb") as f:
    #     pickle.dump(special_case_dict, f)

    with (out_dir / f"{grid_dim}x{grid_dim}_special_cases.pkl").open("rb") as f:
        special_case_dict = pickle.load(f)

    viz_out_dir = script_dir / "viz_4x4_disambiguities"
    viz_out_dir.mkdir(exist_ok=True)

    for special_case_type, special_cases in special_case_dict.items():

        if len(special_cases) > 30:
            special_cases = special_cases[:30]

        special_model_examples = []
        for h0_gs, h0_oi, h1_oi in special_cases:

            h0_label = h0_oi.steps_to_result * (
                1 if h0_oi.result == GameResult.HERO_WIN else -1
            )

            h1_label = h1_oi.steps_to_result * (
                1 if h1_oi.result == GameResult.HERO_WIN else -1
            )

            h0_label = (
                abs(h0_oi.steps_to_result * 1000)
                if h0_oi.result == GameResult.TIE
                else h0_label
            )
            h1_label = (
                abs(h1_oi.steps_to_result * 1000)
                if h1_oi.result == GameResult.TIE
                else h1_label
            )

            special_model_examples.append(
                ModelExample(
                    LabeledExample(
                        PovGameState(h0_gs, 0, 1),
                        h0_label,
                    ),
                    prediction=h1_label,
                )
            )

        viz_grid_dim = int(len(special_model_examples) ** 0.5) + 1
        # random.shuffle(diff_examples)
        viz = render_model_example_image(
            special_model_examples, num_rows=viz_grid_dim, num_cols=viz_grid_dim
        )

        imageio.imwrite(
            viz_out_dir / f"{special_case_type.name}_{grid_dim}x{grid_dim}.png", viz
        )


if __name__ == "__main__":
    main()
