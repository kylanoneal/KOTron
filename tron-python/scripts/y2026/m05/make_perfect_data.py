import pickle
from pathlib import Path

from tron.utils.data_utils import label_every_gamestate


def main():


    script_dir = Path(__file__).resolve().parent

    out_dir = script_dir / "perfect_data_oracle"
    out_dir.mkdir(exist_ok=True)


    for i in range(3, 5):

        oracle_table = label_every_gamestate(i)

        out_path = out_dir / f"{i}x{i}.pkl"

            # Pickle to file
        with out_path.open("wb") as f:
            pickle.dump(oracle_table, f)


if __name__=="__main__":
    main()