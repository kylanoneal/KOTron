import pickle
from pathlib import Path

from tron.utils.OLD_DATA_UTILS import label_every_gamestate


def main():


    script_dir = Path(__file__).resolve().parent

    out_dir = script_dir / "perfect_data"
    out_dir.mkdir(exist_ok=True)


    for i in range(4, 5):

        unique_examples = label_every_gamestate(i)


        # out_path = out_dir / f"{i}x{i}.pkl"

        #     # Pickle to file
        # with out_path.open("wb") as f:
        #     pickle.dump(unique_examples, f)


if __name__=="__main__":
    main()