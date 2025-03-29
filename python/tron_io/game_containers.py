import json

from enum import Enum
from pathlib import Path
from jsonschema import validate

from game.tron import Tron
from containers.game_schemas import GAME_COLLECTION_SCHEMA, GAME_SCHEMA

class GameResult(Enum):
    P1_WIN = "p1_win"
    P2_WIN = "p2_win"
    P3_WIN = "p3_win"
    P4_WIN = "p4_win"
    TIE = "tie"


class GameContainer:

    def __init__(self, game_width: int, game_height: int, game_result: GameResult, positions: list[Tron]) -> None:
        self.game_width = game_width
        self.game_height = game_height
        self.game_result = game_result
        self.positions = positions

    
    def to_json(self, out_file: Path) -> str:

        json_dict = {
            "game_width": self.game_width,
            "game_height": self.game_height,
            "game_result": self.game_result.value,
            "positions": [position.to_json() for position in self.positions]
        }

        json.dumps()

    @staticmethod
    def from_json(json_obj: dict, do_validation=True) -> 'GameContainer':

        if do_validation:
            validate(instance=json_obj, schema=GAME_SCHEMA)

        return GameContainer(game_width=json_obj['game_width'],
                             game_height=json_obj['game_height'],
                             game_result=GameResult(json_obj['game_result']),
                             positions=[Tron.from_json() for position in json_obj['positions']]
                            )


class GameCollection:

    def __init__(self) -> None:
        self.game_containers = []

    def __init__(self, game_containers: list[GameContainer]):
        self.game_containers = game_containers

    @staticmethod
    def load(file_path: Path) -> GameCollection:

        with open(file_path, "r") as f:
            json_game_coll = json.load(f)


        validate(instance=json_game_coll, schema=GAME_COLLECTION_SCHEMA)


        game_collection = [GameContainer.from_json() for json_game in json_game_coll]

        return GameCollection(game_collection)
