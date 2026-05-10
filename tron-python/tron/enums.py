from enum import Enum, auto

import random


class Direction(Enum):
    """(row, col) from top left"""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @staticmethod
    def are_opposite_directions(d1: "Direction", d2: "Direction") -> bool:
        dr1, dc1 = d1.value
        dr2, dc2 = d2.value
        return (dr1 + dr2 == 0) and (dc1 + dc2 == 0)

    @staticmethod
    def get_random_direction() -> "Direction":
        return random.choice(list(Direction))

    @staticmethod
    def fliplr(direction: "Direction") -> "Direction":

        if direction in [Direction.UP, Direction.DOWN]:
            return direction
        elif direction == Direction.LEFT:
            return Direction.RIGHT
        elif direction == Direction.RIGHT:
            return Direction.LEFT
        else:
            raise ValueError(f"Unexpected direction: {direction}")

    @staticmethod
    def rot90_counterclockwise(direction: "Direction") -> "Direction":

        if direction == Direction.UP:
            return Direction.LEFT
        elif direction == Direction.LEFT:
            return Direction.DOWN
        elif direction == Direction.DOWN:
            return Direction.RIGHT
        elif direction == Direction.RIGHT:
            return Direction.UP
        else:
            raise ValueError(f"Unexpected direction: {direction}")

    def transform(
        directions: list["Direction"],
        do_lr_flip: bool,
        n_rot_90: int,
    ) -> list["Direction"]:

        transformed_dirs = directions.copy()

        if do_lr_flip:
            transformed_dirs = [Direction.fliplr(d) for d in transformed_dirs]

        for _ in range(n_rot_90):
            transformed_dirs = [
                Direction.rot90_counterclockwise(d) for d in transformed_dirs
            ]

        return transformed_dirs
    

class GameStatus(Enum):

    IN_PROGRESS = auto()
    TIE = auto()
    WINNER = auto()

class PovGameResult(Enum):

    WINNER = auto()
    LOSER = auto()
    TIE = auto()