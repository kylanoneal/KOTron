# src/tron/__init__.py
# import the things you want to expose at "import tron" level
from .game import GameState, PovGameState, Player, get_status, next, get_possible_directions, Direction, StatusInfo, GameStatus

from .io.proto import to_proto, from_proto
