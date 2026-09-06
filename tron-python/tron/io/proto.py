from collections.abc import Sequence

import tron.game as tron_game
from tron.io import tron_pb2

from tron.ai.minimax_oracle_pessimistic import OracleGameState, GameResult


GameData = Sequence[Sequence[tron_game.GameState]]


def to_proto(game_data: GameData) -> bytes:
    """Serialize game data to protobuf bytes.

    Args:
        game_data: Nested game-state sequences grouped by game.

    Returns:
        Serialized Games protobuf bytes.
    """
    games_pb = tron_pb2.Games()

    for game_states in game_data:
        game_pb = games_pb.games.add()

        for game_state in game_states:
            game_state_to_proto(game_state, game_pb.game_states.add())

    return games_pb.SerializeToString()


def from_proto(serialized_data: bytes) -> list[list[tron_game.GameState]]:
    """Deserialize protobuf bytes into game data.

    Args:
        serialized_data: Serialized Games protobuf bytes.

    Returns:
        Nested game states grouped by game.
    """
    games_pb = tron_pb2.Games()
    games_pb.ParseFromString(serialized_data)

    return [
        [game_state_from_proto(game_state_pb) for game_state_pb in game_pb.game_states]
        for game_pb in games_pb.games
    ]


def game_state_to_proto(
    game_state: tron_game.GameState,
    game_state_pb: tron_pb2.GameState | None = None,
) -> tron_pb2.GameState:
    """Convert a game state to its protobuf message.

    Args:
        game_state: Game state to convert.
        game_state_pb: Optional protobuf message to populate.

    Returns:
        Populated GameState protobuf message.
    """
    if not isinstance(game_state, tron_game.GameState):
        raise TypeError(
            "game_state must be a GameState instance, "
            f"got {type(game_state).__name__}"
        )

    if game_state.board < 0:
        raise ValueError("game_state.board must be non-negative")

    if game_state_pb is None:
        game_state_pb = tron_pb2.GameState()
    else:
        game_state_pb.Clear()

    game_state_pb.num_rows = game_state.num_rows
    game_state_pb.num_cols = game_state.num_cols
    game_state_pb.board = _unsigned_int_to_bytes(game_state.board)

    for player in game_state.players:
        if not isinstance(player, tron_game.Player):
            raise TypeError(
                "game_state.players must contain Player instances, "
                f"got {type(player).__name__}"
            )

        player_pb = game_state_pb.players.add()
        player_pb.idx = player.idx
        player_pb.can_move = player.can_move

    return game_state_pb


def game_state_from_proto(game_state_pb: tron_pb2.GameState) -> tron_game.GameState:
    """Convert a protobuf message to a game state.

    Args:
        game_state_pb: GameState protobuf message to convert.

    Returns:
        Reconstructed game state.
    """
    players = tuple(
        tron_game.Player(player_pb.idx, player_pb.can_move)
        for player_pb in game_state_pb.players
    )

    return tron_game.GameState(
        num_rows=game_state_pb.num_rows,
        num_cols=game_state_pb.num_cols,
        board=_unsigned_int_from_bytes(game_state_pb.board),
        players=players,
    )


def labeled_game_state_to_proto(
    labeled_game_state: OracleGameState,
    labeled_game_state_pb: tron_pb2.LabeledGameState | None = None,
) -> tron_pb2.LabeledGameState:
    """Convert a labeled game state to its protobuf message."""
    if not isinstance(labeled_game_state, OracleGameState):
        raise TypeError(
            "labeled_game_state must be a LabeledGameState instance, "
            f"got {type(labeled_game_state).__name__}"
        )

    if labeled_game_state_pb is None:
        labeled_game_state_pb = tron_pb2.LabeledGameState()
    else:
        labeled_game_state_pb.Clear()

    game_state_to_proto(
        labeled_game_state.game,
        labeled_game_state_pb.game_state,
    )

    labeled_game_state_pb.result = labeled_game_state.result.value
    labeled_game_state_pb.steps_to_result = labeled_game_state.steps_to_result

    return labeled_game_state_pb


def labeled_game_state_from_proto(
    labeled_game_state_pb: tron_pb2.LabeledGameState,
) -> OracleGameState:
    """Convert a protobuf message to a labeled game state."""

    if labeled_game_state_pb.result == 0:
        raise ValueError("Result should not be the unspecified value")
    
    return OracleGameState(
        game=game_state_from_proto(labeled_game_state_pb.game_state),
        result=GameResult(labeled_game_state_pb.result),
        steps_to_result=labeled_game_state_pb.steps_to_result,
    )


def labeled_game_states_to_proto(
    labeled_game_states: Sequence[OracleGameState],
) -> bytes:
    """Serialize labeled game states to protobuf bytes."""
    labeled_game_states_pb = tron_pb2.LabeledGameStates()

    for labeled_game_state in labeled_game_states:
        labeled_game_state_to_proto(
            labeled_game_state,
            labeled_game_states_pb.labeled_states.add(),
        )

    return labeled_game_states_pb.SerializeToString()


def labeled_game_states_from_proto(
    serialized_data: bytes,
) -> list[OracleGameState]:
    """Deserialize protobuf bytes into labeled game states."""
    labeled_game_states_pb = tron_pb2.LabeledGameStates()
    labeled_game_states_pb.ParseFromString(serialized_data)

    return [
        labeled_game_state_from_proto(labeled_game_state_pb)
        for labeled_game_state_pb in labeled_game_states_pb.labeled_states
    ]

def _unsigned_int_to_bytes(value: int) -> bytes:
    """Encode an unsigned integer as little-endian bytes.

    Args:
        value: Non-negative integer to encode.

    Returns:
        Little-endian byte representation.
    """
    byte_length = (value.bit_length() + 7) // 8
    return value.to_bytes(byte_length, byteorder="little", signed=False)


def _unsigned_int_from_bytes(value: bytes) -> int:
    """Decode little-endian bytes into an unsigned integer.

    Args:
        value: Little-endian byte representation.

    Returns:
        Decoded non-negative integer.
    """
    return int.from_bytes(value, byteorder="little", signed=False)
