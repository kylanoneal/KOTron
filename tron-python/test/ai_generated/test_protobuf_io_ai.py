import unittest

from google.protobuf.descriptor import FieldDescriptor

import tron.game as tron_game
from tron.io import proto as proto_io
from tron.io import tron_pb2


def idx(row: int, col: int, num_cols: int) -> int:
    return row * num_cols + col


def board_from_indices(*indices: int) -> int:
    board = 0
    for index in indices:
        board |= tron_game.BIT_MASKS[index]
    return board


def make_game_state(
    num_rows: int,
    num_cols: int,
    player_specs: tuple[tuple[int, bool], ...],
    extra_walls: tuple[int, ...] = (),
) -> tron_game.GameState:
    players = tuple(
        tron_game.Player(idx=player_idx, can_move=can_move)
        for player_idx, can_move in player_specs
    )
    board = board_from_indices(*extra_walls, *(player.idx for player in players))

    return tron_game.GameState(
        num_rows=num_rows,
        num_cols=num_cols,
        board=board,
        players=players,
    )


class TestProtoSchema(unittest.TestCase):
    def test_game_state_messages_are_available_on_generated_module(self):
        self.assertTrue(hasattr(tron_pb2, "Player"))
        self.assertTrue(hasattr(tron_pb2, "GameState"))
        self.assertTrue(hasattr(tron_pb2, "Game"))
        self.assertTrue(hasattr(tron_pb2, "Games"))

    def test_2d_and_bitboard_specific_messages_are_not_available(self):
        self.assertFalse(hasattr(tron_pb2, "GridRow"))
        self.assertFalse(hasattr(tron_pb2, "BitboardPlayer"))
        self.assertFalse(hasattr(tron_pb2, "BitboardGameState"))
        self.assertFalse(hasattr(tron_pb2, "BitboardGame"))
        self.assertFalse(hasattr(tron_pb2, "BitboardGames"))

    def test_game_state_descriptor_matches_game_dataclass_shape(self):
        fields = tron_pb2.GameState.DESCRIPTOR.fields_by_name

        self.assertEqual(fields["num_rows"].type, FieldDescriptor.TYPE_INT32)
        self.assertEqual(fields["num_cols"].type, FieldDescriptor.TYPE_INT32)
        self.assertEqual(fields["board"].type, FieldDescriptor.TYPE_BYTES)
        self.assertTrue(fields["players"].is_repeated)
        self.assertEqual(fields["players"].message_type.name, "Player")

    def test_player_descriptor_matches_player_dataclass_shape(self):
        fields = tron_pb2.Player.DESCRIPTOR.fields_by_name

        self.assertEqual(fields["idx"].type, FieldDescriptor.TYPE_INT32)
        self.assertEqual(fields["can_move"].type, FieldDescriptor.TYPE_BOOL)
        self.assertNotIn("row", fields)
        self.assertNotIn("col", fields)


class TestGameStateProtoHelpers(unittest.TestCase):
    def test_game_state_to_proto_writes_dimensions_board_and_players(self):
        game_state = make_game_state(
            10,
            13,
            ((idx(0, 0, 13), True), (idx(9, 12, 13), False)),
            extra_walls=(idx(4, 7, 13), idx(6, 10, 13)),
        )

        game_state_pb = proto_io.game_state_to_proto(game_state)

        self.assertEqual(game_state_pb.num_rows, 10)
        self.assertEqual(game_state_pb.num_cols, 13)
        self.assertEqual(int.from_bytes(game_state_pb.board, "little"), game_state.board)
        self.assertGreater(len(game_state_pb.board), 8)
        self.assertEqual(len(game_state_pb.players), 2)
        self.assertEqual(game_state_pb.players[0].idx, idx(0, 0, 13))
        self.assertTrue(game_state_pb.players[0].can_move)
        self.assertEqual(game_state_pb.players[1].idx, idx(9, 12, 13))
        self.assertFalse(game_state_pb.players[1].can_move)

    def test_game_state_to_proto_populates_provided_message(self):
        game_state = make_game_state(2, 2, ((0, True), (3, True)))
        game_state_pb = tron_pb2.GameState()

        returned_pb = proto_io.game_state_to_proto(game_state, game_state_pb)

        self.assertIs(returned_pb, game_state_pb)
        self.assertEqual(game_state_pb.num_rows, 2)
        self.assertEqual(game_state_pb.num_cols, 2)
        self.assertEqual(int.from_bytes(game_state_pb.board, "little"), game_state.board)

    def test_game_state_from_proto_reconstructs_game_state(self):
        expected = make_game_state(
            4,
            5,
            ((idx(0, 4, 5), True), (idx(3, 0, 5), False)),
            extra_walls=(idx(1, 1, 5), idx(2, 3, 5)),
        )
        game_state_pb = proto_io.game_state_to_proto(expected)

        actual = proto_io.game_state_from_proto(game_state_pb)

        self.assertEqual(actual, expected)

    def test_game_state_from_proto_raises_for_invalid_post_init_state(self):
        game_state_pb = tron_pb2.GameState()
        game_state_pb.num_rows = 2
        game_state_pb.num_cols = 2
        game_state_pb.board = board_from_indices(0).to_bytes(1, "little")
        game_state_pb.players.add(idx=0, can_move=True)
        game_state_pb.players.add(idx=3, can_move=True)

        with self.assertRaises(ValueError):
            proto_io.game_state_from_proto(game_state_pb)

    def test_game_state_to_proto_rejects_unknown_state_type(self):
        with self.assertRaises(TypeError):
            proto_io.game_state_to_proto(object())

    def test_removed_2d_helpers_are_not_available(self):
        self.assertFalse(hasattr(proto_io, "to_2d_proto"))
        self.assertFalse(hasattr(proto_io, "from_2d_proto"))
        self.assertFalse(hasattr(proto_io, "game_state_2d_to_proto"))
        self.assertFalse(hasattr(proto_io, "game_state_2d_from_proto"))

    def test_removed_bitboard_names_are_not_available(self):
        self.assertFalse(hasattr(proto_io, "to_bitboard_proto"))
        self.assertFalse(hasattr(proto_io, "from_bitboard_proto"))
        self.assertFalse(hasattr(proto_io, "bitboard_game_state_to_proto"))
        self.assertFalse(hasattr(proto_io, "bitboard_game_state_from_proto"))


class TestGameCollectionProto(unittest.TestCase):
    def test_to_proto_and_from_proto_round_trip_nested_games(self):
        game_data = [
            [
                make_game_state(
                    3,
                    4,
                    ((idx(0, 1, 4), True), (idx(2, 3, 4), True)),
                    (idx(1, 1, 4),),
                ),
                make_game_state(
                    3,
                    4,
                    ((idx(0, 2, 4), True), (idx(2, 2, 4), False)),
                    (idx(1, 1, 4), idx(2, 3, 4)),
                ),
            ],
            [
                make_game_state(
                    6,
                    7,
                    ((idx(0, 0, 7), False), (idx(5, 6, 7), True)),
                    (idx(3, 3, 7),),
                ),
            ],
        ]

        serialized = proto_io.to_proto(game_data)
        actual = proto_io.from_proto(serialized)

        self.assertEqual(actual, game_data)

    def test_to_proto_handles_empty_input(self):
        serialized = proto_io.to_proto([])

        games_pb = tron_pb2.Games()
        games_pb.ParseFromString(serialized)

        self.assertEqual(len(games_pb.games), 0)
        self.assertEqual(proto_io.from_proto(serialized), [])

    def test_to_proto_can_store_boards_wider_than_uint64(self):
        high_index = 149
        game_state = tron_game.GameState(
            num_rows=15,
            num_cols=10,
            board=board_from_indices(0, 64, high_index),
            players=(tron_game.Player(0, True), tron_game.Player(high_index, True)),
        )

        serialized = proto_io.to_proto([[game_state]])
        actual = proto_io.from_proto(serialized)

        self.assertEqual(actual, [[game_state]])

    def test_to_proto_rejects_unknown_game_state_type(self):
        with self.assertRaises(TypeError):
            proto_io.to_proto([[object()]])

    def test_raw_games_message_parses_serialized_collection(self):
        game_state = make_game_state(2, 2, ((0, True), (3, True)), (1,))
        serialized = proto_io.to_proto([[game_state]])

        games_pb = tron_pb2.Games()
        games_pb.ParseFromString(serialized)

        self.assertEqual(len(games_pb.games), 1)
        self.assertEqual(len(games_pb.games[0].game_states), 1)
        self.assertEqual(
            int.from_bytes(games_pb.games[0].game_states[0].board, "little"),
            game_state.board,
        )


if __name__ == "__main__":
    unittest.main()
