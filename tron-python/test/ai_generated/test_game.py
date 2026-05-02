import random
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np

import tron.game as bit_game
import tron.game_2d as grid_game
from tron.enums import Direction, GameStatus


def idx(row: int, col: int, num_cols: int) -> int:
    return row * num_cols + col


def board_from_indices(*indices: int) -> int:
    board = 0
    for index in indices:
        board |= bit_game.BIT_MASKS[index]
    return board


def make_2d_game(
    num_rows: int,
    num_cols: int,
    player_specs: tuple[tuple[int, int, bool], ...],
    extra_walls: tuple[tuple[int, int], ...] = (),
) -> grid_game.GameState2D:
    grid = np.zeros((num_rows, num_cols), dtype=bool)

    for row, col in extra_walls:
        grid[row, col] = True

    players = tuple(
        grid_game.Player2D(row=row, col=col, can_move=can_move)
        for row, col, can_move in player_specs
    )

    for player in players:
        grid[player.row, player.col] = True

    return grid_game.GameState2D(grid=grid, players=players)


def make_bit_game(
    num_rows: int,
    num_cols: int,
    player_specs: tuple[tuple[int, bool], ...],
    extra_walls: tuple[int, ...] = (),
) -> bit_game.GameState:
    player_indices = tuple(player_idx for player_idx, _ in player_specs)
    board = board_from_indices(*extra_walls, *player_indices)
    players = tuple(
        bit_game.Player(idx=player_idx, can_move=can_move)
        for player_idx, can_move in player_specs
    )
    return bit_game.GameState(num_rows, num_cols, board=board, players=players)


def expected_transformed_coord(
    shape: tuple[int, int],
    row: int,
    col: int,
    do_lr_flip: bool,
    n_rot_90: int,
) -> tuple[int, int]:
    marker = np.zeros(shape, dtype=bool)
    marker[row, col] = True

    if do_lr_flip:
        marker = np.fliplr(marker)

    marker = np.rot90(marker, k=n_rot_90)
    transformed_row, transformed_col = np.argwhere(marker).squeeze()
    return int(transformed_row), int(transformed_col)


def neutral_counterparts(row: int, col: int, board_size: int) -> set[tuple[int, int]]:
    counterparts = set()
    for do_lr_flip in (False, True):
        rotations = range(4) if do_lr_flip else range(1, 4)
        for n_rot_90 in rotations:
            transformed = expected_transformed_coord(
                (board_size, board_size),
                row,
                col,
                do_lr_flip=do_lr_flip,
                n_rot_90=n_rot_90,
            )
            if transformed != (row, col):
                counterparts.add(transformed)
    return counterparts


class DirectionTests(unittest.TestCase):
    def test_direction_values_are_row_col_deltas(self):
        self.assertEqual(Direction.UP.value, (-1, 0))
        self.assertEqual(Direction.DOWN.value, (1, 0))
        self.assertEqual(Direction.LEFT.value, (0, -1))
        self.assertEqual(Direction.RIGHT.value, (0, 1))
        self.assertEqual(
            list(Direction),
            [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT],
        )

    def test_are_opposite_directions_for_every_pair(self):
        expected_opposites = {
            (Direction.UP, Direction.DOWN),
            (Direction.DOWN, Direction.UP),
            (Direction.LEFT, Direction.RIGHT),
            (Direction.RIGHT, Direction.LEFT),
        }

        for d1 in Direction:
            for d2 in Direction:
                with self.subTest(d1=d1, d2=d2):
                    self.assertEqual(
                        Direction.are_opposite_directions(d1, d2),
                        (d1, d2) in expected_opposites,
                    )

    def test_get_random_direction_delegates_to_random_choice(self):
        with patch("tron.enums.random.choice", return_value=Direction.LEFT) as choice:
            self.assertIs(Direction.get_random_direction(), Direction.LEFT)
            choice.assert_called_once()
            self.assertEqual(list(choice.call_args.args[0]), list(Direction))

    def test_fliplr_mirrors_horizontal_directions_only(self):
        self.assertIs(Direction.fliplr(Direction.UP), Direction.UP)
        self.assertIs(Direction.fliplr(Direction.DOWN), Direction.DOWN)
        self.assertIs(Direction.fliplr(Direction.LEFT), Direction.RIGHT)
        self.assertIs(Direction.fliplr(Direction.RIGHT), Direction.LEFT)

    def test_fliplr_rejects_unknown_direction(self):
        with self.assertRaises(ValueError):
            Direction.fliplr("LEFT")

    def test_rot90_counterclockwise_cycles_directions(self):
        self.assertIs(Direction.rot90_counterclockwise(Direction.UP), Direction.LEFT)
        self.assertIs(Direction.rot90_counterclockwise(Direction.LEFT), Direction.DOWN)
        self.assertIs(Direction.rot90_counterclockwise(Direction.DOWN), Direction.RIGHT)
        self.assertIs(Direction.rot90_counterclockwise(Direction.RIGHT), Direction.UP)

    def test_rot90_counterclockwise_rejects_unknown_direction(self):
        with self.assertRaises(ValueError):
            Direction.rot90_counterclockwise("UP")

    def test_transform_returns_copy_without_mutating_input(self):
        directions = [Direction.UP, Direction.RIGHT]

        transformed = Direction.transform(directions, do_lr_flip=False, n_rot_90=0)

        self.assertEqual(transformed, directions)
        self.assertIsNot(transformed, directions)

    def test_transform_applies_flip_before_rotations(self):
        self.assertEqual(
            Direction.transform(
                [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT],
                do_lr_flip=True,
                n_rot_90=0,
            ),
            [Direction.UP, Direction.DOWN, Direction.RIGHT, Direction.LEFT],
        )
        self.assertEqual(
            Direction.transform([Direction.LEFT], do_lr_flip=True, n_rot_90=1),
            [Direction.UP],
        )

    def test_transform_rotates_multiple_steps(self):
        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]

        self.assertEqual(
            Direction.transform(directions, do_lr_flip=False, n_rot_90=1),
            [Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN],
        )
        self.assertEqual(
            Direction.transform(directions, do_lr_flip=False, n_rot_90=2),
            [Direction.DOWN, Direction.LEFT, Direction.UP, Direction.RIGHT],
        )
        self.assertEqual(
            Direction.transform(directions, do_lr_flip=False, n_rot_90=4),
            directions,
        )

    def test_game_status_members_are_unique(self):
        self.assertEqual(
            set(GameStatus),
            {GameStatus.IN_PROGRESS, GameStatus.TIE, GameStatus.WINNER},
        )
        self.assertEqual(len({status.value for status in GameStatus}), 3)


class BitboardDataModelTests(unittest.TestCase):
    def test_bit_masks_cover_the_declared_max_board(self):
        self.assertEqual(
            len(bit_game.BIT_MASKS),
            bit_game.MAX_ROWS * bit_game.MAX_COLS,
        )
        self.assertEqual(bit_game.BIT_MASKS[0], 1)
        self.assertEqual(bit_game.BIT_MASKS[7], 1 << 7)
        self.assertEqual(bit_game.BIT_MASKS[-1], 1 << (bit_game.MAX_ROWS * bit_game.MAX_COLS - 1))

    def test_player_and_game_state_are_frozen_value_objects(self):
        player = bit_game.Player(idx=3, can_move=True)
        game = bit_game.GameState(
            num_rows=2,
            num_cols=3,
            board=board_from_indices(3),
            players=(player,),
        )

        with self.assertRaises(FrozenInstanceError):
            player.idx = 4
        with self.assertRaises(FrozenInstanceError):
            game.board = 0

        self.assertEqual(player, bit_game.Player(idx=3, can_move=True))
        self.assertEqual(
            game,
            bit_game.GameState(2, 3, board_from_indices(3), (player,)),
        )
        self.assertEqual(hash(player), hash(bit_game.Player(3, True)))
        self.assertEqual(hash(game), hash(bit_game.GameState(2, 3, board_from_indices(3), (player,))))

    def test_pov_game_state_and_status_info_store_requested_fields(self):
        game = make_bit_game(2, 2, ((0, True), (3, True)))
        pov = bit_game.PovGameState(game_state=game, hero_index=0, opponent_index=1)
        status_info = bit_game.StatusInfo(GameStatus.WINNER, winner_index=1)

        self.assertIs(pov.game_state, game)
        self.assertEqual(pov.hero_index, 0)
        self.assertEqual(pov.opponent_index, 1)
        self.assertEqual(status_info.status, GameStatus.WINNER)
        self.assertEqual(status_info.winner_index, 1)
        self.assertIsNone(bit_game.StatusInfo(GameStatus.TIE).winner_index)

    def test_get_status_for_tie_winner_and_in_progress(self):
        cases = [
            (
                make_bit_game(2, 2, ((0, False), (3, False))),
                GameStatus.TIE,
                None,
            ),
            (
                make_bit_game(2, 2, ((0, False), (3, True))),
                GameStatus.WINNER,
                1,
            ),
            (
                make_bit_game(2, 2, ((0, True), (3, True))),
                GameStatus.IN_PROGRESS,
                None,
            ),
            (
                bit_game.GameState(2, 2, board=0, players=()),
                GameStatus.TIE,
                None,
            ),
        ]

        for game, expected_status, expected_winner in cases:
            with self.subTest(game=game):
                status_info = bit_game.get_status(game)
                self.assertIs(status_info.status, expected_status)
                self.assertEqual(status_info.winner_index, expected_winner)

    def test_get_status_uses_last_active_player_as_winner_index(self):
        game = make_bit_game(2, 3, ((0, False), (1, True), (2, False)))

        status_info = bit_game.get_status(game)

        self.assertIs(status_info.status, GameStatus.WINNER)
        self.assertEqual(status_info.winner_index, 1)


class BitboardFunctionTests(unittest.TestCase):
    def test_get_bit_reads_individual_occupancy_bits(self):
        board = board_from_indices(0, 5, 8)

        for index in range(9):
            with self.subTest(index=index):
                self.assertEqual(bit_game.get_bit(board, index), index in {0, 5, 8})

    def test_get_wall_indices_returns_only_bits_inside_board_dimensions(self):
        board = board_from_indices(0, 2, 4, 8, 12)
        game = bit_game.GameState(3, 3, board=board, players=())

        self.assertEqual(bit_game.get_wall_indices(game), [0, 2, 4, 8])

    def test_get_next_position_from_center(self):
        game = make_bit_game(3, 4, ((idx(1, 1, 4), True),))
        expected = {
            Direction.UP: (idx(0, 1, 4), False),
            Direction.DOWN: (idx(2, 1, 4), False),
            Direction.LEFT: (idx(1, 0, 4), False),
            Direction.RIGHT: (idx(1, 2, 4), False),
        }

        for direction, expected_result in expected.items():
            with self.subTest(direction=direction):
                self.assertEqual(
                    bit_game.get_next_position(game, 0, direction),
                    expected_result,
                )

    def test_get_next_position_marks_board_edges_out_of_bounds(self):
        edge_cases = [
            (
                make_bit_game(3, 4, ((idx(0, 2, 4), True),)),
                Direction.UP,
                (idx(-1, 2, 4), True),
            ),
            (
                make_bit_game(3, 4, ((idx(2, 2, 4), True),)),
                Direction.DOWN,
                (idx(3, 2, 4), True),
            ),
            (
                make_bit_game(3, 4, ((idx(1, 0, 4), True),)),
                Direction.LEFT,
                (idx(1, -1, 4), True),
            ),
            (
                make_bit_game(3, 4, ((idx(1, 3, 4), True),)),
                Direction.RIGHT,
                (idx(1, 4, 4), True),
            ),
        ]

        for game, direction, expected_result in edge_cases:
            with self.subTest(direction=direction, player=game.players[0]):
                self.assertEqual(
                    bit_game.get_next_position(game, 0, direction),
                    expected_result,
                )

    def test_get_next_position_rejects_unknown_direction(self):
        game = make_bit_game(3, 3, ((4, True),))

        with self.assertRaises(ValueError):
            bit_game.get_next_position(game, 0, "UP")

    def test_get_next_player_moves_into_open_cell(self):
        game = make_bit_game(3, 3, ((idx(1, 1, 3), True),))

        self.assertEqual(
            bit_game.get_next_player(game, 0, Direction.RIGHT),
            bit_game.Player(idx=idx(1, 2, 3), can_move=True),
        )

    def test_get_next_player_stays_and_stops_when_hitting_wall(self):
        game = make_bit_game(
            3,
            3,
            ((idx(1, 1, 3), True),),
            extra_walls=(idx(1, 2, 3),),
        )

        self.assertEqual(
            bit_game.get_next_player(game, 0, Direction.RIGHT),
            bit_game.Player(idx=idx(1, 1, 3), can_move=False),
        )

    def test_get_next_player_stays_and_stops_when_leaving_board(self):
        game = make_bit_game(3, 3, ((idx(0, 1, 3), True),))

        self.assertEqual(
            bit_game.get_next_player(game, 0, Direction.UP),
            bit_game.Player(idx=idx(0, 1, 3), can_move=False),
        )

    def test_get_next_player_ignores_direction_for_inactive_player(self):
        game = make_bit_game(3, 3, ((idx(1, 1, 3), False),))

        self.assertEqual(
            bit_game.get_next_player(game, 0, Direction.RIGHT),
            bit_game.Player(idx=idx(1, 1, 3), can_move=False),
        )

    def test_next_moves_all_active_players_and_adds_new_heads_to_board(self):
        game = make_bit_game(
            3,
            4,
            ((idx(1, 1, 4), True), (idx(1, 2, 4), True)),
        )

        next_game = bit_game.next(game, (Direction.UP, Direction.DOWN))

        self.assertEqual(
            next_game.players,
            (
                bit_game.Player(idx(0, 1, 4), True),
                bit_game.Player(idx(2, 2, 4), True),
            ),
        )
        self.assertEqual(
            bit_game.get_wall_indices(next_game),
            [idx(0, 1, 4), idx(1, 1, 4), idx(1, 2, 4), idx(2, 2, 4)],
        )
        self.assertEqual(next_game.num_rows, game.num_rows)
        self.assertEqual(next_game.num_cols, game.num_cols)

    def test_next_stops_players_that_hit_old_walls_or_boundaries(self):
        game = make_bit_game(
            3,
            3,
            ((idx(0, 1, 3), True), (idx(1, 1, 3), True)),
            extra_walls=(idx(1, 2, 3),),
        )

        next_game = bit_game.next(game, (Direction.UP, Direction.RIGHT))

        self.assertEqual(
            next_game.players,
            (
                bit_game.Player(idx(0, 1, 3), False),
                bit_game.Player(idx(1, 1, 3), False),
            ),
        )
        self.assertEqual(
            bit_game.get_wall_indices(next_game),
            [idx(0, 1, 3), idx(1, 1, 3), idx(1, 2, 3)],
        )

    def test_next_marks_head_on_collision_square_and_stops_both_players(self):
        game = make_bit_game(
            3,
            3,
            ((idx(1, 0, 3), True), (idx(1, 2, 3), True)),
        )

        next_game = bit_game.next(game, (Direction.RIGHT, Direction.LEFT))

        self.assertEqual(
            next_game.players,
            (
                bit_game.Player(idx(1, 1, 3), False),
                bit_game.Player(idx(1, 1, 3), False),
            ),
        )
        self.assertEqual(
            bit_game.get_wall_indices(next_game),
            [idx(1, 0, 3), idx(1, 1, 3), idx(1, 2, 3)],
        )

    def test_next_does_not_allow_players_to_swap_through_existing_heads(self):
        game = make_bit_game(
            1,
            3,
            ((idx(0, 0, 3), True), (idx(0, 1, 3), True)),
        )

        next_game = bit_game.next(game, (Direction.RIGHT, Direction.LEFT))

        self.assertEqual(
            next_game.players,
            (
                bit_game.Player(idx(0, 0, 3), False),
                bit_game.Player(idx(0, 1, 3), False),
            ),
        )
        self.assertEqual(bit_game.get_wall_indices(next_game), [0, 1])

    def test_get_possible_directions_respects_bounds_and_occupied_cells(self):
        game = make_bit_game(
            3,
            3,
            ((idx(1, 1, 3), True), (idx(2, 2, 3), True)),
            extra_walls=(idx(0, 1, 3), idx(1, 0, 3)),
        )

        self.assertEqual(
            bit_game.get_possible_directions(game, player_index=0),
            [Direction.DOWN, Direction.RIGHT],
        )

    def test_get_possible_directions_uses_player_position_even_if_inactive(self):
        game = make_bit_game(
            3,
            3,
            ((idx(1, 1, 3), False), (idx(2, 2, 3), True)),
            extra_walls=(idx(0, 1, 3), idx(1, 0, 3)),
        )

        self.assertEqual(
            bit_game.get_possible_directions(game, player_index=0),
            [Direction.DOWN, Direction.RIGHT],
        )


class GameState2DDataModelTests(unittest.TestCase):
    def test_player2d_equality_hashing_and_frozen_fields(self):
        player = grid_game.Player2D(1, 2, True)

        self.assertEqual(player, grid_game.Player2D(1, 2, True))
        self.assertNotEqual(player, grid_game.Player2D(1, 2, False))
        self.assertNotEqual(player, (1, 2, True))
        self.assertEqual(hash(player), hash(grid_game.Player2D(1, 2, True)))

        with self.assertRaises(FrozenInstanceError):
            player.row = 0

    def test_game_state2d_equality_hashing_and_frozen_fields(self):
        game = make_2d_game(2, 3, ((0, 0, True), (1, 2, True)))
        same_game = grid_game.GameState2D(game.grid.copy(), game.players)
        different_grid_array = game.grid.copy()
        different_grid_array[0, 1] = True
        different_grid = grid_game.GameState2D(
            different_grid_array,
            game.players,
        )

        self.assertEqual(game, same_game)
        self.assertNotEqual(game, different_grid)
        self.assertNotEqual(game, object())
        self.assertEqual(hash(game), hash(same_game))

        with self.assertRaises(FrozenInstanceError):
            game.players = ()

    def test_game_state2d_hash_tracks_mutable_grid_contents(self):
        game = make_2d_game(2, 2, ((0, 0, True), (1, 1, True)))
        original_hash = hash(game)

        game.grid[0, 1] = True

        self.assertNotEqual(hash(game), original_hash)

    def test_str_contains_grid_rows_and_player_locations(self):
        game = make_2d_game(2, 2, ((0, 0, True), (1, 1, False)))

        text = str(game)

        self.assertIn("[True, False]", text)
        self.assertIn("[False, True]", text)
        self.assertIn("Player2D 1: (0, 0)", text)
        self.assertIn("Player2D 2: (1, 1)", text)


class GameState2DPostInitValidationTests(unittest.TestCase):
    def test_post_init_accepts_valid_two_player_game(self):
        game = make_2d_game(
            3,
            3,
            ((0, 0, True), (2, 2, False)),
            extra_walls=((1, 1),),
        )

        self.assertEqual(game.grid.shape, (3, 3))
        self.assertEqual(len(game.players), 2)

    def test_post_init_rejects_non_tuple_players(self):
        with self.assertRaises(TypeError):
            grid_game.GameState2D(
                grid=np.ones((2, 2), dtype=bool),
                players=[
                    grid_game.Player2D(0, 0, True),
                    grid_game.Player2D(1, 1, True),
                ],
            )

    def test_post_init_rejects_non_numpy_grid(self):
        with self.assertRaises(TypeError):
            grid_game.GameState2D(
                grid=[[True, False], [False, True]],
                players=(
                    grid_game.Player2D(0, 0, True),
                    grid_game.Player2D(1, 1, True),
                ),
            )

    def test_post_init_rejects_non_bool_grid(self):
        with self.assertRaises(TypeError):
            grid_game.GameState2D(
                grid=np.ones((2, 2), dtype=np.uint8),
                players=(
                    grid_game.Player2D(0, 0, True),
                    grid_game.Player2D(1, 1, True),
                ),
            )

    def test_post_init_rejects_non_2d_grid(self):
        with self.assertRaises(ValueError):
            grid_game.GameState2D(
                grid=np.ones((2, 2, 1), dtype=bool),
                players=(
                    grid_game.Player2D(0, 0, True),
                    grid_game.Player2D(1, 1, True),
                ),
            )

    def test_post_init_rejects_non_player2d_elements(self):
        with self.assertRaises(TypeError):
            grid_game.GameState2D(
                grid=np.ones((2, 2), dtype=bool),
                players=(grid_game.Player2D(0, 0, True), (1, 1, True)),
            )

    def test_post_init_rejects_out_of_bounds_player_coordinates(self):
        with self.assertRaises(IndexError):
            grid_game.GameState2D(
                grid=np.ones((2, 2), dtype=bool),
                players=(
                    grid_game.Player2D(-1, 0, True),
                    grid_game.Player2D(1, 1, True),
                ),
            )
        with self.assertRaises(IndexError):
            grid_game.GameState2D(
                grid=np.ones((2, 2), dtype=bool),
                players=(
                    grid_game.Player2D(0, 0, True),
                    grid_game.Player2D(1, 2, True),
                ),
            )

    def test_post_init_rejects_grid_without_player_head_occupied(self):
        grid = np.zeros((2, 2), dtype=bool)
        grid[0, 0] = True

        with self.assertRaises(ValueError):
            grid_game.GameState2D(
                grid=grid,
                players=(
                    grid_game.Player2D(0, 0, True),
                    grid_game.Player2D(1, 1, True),
                ),
            )

    def test_post_init_rejects_duplicate_square_when_either_player_is_active(self):
        grid = np.zeros((2, 2), dtype=bool)
        grid[0, 0] = True

        for players in (
            (grid_game.Player2D(0, 0, True), grid_game.Player2D(0, 0, True)),
            (grid_game.Player2D(0, 0, True), grid_game.Player2D(0, 0, False)),
            (grid_game.Player2D(0, 0, False), grid_game.Player2D(0, 0, True)),
        ):
            with self.subTest(players=players):
                with self.assertRaises(ValueError):
                    grid_game.GameState2D(grid.copy(), players)

    def test_post_init_allows_duplicate_square_when_both_players_are_inactive(self):
        grid = np.zeros((2, 2), dtype=bool)
        grid[0, 0] = True
        game = grid_game.GameState2D(
            grid=grid,
            players=(grid_game.Player2D(0, 0, False), grid_game.Player2D(0, 0, False)),
        )

        self.assertEqual(
            game.players,
            (
                grid_game.Player2D(0, 0, False),
                grid_game.Player2D(0, 0, False),
            ),
        )

    def test_post_init_currently_supports_exactly_two_players(self):
        grid = np.ones((2, 2), dtype=bool)

        with self.assertRaises(NotImplementedError):
            grid_game.GameState2D(
                grid=grid,
                players=(grid_game.Player2D(0, 0, True),),
            )
        with self.assertRaises(NotImplementedError):
            grid_game.GameState2D(
                grid=grid,
                players=(
                    grid_game.Player2D(0, 0, True),
                    grid_game.Player2D(0, 1, True),
                    grid_game.Player2D(1, 0, True),
                ),
            )


class GameState2DFactoryTests(unittest.TestCase):
    def test_from_players_creates_grid_with_player_heads(self):
        players = (grid_game.Player2D(0, 1, True), grid_game.Player2D(2, 3, True))

        game = grid_game.GameState2D.from_players(players, num_rows=3, num_cols=4)

        expected_grid = np.zeros((3, 4), dtype=bool)
        expected_grid[0, 1] = True
        expected_grid[2, 3] = True
        np.testing.assert_array_equal(game.grid, expected_grid)
        self.assertEqual(game.players, players)

    def test_from_players_rejects_invalid_inputs_with_assertions(self):
        valid_player = grid_game.Player2D(0, 0, True)

        invalid_inputs = [
            ([valid_player], 2, 2),
            ((object(),), 2, 2),
            ((grid_game.Player2D(2, 0, True),), 2, 2),
            ((grid_game.Player2D(0, 0, False),), 2, 2),
            ((grid_game.Player2D(0, 0, True), grid_game.Player2D(0, 0, True)), 2, 2),
        ]

        for players, num_rows, num_cols in invalid_inputs:
            with self.subTest(players=players):
                with self.assertRaises(AssertionError):
                    grid_game.GameState2D.from_players(players, num_rows, num_cols)

    def test_new_game_random_starts_creates_distinct_active_players(self):
        np.random.seed(7)

        game = grid_game.GameState2D.new_game(
            num_players=2,
            num_rows=4,
            num_cols=5,
            random_starts=True,
        )

        self.assertEqual(game.grid.shape, (4, 5))
        self.assertEqual(len(game.players), 2)
        self.assertEqual(len({(p.row, p.col) for p in game.players}), 2)
        self.assertTrue(all(player.can_move for player in game.players))
        self.assertTrue(all(game.grid[player.row, player.col] for player in game.players))

    def test_new_game_random_starts_with_obstacles_keeps_bool_grid_and_heads(self):
        np.random.seed(11)

        game = grid_game.GameState2D.new_game(
            num_players=2,
            num_rows=5,
            num_cols=4,
            random_starts=True,
            obstacle_density=0.25,
        )

        self.assertEqual(game.grid.dtype, bool)
        self.assertGreaterEqual(int(game.grid.sum()), 2)
        self.assertLessEqual(int(game.grid.sum()), 2 + int(5 * 4 * 0.25))
        self.assertTrue(all(game.grid[player.row, player.col] for player in game.players))

    def test_new_game_rejects_unsupported_or_impossible_requests(self):
        with self.assertRaises(ValueError):
            grid_game.GameState2D.new_game(obstacle_density=0.81, random_starts=True)
        with self.assertRaises(ValueError):
            grid_game.GameState2D.new_game(num_players=5, num_rows=2, num_cols=2, random_starts=True)
        with self.assertRaises(NotImplementedError):
            grid_game.GameState2D.new_game(random_starts=False)
        with self.assertRaises(NotImplementedError):
            grid_game.GameState2D.new_game(num_players=3, random_starts=True, neutral_starts=True)

    def test_new_game_neutral_starts_place_second_player_on_a_symmetry(self):
        board_size = 7

        for seed in range(20):
            with self.subTest(seed=seed):
                random.seed(seed)
                game = grid_game.GameState2D.new_game(
                    num_players=2,
                    num_rows=board_size,
                    num_cols=board_size,
                    random_starts=True,
                    neutral_starts=True,
                )
                p0, p1 = game.players

                self.assertNotEqual((p0.row, p0.col), (p1.row, p1.col))
                self.assertIn(
                    (p1.row, p1.col),
                    neutral_counterparts(p0.row, p0.col, board_size),
                )

    def test_new_game_neutral_starts_raises_after_retry_exhaustion(self):
        with patch("tron.game_2d.random.randrange", return_value=1):
            with patch("tron.game_2d.random.random", return_value=0.0):
                with self.assertRaises(RuntimeError):
                    grid_game.GameState2D.new_game(
                        num_players=2,
                        num_rows=3,
                        num_cols=3,
                        random_starts=True,
                        neutral_starts=True,
                    )


class GameState2DTransformTests(unittest.TestCase):
    def test_transform_matches_numpy_grid_transform_for_all_symmetries(self):
        game = make_2d_game(
            2,
            3,
            ((0, 0, True), (1, 2, False)),
            extra_walls=((0, 2),),
        )

        for do_lr_flip in (False, True):
            for n_rot_90 in range(4):
                with self.subTest(do_lr_flip=do_lr_flip, n_rot_90=n_rot_90):
                    transformed = grid_game.GameState2D.transform(
                        game,
                        do_lr_flip=do_lr_flip,
                        n_rot_90=n_rot_90,
                    )
                    expected_grid = game.grid.copy()
                    if do_lr_flip:
                        expected_grid = np.fliplr(expected_grid)
                    expected_grid = np.rot90(expected_grid, k=n_rot_90)

                    np.testing.assert_array_equal(transformed.grid, expected_grid)
                    self.assertEqual(
                        transformed.players,
                        tuple(
                            grid_game.Player2D(
                                *expected_transformed_coord(
                                    game.grid.shape,
                                    player.row,
                                    player.col,
                                    do_lr_flip,
                                    n_rot_90,
                                ),
                                player.can_move,
                            )
                            for player in game.players
                        ),
                    )

    def test_transform_does_not_mutate_original_game(self):
        game = make_2d_game(3, 3, ((0, 1, True), (2, 1, True)), extra_walls=((1, 1),))
        original_grid = game.grid.copy()
        original_players = game.players

        grid_game.GameState2D.transform(game, do_lr_flip=True, n_rot_90=1)

        np.testing.assert_array_equal(game.grid, original_grid)
        self.assertEqual(game.players, original_players)

    def test_transform_four_rotations_or_two_flips_return_original(self):
        game = make_2d_game(3, 4, ((0, 1, True), (2, 3, True)), extra_walls=((1, 2),))

        rotated = grid_game.GameState2D.transform(game, do_lr_flip=False, n_rot_90=4)
        flipped_twice = grid_game.GameState2D.transform(
            grid_game.GameState2D.transform(game, do_lr_flip=True, n_rot_90=0),
            do_lr_flip=True,
            n_rot_90=0,
        )

        self.assertEqual(rotated, game)
        self.assertEqual(flipped_twice, game)


class GameState2DFunctionTests(unittest.TestCase):
    def test_next_is_a_plain_module_function(self):
        self.assertFalse(isinstance(grid_game.next, staticmethod))
        self.assertTrue(callable(grid_game.next))

    def test_in_bounds_for_corners_edges_and_negative_indices(self):
        grid = np.zeros((2, 3), dtype=bool)

        self.assertTrue(grid_game.in_bounds(grid, 0, 0))
        self.assertTrue(grid_game.in_bounds(grid, 1, 2))
        self.assertFalse(grid_game.in_bounds(grid, -1, 0))
        self.assertFalse(grid_game.in_bounds(grid, 0, -1))
        self.assertFalse(grid_game.in_bounds(grid, 2, 0))
        self.assertFalse(grid_game.in_bounds(grid, 0, 3))

    def test_get_possible_directions_respects_bounds_and_occupied_cells(self):
        game = make_2d_game(
            3,
            3,
            ((1, 1, True), (2, 2, True)),
            extra_walls=((0, 1), (1, 0)),
        )

        self.assertEqual(
            grid_game.get_possible_directions(game, player_index=0),
            [Direction.DOWN, Direction.RIGHT],
        )

    def test_get_possible_directions_uses_player_position_even_if_inactive(self):
        game = make_2d_game(
            3,
            3,
            ((1, 1, False), (2, 2, True)),
            extra_walls=((0, 1), (1, 0)),
        )

        self.assertEqual(
            grid_game.get_possible_directions(game, player_index=0),
            [Direction.DOWN, Direction.RIGHT],
        )

    def test_next_requires_one_direction_per_player(self):
        game = make_2d_game(2, 2, ((0, 0, True), (1, 1, True)))

        with self.assertRaises(AssertionError):
            grid_game.next(game, (Direction.RIGHT,))

    def test_next_moves_active_players_and_adds_new_heads_to_grid(self):
        game = make_2d_game(3, 4, ((1, 1, True), (1, 2, True)))

        next_game = grid_game.next(game, (Direction.UP, Direction.DOWN))

        self.assertEqual(
            next_game.players,
            (
                grid_game.Player2D(0, 1, True),
                grid_game.Player2D(2, 2, True),
            ),
        )
        expected_grid = np.zeros((3, 4), dtype=bool)
        for row, col in ((1, 1), (1, 2), (0, 1), (2, 2)):
            expected_grid[row, col] = True
        np.testing.assert_array_equal(next_game.grid, expected_grid)

    def test_next_stops_players_that_hit_walls_or_boundaries(self):
        game = make_2d_game(
            3,
            3,
            ((0, 1, True), (1, 1, True)),
            extra_walls=((1, 2),),
        )

        next_game = grid_game.next(game, (Direction.UP, Direction.RIGHT))

        self.assertEqual(
            next_game.players,
            (
                grid_game.Player2D(0, 1, False),
                grid_game.Player2D(1, 1, False),
            ),
        )
        expected_grid = np.zeros((3, 3), dtype=bool)
        for row, col in ((0, 1), (1, 1), (1, 2)):
            expected_grid[row, col] = True
        np.testing.assert_array_equal(next_game.grid, expected_grid)

    def test_next_keeps_inactive_players_in_place(self):
        game = make_2d_game(3, 3, ((1, 1, False), (2, 2, True)))

        next_game = grid_game.next(game, (Direction.RIGHT, Direction.LEFT))

        self.assertEqual(next_game.players[0], grid_game.Player2D(1, 1, False))
        self.assertEqual(next_game.players[1], grid_game.Player2D(2, 1, True))

    def test_next_marks_head_on_collision_square_and_stops_both_players(self):
        game = make_2d_game(3, 3, ((1, 0, True), (1, 2, True)))

        next_game = grid_game.next(game, (Direction.RIGHT, Direction.LEFT))

        self.assertEqual(
            next_game.players,
            (
                grid_game.Player2D(1, 1, False),
                grid_game.Player2D(1, 1, False),
            ),
        )
        self.assertTrue(next_game.grid[1, 1])

    def test_next_does_not_allow_players_to_swap_through_existing_heads(self):
        game = make_2d_game(1, 3, ((0, 0, True), (0, 1, True)))

        next_game = grid_game.next(game, (Direction.RIGHT, Direction.LEFT))

        self.assertEqual(
            next_game.players,
            (
                grid_game.Player2D(0, 0, False),
                grid_game.Player2D(0, 1, False),
            ),
        )
        np.testing.assert_array_equal(next_game.grid, game.grid)


class ConversionTests(unittest.TestCase):
    def test_from_2d_game_state_encodes_grid_and_players_as_row_major_bits(self):
        game_2d = make_2d_game(
            3,
            4,
            ((0, 1, True), (2, 3, False)),
            extra_walls=((1, 2),),
        )

        game = bit_game.from_2d_game_state(game_2d)

        self.assertEqual(game.num_rows, 3)
        self.assertEqual(game.num_cols, 4)
        self.assertEqual(
            game.board,
            board_from_indices(idx(0, 1, 4), idx(1, 2, 4), idx(2, 3, 4)),
        )
        self.assertEqual(
            game.players,
            (
                bit_game.Player(idx(0, 1, 4), True),
                bit_game.Player(idx(2, 3, 4), False),
            ),
        )

    def test_from_bitboard_decodes_row_major_bits_and_players(self):
        game = make_bit_game(
            3,
            4,
            ((idx(0, 1, 4), True), (idx(2, 3, 4), False)),
            extra_walls=(idx(1, 2, 4),),
        )

        game_2d = bit_game.from_bitboard(game)

        expected = make_2d_game(
            3,
            4,
            ((0, 1, True), (2, 3, False)),
            extra_walls=((1, 2),),
        )
        self.assertEqual(game_2d, expected)

    def test_round_trip_2d_to_bitboard_to_2d_preserves_state(self):
        game = make_2d_game(
            4,
            5,
            ((0, 3, True), (3, 1, False)),
            extra_walls=((1, 1), (2, 4)),
        )

        round_tripped = bit_game.from_bitboard(bit_game.from_2d_game_state(game))

        self.assertEqual(round_tripped, game)

    def test_round_trip_bitboard_to_2d_to_bitboard_preserves_state(self):
        game = make_bit_game(
            4,
            5,
            ((idx(0, 3, 5), True), (idx(3, 1, 5), False)),
            extra_walls=(idx(1, 1, 5), idx(2, 4, 5)),
        )

        round_tripped = bit_game.from_2d_game_state(bit_game.from_bitboard(game))

        self.assertEqual(round_tripped, game)

    def test_conversion_preserves_inactive_duplicate_collision_heads(self):
        game = make_bit_game(
            3,
            3,
            ((idx(1, 1, 3), False), (idx(1, 1, 3), False)),
            extra_walls=(idx(0, 0, 3),),
        )

        game_2d = bit_game.from_bitboard(game)

        self.assertEqual(
            game_2d.players,
            (
                grid_game.Player2D(1, 1, False),
                grid_game.Player2D(1, 1, False),
            ),
        )
        self.assertTrue(game_2d.grid[0, 0])
        self.assertTrue(game_2d.grid[1, 1])
        


class BitboardAnd2DParityTests(unittest.TestCase):
    def test_get_possible_directions_matches_between_representations(self):
        cases = [
            make_2d_game(3, 3, ((1, 1, True), (2, 2, True))),
            make_2d_game(3, 3, ((1, 1, True), (0, 2, False)), extra_walls=((0, 1),)),
            make_2d_game(4, 5, ((0, 0, True), (3, 4, True)), extra_walls=((1, 0), (2, 3))),
        ]

        for game_2d in cases:
            game = bit_game.from_2d_game_state(game_2d)
            for player_index in range(len(game.players)):
                with self.subTest(game=game_2d, player_index=player_index):
                    self.assertEqual(
                        bit_game.get_possible_directions(game, player_index),
                        grid_game.get_possible_directions(game_2d, player_index),
                    )

    def test_next_matches_between_representations_for_exhaustive_small_states(self):
        num_rows = 3
        num_cols = 3
        all_cells = [(row, col) for row in range(num_rows) for col in range(num_cols)]
        obstacle_sets = [
            (),
            ((1, 1),),
            ((0, 1), (2, 1)),
            ((1, 0), (1, 2)),
        ]

        checked_cases = 0
        for p0_row, p0_col in all_cells:
            for p1_row, p1_col in all_cells:
                if (p0_row, p0_col) == (p1_row, p1_col):
                    continue
                for extra_walls in obstacle_sets:
                    if (p0_row, p0_col) in extra_walls or (p1_row, p1_col) in extra_walls:
                        continue
                    game_2d = make_2d_game(
                        num_rows,
                        num_cols,
                        ((p0_row, p0_col, True), (p1_row, p1_col, True)),
                        extra_walls=extra_walls,
                    )
                    game = bit_game.from_2d_game_state(game_2d)

                    for p0_direction in Direction:
                        for p1_direction in Direction:
                            directions = (p0_direction, p1_direction)
                            with self.subTest(
                                p0=(p0_row, p0_col),
                                p1=(p1_row, p1_col),
                                extra_walls=extra_walls,
                                directions=directions,
                            ):
                                next_2d = grid_game.next(game_2d, directions)
                                next_bitboard_as_2d = bit_game.from_bitboard(
                                    bit_game.next(game, directions)
                                )
                                self.assertEqual(next_bitboard_as_2d, next_2d)
                                checked_cases += 1

        self.assertEqual(checked_cases, 3392)

    def test_multi_step_gameplay_stays_equivalent_between_representations(self):
        game_2d = make_2d_game(
            5,
            5,
            ((1, 1, True), (3, 3, True)),
            extra_walls=((2, 2),),
        )
        game = bit_game.from_2d_game_state(game_2d)
        direction_sequence = [
            (Direction.RIGHT, Direction.LEFT),
            (Direction.UP, Direction.DOWN),
            (Direction.RIGHT, Direction.LEFT),
            (Direction.DOWN, Direction.UP),
        ]

        for directions in direction_sequence:
            with self.subTest(directions=directions):
                game_2d = grid_game.next(game_2d, directions)
                game = bit_game.next(game, directions)
                self.assertEqual(bit_game.from_bitboard(game), game_2d)


if __name__ == "__main__":
    unittest.main()
