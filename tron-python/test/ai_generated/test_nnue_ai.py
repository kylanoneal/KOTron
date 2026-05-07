import pytest
import torch
from types import SimpleNamespace

import tron.game as game
import tron.ai.nnue as nnue
from tron.enums import Direction


def board_from_indices(*indices: int) -> int:
    board = 0
    for index in indices:
        board |= game.BIT_MASKS[index]
    return board


def make_game_state(
    num_rows: int = 3,
    num_cols: int = 3,
    player_specs: tuple[tuple[int, bool], tuple[int, bool]] = ((4, True), (8, True)),
    extra_walls: tuple[int, ...] = (),
) -> game.GameState:
    players = tuple(
        game.Player(idx=player_idx, can_move=can_move)
        for player_idx, can_move in player_specs
    )
    board = board_from_indices(*extra_walls, *(player.idx for player in players))

    return game.GameState(
        num_rows=num_rows,
        num_cols=num_cols,
        board=board,
        players=players,
    )


def make_pov(
    game_state: game.GameState | None = None,
    hero_index: int = 0,
    opponent_index: int = 1,
) -> game.PovGameState:
    return game.PovGameState(
        game_state or make_game_state(),
        hero_index=hero_index,
        opponent_index=opponent_index,
    )


def make_deterministic_model(scale_weights: float = 1.0) -> nnue.NnueTronModel:
    model = nnue.NnueTronModel(num_rows=3, num_cols=3, acc_dim=4)

    with torch.no_grad():
        values = torch.arange(model.embedding.weight.numel(), dtype=torch.float32)
        values = ((values % 17) - 8.0) * 0.0025 * scale_weights
        model.embedding.weight.copy_(values.view_as(model.embedding.weight))
        model.fc1.weight.copy_(
            torch.tensor([[0.35, -0.2, 0.125, 0.05]], dtype=torch.float32)
        )
        model.fc1.bias.copy_(torch.tensor([0.075], dtype=torch.float32))

    model.eval()
    return model


def test_nnue_model_is_torch_module_with_expected_layers_and_dimensions():
    model = nnue.NnueTronModel(num_rows=4, num_cols=5, acc_dim=7)

    assert isinstance(model, nnue.TronModel)
    assert isinstance(model, torch.nn.Module)
    assert model.num_rows == 4
    assert model.num_cols == 5
    assert model.num_cells == 20
    assert model.embedding.num_embeddings == 60
    assert model.embedding.embedding_dim == 7
    assert model.fc1.in_features == 7
    assert model.fc1.out_features == 1


def test_feature_index_helpers_partition_wall_hero_and_opponent_features():
    model = nnue.NnueTronModel(num_rows=3, num_cols=4, acc_dim=2)

    assert model.emb_idx_wall(5) == 5
    assert model.emb_idx_hero_head(5) == 17
    assert model.emb_idx_opponent_head(5) == 29


@pytest.mark.parametrize(
    ("pov", "expected"),
    [
        (
            make_pov(
                make_game_state(
                    player_specs=((4, True), (8, True)),
                    extra_walls=(0, 1),
                ),
                hero_index=0,
                opponent_index=1,
            ),
            [13, 26, 0, 1, 4, 8],
        ),
        (
            make_pov(
                make_game_state(
                    player_specs=((4, True), (8, True)),
                    extra_walls=(0, 1),
                ),
                hero_index=1,
                opponent_index=0,
            ),
            [17, 22, 0, 1, 4, 8],
        ),
    ],
)
def test_get_active_indices_encodes_pov_heads_and_all_occupied_wall_bits(
    pov: game.PovGameState,
    expected: list[int],
):
    model = nnue.NnueTronModel(num_rows=3, num_cols=3, acc_dim=4)

    assert model.get_active_indices(pov) == expected


def test_get_active_indices_rejects_non_two_player_state():
    model = nnue.NnueTronModel(num_rows=3, num_cols=3, acc_dim=4)
    malformed_pov = SimpleNamespace(
        game_state=SimpleNamespace(players=(game.Player(0, True),)),
        hero_index=0,
        opponent_index=1,
    )

    with pytest.raises(NotImplementedError):
        model.get_active_indices(malformed_pov)


def test_initialize_acc_sums_active_feature_embeddings():
    model = make_deterministic_model()
    pov = make_pov(make_game_state(extra_walls=(0, 1)))
    active_indices = model.get_active_indices(pov)

    acc = model.initialize_acc(pov)
    expected_acc = model.embedding(torch.tensor(active_indices)).sum(dim=0)

    assert acc.shape == (4,)
    assert torch.allclose(acc, expected_acc)


def test_update_acc_matches_recomputed_accumulator_for_single_game_step():
    model = make_deterministic_model()
    prev_state = make_game_state(player_specs=((4, True), (8, True)))
    next_state = game.next(prev_state, (Direction.LEFT, Direction.UP))
    prev_pov = make_pov(prev_state)
    next_pov = make_pov(next_state)

    prev_acc = model.initialize_acc(prev_pov)
    to_remove = [
        model.emb_idx_hero_head(prev_state.players[0].idx),
        model.emb_idx_opponent_head(prev_state.players[1].idx),
    ]
    to_add = [
        model.emb_idx_wall(next_state.players[0].idx),
        model.emb_idx_wall(next_state.players[1].idx),
        model.emb_idx_hero_head(next_state.players[0].idx),
        model.emb_idx_opponent_head(next_state.players[1].idx),
    ]

    updated_acc = model.update_acc(prev_acc, to_remove=to_remove, to_add=to_add)

    assert torch.allclose(updated_acc, model.initialize_acc(next_pov))


def test_get_model_input_stacks_one_accumulator_per_pov_state():
    model = make_deterministic_model()
    povs = [
        make_pov(make_game_state(extra_walls=(0,))),
        make_pov(make_game_state(player_specs=((3, True), (8, True)), extra_walls=(4,))),
    ]

    model_input = model.get_model_input(povs)
    expected = torch.stack([model.initialize_acc(pov) for pov in povs])

    assert model_input.shape == (2, 4)
    assert torch.allclose(model_input, expected)


def test_forward_clamps_accumulator_to_unit_interval_before_linear_layer():
    model = nnue.NnueTronModel(num_rows=1, num_cols=1, acc_dim=3)

    with torch.no_grad():
        model.fc1.weight.fill_(1.0)
        model.fc1.bias.zero_()

    output = model(torch.tensor([-1.0, 0.25, 2.0]))
    batch_output = model(torch.tensor([[-1.0, 0.25, 2.0], [0.5, 0.5, 0.5]]))

    assert output.shape == torch.Size([])
    assert output.item() == pytest.approx(1.25)
    assert batch_output.shape == (2,)
    assert torch.allclose(batch_output, torch.tensor([1.25, 1.5]))


def test_run_inference_returns_forward_of_initialized_accumulator_as_float():
    model = make_deterministic_model()
    pov = make_pov(make_game_state(extra_walls=(0, 1)))

    result = model.run_inference(pov)
    expected = model(model.initialize_acc(pov)).detach().item()

    assert isinstance(result, float)
    assert result == pytest.approx(expected)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_load_state_dict_delegates_to_torch_and_loads_parameters():
    source = make_deterministic_model(scale_weights=2.0)
    target = nnue.NnueTronModel(num_rows=3, num_cols=3, acc_dim=4)

    result = target.load_state_dict(source.state_dict())

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        assert torch.equal(source_param, target_param)


def test_quantized_model_requires_float_nnue_model():
    with pytest.raises(AssertionError):
        nnue.QuantizedNnueTronModel(object())


def test_quantized_model_stores_rounded_integer_parameters():
    model = make_deterministic_model()
    quantized_model = nnue.QuantizedNnueTronModel(model, scale=128)

    assert quantized_model.raw_model is model
    assert quantized_model.scale == 128
    assert quantized_model.embed_weights.dtype == torch.int64
    assert quantized_model.linear_weights.dtype == torch.int64
    assert quantized_model.linear_bias.dtype == torch.int64
    assert torch.equal(
        quantized_model.embed_weights,
        torch.round(model.embedding.weight * 128).to(torch.int64),
    )
    assert torch.equal(
        quantized_model.linear_weights,
        torch.round(model.fc1.weight * 128).to(torch.int64),
    )
    assert torch.equal(
        quantized_model.linear_bias,
        torch.round(model.fc1.bias * 128 * 128).to(torch.int64),
    )


def test_quantized_initialize_acc_sums_quantized_active_embeddings():
    model = make_deterministic_model()
    quantized_model = nnue.QuantizedNnueTronModel(model, scale=256)
    pov = make_pov(make_game_state(extra_walls=(0, 1)))
    active_indices = model.get_active_indices(pov)

    acc = quantized_model.initialize_acc(pov)
    expected_acc = quantized_model.embed_weights[active_indices].sum(dim=0)

    assert acc.dtype == expected_acc.dtype
    assert torch.equal(acc, expected_acc)


def test_quantized_run_inference_acc_matches_manual_integer_math():
    model = make_deterministic_model()
    quantized_model = nnue.QuantizedNnueTronModel(model, scale=16)
    acc = torch.tensor([-5, 0, 8, 40], dtype=torch.int64)

    result = quantized_model.run_inference_acc(acc)

    clamped_acc = torch.clamp(acc, 0, 16).to(torch.int64)
    expected_int = quantized_model.linear_weights @ clamped_acc
    expected_int = expected_int + quantized_model.linear_bias
    expected = (expected_int.float() / (16 * 16)).item()
    assert result == expected


def test_quantized_run_inference_is_close_to_float_model_output():
    model = make_deterministic_model()
    quantized_model = nnue.QuantizedNnueTronModel(model, scale=4096)
    povs = [
        make_pov(make_game_state(extra_walls=(0, 1))),
        make_pov(make_game_state(player_specs=((3, True), (8, True)), extra_walls=(4,))),
        make_pov(make_game_state(player_specs=((0, False), (8, True)), extra_walls=(1, 3))),
    ]

    for pov in povs:
        float_eval = model.run_inference(pov)
        quantized_eval = quantized_model.run_inference(pov)

        assert quantized_eval == pytest.approx(float_eval, abs=7e-4)


def test_quantized_incremental_update_matches_recomputed_quantized_accumulator():
    model = make_deterministic_model()
    quantized_model = nnue.QuantizedNnueTronModel(model, scale=256)
    prev_state = make_game_state(player_specs=((4, True), (8, True)))
    next_state = game.next(prev_state, (Direction.LEFT, Direction.UP))
    prev_pov = make_pov(prev_state)
    next_pov = make_pov(next_state)

    prev_acc = quantized_model.initialize_acc(prev_pov)
    updated_acc = quantized_model.update_acc(
        prev_acc=prev_acc,
        prev_game_state=prev_state,
        next_pov_game_state=next_pov,
    )

    assert torch.equal(updated_acc, quantized_model.initialize_acc(next_pov))


def test_quantized_run_inference_acc_uses_incrementally_updated_accumulator():
    model = make_deterministic_model()
    quantized_model = nnue.QuantizedNnueTronModel(model, scale=1024)
    prev_state = make_game_state(player_specs=((4, True), (8, True)))
    next_state = game.next(prev_state, (Direction.LEFT, Direction.UP))
    prev_pov = make_pov(prev_state)
    next_pov = make_pov(next_state)
    prev_acc = quantized_model.initialize_acc(prev_pov)
    updated_acc = quantized_model.update_acc(prev_acc, prev_state, next_pov)

    incremental_eval = quantized_model.run_inference_acc(updated_acc)
    fresh_eval = quantized_model.run_inference(next_pov)

    assert incremental_eval == fresh_eval


def test_quantized_model_get_model_input_is_not_for_training():
    model = make_deterministic_model()
    quantized_model = nnue.QuantizedNnueTronModel(model)

    with pytest.raises(RuntimeError, match="not used for training"):
        quantized_model.get_model_input([make_pov()])
