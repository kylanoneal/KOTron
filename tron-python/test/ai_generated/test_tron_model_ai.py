import pytest
import torch

import tron.ai.tron_model as tron_model
import tron.game as game


def board_from_indices(*indices: int) -> int:
    board = 0
    for index in indices:
        board |= game.BIT_MASKS[index]
    return board


def make_game_state(
    num_rows: int = 2,
    num_cols: int = 2,
    player_specs: tuple[tuple[int, bool], tuple[int, bool]] = ((0, True), (3, True)),
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


class MinimalTronModel(tron_model.TronModel):
    def __init__(self, evaluation: float = 0.0):
        super().__init__()
        self.evaluation = evaluation
        self.model_input_calls = []
        self.inference_calls = []

    def get_model_input(
        self,
        pov_game_states: list[game.PovGameState],
    ) -> torch.Tensor:
        self.model_input_calls.append(pov_game_states)
        return torch.tensor([len(pov_game_states)], dtype=torch.float32)

    def run_inference(self, pov_game_state: game.PovGameState) -> float:
        self.inference_calls.append(pov_game_state)
        return self.evaluation


def test_tron_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError, match="abstract"):
        tron_model.TronModel()


def test_subclass_must_implement_get_model_input():
    class MissingInputModel(tron_model.TronModel):
        def run_inference(self, pov_game_state: game.PovGameState) -> float:
            return 0.0

    with pytest.raises(TypeError, match="abstract"):
        MissingInputModel()


def test_subclass_must_implement_run_inference():
    class MissingInferenceModel(tron_model.TronModel):
        def get_model_input(
            self,
            pov_game_states: list[game.PovGameState],
        ) -> torch.Tensor:
            return torch.empty(0)

    with pytest.raises(TypeError, match="abstract"):
        MissingInferenceModel()


def test_concrete_tron_model_is_torch_module_and_uses_subclass_methods():
    game_state = make_game_state()
    pov_game_state = game.PovGameState(game_state, hero_index=0, opponent_index=1)
    model = MinimalTronModel(evaluation=2.5)

    model_input = model.get_model_input([pov_game_state])
    evaluation = model.run_inference(pov_game_state)

    assert isinstance(model, torch.nn.Module)
    assert isinstance(model, tron_model.TronModel)
    assert torch.equal(model_input, torch.tensor([1.0]))
    assert evaluation == 2.5
    assert model.model_input_calls == [[pov_game_state]]
    assert model.inference_calls == [pov_game_state]


def test_base_run_inference_acc_reports_missing_override():
    model = MinimalTronModel()

    with pytest.raises(
        NotImplementedError,
        match="MinimalTronModel does not implement run_inference_acc",
    ):
        model.run_inference_acc(torch.tensor([1.0]))


def test_base_update_acc_reports_missing_override():
    game_state = make_game_state()
    pov_game_state = game.PovGameState(game_state, hero_index=0, opponent_index=1)
    model = MinimalTronModel()

    with pytest.raises(
        NotImplementedError,
        match="MinimalTronModel does not implement update_acc",
    ):
        model.update_acc(
            prev_acc=torch.tensor([1.0]),
            prev_game_state=game_state,
            next_pov_game_state=pov_game_state,
        )


def test_random_tron_model_is_instantiable_tron_model():
    model = tron_model.RandomTronModel()

    assert isinstance(model, tron_model.TronModel)
    assert isinstance(model, torch.nn.Module)


def test_random_tron_model_get_model_input_is_not_implemented():
    model = tron_model.RandomTronModel()

    with pytest.raises(NotImplementedError):
        model.get_model_input([])


def test_random_tron_model_returns_float_evaluation():
    game_state = make_game_state()
    pov_game_state = game.PovGameState(game_state, hero_index=0, opponent_index=1)
    model = tron_model.RandomTronModel()

    evaluation = model.run_inference(pov_game_state)

    assert isinstance(evaluation, float)


def test_random_tron_model_is_deterministic_for_same_pov_game_state():
    game_state = make_game_state()
    pov_game_state = game.PovGameState(game_state, hero_index=0, opponent_index=1)
    model = tron_model.RandomTronModel()

    first_eval = model.run_inference(pov_game_state)
    second_eval = model.run_inference(pov_game_state)

    assert first_eval == second_eval


def test_random_tron_model_is_deterministic_for_equal_game_states():
    game_state = make_game_state()
    equal_game_state = make_game_state()
    model = tron_model.RandomTronModel()

    first_eval = model.run_inference(
        game.PovGameState(game_state, hero_index=0, opponent_index=1)
    )
    second_eval = model.run_inference(
        game.PovGameState(equal_game_state, hero_index=0, opponent_index=1)
    )

    assert game_state == equal_game_state
    assert first_eval == second_eval


def test_random_tron_model_seeds_rng_from_pov_game_state(
    monkeypatch: pytest.MonkeyPatch,
):
    seeds = []
    normalvariate_args = []

    class FakeRandom:
        def __init__(self, seed: int):
            self.seed = seed
            seeds.append(seed)

        def normalvariate(self, mu: float, sigma: float) -> float:
            normalvariate_args.append((mu, sigma))
            return 42.0

    monkeypatch.setattr(tron_model.random, "Random", FakeRandom)

    game_state = make_game_state()
    pov_game_state = game.PovGameState(game_state, hero_index=0, opponent_index=1)
    model = tron_model.RandomTronModel()

    evaluation = model.run_inference(pov_game_state)

    assert evaluation == 42.0
    assert seeds == [hash(pov_game_state)]
    assert normalvariate_args == [(0, 1)]


def test_random_tron_model_uses_different_seed_for_different_player_pov(
    monkeypatch: pytest.MonkeyPatch,
):
    seeds = []

    class FakeRandom:
        def __init__(self, seed: int):
            self.seed = seed
            seeds.append(seed)

        def normalvariate(self, mu: float, sigma: float) -> float:
            return float(len(seeds))

    monkeypatch.setattr(tron_model.random, "Random", FakeRandom)

    game_state = make_game_state()
    hero_pov = game.PovGameState(game_state, hero_index=0, opponent_index=1)
    opponent_pov = game.PovGameState(game_state, hero_index=1, opponent_index=0)
    model = tron_model.RandomTronModel()

    hero_eval = model.run_inference(hero_pov)
    opponent_eval = model.run_inference(opponent_pov)

    assert seeds == [
        hash(hero_pov),
        hash(opponent_pov),
    ]
    assert hero_eval == 1.0
    assert opponent_eval == 2.0


def test_random_tron_model_uses_different_seed_for_different_game_state(
    monkeypatch: pytest.MonkeyPatch,
):
    seeds = []

    class FakeRandom:
        def __init__(self, seed: int):
            seeds.append(seed)

        def normalvariate(self, mu: float, sigma: float) -> float:
            return 0.0

    monkeypatch.setattr(tron_model.random, "Random", FakeRandom)

    first_game_state = make_game_state()
    second_game_state = make_game_state(extra_walls=(1,))
    first_pov = game.PovGameState(first_game_state, 0, 1)
    second_pov = game.PovGameState(second_game_state, 0, 1)
    model = tron_model.RandomTronModel()

    model.run_inference(first_pov)
    model.run_inference(second_pov)

    assert seeds == [
        hash(first_pov),
        hash(second_pov),
    ]
    assert first_game_state != second_game_state
