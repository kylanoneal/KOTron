import itertools
import random

import numpy as np
from scipy.optimize import linprog


from typing import Optional
from dataclasses import dataclass
from enum import Enum, auto

import tron
from tron.game import GameState, StatusInfo, GameStatus, Direction, Player

from tron.ai.tron_model import TronModel, PovGameState

from tron.enums import PovGameResult


@dataclass(frozen=True)
class MatrixGameSolution:
    # The guaranteed expected value for the hero if both players play optimally.
    value: float

    result: Optional[PovGameResult] = None
    result_depth: Optional[int] = None

    # Probability distribution over hero moves.
    # Example: [0.7, 0.3, 0.0] means:
    #   play hero move 0 with probability 70%
    #   play hero move 1 with probability 30%
    #   never play hero move 2
    hero_strategy: np.ndarray

    # Probability distribution over opponent moves.
    # This is useful for debugging / analysis.
    opponent_strategy: np.ndarray


@dataclass
class OracleInfo:
    solution: MatrixGameSolution
    hero_player: Player = None
    oppo_player: Player = None
    # steps_to_result: int


@dataclass
class MinimaxContext:
    model: TronModel
    hero_index: int
    opponent_index: int
    oracle_table: dict[GameState, OracleInfo]
    win_magnitude: float = 10_000.0

    def __post_init__(self):

        assert (0 <= self.hero_index < 2) and (0 <= self.opponent_index < 2)

        assert self.hero_index != self.opponent_index

        assert self.win_magnitude > 1.0


def swap_perspective(solution: MatrixGameSolution):

    new_val = -solution.value

    if solution.result != PovGameResult.TIE:

        new_result = (
            PovGameResult.WIN
            if solution.result == PovGameResult.LOSS
            else PovGameResult.LOSS
        )
    else:
        new_result = solution.result

    return MatrixGameSolution(new_val, new_result, solution.result_depth)


def solve_zero_sum_matrix_game(
    payoff_matrix: np.ndarray, is_fallback: bool = False
) -> MatrixGameSolution:
    """
    Solve a simultaneous-move zero-sum matrix game.

    payoff_matrix[i, j] is the value for the HERO when:
        - hero chooses move i
        - opponent chooses move j

    The hero wants high values.
    The opponent wants low values.

    This function returns:
        - the value of the game
        - the hero's optimal randomized strategy
        - the opponent's optimal randomized strategy

    Conceptually, the hero is asking:

        "What probability distribution over my moves gives me the best
         guaranteed expected value, even if the opponent chooses the
         best counter-strategy?"
    """

    # Convert the input to a NumPy float array.
    # We want floats because the solver works with real-valued probabilities.
    M = np.asarray(payoff_matrix, dtype=float)

    # Basic validation: the matrix should be 2D.
    #
    # Rows are hero moves.
    # Columns are opponent moves.
    if M.ndim != 2:
        raise ValueError("payoff_matrix must be a 2D array")

    # Number of possible hero moves.
    num_hero_moves = M.shape[0]

    # Number of possible opponent moves.
    num_opp_moves = M.shape[1]

    if num_hero_moves == 0 or num_opp_moves == 0:
        raise ValueError("payoff_matrix must have at least one row and one column")

    # -------------------------------------------------------------------------
    # PART 1:
    # Solve for the HERO'S optimal mixed strategy.
    # -------------------------------------------------------------------------
    #
    # The hero is choosing probabilities over rows.
    #
    # Example:
    #
    #   hero_strategy = [0.5, 0.5, 0.0]
    #
    # means:
    #
    #   play row 0 with probability 50%
    #   play row 1 with probability 50%
    #   play row 2 with probability 0%
    #
    # The hero wants to maximize the worst-case expected payoff.
    #
    # We introduce a variable called v.
    #
    # v means:
    #
    #   "the payoff I can guarantee myself"
    #
    # The hero wants to make v as large as possible.
    #
    # Variables for the linear program:
    #
    #   [p_0, p_1, p_2, ..., p_n, v]
    #
    # where:
    #
    #   p_i = probability of playing hero move i
    #   v   = guaranteed expected value
    #
    # For example, if the hero has 3 moves, the variables are:
    #
    #   [p_0, p_1, p_2, v]
    #
    # The constraints are:
    #
    #   1. probabilities must sum to 1
    #
    #          p_0 + p_1 + p_2 + ... = 1
    #
    #   2. probabilities must be nonnegative
    #
    #          p_i >= 0
    #
    #   3. against every opponent move, the expected value must be at least v
    #
    #          expected payoff if opponent plays column j >= v
    #
    # The solver will choose probabilities that make v as large as possible.
    # -------------------------------------------------------------------------

    # Objective vector.
    #
    # scipy.optimize.linprog only minimizes.
    #
    # But the hero wants to maximize v.
    #
    # Maximizing v is the same as minimizing -v.
    #
    # So we set the coefficient of v to -1.
    #
    # Variables are:
    #
    #   [p_0, p_1, ..., p_n, v]
    #
    # So the last variable is v.
    c = np.zeros(num_hero_moves + 1)
    c[-1] = -1.0

    # Inequality constraints for linprog have the form:
    #
    #   A_ub @ x <= b_ub
    #
    # We need one inequality for each opponent move.
    #
    # For each opponent move j, we want:
    #
    #   expected payoff against j >= v
    #
    # The expected payoff against opponent column j is:
    #
    #   p_0 * M[0, j] + p_1 * M[1, j] + ... + p_n * M[n, j]
    #
    # So the constraint is:
    #
    #   p_0 * M[0, j] + p_1 * M[1, j] + ... >= v
    #
    # linprog wants <= constraints, so rearrange:
    #
    #   -p_0 * M[0, j] - p_1 * M[1, j] - ... + v <= 0
    #
    # That is what we build below.
    A_ub = []
    b_ub = []

    for j in range(num_opp_moves):
        # This row represents one inequality constraint.
        #
        # It has one coefficient per variable:
        #
        #   [coefficient for p_0,
        #    coefficient for p_1,
        #    ...
        #    coefficient for p_n,
        #    coefficient for v]
        row = np.zeros(num_hero_moves + 1)

        # Coefficients for the probability variables.
        #
        # We use negative values because we rearranged:
        #
        #   expected_payoff >= v
        #
        # into:
        #
        #   -expected_payoff + v <= 0
        row[:num_hero_moves] = -M[:, j]

        # Coefficient for v.
        #
        # The rearranged constraint has:
        #
        #   +v
        row[-1] = 1.0

        # Right side of the inequality is 0:
        #
        #   -expected_payoff + v <= 0
        A_ub.append(row)
        b_ub.append(0.0)

    # Equality constraint:
    #
    #   p_0 + p_1 + ... + p_n = 1
    #
    # Again, variables are:
    #
    #   [p_0, p_1, ..., p_n, v]
    #
    # So the probability variables get coefficient 1.
    # The v variable gets coefficient 0.
    A_eq = np.zeros((1, num_hero_moves + 1))
    A_eq[0, :num_hero_moves] = 1.0

    b_eq = np.array([1.0])

    # Bounds for each variable.
    #
    # Each probability p_i must be between 0 and 1.
    #
    # The value v can be anything:
    #   - negative
    #   - zero
    #   - positive
    #
    # So v gets bounds (None, None).
    bounds = [(0.0, 1.0)] * num_hero_moves + [(None, None)]

    # Solve the hero's linear program.
    hero_result = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    # If the solver failed, stop immediately.
    if not hero_result.success:
        raise RuntimeError(f"Failed to solve hero strategy: {hero_result.message}")

    # Extract the hero probabilities.
    hero_strategy = hero_result.x[:num_hero_moves]

    # Extract the guaranteed value v.
    value = hero_result.x[-1]

    # Clean up tiny numerical noise.
    #
    # Linear solvers often return values like:
    #
    #   0.000000000000003
    #
    # instead of exactly:
    #
    #   0.0
    #
    # We treat tiny values as zero.
    hero_strategy[np.abs(hero_strategy) < 1e-10] = 0.0

    # Renormalize the probabilities so they sum to exactly 1.
    #
    # This protects against tiny floating-point drift.
    hero_strategy = hero_strategy / hero_strategy.sum()

    # -------------------------------------------------------------------------
    # PART 2:
    # Solve for the OPPONENT'S optimal mixed strategy.
    # -------------------------------------------------------------------------
    #
    # This part is not strictly required if all you need is:
    #
    #   - hero_strategy
    #   - game value
    #
    # But it is very useful for debugging.
    #
    # The opponent chooses probabilities over columns.
    #
    # Example:
    #
    #   opponent_strategy = [0.25, 0.75]
    #
    # means:
    #
    #   opponent plays column 0 with probability 25%
    #   opponent plays column 1 with probability 75%
    #
    # The opponent wants to MINIMIZE the hero's expected payoff.
    #
    # We introduce a variable called w.
    #
    # w means:
    #
    #   "the maximum payoff the hero can get against my opponent strategy"
    #
    # The opponent wants to make w as small as possible.
    #
    # Variables:
    #
    #   [q_0, q_1, q_2, ..., q_m, w]
    #
    # where:
    #
    #   q_j = probability of opponent move j
    #   w   = upper bound on hero payoff
    # -------------------------------------------------------------------------

    # Objective vector.
    #
    # This time we really are minimizing w.
    #
    # Variables are:
    #
    #   [q_0, q_1, ..., q_m, w]
    #
    # So the last variable is w, and its coefficient is +1.
    c = np.zeros(num_opp_moves + 1)
    c[-1] = 1.0

    # Inequality constraints.
    #
    # We need one constraint for each hero move.
    #
    # For every hero row i, we want:
    #
    #   expected payoff for hero row i <= w
    #
    # The expected payoff if hero plays row i and opponent randomizes is:
    #
    #   q_0 * M[i, 0] + q_1 * M[i, 1] + ... + q_m * M[i, m]
    #
    # Constraint:
    #
    #   q_0 * M[i, 0] + q_1 * M[i, 1] + ... <= w
    #
    # Rearranged into linprog's <= form:
    #
    #   q_0 * M[i, 0] + q_1 * M[i, 1] + ... - w <= 0
    A_ub = []
    b_ub = []

    for i in range(num_hero_moves):
        row = np.zeros(num_opp_moves + 1)

        # Coefficients for q_0, q_1, ..., q_m.
        row[:num_opp_moves] = M[i, :]

        # Coefficient for w.
        row[-1] = -1.0

        # Right side is 0:
        #
        #   expected_payoff - w <= 0
        A_ub.append(row)
        b_ub.append(0.0)

    # Equality constraint:
    #
    #   q_0 + q_1 + ... + q_m = 1
    A_eq = np.zeros((1, num_opp_moves + 1))
    A_eq[0, :num_opp_moves] = 1.0

    b_eq = np.array([1.0])

    # Bounds:
    #
    # Opponent probabilities must be between 0 and 1.
    # w can be any real number.
    bounds = [(0.0, 1.0)] * num_opp_moves + [(None, None)]

    # Solve the opponent's linear program.
    opp_result = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not opp_result.success:
        raise RuntimeError(f"Failed to solve opponent strategy: {opp_result.message}")

    # Extract opponent probabilities.
    opponent_strategy = opp_result.x[:num_opp_moves]

    # Clean up tiny numerical noise.
    opponent_strategy[np.abs(opponent_strategy) < 1e-10] = 0.0

    # Renormalize.
    opponent_strategy = opponent_strategy / opponent_strategy.sum()

    assert hero_strategy.sum() == 1.0 and opponent_strategy.sum() == 1.0

    if is_fallback:
        single_hero_strat = sum([p == 1.0 for p in hero_strategy]) == 1
        single_oppo_strat = sum([p == 1.0 for p in opponent_strategy]) == 1

        assert not single_hero_strat and not single_oppo_strat
        print(f"\n{hero_strategy=}\n{opponent_strategy=}\n{value=}\n")
        if single_hero_strat and single_oppo_strat:
            pass
        elif single_hero_strat or single_oppo_strat:
            print(f"One sided mixed strat found!")
        else:
            print(f"Mixed strat found!")

    # return float(value)
    return MatrixGameSolution(
        value=float(value),
        hero_strategy=hero_strategy,
        opponent_strategy=opponent_strategy,
    )


# Looks for disagreement then falls back on matrix solve
def pessimistic_solve(payoff_matrix: np.ndarray) -> float:

    best_oppo_eval = float("-inf")

    for i, row in enumerate(payoff_matrix):

        min_move = int(np.argmin(row))

        if row[min_move] > best_oppo_eval:

            best_oppo_eval = row[min_move]
            hero_variation = (i, min_move)

    best_hero_eval = float("inf")

    for j in range(payoff_matrix.shape[1]):
        col = payoff_matrix[:, j]

        max_move = int(np.argmax(col))

        if col[max_move] < best_hero_eval:

            best_hero_eval = col[max_move]
            oppo_variation = (max_move, j)

    hero_value = payoff_matrix[hero_variation[0]][hero_variation[1]]
    oppo_value = payoff_matrix[oppo_variation[0]][oppo_variation[1]]

    if hero_value == oppo_value:

        return hero_value
    else:
        print(f"\n\n{'-' * 15}\n")
        print(f"Differing opinions, falling back to matrix solve!")
        print(f"Payoff matrix: \n{payoff_matrix}")
        print(f"{hero_variation=}, {oppo_variation=}")

        return solve_zero_sum_matrix_game(payoff_matrix, is_fallback=True)


def oracle_minimax(
    game_state: GameState,
    depth: int,
    context: MinimaxContext = None,
    return_pv: bool = False,
) -> MatrixGameSolution:

    assert depth > 0, "Oracle minimax should not reach depth 0"
    assert context is not None, "Context must be passed"

    hero_index = context.hero_index
    opponent_index = context.opponent_index

    status_info: StatusInfo = tron.get_status(game_state)

    if status_info.status != GameStatus.IN_PROGRESS:

        assert (
            status_info.status == GameStatus.TIE
        ), "Winning terminal state should not be reached here"

        return MatrixGameSolution(0.0, result=PovGameResult.TIE, result_depth=depth)

    # Lookup and return oracle info if we already have it
    oracle_info_lookup = context.oracle_table.get(game_state)

    if oracle_info_lookup is not None:

        if oracle_info_lookup.hero_player == game_state.players[hero_index]:
            assert oracle_info_lookup.oppo_player == game_state.players[opponent_index]

            # Perspective matches
            return oracle_info_lookup.solution
        elif oracle_info_lookup.hero_player == game_state.players[opponent_index]:

            assert oracle_info_lookup.oppo_player == game_state.players[hero_index]
            # Negate value for perspective swap
            return swap_perspective(oracle_info_lookup.solution)

        else:
            raise AssertionError()

    # Otherwise proceed with minimax

    # Check for a winner

    hero_possible_directions = tron.get_possible_directions(game_state, hero_index)

    oppo_possible_directions = tron.get_possible_directions(game_state, opponent_index)

    hero_has_options = len(hero_possible_directions) > 0
    oppo_has_options = len(oppo_possible_directions) > 0

    if not hero_has_options or not oppo_has_options:

        if not hero_has_options and oppo_has_options:

            value = -1.0
            result = PovGameResult.LOSS

        elif not oppo_has_options and hero_has_options:
            value = 1.0
            result = PovGameResult.WIN
        elif not hero_has_options and not oppo_has_options:
            value = 0.0
            result = PovGameResult.TIE
        else:
            raise AssertionError()

        solution = MatrixGameSolution(value, result, depth - 1)
        context.oracle_table[game_state] = OracleInfo(
            solution,
            hero_player=game_state.players[hero_index],
            oppo_player=game_state.players[opponent_index],
        )

        return solution

    # No winner found, go deeper

    move_matrix = np.zeros(
        (len(hero_possible_directions), len(oppo_possible_directions)), dtype=np.float32
    )

    # Fill in matrix:

    for i in range(len(hero_possible_directions)):

        for j in range(len(oppo_possible_directions)):

            directions = [None, None]
            directions[hero_index] = hero_possible_directions[i]
            directions[opponent_index] = oppo_possible_directions[j]

            child_state = tron.next(game_state, directions=tuple(directions))

            move_matrix[i][j] = oracle_minimax(child_state, depth - 1, context=context)

    matrix_solution = solve_zero_sum_matrix_game(move_matrix)

    # pessimistic_solved_value = pessimistic_solve(move_matrix)

    # assert pessimistic_solved_value == matrix_solved_value

    context.oracle_table[game_state] = OracleInfo(
        matrix_solution,
        hero_player=game_state.players[hero_index],
        oppo_player=game_state.players[opponent_index],
    )

    return matrix_solution

    # if return_pv:

    #     hero_strat = matrix_solve_result.hero_strategy
    #     oppo_strat = matrix_solve_result.opponent_strategy

    #     single_hero_strat = sum([p == 1.0 for p in hero_strat]) == 1
    #     single_oppo_strat = sum([p == 1.0 for p in oppo_strat]) == 1

    #     if single_oppo_strat:

    #     hero_dir = np.random.choice(len(hero_possible_directions), p=hero_strat)

    # else:

    #     return matrix_solve_result.value
