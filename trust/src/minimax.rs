use crate::tron;
use crate::{
    models::Model,
    tron::{Direction, GameState, GameStatus},
};

use im::{vector, Vector};
use rand::Rng;
use strum::IntoEnumIterator;

#[derive(Debug, Clone)]
pub struct MinimaxResult {
    pub evaluation: f32,
    pub principal_variation: Option<Direction>,
}

pub struct MinimaxContext<'a> {
    pub model: &'a dyn Model,
    pub maximizing_player: usize,
    pub minimizing_player: usize,
}

/// Performs a recursive minimax search with alpha-beta pruning.
///
/// # Parameters
/// - `game_state`: The current game state.
/// - `depth`: The remaining search depth.
/// - `is_maximizing_player`: True if we are evaluating a maximizing node.
/// - `alpha`: The best value that the maximizer currently can guarantee.
/// - `beta`: The best value that the minimizer currently can guarantee.
/// - `maximizing_player_move`: In the maximizing branch, the move being considered.
/// - `context`: A reference to the minimax context.
///
/// # Returns
/// A `MinimaxResult` containing the evaluation and the principal variation.
pub fn minimax_alpha_beta_eval_all(
    game_state: &GameState,
    depth: u32,
    is_maximizing_player: bool,
    mut alpha: f32,
    mut beta: f32,
    maximizing_player_move: Option<Direction>,
    context: &MinimaxContext,
) -> MinimaxResult {
    let maximizing_player = context.maximizing_player;
    let minimizing_player = context.minimizing_player;

    let game_status = tron::get_status(&game_state);
    match game_status.status {
        GameStatus::Tie => {
            return MinimaxResult {
                evaluation: 0.0,
                principal_variation: None,
            }
        }
        GameStatus::Winner => panic!("Winning terminal state should never be reached here."),
        _ => {}
    }

    if depth == 0 {
        let eval: f32 = (context
            .model
            .run_inference(&vec![game_state], maximizing_player)
            .expect("failed to run inference"))[0];
        return MinimaxResult {
            evaluation: eval,
            principal_variation: None,
        };
    }

    if is_maximizing_player {
        let mut possible_directions = tron::get_possible_directions(&game_state, maximizing_player);
        // No available moves for maximizing player.
        if possible_directions.is_empty() {
            let opponent_possible_directions: Vec<Direction> =
                tron::get_possible_directions(&game_state, minimizing_player);
            if opponent_possible_directions.is_empty() {
                return MinimaxResult {
                    evaluation: 0.0,
                    principal_variation: None,
                };
            } else {
                return MinimaxResult {
                    evaluation: -100.0 * (depth as f32),
                    principal_variation: None,
                };
            }
        }

        let mut max_eval = f32::NEG_INFINITY;
        let mut best_dir = None;
        for direction in possible_directions {
            let mm_result = minimax_alpha_beta_eval_all(
                game_state,
                depth,
                false,
                alpha,
                beta,
                Some(direction),
                context,
            );

            if mm_result.evaluation > max_eval {
                max_eval = mm_result.evaluation;
                best_dir = Some(direction);
            }
            alpha = alpha.max(mm_result.evaluation);
            if beta <= alpha {
                break;
            }
        }
        MinimaxResult {
            evaluation: max_eval,
            principal_variation: best_dir,
        }
    } else {
        let possible_directions = tron::get_possible_directions(&game_state, minimizing_player);
        // If the minimizing player has no moves, it is a guaranteed win for the maximizer.
        if possible_directions.is_empty() {
            return MinimaxResult {
                evaluation: 100.0 * (depth as f32),
                principal_variation: None,
            };
        }

        // Build child states by applying moves from the minimizing player.
        let mut child_states: Vec<GameState> = Vec::new();
        for &direction in &possible_directions {
            // Assume two players (indices 0 and 1).
            let mut directions: [Direction; 2] = [Direction::Up, Direction::Up];
            directions[maximizing_player] =
                maximizing_player_move.expect("Maximizing player should always be passed here.");
            directions[minimizing_player] = direction;
            child_states.push(tron::next(&game_state, &directions));
        }

        // Can do some sorting here

        let mut min_eval = f32::INFINITY;
        let mut best_dir = None;
        for (direction, child_state) in possible_directions.into_iter().zip(child_states.iter()) {
            let mm_result = minimax_alpha_beta_eval_all(
                child_state,
                depth - 1,
                true,
                alpha,
                beta,
                None,
                context,
            );
            if mm_result.evaluation < min_eval {
                min_eval = mm_result.evaluation;
                best_dir = Some(direction);
            }
            beta = beta.min(mm_result.evaluation);
            if beta <= alpha {
                break;
            }
        }
        MinimaxResult {
            evaluation: min_eval,
            principal_variation: best_dir,
        }
    }
}
