use crate::{game::{GameState, Direction, DirectionUpdate, Player}, models::Model}; 

use strum::IntoEnumIterator;
use im::{vector, Vector};
use rand::Rng;


pub fn choose_direction_random(game: &GameState, player_index: usize) -> DirectionUpdate {
    let possible_moves: Vec<Direction> = get_possible_moves(
        game.players[player_index].row,
        game.players[player_index].col,
        &game.grid,
    );

    let mut rng = rand::thread_rng(); // Create a random number generator
    let chosen_direction: Direction;

    if !possible_moves.is_empty() {
        let random_index = rng.gen_range(0..possible_moves.len());
        chosen_direction = possible_moves[random_index];
    } else {
        chosen_direction = game.players[player_index].direction;
    }

    DirectionUpdate {
        player_index: player_index,
        direction: chosen_direction,
    }
}

pub fn choose_direction_model_naive(
    game: &GameState,
    player_index: usize,
    model: &dyn Model,
) -> DirectionUpdate {
    // CLean
    if game.players[player_index].can_move {
        let possible_moves: Vec<Direction> = get_possible_moves(
            game.players[player_index].row,
            game.players[player_index].col,
            &game.grid,
        );

        // CLean
        if possible_moves.is_empty() {
            return DirectionUpdate {
                player_index: player_index,
                direction: game.players[player_index].direction,
            };
        }

        let mut evaluations: Vec<f32> = Vec::new();

        for possible_move in possible_moves.iter() {
            // Copy game state
            let next_game = GameState::next(
                game,
                &vec![DirectionUpdate {
                    player_index: player_index,
                    direction: *possible_move,
                }],
            );

            // Running inference one a time for now
            let eval: f32 = (model.run_inference(&vec![next_game], player_index).expect("failed to run inference"))[0];
            evaluations.push(eval);
        }

        // Update direction to possible_moves[argmax(evaluations)]
        return DirectionUpdate {
            player_index: player_index,
            direction: possible_moves[argmax(&evaluations)],
        };
    } else {
        // Return current direction if it can't move
        DirectionUpdate {
            player_index: player_index,
            direction: game.players[player_index].direction,
        }
    }
}

pub fn argmax(vec: &Vec<f32>) -> usize {
    let mut max_index = 0;
    let mut max_val = vec[0];

    // Probably get rid of this if it's not possible
    assert!(!vec[0].is_nan());

    for i in 1..vec.len() {
        // Probably get rid of this if it's not possible
        assert!(!vec[i].is_nan());

        if vec[i] > max_val {
            max_index = i;
            max_val = vec[i];
        }
    }

    max_index
}


// Move this into game and change signature to pub fn get_possible_moves(game, player_index)
pub fn get_possible_moves(row: usize, col: usize, grid: &Vector<Vector<bool>>) -> Vec<Direction> {
    let curr_row = row as i8;
    let curr_col = col as i8;

    let mut possible_moves: Vec<Direction> = Vec::new();
    for direction in Direction::iter() {
        let new_row = curr_row + direction.value().0;
        let new_col = curr_col + direction.value().1;

        let out_of_bounds: bool = new_row < 0
            || new_col < 0
            || new_row >= (grid.len() as i8)
            || new_col >= (grid[0].len() as i8);

        if !(out_of_bounds || grid[new_row as usize][new_col as usize]) {
            possible_moves.push(direction);
        }
    }
    return possible_moves;
}