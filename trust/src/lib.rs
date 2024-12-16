pub mod algos;
pub mod game;
pub mod models;

use console_error_panic_hook;
use once_cell::sync::OnceCell;
use std::panic;

use wasm_bindgen::prelude::*;

use im::{vector, Vector};

use algos::choose_direction_model_naive;
use game::{Direction, DirectionUpdate, GameState, Player};
use models::TractModel;

static TRACT_MODEL: OnceCell<TractModel> = OnceCell::new();

// Init model when the WASM module is instantiated.
#[wasm_bindgen(start)]
pub fn start() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    // Embed the ONNX model file at compile time
    const MODEL_BYTES: &[u8] = include_bytes!("../tron_model_v2.onnx");

    //let model = TractModel::new("tron_model_v2.onnx").expect("failed to init tract model.");

    let model = TractModel::from_bytes(MODEL_BYTES).expect("failed to init model from bytes");

    TRACT_MODEL
        .set(model)
        .expect("TRACT_MODEL was already set, this should never happen!");
}

// Figure out how to use Direction enum and how that would work on the JavaScript side
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub row_offset: i8,
    pub col_offset: i8,
}

#[wasm_bindgen]
impl Move {
    #[wasm_bindgen(constructor)]
    pub fn new(row_offset: i8, col_offset: i8) -> Move {
        Move {
            row_offset,
            col_offset,
        }
    }
}

#[wasm_bindgen]
pub fn run_engine(
    data: &[u8],
    num_rows: usize,
    num_cols: usize,
    player_row: usize,
    player_col: usize,
    opponent_row: usize,
    opponent_col: usize,
) -> Move {
    let model: &TractModel = TRACT_MODEL
        .get()
        .expect("TRACT_MODEL is not initialized. Did #[wasm_bindgen(start)] run?");

    let p1 = Player {
        row: player_row,
        col: player_col,
        direction: Direction::Up,
        can_move: true,
    };

    let p2 = Player {
        row: opponent_row,
        col: opponent_col,
        direction: Direction::Up,
        can_move: true,
    };

    let players: Vector<Player> = vector![p1, p2];

    let grid: Vector<Vector<bool>> = flatten_to_im_vector(data, num_rows, num_cols);

    let mut game: GameState = GameState::new(players, 10, 10);
    // Make a different constructor for this purpose
    game.grid = grid;

    let dir_update: DirectionUpdate = choose_direction_model_naive(&game, 0, model);

    let (row_offset, col_offset) = dir_update.direction.value();

    return Move {
        row_offset: row_offset,
        col_offset: col_offset,
    };

}

fn flatten_to_im_vector(data: &[u8], rows: usize, cols: usize) -> Vector<Vector<bool>> {
    assert!(
        data.len() == rows * cols,
        "Data length ({}) does not match rows * cols ({})",
        data.len(),
        rows * cols
    );

    // Initialize the outer Vector
    let mut grid = Vector::new();

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        // Slice the data for the current row
        let row_slice = &data[start..end];
        // Convert to Vector<bool>
        let row_vector: Vector<bool> = row_slice.iter().map(|&b| b != 0).collect();
        grid.push_back(row_vector);
    }

    grid
}
