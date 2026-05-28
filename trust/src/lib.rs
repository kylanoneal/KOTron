pub mod algos;
pub mod alphabeta;
pub mod io;
pub mod model;
pub mod nnue;
pub mod tron_2d;

pub mod tron_pb {
    include!(concat!(env!("OUT_DIR"), "/tron_pb.rs"));
}

use console_error_panic_hook;
use once_cell::sync::OnceCell;
use std::panic;

use wasm_bindgen::prelude::*;

use im::{vector, Vector};

// use algos::choose_direction_model_naive;
use alphabeta::{alphabeta, MinimaxContext, MinimaxResult};
use nnue::QuantizedNnue;
use tron_2d::{Direction, GameState, Player};

static NNUE_MODEL_3X3: OnceCell<QuantizedNnue> = OnceCell::new();
static NNUE_MODEL_4X4: OnceCell<QuantizedNnue> = OnceCell::new();

#[wasm_bindgen(start)]
pub fn start() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));

    const MODEL_3X3_BYTES: &[u8] =
        include_bytes!(r#"C:\Users\kylan\code\KOTron\tron-python\models\3x3.npz"#);

    const MODEL_4X4_BYTES: &[u8] =
        include_bytes!(r#"C:\Users\kylan\code\KOTron\tron-python\models\4x4.npz"#);

    let model_3x3 = QuantizedNnue::from_bytes(MODEL_3X3_BYTES)
        .expect("failed to init 3x3 QuantizedNnue from embedded npz bytes");

    let model_4x4 = QuantizedNnue::from_bytes(MODEL_4X4_BYTES)
        .expect("failed to init 4x4 QuantizedNnue from embedded npz bytes");

    NNUE_MODEL_3X3
        .set(model_3x3)
        .expect("NNUE_MODEL_3X3 was already set");

    NNUE_MODEL_4X4
        .set(model_4x4)
        .expect("NNUE_MODEL_4X4 was already set");
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

fn get_nnue_model(num_rows: usize, num_cols: usize) -> &'static QuantizedNnue {
    match (num_rows, num_cols) {
        (3, 3) => NNUE_MODEL_3X3
            .get()
            .expect("NNUE_MODEL_3X3 is not initialized. Did #[wasm_bindgen(start)] run?"),

        (4, 4) => NNUE_MODEL_4X4
            .get()
            .expect("NNUE_MODEL_4X4 is not initialized. Did #[wasm_bindgen(start)] run?"),

        _ => panic!(
            "No NNUE model available for board size {}x{}",
            num_rows, num_cols
        ),
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
    let model: &QuantizedNnue = get_nnue_model(num_rows, num_cols);
    let hero = Player {
        row: player_row,
        col: player_col,
        can_move: true,
    };

    let villain = Player {
        row: opponent_row,
        col: opponent_col,
        can_move: true,
    };

    let players: Vector<Player> = vector![hero, villain];

    let grid: Vector<Vector<bool>> = flatten_to_im_vector(data, num_rows, num_cols);

    let mut game: GameState = tron_2d::new_game(players, num_rows, num_cols);
    // Make a different constructor for this purpose
    game.grid = grid;

    let hero_mm_result: MinimaxResult = alphabeta(
        &game,
        1,
        true,
        f32::NEG_INFINITY,
        f32::INFINITY,
        None,
        &MinimaxContext {
            model: model,
            maximizing_player: 0,
            minimizing_player: 1,
        },
    );

    let hero_direction: Direction = hero_mm_result.principal_variation.unwrap_or(Direction::Up);

    let (row_offset, col_offset) = hero_direction.value();

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
