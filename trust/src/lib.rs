pub mod algos;
pub mod alphabeta;
pub mod io;
pub mod model;
pub mod nnue;
pub mod tron;
pub mod tron_2d;

pub mod tron_pb {
    include!(concat!(env!("OUT_DIR"), "/tron_pb.rs"));
}

use console_error_panic_hook;
use once_cell::sync::OnceCell;
use std::panic;

use wasm_bindgen::prelude::*;

// use algos::choose_direction_model_naive;
use alphabeta::{alphabeta, MinimaxContext, MinimaxResult};
use nnue::QuantizedNnue;
use tron::{BitBoard, Direction, GameState, Player};

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
    let hero_idx = player_row * num_cols + player_col;
    let villain_idx = opponent_row * num_cols + opponent_col;

    let hero = Player {
        idx: hero_idx,
        can_move: true,
    };

    let villain = Player {
        idx: villain_idx,
        can_move: true,
    };

    let num_cells = num_rows * num_cols;
    let mut board =
        BitBoard::from_flat_occupancy(data, num_cells).expect("invalid flat occupancy board");
    board.set(hero_idx);
    board.set(villain_idx);

    let game = GameState::new(num_rows, num_cols, board, vec![hero, villain]);

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
