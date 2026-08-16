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
use tron::{self as bit_tron, BitBoard, Direction, GameState, GameStatus, Player};

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

#[wasm_bindgen]
pub struct BotSession {
    game_state: GameState,
    bot_index: usize,
    human_index: usize,
    best_move: Option<Direction>,
    search_depth: u32,
}

#[wasm_bindgen]
impl BotSession {
    #[wasm_bindgen(constructor)]
    pub fn new(
        data: &[u8],
        num_rows: usize,
        num_cols: usize,
        bot_row: usize,
        bot_col: usize,
        human_row: usize,
        human_col: usize,
    ) -> BotSession {
        init_bot(
            data, num_rows, num_cols, bot_row, bot_col, human_row, human_col,
        )
    }

    pub fn search_step(&mut self) -> Move {
        let next_depth = self.search_depth.saturating_add(1).max(1);
        self.search_at_depth(next_depth)
    }

    pub fn search_at_depth(&mut self, depth: u32) -> Move {
        if bit_tron::get_status(&self.game_state).status != GameStatus::InProgress {
            return self.current_best_move();
        }

        self.search_depth = depth.max(1);
        self.best_move = search_best_move(
            &self.game_state,
            self.bot_index,
            self.human_index,
            self.search_depth,
        );

        self.current_best_move()
    }

    pub fn get_move(&mut self, human_move: Move) -> Move {
        if self.best_move.is_none()
            && bit_tron::get_status(&self.game_state).status == GameStatus::InProgress
        {
            self.search_at_depth(1);
        }

        let bot_direction = self.best_move.unwrap_or_else(|| {
            fallback_direction(&self.game_state, self.bot_index).unwrap_or(Direction::Up)
        });
        let human_direction = direction_from_move(human_move);

        let mut directions = [Direction::Up, Direction::Up];
        directions[self.bot_index] = bot_direction;
        directions[self.human_index] = human_direction;

        self.game_state = bit_tron::next(&self.game_state, &directions);
        self.best_move = None;
        self.search_depth = 0;

        move_from_direction(bot_direction)
    }

    pub fn current_best_move(&self) -> Move {
        move_from_direction(
            self.best_move
                .or_else(|| fallback_direction(&self.game_state, self.bot_index))
                .unwrap_or(Direction::Up),
        )
    }

    pub fn search_depth(&self) -> u32 {
        self.search_depth
    }

    pub fn is_finished(&self) -> bool {
        bit_tron::get_status(&self.game_state).status != GameStatus::InProgress
    }

    pub fn status(&self) -> u8 {
        match bit_tron::get_status(&self.game_state).status {
            GameStatus::InProgress => 0,
            GameStatus::Tie => 1,
            GameStatus::Winner => 2,
        }
    }

    pub fn winner_index(&self) -> i32 {
        bit_tron::get_status(&self.game_state)
            .winner_index
            .map(|idx| idx as i32)
            .unwrap_or(-1)
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
pub fn init_bot(
    data: &[u8],
    num_rows: usize,
    num_cols: usize,
    bot_row: usize,
    bot_col: usize,
    human_row: usize,
    human_col: usize,
) -> BotSession {
    BotSession {
        game_state: game_state_from_flat_occupancy(
            data, num_rows, num_cols, bot_row, bot_col, human_row, human_col,
        ),
        bot_index: 0,
        human_index: 1,
        best_move: None,
        search_depth: 0,
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
    let game = game_state_from_flat_occupancy(
        data,
        num_rows,
        num_cols,
        player_row,
        player_col,
        opponent_row,
        opponent_col,
    );

    move_from_direction(
        search_best_move(&game, 0, 1, 1)
            .unwrap_or_else(|| fallback_direction(&game, 0).unwrap_or(Direction::Up)),
    )
}

fn game_state_from_flat_occupancy(
    data: &[u8],
    num_rows: usize,
    num_cols: usize,
    player_row: usize,
    player_col: usize,
    opponent_row: usize,
    opponent_col: usize,
) -> GameState {
    let player_idx = player_row * num_cols + player_col;
    let opponent_idx = opponent_row * num_cols + opponent_col;

    let mut board = BitBoard::from_flat_occupancy(data, num_rows * num_cols)
        .expect("invalid flat occupancy board");
    board.set(player_idx);
    board.set(opponent_idx);

    GameState::new(
        num_rows,
        num_cols,
        board,
        vec![
            Player::new(player_idx, true),
            Player::new(opponent_idx, true),
        ],
    )
}

fn search_best_move(
    game: &GameState,
    bot_index: usize,
    human_index: usize,
    depth: u32,
) -> Option<Direction> {
    if bit_tron::get_status(game).status != GameStatus::InProgress {
        return None;
    }

    let model = get_nnue_model(game.num_rows, game.num_cols);
    let result: MinimaxResult = alphabeta(
        game,
        depth.max(1),
        true,
        f32::NEG_INFINITY,
        f32::INFINITY,
        None,
        &MinimaxContext {
            model,
            maximizing_player: bot_index,
            minimizing_player: human_index,
        },
    );

    result.principal_variation
}

fn fallback_direction(game: &GameState, player_index: usize) -> Option<Direction> {
    bit_tron::get_possible_directions(game, player_index)
        .into_iter()
        .next()
}

fn direction_from_move(mv: Move) -> Direction {
    match (mv.row_offset, mv.col_offset) {
        (-1, 0) => Direction::Up,
        (1, 0) => Direction::Down,
        (0, -1) => Direction::Left,
        (0, 1) => Direction::Right,
        _ => panic!(
            "invalid move offset ({}, {}), expected one cardinal step",
            mv.row_offset, mv.col_offset
        ),
    }
}

fn move_from_direction(direction: Direction) -> Move {
    let (row_offset, col_offset) = direction.value();
    Move {
        row_offset,
        col_offset,
    }
}
