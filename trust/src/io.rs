use crate::tron::{GameState, Player};
use crate::tron_pb;
use im::Vector;
use std::convert::{TryFrom, TryInto};

impl From<GameState> for tron_pb::GameState {
    fn from(state: GameState) -> Self {
        let num_rows = state.grid.len();
        let num_cols = if num_rows == 0 { 0 } else { state.grid[0].len() };

        tron_pb::GameState {
            num_rows: num_rows as i32,
            num_cols: num_cols as i32,
            board: board_to_bytes(&state.grid, num_cols),
            players: state
                .players
                .iter()
                .map(|player| player_to_proto(*player, num_cols))
                .collect(),
        }
    }
}

impl TryFrom<tron_pb::GameState> for GameState {
    type Error = &'static str;

    fn try_from(proto: tron_pb::GameState) -> Result<Self, Self::Error> {
        if proto.num_rows < 0 || proto.num_cols < 0 {
            return Err("grid dimensions cannot be negative");
        }

        let num_rows = proto.num_rows as usize;
        let num_cols = proto.num_cols as usize;
        let num_cells = num_rows.checked_mul(num_cols).ok_or("grid is too large")?;

        let mut grid: Vector<Vector<bool>> = Vector::new();
        for row in 0..num_rows {
            let cells: Vec<bool> = (0..num_cols)
                .map(|col| bit_is_set(&proto.board, row * num_cols + col))
                .collect();
            grid.push_back(Vector::from(cells));
        }

        let players_vec: Vec<Player> = proto
            .players
            .into_iter()
            .map(|player| player_from_proto(player, num_cols, num_cells))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(GameState {
            grid,
            players: Vector::from(players_vec),
        })
    }
}

impl From<Vec<GameState>> for tron_pb::Game {
    fn from(states: Vec<GameState>) -> Self {
        tron_pb::Game {
            game_states: states.into_iter().map(|state| state.into()).collect(),
        }
    }
}

impl TryFrom<tron_pb::Game> for Vec<GameState> {
    type Error = &'static str;

    fn try_from(proto: tron_pb::Game) -> Result<Self, Self::Error> {
        proto
            .game_states
            .into_iter()
            .map(|game_state| game_state.try_into())
            .collect::<Result<Vec<_>, _>>()
    }
}

impl From<Vec<Vec<GameState>>> for tron_pb::Games {
    fn from(games: Vec<Vec<GameState>>) -> Self {
        tron_pb::Games {
            games: games.into_iter().map(|game| game.into()).collect(),
        }
    }
}

impl TryFrom<tron_pb::Games> for Vec<Vec<GameState>> {
    type Error = &'static str;

    fn try_from(proto: tron_pb::Games) -> Result<Self, Self::Error> {
        proto
            .games
            .into_iter()
            .map(|game| game.try_into())
            .collect::<Result<Vec<_>, _>>()
    }
}

fn player_to_proto(player: Player, num_cols: usize) -> tron_pb::Player {
    tron_pb::Player {
        idx: (player.row * num_cols + player.col) as i32,
        can_move: player.can_move,
    }
}

fn player_from_proto(
    proto: tron_pb::Player,
    num_cols: usize,
    num_cells: usize,
) -> Result<Player, &'static str> {
    if proto.idx < 0 {
        return Err("player idx cannot be negative");
    }

    let idx = proto.idx as usize;
    if idx >= num_cells {
        return Err("player idx is out of bounds");
    }

    if num_cols == 0 {
        return Err("player idx requires positive num_cols");
    }

    Ok(Player {
        row: idx / num_cols,
        col: idx % num_cols,
        can_move: proto.can_move,
    })
}

fn board_to_bytes(grid: &Vector<Vector<bool>>, num_cols: usize) -> Vec<u8> {
    let num_cells = grid.len() * num_cols;
    let mut board = vec![0u8; (num_cells + 7) / 8];

    for (row_index, row) in grid.iter().enumerate() {
        for (col_index, occupied) in row.iter().enumerate() {
            if *occupied {
                let idx = row_index * num_cols + col_index;
                board[idx / 8] |= 1u8 << (idx % 8);
            }
        }
    }

    while board.last() == Some(&0) {
        board.pop();
    }

    board
}

fn bit_is_set(board: &[u8], idx: usize) -> bool {
    board
        .get(idx / 8)
        .map_or(false, |byte| (byte & (1u8 << (idx % 8))) != 0)
}
