use im::Vector;
use strum_macros::EnumIter;
use strum::IntoEnumIterator;

#[derive(Debug, EnumIter, Copy, Clone)]
pub enum Direction {
    Up,
    Right,
    Down,
    Left,
}

impl Direction {
    pub fn value(&self) -> (i8, i8) {
        match self {
            Direction::Up => (-1, 0),
            Direction::Right => (0, 1),
            Direction::Down => (1, 0),
            Direction::Left => (0, -1),
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum GameStatus {
    InProgress,
    Tie,
    Winner,
}

pub struct StatusInfo {
    pub status: GameStatus,
    pub winner_index: Option<usize>,
}

#[derive(Copy, Clone)]
pub struct Player {
    pub row: usize,
    pub col: usize,
    pub can_move: bool,
}

#[derive(Clone)]
pub struct GameState {
    // These should all be private?
    pub grid: Vector<Vector<bool>>,
    pub players: Vector<Player>,
}

#[derive(Copy, Clone)]
pub struct DirectionUpdate {
    pub player_index: usize,
    pub direction: Direction,
}

/// Creates a new `GameState` with the given players
pub fn new_game(players: Vector<Player>, num_rows: usize, num_cols: usize) -> GameState {
    let mut grid: Vector<Vector<bool>> = Vector::new();
    // Initialize the grid
    for _ in 0..num_rows {
        grid.push_back(Vector::from(vec![false; num_cols]));
    }

    // Mark the players' starting positions
    for player in players.iter() {
        let grid_row = grid[player.row].update(player.col, true);
        grid = grid.update(player.row, grid_row);
    }

    GameState { players, grid }
}

// Moves players in their direction, also does collision checking
pub fn next(game: &GameState, directions: &[Direction]) -> GameState {
    assert!(directions.len() == game.players.len());

    let mut next_grid = game.grid.clone();

    let mut next_players: Vector<Player> = Vector::new();

    // Check if player is going to go out of bounds or hit wall
    for (player, direction) in game.players.iter().zip(directions.iter()) {
        let mut next_can_move: bool = player.can_move;
        let mut next_row: usize = player.row;
        let mut next_col: usize = player.col;

        if player.can_move {
            let (delta_row, delta_col) = direction.value();
            let new_row = (player.row as i8) + delta_row;
            let new_col = (player.col as i8) + delta_col;

            if !in_bounds(&game, new_row, new_col)
                || game.grid[new_row as usize][new_col as usize]
            {
                next_can_move = false;
            } else {
                next_row = new_row as usize;
                next_col = new_col as usize;
            }
        }

        next_players.push_back(Player {
            row: next_row,
            col: next_col,
            can_move: next_can_move,
        });
    }

    // Handle case where 2 or more players try to occupy same square
    for i in 0..next_players.len() {
        let pi: Player = next_players[i];

        if pi.can_move {
            for j in (0..next_players.len()).filter(|&j| j != i) {
                let pj: Player = next_players[j];

                if pj.can_move {
                    if pi.row == pj.row && pi.col == pj.col {
                        next_players[i] = Player {
                            row: pi.row,
                            col: pi.col,
                            can_move: false,
                        };
                        next_players[j] = Player {
                            row: pj.row,
                            col: pj.col,
                            can_move: false,
                        };
                    }
                }
            }
        }
    }

    // Add new walls to self.grid
    // You can add this to the previous loop if ya want
    for player in next_players.iter() {
        let new_grid_row = next_grid[player.row].update(player.col, true);
        next_grid = next_grid.update(player.row, new_grid_row);
    }

    GameState {
        grid: next_grid,
        players: next_players,
    }
}

pub fn in_bounds(game: &GameState, row: i8, col: i8) -> bool {
    // Fix the cringe i8 bullshit
    row >= 0 && col >= 0 && row < (game.grid.len() as i8) && col < (game.grid[0].len() as i8)
}

// Move this into game and change signature to pub fn get_possible_moves(game, player_index)
pub fn get_possible_directions(game: &GameState, player_index: usize) -> Vec<Direction> {

    let p_row: usize = game.players[player_index].row;
    let p_col: usize = game.players[player_index].col;

    let mut possible_moves: Vec<Direction> = Vec::new();
    for direction in Direction::iter() {
        let new_row: i8 = (p_row as i8) + direction.value().0;
        let new_col: i8 = (p_col as i8) + direction.value().1;

        if in_bounds(game, new_row, new_col) && !game.grid[new_row as usize][new_col as usize] {
            possible_moves.push(direction);
        }
    }
    return possible_moves;
}

pub fn get_status(game_state: &GameState) -> StatusInfo {
    let mut num_players_can_move = 0;
    let mut winner_index: usize = 0;
    let status: StatusInfo;

    for (i, player) in game_state.players.iter().enumerate() {
        // player is a reference to an element in players
        if player.can_move {
            num_players_can_move += 1;
            winner_index = i;
        }
    }

    if num_players_can_move == 0 {
        status = StatusInfo {
            status: GameStatus::Tie,
            winner_index: None,
        };
    } else if num_players_can_move == 1 {
        // Assuming 2 players only

        status = StatusInfo {
            status: GameStatus::Winner,
            winner_index: Some(winner_index),
        };
    } else {
        status = StatusInfo {
            status: GameStatus::InProgress,
            winner_index: None,
        };
    }

    return status;
}

/// Prints the current state of the grid
pub fn print_grid(game_state: &GameState) {
    for row in game_state.grid.iter() {
        for &cell in row.iter() {
            print!("{}", if cell { "1 " } else { "0 " });
        }
        println!();
    }
    println!("\n-----------------------------\n");
}
