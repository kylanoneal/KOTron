use im::{Vector, vector};
use strum_macros::EnumIter;

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
    P1Win,
    P2Win,
}

#[derive(Copy, Clone)]
pub struct Player {
    pub row: usize,
    pub col: usize,
    pub direction: Direction,
    pub can_move: bool,
}

#[derive(Clone)]
pub struct GameState {
    pub players: Vector<Player>,
    pub grid: Vector<Vector<bool>>,
    pub status: GameStatus,
}

#[derive(Copy, Clone)]
pub struct DirectionUpdate {
    pub player_index: usize,
    pub direction: Direction,
}

impl GameState {
    /// Creates a new `GameState` with the given players
    pub fn new(players: Vector<Player>, num_rows: usize, num_cols: usize) -> Self {
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

        GameState {
            players,
            grid,
            status: GameStatus::InProgress,
        }
    }

    // Moves players in their direction, also does collision checking
    pub fn next(game: &GameState, direction_updates: &Vec<DirectionUpdate>) -> Self {


        // let new_p1: Player = Player {
        //     direction: p1_direction,              
        //     ..game.players[0].clone()        
        // };
    
        // let new_p2: Player = Player {
        //     direction: p2_direction,               
        //     ..game.players[1].clone()          
        // };
    
        // let mut new_players: Vector<Player> = vector![new_p1, new_p2];

        // Just clone the whole game state instead of clone players and grid individually?

        // Does this work?
        let mut new_players = game.players.clone();

        for direction_update in direction_updates.iter()
        {
            new_players[direction_update.player_index].direction = direction_update.direction;
        }

        // You need to figure out what's going on with these "immutable" arrays
        // that seem to be completely mutable...
        let mut new_grid = game.grid.clone();

        // Check if player is going to go out of bounds or hit wall
        for new_player in new_players.iter_mut() {

            if new_player.can_move {
                let (delta_row, delta_col) = new_player.direction.value();
                let new_row = (new_player.row as i8) + delta_row;
                let new_col = (new_player.col as i8) + delta_col;

                if Self::out_of_bounds(&game, new_row, new_col)
                    || game.grid[new_row as usize][new_col as usize]
                {
                    new_player.can_move = false;
                } else {
                    new_player.row = new_row as usize;
                    new_player.col = new_col as usize;
                }
            }

        }

        for i in 0..new_players.len() {
            if new_players[i].can_move {
                for j in (0..new_players.len()).filter(|&j| j != i) {
                    if new_players[j].can_move {
                        if new_players[i].row == new_players[j].row
                            && new_players[i].col == new_players[j].col
                        {
                            new_players[i].can_move = false;
                            new_players[j].can_move = false;
                        }
                    }
                }
            }
        }

        // Add new walls to self.grid
        // You can add this to the previous loop if ya want
        for player in new_players.iter() {
            let new_grid_row = new_grid[player.row].update(player.col, true);
            new_grid = new_grid.update(player.row, new_grid_row);
        }

        let new_status: GameStatus = Self::get_status(&new_players);

        GameState {
            players: new_players,
            grid: new_grid,
            status: new_status,
        }
    }

    pub fn out_of_bounds(game: &GameState, row: i8, col: i8) -> bool {
        // Fix the cringe i8 bullshit
        row < 0 || col < 0 || row >= (game.grid.len() as i8) || col >= (game.grid[0].len() as i8)
    }

    fn get_status(players: &Vector<Player>) -> GameStatus {
        let mut num_players_can_move = 0;
        let status: GameStatus;

        for player in players.iter() {
            // player is a reference to an element in players
            if player.can_move {
                num_players_can_move += 1
            }
        }

        if num_players_can_move == 0 {
            status = GameStatus::Tie
        } else if num_players_can_move == 1 {
            // Assuming 2 players only

            status = if players[0].can_move {
                GameStatus::P1Win
            } else {
                GameStatus::P2Win
            };
        } else {
            status = GameStatus::InProgress;
        }

        return status;
    }


    /// Prints the current state of the grid
    pub fn print_grid(&self) {
        for row in &self.grid {
            for &cell in row {
                print!("{}", if cell { "1 " } else { "0 " });
            }
            println!();
        }
        println!("\n-----------------------------\n");
    }
}
