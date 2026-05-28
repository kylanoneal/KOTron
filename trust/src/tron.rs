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

#[derive(Clone)]
pub struct PovGameState {
    pub game_state: GameState,
    pub hero_index: usize,
    pub opponent_index: usize,
}
