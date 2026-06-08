use im::Vector;
use rand::seq::SliceRandom;
use rand::Rng;
use std::hash::{Hash, Hasher};
use strum_macros::EnumIter;

use crate::tron_2d::{GameState2D, Player as Player2D};

pub const MAX_ROWS: usize = 50;
pub const MAX_COLS: usize = 50;
pub const MAX_CELLS: usize = MAX_ROWS * MAX_COLS;

#[derive(Debug, EnumIter, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub const ALL: [Direction; 4] = [
        Direction::Up,
        Direction::Down,
        Direction::Left,
        Direction::Right,
    ];

    pub fn value(&self) -> (i8, i8) {
        match self {
            Direction::Up => (-1, 0),
            Direction::Down => (1, 0),
            Direction::Left => (0, -1),
            Direction::Right => (0, 1),
        }
    }

    pub fn are_opposite_directions(d1: Direction, d2: Direction) -> bool {
        let (dr1, dc1) = d1.value();
        let (dr2, dc2) = d2.value();
        dr1 + dr2 == 0 && dc1 + dc2 == 0
    }

    pub fn get_random_direction() -> Direction {
        *Self::ALL
            .choose(&mut rand::thread_rng())
            .expect("Direction::ALL is never empty")
    }

    pub fn fliplr(direction: Direction) -> Direction {
        match direction {
            Direction::Up | Direction::Down => direction,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }

    pub fn rot90_counterclockwise(direction: Direction) -> Direction {
        match direction {
            Direction::Up => Direction::Left,
            Direction::Left => Direction::Down,
            Direction::Down => Direction::Right,
            Direction::Right => Direction::Up,
        }
    }

    pub fn transform(
        directions: &[Direction],
        do_lr_flip: bool,
        n_rot_90: usize,
    ) -> Vec<Direction> {
        let mut transformed_dirs = directions.to_vec();

        if do_lr_flip {
            transformed_dirs = transformed_dirs
                .into_iter()
                .map(Direction::fliplr)
                .collect();
        }

        for _ in 0..(n_rot_90 % 4) {
            transformed_dirs = transformed_dirs
                .into_iter()
                .map(Direction::rot90_counterclockwise)
                .collect();
        }

        transformed_dirs
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub enum GameStatus {
    InProgress,
    Tie,
    Winner,
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub struct StatusInfo {
    pub status: GameStatus,
    pub winner_index: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PovGameState {
    pub game_state: GameState,
    pub hero_index: usize,
    pub opponent_index: usize,
}

impl PovGameState {
    pub fn new(game_state: GameState, hero_index: usize, opponent_index: usize) -> Self {
        assert!(hero_index < 2, "hero_index must be 0 or 1");
        assert!(opponent_index < 2, "opponent_index must be 0 or 1");
        assert_ne!(
            hero_index, opponent_index,
            "hero_index and opponent_index must differ"
        );

        Self {
            game_state,
            hero_index,
            opponent_index,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Player {
    pub idx: usize,
    pub can_move: bool,
}

impl Player {
    pub fn new(idx: usize, can_move: bool) -> Self {
        Self { idx, can_move }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct BitBoard {
    words: Vec<u64>,
}

impl BitBoard {
    pub fn empty() -> Self {
        Self { words: Vec::new() }
    }

    pub fn from_words(words: Vec<u64>) -> Self {
        let mut board = Self { words };
        board.trim_trailing_zero_words();
        board
    }

    pub fn from_indices<I>(indices: I) -> Self
    where
        I: IntoIterator<Item = usize>,
    {
        let mut board = Self::empty();

        for idx in indices {
            board.set(idx);
        }

        board
    }

    pub fn from_flat_occupancy(data: &[u8], num_cells: usize) -> Result<Self, String> {
        if data.len() != num_cells {
            return Err(format!(
                "occupancy data has length {}, expected {}",
                data.len(),
                num_cells
            ));
        }

        let mut board = Self::empty();
        for (idx, occupied) in data.iter().enumerate() {
            if *occupied != 0 {
                board.set(idx);
            }
        }

        Ok(board)
    }

    pub fn from_le_bytes(num_cells: usize, bytes: &[u8]) -> Result<Self, String> {
        let mut words = vec![0u64; (bytes.len() + 7) / 8];

        for (byte_idx, byte) in bytes.iter().enumerate() {
            let word_idx = byte_idx / 8;
            let shift = (byte_idx % 8) * 8;
            words[word_idx] |= (*byte as u64) << shift;
        }

        let board = Self::from_words(words);
        if board.has_bits_outside(num_cells) {
            return Err(format!(
                "board has bits set outside the first {} cells",
                num_cells
            ));
        }

        Ok(board)
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.words.len() * 8);

        for word in &self.words {
            for byte_offset in 0..8 {
                bytes.push((word >> (byte_offset * 8)) as u8);
            }
        }

        while bytes.last() == Some(&0) {
            bytes.pop();
        }

        bytes
    }

    pub fn get(&self, idx: usize) -> bool {
        let word_idx = idx / 64;
        let bit_idx = idx % 64;

        self.words
            .get(word_idx)
            .map_or(false, |word| (word & (1u64 << bit_idx)) != 0)
    }

    pub fn set(&mut self, idx: usize) {
        let word_idx = idx / 64;
        let bit_idx = idx % 64;

        if self.words.len() <= word_idx {
            self.words.resize(word_idx + 1, 0);
        }

        self.words[word_idx] |= 1u64 << bit_idx;
    }

    pub fn with_set(&self, idx: usize) -> Self {
        let mut next = self.clone();
        next.set(idx);
        next
    }

    pub fn occupied_indices_within(&self, num_cells: usize) -> Vec<usize> {
        let mut indices = Vec::new();

        for (word_idx, word) in self.words.iter().enumerate() {
            let base_idx = word_idx * 64;

            if base_idx >= num_cells {
                break;
            }

            let mut masked_word = *word;
            let remaining_cells = num_cells - base_idx;

            if remaining_cells < 64 {
                masked_word &= (1u64 << remaining_cells) - 1;
            }

            while masked_word != 0 {
                let bit_idx = masked_word.trailing_zeros() as usize;
                indices.push(base_idx + bit_idx);
                masked_word &= masked_word - 1;
            }
        }

        indices
    }

    pub fn has_bits_outside(&self, num_cells: usize) -> bool {
        let full_words = num_cells / 64;
        let remaining_bits = num_cells % 64;

        if remaining_bits == 0 {
            return self.words.iter().skip(full_words).any(|word| *word != 0);
        }

        if let Some(word) = self.words.get(full_words) {
            let valid_mask = (1u64 << remaining_bits) - 1;
            if (word & !valid_mask) != 0 {
                return true;
            }
        }

        self.words
            .iter()
            .skip(full_words + 1)
            .any(|word| *word != 0)
    }

    fn trim_trailing_zero_words(&mut self) {
        while self.words.last() == Some(&0) {
            self.words.pop();
        }
    }
}

impl From<u64> for BitBoard {
    fn from(word: u64) -> Self {
        Self::from_words(vec![word])
    }
}

impl From<Vec<u64>> for BitBoard {
    fn from(words: Vec<u64>) -> Self {
        Self::from_words(words)
    }
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub num_rows: usize,
    pub num_cols: usize,
    pub board: BitBoard,
    pub players: Vec<Player>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CanonicalKey {
    pub num_rows: usize,
    pub num_cols: usize,
    pub board: BitBoard,
    pub player_indices: [usize; 2],
}

impl GameState {
    pub fn new(num_rows: usize, num_cols: usize, board: BitBoard, players: Vec<Player>) -> Self {
        Self::try_new(num_rows, num_cols, board, players).expect("invalid GameState")
    }

    pub fn try_new(
        num_rows: usize,
        num_cols: usize,
        board: BitBoard,
        players: Vec<Player>,
    ) -> Result<Self, String> {
        let state = Self {
            num_rows,
            num_cols,
            board: BitBoard::from_words(board.words),
            players,
        };

        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.num_rows == 0 {
            return Err("num_rows must be positive".to_string());
        }
        if self.num_cols == 0 {
            return Err("num_cols must be positive".to_string());
        }

        let num_cells = self
            .num_rows
            .checked_mul(self.num_cols)
            .ok_or_else(|| "board dimensions overflow usize".to_string())?;

        if num_cells > MAX_CELLS {
            return Err(format!(
                "board has {} cells, but only {} cells are supported",
                num_cells, MAX_CELLS
            ));
        }

        if self.board.has_bits_outside(num_cells) {
            return Err(format!(
                "board has bits set outside the {}x{} grid",
                self.num_rows, self.num_cols
            ));
        }

        for (player_index, player) in self.players.iter().enumerate() {
            if player.idx >= num_cells {
                return Err(format!(
                    "Player {} index out of bounds: {} not in [0, {}]",
                    player_index,
                    player.idx,
                    num_cells - 1
                ));
            }

            if !self.board.get(player.idx) {
                return Err(format!(
                    "board bit at index {} must be set for a player head",
                    player.idx
                ));
            }

            for other_player in self.players.iter().skip(player_index + 1) {
                if player.idx == other_player.idx && (player.can_move || other_player.can_move) {
                    return Err("Active players occupying same square".to_string());
                }
            }
        }

        if self.players.len() != 2 {
            return Err("only two-player games are currently supported".to_string());
        }

        Ok(())
    }

    pub fn canonical_key(&self) -> CanonicalKey {
        self.canonical_key_if_active_two_player()
            .expect("canonical_key requires exactly two active players")
    }

    pub fn new_game(
        num_players: usize,
        num_rows: usize,
        num_cols: usize,
        random_starts: bool,
        neutral_starts: bool,
        obstacle_density: f64,
    ) -> Self {
        Self::try_new_game(
            num_players,
            num_rows,
            num_cols,
            random_starts,
            neutral_starts,
            obstacle_density,
        )
        .expect("invalid new game options")
    }

    pub fn try_new_game(
        num_players: usize,
        num_rows: usize,
        num_cols: usize,
        random_starts: bool,
        neutral_starts: bool,
        obstacle_density: f64,
    ) -> Result<Self, String> {
        if obstacle_density > 0.8 {
            return Err("Too many obstacles.".to_string());
        }

        if !obstacle_density.is_finite() {
            return Err("obstacle_density must be finite".to_string());
        }

        let num_cells = num_rows
            .checked_mul(num_cols)
            .ok_or_else(|| "board dimensions overflow usize".to_string())?;

        if num_cells < num_players {
            return Err("Too many players for grid size.".to_string());
        }

        if num_players != 2 {
            return Err("only two-player games are currently supported".to_string());
        }

        let mut board = BitBoard::empty();
        let mut rng = rand::thread_rng();

        if obstacle_density > 0.0 {
            let num_obstacles = (num_cells as f64 * obstacle_density) as usize;
            let mut obstacle_indices: Vec<usize> = (0..num_cells).collect();
            obstacle_indices.shuffle(&mut rng);

            for idx in obstacle_indices.into_iter().take(num_obstacles) {
                board.set(idx);
            }
        }

        let players = if random_starts && !neutral_starts {
            let mut start_indices: Vec<usize> = (0..num_cells).collect();
            start_indices.shuffle(&mut rng);
            start_indices
                .into_iter()
                .take(num_players)
                .map(|idx| Player::new(idx, true))
                .collect()
        } else if random_starts && neutral_starts {
            let retries = 200;
            let mut starts = None;

            for _ in 0..retries {
                let rand_row = rng.gen_range(0..num_rows);
                let rand_col = rng.gen_range(0..num_cols);
                let do_lr_flip = rng.gen_bool(0.5);
                let n_rot_90 = if do_lr_flip {
                    rng.gen_range(0..4)
                } else {
                    rng.gen_range(1..4)
                };

                let (_, _, oppo_row, oppo_col) =
                    transform_coord(num_rows, num_cols, rand_row, rand_col, do_lr_flip, n_rot_90);

                if oppo_row < num_rows
                    && oppo_col < num_cols
                    && (rand_row != oppo_row || rand_col != oppo_col)
                {
                    starts = Some((rand_row, rand_col, oppo_row, oppo_col));
                    break;
                }
            }

            let (rand_row, rand_col, oppo_row, oppo_col) =
                starts.ok_or_else(|| format!("Neutral start not found after {}.", retries))?;

            vec![
                Player::new(rand_row * num_cols + rand_col, true),
                Player::new(oppo_row * num_cols + oppo_col, true),
            ]
        } else {
            return Err("default starts are not implemented".to_string());
        };

        for player in &players {
            board.set(player.idx);
        }

        Self::try_new(num_rows, num_cols, board, players)
    }

    pub fn from_players<I>(players: I, num_rows: usize, num_cols: usize) -> Self
    where
        I: IntoIterator<Item = Player>,
    {
        let players: Vec<Player> = players.into_iter().collect();
        let num_cells = num_rows
            .checked_mul(num_cols)
            .expect("board dimensions overflow usize");

        let mut board = BitBoard::empty();

        for (i, player) in players.iter().enumerate() {
            assert!(player.idx < num_cells, "player index out of bounds");
            assert!(player.can_move, "from_players expects active players");

            for other_player in players.iter().skip(i + 1) {
                assert_ne!(
                    player.idx, other_player.idx,
                    "players must occupy distinct cells"
                );
            }

            board.set(player.idx);
        }

        Self::new(num_rows, num_cols, board, players)
    }

    pub fn transform(game_state: &GameState, do_lr_flip: bool, n_rot_90: usize) -> GameState {
        let n_rot_90 = n_rot_90 % 4;
        let (new_num_rows, new_num_cols) =
            transformed_dimensions(game_state.num_rows, game_state.num_cols, n_rot_90);

        let mut next_board = BitBoard::empty();

        for idx in get_wall_indices(game_state) {
            let row = idx / game_state.num_cols;
            let col = idx % game_state.num_cols;
            let (_, _, next_row, next_col) = transform_coord(
                game_state.num_rows,
                game_state.num_cols,
                row,
                col,
                do_lr_flip,
                n_rot_90,
            );
            next_board.set(next_row * new_num_cols + next_col);
        }

        let next_players = game_state
            .players
            .iter()
            .map(|player| {
                let row = player.idx / game_state.num_cols;
                let col = player.idx % game_state.num_cols;
                let (_, _, next_row, next_col) = transform_coord(
                    game_state.num_rows,
                    game_state.num_cols,
                    row,
                    col,
                    do_lr_flip,
                    n_rot_90,
                );

                Player::new(next_row * new_num_cols + next_col, player.can_move)
            })
            .collect();

        GameState::new(new_num_rows, new_num_cols, next_board, next_players)
    }

    pub fn to_2d(&self) -> GameState2D {
        from_bitboard(self)
    }

    pub fn from_2d(game_state: &GameState2D) -> Self {
        from_2d_game_state(game_state)
    }

    fn canonical_key_if_active_two_player(&self) -> Option<CanonicalKey> {
        if self.players.len() != 2 || !self.players.iter().all(|player| player.can_move) {
            return None;
        }

        let player_indices = if self.players[0].idx > self.players[1].idx {
            [self.players[0].idx, self.players[1].idx]
        } else {
            [self.players[1].idx, self.players[0].idx]
        };

        Some(CanonicalKey {
            num_rows: self.num_rows,
            num_cols: self.num_cols,
            board: self.board.clone(),
            player_indices,
        })
    }
}

impl PartialEq for GameState {
    fn eq(&self, other: &Self) -> bool {
        match (
            self.canonical_key_if_active_two_player(),
            other.canonical_key_if_active_two_player(),
        ) {
            (Some(left), Some(right)) => left == right,
            _ => {
                self.num_rows == other.num_rows
                    && self.num_cols == other.num_cols
                    && self.board == other.board
                    && self.players == other.players
            }
        }
    }
}

impl Eq for GameState {}

impl Hash for GameState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if let Some(key) = self.canonical_key_if_active_two_player() {
            key.hash(state);
            return;
        }

        self.num_rows.hash(state);
        self.num_cols.hash(state);
        self.board.hash(state);
        self.players.hash(state);
    }
}

pub fn get_status(game: &GameState) -> StatusInfo {
    let mut num_players_can_move = 0;
    let mut winner_index = None;

    for (i, player) in game.players.iter().enumerate() {
        if player.can_move {
            num_players_can_move += 1;
            winner_index = Some(i);
        }
    }

    if num_players_can_move == 0 {
        StatusInfo {
            status: GameStatus::Tie,
            winner_index: None,
        }
    } else if num_players_can_move == 1 {
        StatusInfo {
            status: GameStatus::Winner,
            winner_index,
        }
    } else {
        StatusInfo {
            status: GameStatus::InProgress,
            winner_index: None,
        }
    }
}

pub fn get_bit(board: &BitBoard, idx: usize) -> bool {
    board.get(idx)
}

pub fn get_wall_indices(game_state: &GameState) -> Vec<usize> {
    game_state
        .board
        .occupied_indices_within(game_state.num_rows * game_state.num_cols)
}

pub fn get_next_position(
    game: &GameState,
    player_index: usize,
    direction: Direction,
) -> (isize, bool) {
    let player = game.players[player_index];
    let player_idx = player.idx as isize;
    let num_cols = game.num_cols as isize;
    let last_row_start = game.num_cols * (game.num_rows - 1);

    match direction {
        Direction::Up => (player_idx - num_cols, player.idx < game.num_cols),
        Direction::Down => (player_idx + num_cols, player.idx >= last_row_start),
        Direction::Left => (player_idx - 1, player.idx % game.num_cols == 0),
        Direction::Right => (
            player_idx + 1,
            player.idx % game.num_cols == game.num_cols - 1,
        ),
    }
}

pub fn get_next_player(game: &GameState, player_index: usize, direction: Direction) -> Player {
    let player = game.players[player_index];
    let mut next_idx = player.idx;
    let mut next_can_move = player.can_move;

    if player.can_move {
        let (new_idx, is_oob) = get_next_position(game, player_index, direction);

        if is_oob {
            next_can_move = false;
        } else {
            next_idx = new_idx as usize;
        }
    }

    if next_can_move && game.board.get(next_idx) {
        next_can_move = false;
        next_idx = player.idx;
    }

    Player::new(next_idx, next_can_move)
}

pub fn next(game: &GameState, directions: &[Direction]) -> GameState {
    assert_eq!(
        directions.len(),
        game.players.len(),
        "directions must have one move per player"
    );

    let mut next_players: Vec<Player> = directions
        .iter()
        .enumerate()
        .map(|(i, direction)| get_next_player(game, i, *direction))
        .collect();
    let mut next_board = game.board.clone();

    for i in 0..next_players.len() {
        let pi = next_players[i];
        next_board.set(pi.idx);

        if pi.can_move {
            for j in (i + 1)..next_players.len() {
                let pj = next_players[j];

                if pj.can_move && pi.idx == pj.idx {
                    for player in &mut next_players {
                        if player.idx == pi.idx {
                            player.can_move = false;
                        }
                    }
                    break;
                }
            }
        }
    }

    GameState::new(game.num_rows, game.num_cols, next_board, next_players)
}

pub fn get_possible_directions(game: &GameState, player_index: usize) -> Vec<Direction> {
    let mut available_directions = Vec::new();

    for direction in Direction::ALL {
        let (new_idx, is_oob) = get_next_position(game, player_index, direction);

        if !is_oob && !game.board.get(new_idx as usize) {
            available_directions.push(direction);
        }
    }

    available_directions
}

pub fn in_bounds(game: &GameState, row: isize, col: isize) -> bool {
    row >= 0 && col >= 0 && row < game.num_rows as isize && col < game.num_cols as isize
}

pub fn from_2d_game_state(game: &GameState2D) -> GameState {
    let num_rows = game.grid.len();
    let num_cols = if num_rows == 0 { 0 } else { game.grid[0].len() };
    let mut board = BitBoard::empty();

    for row in 0..num_rows {
        for col in 0..num_cols {
            if game.grid[row][col] {
                board.set(row * num_cols + col);
            }
        }
    }

    let players = game
        .players
        .iter()
        .map(|player| Player::new(player.row * num_cols + player.col, player.can_move))
        .collect();

    GameState::new(num_rows, num_cols, board, players)
}

pub fn from_bitboard(game: &GameState) -> GameState2D {
    let mut grid: Vector<Vector<bool>> = Vector::new();

    for _ in 0..game.num_rows {
        grid.push_back(Vector::from(vec![false; game.num_cols]));
    }

    for idx in get_wall_indices(game) {
        let row = idx / game.num_cols;
        let col = idx % game.num_cols;
        let grid_row = grid[row].update(col, true);
        grid = grid.update(row, grid_row);
    }

    let players = Vector::from(
        game.players
            .iter()
            .map(|player| Player2D {
                row: player.idx / game.num_cols,
                col: player.idx % game.num_cols,
                can_move: player.can_move,
            })
            .collect::<Vec<_>>(),
    );

    GameState2D { grid, players }
}

pub fn new_game<I>(players: I, num_rows: usize, num_cols: usize) -> GameState
where
    I: IntoIterator<Item = Player>,
{
    GameState::from_players(players, num_rows, num_cols)
}

fn transformed_dimensions(num_rows: usize, num_cols: usize, n_rot_90: usize) -> (usize, usize) {
    if n_rot_90 % 2 == 0 {
        (num_rows, num_cols)
    } else {
        (num_cols, num_rows)
    }
}

fn transform_coord(
    num_rows: usize,
    num_cols: usize,
    row: usize,
    col: usize,
    do_lr_flip: bool,
    n_rot_90: usize,
) -> (usize, usize, usize, usize) {
    let mut current_num_rows = num_rows;
    let mut current_num_cols = num_cols;
    let mut current_row = row;
    let mut current_col = if do_lr_flip { num_cols - 1 - col } else { col };

    for _ in 0..(n_rot_90 % 4) {
        let next_row = current_num_cols - 1 - current_col;
        let next_col = current_row;

        current_row = next_row;
        current_col = next_col;
        std::mem::swap(&mut current_num_rows, &mut current_num_cols);
    }

    (current_num_rows, current_num_cols, current_row, current_col)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(row: usize, col: usize, num_cols: usize) -> usize {
        row * num_cols + col
    }

    fn make_game(
        num_rows: usize,
        num_cols: usize,
        player_specs: &[(usize, bool)],
        extra_walls: &[usize],
    ) -> GameState {
        let mut board = BitBoard::from_indices(extra_walls.iter().copied());
        let players = player_specs
            .iter()
            .map(|(idx, can_move)| {
                board.set(*idx);
                Player::new(*idx, *can_move)
            })
            .collect();

        GameState::new(num_rows, num_cols, board, players)
    }

    #[test]
    fn next_handles_head_on_collision() {
        let game = make_game(3, 3, &[(idx(1, 0, 3), true), (idx(1, 2, 3), true)], &[]);

        let next_game = next(&game, &[Direction::Right, Direction::Left]);

        assert_eq!(
            next_game.players,
            vec![
                Player::new(idx(1, 1, 3), false),
                Player::new(idx(1, 1, 3), false),
            ]
        );
        assert_eq!(
            get_wall_indices(&next_game),
            vec![idx(1, 0, 3), idx(1, 1, 3), idx(1, 2, 3)]
        );
    }

    #[test]
    fn possible_directions_follow_python_order() {
        let game = make_game(
            3,
            3,
            &[(idx(1, 1, 3), true), (idx(2, 2, 3), true)],
            &[idx(0, 1, 3), idx(1, 0, 3)],
        );

        assert_eq!(
            get_possible_directions(&game, 0),
            vec![Direction::Down, Direction::Right]
        );
    }

    #[test]
    fn transform_rotates_rectangular_board_like_numpy_rot90() {
        let game = make_game(
            2,
            3,
            &[(idx(0, 0, 3), true), (idx(1, 2, 3), false)],
            &[idx(0, 2, 3)],
        );

        let transformed = GameState::transform(&game, true, 1);

        assert_eq!(transformed.num_rows, 3);
        assert_eq!(transformed.num_cols, 2);
        assert_eq!(get_wall_indices(&transformed), vec![0, 4, 5]);
        assert_eq!(
            transformed.players,
            vec![Player::new(0, true), Player::new(5, false)]
        );
    }

    #[test]
    fn proto_bytes_round_trip_little_endian_bits() {
        let board = BitBoard::from_indices([0, 8, 70]);
        let bytes = board.to_le_bytes();
        let decoded = BitBoard::from_le_bytes(80, &bytes).unwrap();

        assert_eq!(decoded, board);
        assert!(decoded.get(0));
        assert!(decoded.get(8));
        assert!(decoded.get(70));
    }
}
