use crate::tron::{BitBoard, GameState, Player};
use crate::tron_pb;
use std::convert::{TryFrom, TryInto};

impl From<GameState> for tron_pb::GameState {
    fn from(state: GameState) -> Self {
        tron_pb::GameState {
            num_rows: state.num_rows as i32,
            num_cols: state.num_cols as i32,
            board: state.board.to_le_bytes(),
            players: state.players.into_iter().map(player_to_proto).collect(),
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
        let board = BitBoard::from_le_bytes(num_cells, &proto.board)
            .map_err(|_| "board has bits set outside the grid")?;

        let players_vec: Vec<Player> = proto
            .players
            .into_iter()
            .map(|player| player_from_proto(player, num_cells))
            .collect::<Result<Vec<_>, _>>()?;

        GameState::try_new(num_rows, num_cols, board, players_vec).map_err(|_| "invalid game state")
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

fn player_to_proto(player: Player) -> tron_pb::Player {
    tron_pb::Player {
        idx: player.idx as i32,
        can_move: player.can_move,
    }
}

fn player_from_proto(proto: tron_pb::Player, num_cells: usize) -> Result<Player, &'static str> {
    if proto.idx < 0 {
        return Err("player idx cannot be negative");
    }

    let idx = proto.idx as usize;
    if idx >= num_cells {
        return Err("player idx is out of bounds");
    }

    Ok(Player {
        idx,
        can_move: proto.can_move,
    })
}
