use im::Vector;
use std::convert::{TryFrom, TryInto};
use crate::tron::{GameState, Player};
use crate::tron_pb;

// Conversion from your domain Player to the generated protobuf tron::Player.
impl From<Player> for tron_pb::Player {
    fn from(player: Player) -> Self {
        tron_pb::Player {
            row: player.row as i32,
            col: player.col as i32,
            can_move: player.can_move,
        }
    }
}

// Conversion from the generated protobuf tron::Player to your domain Player.
// We use TryFrom here because converting a signed integer to usize requires a check.
impl TryFrom<tron_pb::Player> for Player {
    type Error = &'static str;

    fn try_from(proto: tron_pb::Player) -> Result<Self, Self::Error> {
        if proto.row < 0 || proto.col < 0 {
            Err("row or col cannot be negative")
        } else {
            Ok(Player {
                row: proto.row as usize,
                col: proto.col as usize,
                can_move: proto.can_move,
            })
        }
    }
}

// Conversion from your domain GameState to the generated protobuf tron::GameState.
impl From<GameState> for tron_pb::GameState {
    fn from(state: GameState) -> Self {
        tron_pb::GameState {
            grid: state.grid.iter().map(|row| {
                // Each grid row is converted into a tron::GridRow
                tron_pb::GridRow {
                    cells: row.iter().copied().collect(),
                }
            }).collect(),
            players: state.players.iter().map(|player| (*player).into()).collect(),
        }
    }
}

// Conversion from the generated protobuf tron::GameState to your domain GameState.
impl TryFrom<tron_pb::GameState> for GameState {
    type Error = &'static str;

    fn try_from(proto: tron_pb::GameState) -> Result<Self, Self::Error> {
        // Convert grid: each tron::GridRow into an im::Vector<bool>
        let grid: Vector<Vector<bool>> = proto.grid.into_iter()
            .map(|row| Vector::from(row.cells))
            .collect();

        // Convert players: since TryFrom is fallible, we collect into a Vec first
        let players_vec: Vec<Player> = proto.players.into_iter()
            .map(|p| p.try_into())
            .collect::<Result<Vec<_>, _>>()?;
        let players = Vector::from(players_vec);

        Ok(GameState {
            grid,
            players,
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
        let states: Vec<GameState> = proto.game_states
            .into_iter()
            .map(|gs| gs.try_into())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(states)
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
        let games: Vec<Vec<GameState>> = proto.games
            .into_iter()
            .map(|gs| gs.try_into())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(games)
    }
}

