use anyhow::Result;
use core::f32;
use im::{vector, Vector};
use prost::Message; // for protobuf serialization
use rand::Rng;
use std::{io::Read, time::Instant};
use trust::tron;
use zmq;

use trust::{
    minimax::{minimax_alpha_beta_eval_all, MinimaxContext, MinimaxResult},
    models::TractModel,
    tron::{Direction, GameState, GameStatus, Player, StatusInfo},
    tron_pb,
};

fn main() {
    let start = Instant::now();
    let _ = run_sims().expect("bruh");

    let duration = start.elapsed();
    println!("Duration: {:?}", duration);
}

// This is cringe. Also should be moved to a GameState constructor probably
fn random_start_players(num_rows: usize, num_cols: usize) -> Vector<Player> {
    let mut start_pos_found = false;
    let mut rng = rand::thread_rng();

    let mut p1: Player;
    let mut p2: Player;
    let mut start_players: Vector<Player> = Vector::new();

    while !start_pos_found {
        p1 = Player {
            row: rng.gen_range(0..=num_rows - 1),
            col: rng.gen_range(0..=num_cols - 1),
            can_move: true,
        };

        p2 = Player {
            row: rng.gen_range(0..=num_rows - 1),
            col: rng.gen_range(0..=num_cols - 1),
            can_move: true,
        };

        if !(p1.row == p2.row && p1.col == p2.col) {
            start_pos_found = true;
            start_players = vector![p1, p2]
        }
    }

    // start_players = vector![p1, p2];
    start_players
}

fn run_sims() -> Result<()> {
    // Create a ZMQ context and a PUSH socket.
    let context = zmq::Context::new();
    let socket = context.socket(zmq::REQ)?;
    // Bind to a TCP address (for example, port 5555).
    socket.connect("tcp://localhost:5555")?;

    let mut tract_model: TractModel =
        TractModel::new("tron_model_v2.onnx").expect("failed to init tract model.");


    let n_sim_train_cycles = 10_000;
    let n_games = 16;
    let start = Instant::now();

    let mut p1_wins = 0;
    let mut p2_wins = 0;
    let mut ties = 0;

    for i in 0..n_sim_train_cycles {

        println!("\n\nStarting sims...");
        let start = Instant::now();

        let mut game_data: Vec<Vec<GameState>> = Vec::new();
        for j in 0..n_games {

            let start_players: Vector<Player> = random_start_players(10, 10);
            let mut curr_game = tron::new_game(start_players, 10, 10);

            let mut curr_status: StatusInfo = tron::get_status(&curr_game);

            let mut game_states: Vec<GameState> = vec![curr_game.clone()]; // Why does this need a clone?

            while curr_status.status == GameStatus::InProgress {
                // bot chooses player directions

                let p1_mm_result: MinimaxResult = minimax_alpha_beta_eval_all(
                    &curr_game,
                    4,
                    true,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    None,
                    &MinimaxContext {
                        model: &tract_model,
                        maximizing_player: 0,
                        minimizing_player: 1,
                    },
                );

                let p2_mm_result: MinimaxResult = minimax_alpha_beta_eval_all(
                    &curr_game,
                    4,
                    true,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    None,
                    &MinimaxContext {
                        model: &tract_model,
                        maximizing_player: 1,
                        minimizing_player: 0,
                    },
                );

                let p1_direction: Direction =
                    p1_mm_result.principal_variation.unwrap_or(Direction::Up);
                let p2_direction: Direction =
                    p2_mm_result.principal_variation.unwrap_or(Direction::Up);

                let directions: [Direction; 2] = [p1_direction, p2_direction];
                curr_game = tron::next(&curr_game, &directions);
                curr_status = tron::get_status(&curr_game);

                game_states.push(curr_game.clone()); // Why does this need a clone?
            }

            game_data.push(game_states);
            if curr_status.status == GameStatus::Tie {
                ties += 1;
            } else {
                let winner_index = curr_status
                    .winner_index
                    .expect("Winner index should be set");
                if winner_index == 0 {
                    p1_wins += 1;
                } else if winner_index == 1 {
                    p2_wins += 1;
                }
            }
        }


        let duration = start.elapsed();

        println!("Simulated {n_games} in {:?} seconds.", duration);
        println!("Games per second: {:?}", (n_games as f32) / duration.as_secs_f32());

        println!("Cycle: {i}. P1 wins: {p1_wins}, P2 wins: {p2_wins}, ties: {ties}.");
        println!("Sending game data...");

        // Convert the domain Game into the protobuf message.
        let game_data_pb: tron_pb::Games = game_data.into();

        // Serialize the protobuf message to a byte vector.
        let mut buf = Vec::new();
        game_data_pb.encode(&mut buf)?;

        // Send the serialized bytes over ZMQ.
        socket.send(buf, 0)?;

        println!("Sent game state sequence over ZMQ");
        println!("Awaiting updated model...");

        let reply = socket.recv_msg(0)?;

        let model_bytes: &[u8] = reply.as_ref();

        tract_model = TractModel::from_bytes(model_bytes).expect("failed to init model from bytes");

        println!("Received new model!");
    }

    // println!("Game completed!: {:?}", game.status);

    let duration = start.elapsed();
    println!("Time to run {} games: {:?}", n_games, duration);

    println!("P1 wins: {}, p2 wins: {}, ties: {}", p1_wins, p2_wins, ties);

    Ok(())
}
