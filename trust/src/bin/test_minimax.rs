use anyhow::Result;
use core::f32;
use im::{vector, Vector};
use std::time::Instant;
use trust::tron;

use trust::{
    minimax::{minimax_alpha_beta_eval_all, MinimaxContext, MinimaxResult},
    models::TractModel,
    tron::{Direction, GameState, GameStatus, Player, StatusInfo},
};

fn main() {
    let start = Instant::now();
    let _ = run_sims().expect("bruh");

    let duration = start.elapsed();
    println!("Duration: {:?}", duration);
}

fn run_sims() -> Result<()> {
    let p1 = Player {
        row: 2,
        col: 2,
        can_move: true,
    };

    let p2 = Player {
        row: 7,
        col: 7,
        can_move: true,
    };



    let tract_model: TractModel =
        TractModel::new("tron_model_v2.onnx").expect("failed to init tract model.");

    println!("starting sims");
    let n_games = 100;
    let start = Instant::now();

    let mut p1_wins = 0;
    let mut p2_wins = 0;
    let mut ties = 0;

    for i in 0..n_games {

        let start_players: Vector<Player> = vector![p1, p2];
        let mut curr_game = tron::new_game(start_players, 10, 10);

        let mut curr_status: StatusInfo = tron::get_status(&curr_game);

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

            let p1_direction: Direction = p1_mm_result.principal_variation.unwrap_or(Direction::Up);
            let p2_direction: Direction = p2_mm_result.principal_variation.unwrap_or(Direction::Up);

            let directions: [Direction; 2] = [p1_direction, p2_direction];
            curr_game = tron::next(&curr_game, &directions);
            curr_status = tron::get_status(&curr_game);
        }

        println!("{i}");
        if curr_status.status == GameStatus::Tie {
            ties += 1;
            println!("Tie");
        } else {
            let winner_index = curr_status
                .winner_index
                .expect("Winner index should be set");
            if winner_index == 0 {
                p1_wins += 1;
                println!("P1 win!");
            } else if winner_index == 1 {
                p2_wins += 1;
                println!("P2 win!");
            }
        }
    }

    // println!("Game completed!: {:?}", game.status);

    let duration = start.elapsed();
    println!("Time to run {} games: {:?}", n_games, duration);

    println!("P1 wins: {}, p2 wins: {}, ties: {}", p1_wins, p2_wins, ties);

    Ok(())
}
