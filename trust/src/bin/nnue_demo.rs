use anyhow::Result;
use std::time::Instant;

use trust::{
    model::Model,
    nnue::QuantizedNnue,
    tron::{BitBoard, GameState, Player, PovGameState},
};

fn main() {
    let start = Instant::now();
    let _ = run_sims().expect("bruh");

    let duration = start.elapsed();
    println!("Duration: {:?}", duration);
}

fn run_sims() -> Result<()> {
    let nnue_model =
        QuantizedNnue::load(r#"C:\Users\kylan\code\KOTron\tron-python\models\3x3.npz"#)?;

    println!("Loaded model:");
    println!("  scale: {}", nnue_model.scale);
    println!("  padding_idx: {}", nnue_model.padding_idx);
    println!("  num_features: {}", nnue_model.num_features());
    println!("  acc_dim: {}", nnue_model.acc_dim());
    println!("  num_fc_layers: {}", nnue_model.num_fc_layers());

    let example = PovGameState {
        game_state: GameState::new(
            3,
            3,
            BitBoard::from_indices([1, 2, 4, 5, 6, 7, 8]),
            vec![
                Player {
                    idx: 6,
                    can_move: true,
                },
                Player {
                    idx: 8,
                    can_move: true,
                },
            ],
        ),
        hero_index: 0,
        opponent_index: 1,
    };

    let mirrored = PovGameState {
        game_state: example.game_state.clone(),
        hero_index: 1,
        opponent_index: 0,
    };

    let eval = nnue_model.run_inference(&example)?;
    let eval_mirrored = nnue_model.run_inference(&mirrored)?;

    println!("  eval: {}", eval);
    println!("  mirrored: {}", eval_mirrored);

    Ok(())

    // let p1 = Player {
    //     row: 0,
    //     col: 0,
    //     can_move: true,
    // };

    // let p2 = Player {
    //     row: 2,
    //     col: 2,
    //     can_move: true,
    // };

    // println!("starting sims");
    // let n_games = 100;
    // let start = Instant::now();

    // let mut p1_wins = 0;
    // let mut p2_wins = 0;
    // let mut ties = 0;

    // for i in 0..n_games {
    //     let start_players: Vector<Player> = vector![p1, p2];
    //     let mut curr_game = tron::new_game(start_players, 3, 3);

    //     let mut curr_status: StatusInfo = tron::get_status(&curr_game);

    //     while curr_status.status == GameStatus::InProgress {
    //         // bot chooses player directions

    //         let p1_mm_result: MinimaxResult = alphabeta(
    //             &curr_game,
    //             1,
    //             true,
    //             f32::NEG_INFINITY,
    //             f32::INFINITY,
    //             None,
    //             &MinimaxContext {
    //                 model: &nnue_model,
    //                 maximizing_player: 0,
    //                 minimizing_player: 1,
    //             },
    //         );

    //         let p2_mm_result: MinimaxResult = alphabeta(
    //             &curr_game,
    //             1,
    //             true,
    //             f32::NEG_INFINITY,
    //             f32::INFINITY,
    //             None,
    //             &MinimaxContext {
    //                 model: &nnue_model,
    //                 maximizing_player: 1,
    //                 minimizing_player: 0,
    //             },
    //         );

    //         let p1_direction: Direction = p1_mm_result.principal_variation.unwrap_or(Direction::Up);
    //         let p2_direction: Direction = p2_mm_result.principal_variation.unwrap_or(Direction::Up);

    //         let directions: [Direction; 2] = [p1_direction, p2_direction];
    //         curr_game = tron::next(&curr_game, &directions);
    //         curr_status = tron::get_status(&curr_game);
    //     }

    //     println!("{i}");
    //     if curr_status.status == GameStatus::Tie {
    //         ties += 1;
    //         println!("Tie");
    //     } else {
    //         let winner_index = curr_status
    //             .winner_index
    //             .expect("Winner index should be set");
    //         if winner_index == 0 {
    //             p1_wins += 1;
    //             println!("P1 win!");
    //         } else if winner_index == 1 {
    //             p2_wins += 1;
    //             println!("P2 win!");
    //         }
    //     }
    // }

    // // println!("Game completed!: {:?}", game.status);

    // let duration = start.elapsed();
    // println!("Time to run {} games: {:?}", n_games, duration);

    // println!("P1 wins: {}, p2 wins: {}, ties: {}", p1_wins, p2_wins, ties);

    // Ok(())
}
