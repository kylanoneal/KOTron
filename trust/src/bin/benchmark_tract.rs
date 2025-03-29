// use anyhow::Result;
// use im::{vector, Vector};
// use std::time::Instant;
// use tract_onnx::prelude::*;


// use trust::{
//     algos::{choose_direction_model_naive, choose_direction_random},
//     tron::{Direction, DirectionUpdate, GameState, GameStatus, Player},
//     models::TractModel,
// };

fn main() {

    println!("bruh");

    // let start = Instant::now();
    // let _ = run_sims().expect("bruh");

    // let duration = start.elapsed();
    // println!("Duration: {:?}", duration);
}

// fn run_sims() -> Result<()> {
//     let p1 = Player {
//         row: 2,
//         col: 2,
//         direction: Direction::Right,
//         can_move: true,
//     };

//     let p2 = Player {
//         row: 7,
//         col: 7,
//         direction: Direction::Left,
//         can_move: true,
//     };

//     let mut players: Vector<Player> = vector![p1, p2];

//     let start_game: GameState = GameState::new(players, 10, 10);

//     let mut curr_game: GameState = start_game.clone();


//     let tract_model: TractModel =
//         TractModel::new("tron_model_v2.onnx").expect("failed to init tract model.");

//     // let model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>> =
//     // tract_onnx::onnx()
//     //     .model_for_path("tron_model_v2.onnx")?
//     //     .into_optimized()?
//     //     .into_runnable()?;
//     // let mut input: tract_ndarray::Array4<f32> = tract_ndarray::Array4::<f32>::zeros((1, 3, 10, 10));

//     // // Probably a better way of doing this...should game.grid be an ndarray to begin with?
//     // for row in 0..curr_game.grid.len() {
//     //     for col in 0..curr_game.grid[0].len() {
//     //         input[[0, 0, row, col]] = if curr_game.grid[row][col] { 1.0 } else { 0.0 };
//     //     }
//     // }

//     // input[[
//     //     0,
//     //     1,
//     //     curr_game.players[0].row,
//     //     curr_game.players[0].col,
//     // ]] = 1.0;

//     // input[[
//     //     0,
//     //     2,
//     //     curr_game.players[1].row,
//     //     curr_game.players[1].col,
//     // ]] = -1.0;

//     // let tensor_input: Tensor = input.into();

//     // let result: tract_data::internal::tract_smallvec::SmallVec<[TValue; 4]> =
//     //     model.run(tvec!(tensor_input.into()))?;

//     // // result is a SmallVec<[TValue; 4]> from self.model.run(...)
//     // let output_tensor = result[0].clone().into_tensor();

//     // // If you know the output is always f32, you can try converting it directly:
//     // let output_array: tract_ndarray::ArrayViewD<f32> = output_tensor.to_array_view::<f32>()?;

//     // // Now you can turn that array into a Vec<f32> by iterating over the elements:
//     // let output_vec: Vec<f32> = output_array.iter().cloned().collect();
//     // println!("Output vec: {:?}", output_vec);

//     println!("starting sims");
//     let n_games = 100;
//     let start = Instant::now();

//     let mut p1_wins = 0;
//     let mut p2_wins = 0;
//     let mut ties = 0;

//     for i in 0..n_games {
//         let mut curr_game = start_game.clone();

//         while curr_game.status == GameStatus::InProgress {
//             // bot chooses player directions

//             let p1_direction_update: DirectionUpdate =
//                 choose_direction_model_naive(&curr_game, 0, &tract_model);
//             let p2_direction_update: DirectionUpdate = choose_direction_random(&curr_game, 1);

//             curr_game =
//                 GameState::next(&curr_game, &vec![p1_direction_update, p2_direction_update]);
//         }

//         if curr_game.status == GameStatus::Tie {
//             ties += 1;
//         } else if curr_game.status == GameStatus::P1Win {
//             p1_wins += 1;
//         } else if curr_game.status == GameStatus::P2Win {
//             p2_wins += 1;
//         }
//     }

//     // println!("Game completed!: {:?}", game.status);

//     let duration = start.elapsed();
//     println!("Time to run {} games: {:?}", n_games, duration);

//     println!("P1 wins: {}, p2 wins: {}, ties: {}", p1_wins, p2_wins, ties);

//     Ok(())
// }
