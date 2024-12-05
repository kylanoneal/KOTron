use ndarray::{Array, Array3, Array4};
use rand::Rng;
use std::ffi::CString;
use std::os::raw::c_char;
use std::time::Instant;
use strum::IntoEnumIterator;
use winapi::um::libloaderapi::LoadLibraryA;

use ort::{
    execution_providers::CUDAExecutionProvider,
    inputs,
    session::{builder::GraphOptimizationLevel, Session, SessionOutputs},
};

use im::{vector, Vector};
use tch::{nn, CModule, Cuda, Device, IndexOp, Kind, Tensor};

mod models;
mod tron;

use models::{Model, OrtModel, TorchScriptModel};
use tron::{Direction, DirectionUpdate, GameState, GameStatus, Player};

use anyhow::{ensure, Result};


    // Hack to make CUDA work
    // let path = CString::new("C:\\libtorch\\lib\\torch_cuda.dll").unwrap();

    // unsafe {
    //     LoadLibraryA(path.as_ptr() as *const c_char);
    // }

    // println!("cuda: {}", tch::Cuda::is_available());
    // println!("cudnn: {}", tch::Cuda::cudnn_is_available());
    // assert!(Cuda::is_available());

fn debug_ort() -> Result<Vec<f32>>
{
    let model = Session::builder()?.commit_from_file("tron_model_dynamic.onnx")?;

    let mut input: Array4<f32> = Array::zeros((2, 3, 10, 10));

    let outputs = model.run(ort::inputs![input]?)?;

    let predictions = outputs[0].try_extract_tensor::<f32>()?;

    let predictions_array = predictions.to_owned(); // This creates an ArrayD<f32>
    let (predictions_vec, _) = predictions_array.into_raw_vec_and_offset();

    println!("{:?}", predictions_vec);

    Ok(predictions_vec)
}

fn main() {

    
    // let result = debug_ort();
    // match result {
    //     Ok(value) => {
    //         // Handle the success case with `value`
    //         println!("Got a value: {:?}", value);
    //     }
    //     Err(error) => {
    //         // Handle the error case
    //         eprintln!("An error occurred: {:?}", error);
    //     }
    // }

    // Init benchmarking game state

    let p1 = Player {
        row: 2,
        col: 2,
        direction: Direction::Right,
        can_move: true,
    };

    let p2 = Player {
        row: 7,
        col: 7,
        direction: Direction::Left,
        can_move: true,
    };

    let mut players: Vector<Player> = vector![p1, p2];

    let start_game: GameState = GameState::new(players, 10, 10);

    let mut curr_game: GameState = start_game.clone();

    let ort_model: OrtModel = OrtModel::new("tron_model_dynamic.onnx").expect("Failed to load model");

    println!("starting sims");
    let n_games = 1000;
    let start = Instant::now();

    let mut p1_wins = 0;
    let mut p2_wins = 0;
    let mut ties = 0;

    for i in 0..n_games{

        let mut curr_game = start_game.clone();

        while curr_game.status == GameStatus::InProgress {
            // bot chooses player directions

            let p1_direction_update: DirectionUpdate = choose_direction_model_naive(&curr_game, 0, &ort_model);
            let p2_direction_update: DirectionUpdate = choose_direction_random(&curr_game, 1);

            curr_game = GameState::next(&curr_game, &vec![p1_direction_update, p2_direction_update]);

        }

        if curr_game.status == GameStatus::Tie{
            ties += 1;
        }
        else if curr_game.status == GameStatus::P1Win {
            p1_wins += 1;
        }
        else if curr_game.status == GameStatus::P2Win {
            p2_wins += 1;
        }

    }

    // println!("Game completed!: {:?}", game.status);

    let duration = start.elapsed();
    println!("Time to run {} games: {:?}", n_games, duration);

    println!("P1 wins: {}, p2 wins: {}, ties: {}", p1_wins, p2_wins, ties);
}

fn choose_direction_random(game: &GameState, player_index: usize) -> DirectionUpdate {
    let possible_moves: Vec<Direction> = get_possible_moves(
        game.players[player_index].row,
        game.players[player_index].col,
        &game.grid,
    );

    let mut rng = rand::thread_rng(); // Create a random number generator
    let chosen_direction: Direction;

    if !possible_moves.is_empty() {
        let random_index = rng.gen_range(0..possible_moves.len());
        chosen_direction = possible_moves[random_index];
    } else {
        chosen_direction = game.players[player_index].direction;
    }

    DirectionUpdate {
        player_index: player_index,
        direction: chosen_direction,
    }
}

fn choose_direction_model_naive(
    game: &GameState,
    player_index: usize,
    model: &dyn Model,
) -> DirectionUpdate {
    // CLean
    if game.players[player_index].can_move {
        let possible_moves: Vec<Direction> = get_possible_moves(
            game.players[player_index].row,
            game.players[player_index].col,
            &game.grid,
        );

        // CLean
        if possible_moves.is_empty() {
            return DirectionUpdate {
                player_index: player_index,
                direction: game.players[player_index].direction,
            };
        }

        let mut game_states_to_eval: Vec<GameState> = Vec::new();

        for possible_move in possible_moves.iter() {
            // Copy game state
            let next_game = GameState::next(
                game,
                &vec![DirectionUpdate {
                    player_index: player_index,
                    direction: *possible_move,
                }],
            );

            game_states_to_eval.push(next_game);
        }

        let evaluations: Vec<f32> = model
            .run_inference(&game_states_to_eval, player_index)
            .expect("Failed to run inference.");

        // Update direction to possible_moves[argmax(evaluations)]
        return DirectionUpdate {
            player_index: player_index,
            direction: possible_moves[argmax(&evaluations)],
        };
    } else {
        // Return current direction if it can't move
        DirectionUpdate {
            player_index: player_index,
            direction: game.players[player_index].direction,
        }
    }
}

fn argmax(vec: &Vec<f32>) -> usize {
    let mut max_index = 0;
    let mut max_val = vec[0];

    // Probably get rid of this if it's not possible
    assert!(!vec[0].is_nan());

    for i in 1..vec.len() {
        // Probably get rid of this if it's not possible
        assert!(!vec[i].is_nan());

        if vec[i] > max_val {
            max_index = i;
            max_val = vec[i];
        }
    }

    max_index
}

fn get_possible_moves(row: usize, col: usize, grid: &Vector<Vector<bool>>) -> Vec<Direction> {
    let curr_row = row as i8;
    let curr_col = col as i8;

    let mut possible_moves: Vec<Direction> = Vec::new();
    for direction in Direction::iter() {
        let new_row = curr_row + direction.value().0;
        let new_col = curr_col + direction.value().1;

        let out_of_bounds: bool = new_row < 0
            || new_col < 0
            || new_row >= (grid.len() as i8)
            || new_col >= (grid[0].len() as i8);

        if !(out_of_bounds || grid[new_row as usize][new_col as usize]) {
            possible_moves.push(direction);
        }
    }
    return possible_moves;
}

// fn ort_vs_tch(){
//     let device = Device::Cuda(0);

//     // Init benchmarking game state

//     let p1 = Player {
//         row: 0,
//         col: 0,
//         direction: Direction::Right,
//         can_move: true,
//     };

//     let p2 = Player {
//         row: 9,
//         col: 9,
//         direction: Direction::Left,
//         can_move: true,
//     };

//     let mut players: Vector<Player> = vector![p1, p2];

//     let start_game: GameState = GameState::new(players, 10, 10);

//     let mut curr_game: GameState = start_game.clone();

//     let no_update: Vec<DirectionUpdate> = vec![];

//     for i in 0..9 {
//         curr_game = GameState::next(&curr_game, &no_update);
//     }

//     let mut benchmark_game_vec: Vec<GameState> = Vec::new();

//     for i in 0..1000 {
//         benchmark_game_vec.push(curr_game.clone());
//     }
//     // Benchmark ORT

//     let ort_model = OrtModel::new("tron_model.onnx").expect("Failed to init ort model");
//     println!("\n--------------------------------------\n");
//     println!("Starting ORT (Parallel)");
//     let start = Instant::now();
//     let n_inf = 10;

//     for _ in 0..n_inf {
//         ort_model.run_inference(&benchmark_game_vec, 0);
//     }

//     let duration = start.elapsed();
//     println!(
//         "Time to run inference {} * {} times: {:?}",
//         n_inf,
//         benchmark_game_vec.len(),
//         duration
//     );
//     println!("\n--------------------------------------\n");
//     println!("Starting ORT (Non-Parallel)");
//     let start = Instant::now();
//     let n_inf = 10 * benchmark_game_vec.len();

//     for _ in 0..n_inf {

//         ort_model.run_inference(&vec![curr_game.clone()], 0);
//     }

//     let duration = start.elapsed();
//     println!("Time to run inference {} times: {:?}", n_inf, duration);

    // Benchmark TCH

    // let tch_model = TorchScriptModel::new("tron_torchscript_model.pt", Device::Cuda(0))
    //     .expect("Failed to init torchscript model.");

    // println!("\n--------------------------------------\n");
    // println!("Starting TCH (Parallel)");
    // let start = Instant::now();
    // let n_inf = 10;

    // for _ in 0..n_inf {
    //     benchmark_model(&benchmark_game_vec, &tch_model);
    // }

    // let duration = start.elapsed();
    // println!(
    //     "Time to run inference {} * {} times: {:?}",
    //     n_inf,
    //     benchmark_game_vec.len(),
    //     duration
    // );

    // println!("\n--------------------------------------\n");
    // println!("Starting TCH (Non-Parallel)");
    // let start = Instant::now();
    // let n_inf = 10 * benchmark_game_vec.len();

    // for _ in 0..n_inf {
    //     benchmark_model(&vec![curr_game.clone()], &tch_model);
    // }

    // let duration = start.elapsed();
    // println!("Time to run inference {} times: {:?}", n_inf, duration);
//}
