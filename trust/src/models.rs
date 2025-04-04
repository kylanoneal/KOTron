use std::io::Cursor;

use crate::tron::GameState;

use ndarray::Array4;
use ndarray::ArrayViewD;

use tract_onnx::prelude::*;
// use ort::session::Session;
// use tch::{CModule, Cuda, Device, IndexOp, Kind, Tensor};

use anyhow::Result;

pub trait Model {
    fn run_inference(&self, game_states: &Vec<&GameState>, player_index: usize) -> Result<Vec<f32>>;
}

#[derive(Debug)]
pub struct TractModel {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl TractModel {
    pub fn new(model_path: &str) -> Result<Self> {
        // Get CUDA working

        let model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>> =
            tract_onnx::onnx()
                .model_for_path(model_path)?
                .into_optimized()?
                .into_runnable()?;

        Ok(Self { model })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // Wrap the byte slice in a Cursor to implement `Read`
        let mut reader: Cursor<&[u8]> = Cursor::new(bytes);
        // Load the model from the provided byte slice
        let model = tract_onnx::onnx()
            .model_for_read(&mut reader)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }
}

impl Model for TractModel {
    fn run_inference(&self, game_states: &Vec<&GameState>, player_index: usize) -> Result<Vec<f32>> {
        // Example input tensor

        let mut input: Array4<f32> =
            tract_ndarray::Array4::<f32>::zeros((game_states.len(), 3, 10, 10));

        for (i, game) in game_states.iter().enumerate() {
            // Probably a better way of doing this...should game.grid be an ndarray to begin with?
            for row in 0..game.grid.len() {
                for col in 0..game.grid[0].len() {
                    input[[i, 0, row, col]] = if game.grid[row][col] { 1.0 } else { 0.0 };
                }
            }

            // Assuming 2 players
            let opponent_index = if player_index == 0 { 1 } else { 0 };

            input[[
                i,
                1,
                game.players[player_index].row,
                game.players[player_index].col,
            ]] = 1.0;

            input[[
                i,
                2,
                game.players[opponent_index].row,
                game.players[opponent_index].col,
            ]] = -1.0;
        }

        let tensor_input: Tensor = input.into();

        let result: tract_data::internal::tract_smallvec::SmallVec<[TValue; 4]> =
            self.model.run(tvec!(tensor_input.into()))?;

        // result is a SmallVec<[TValue; 4]> from self.model.run(...)
        let output_tensor = result[0].clone().into_tensor();

        // If you know the output is always f32, you can try converting it directly:
        let output_array: ArrayViewD<f32> = output_tensor.to_array_view::<f32>()?;

        // Now you can turn that array into a Vec<f32> by iterating over the elements:
        let output_vec: Vec<f32> = output_array.iter().cloned().collect();

        Ok(output_vec)
    }
}

// pub struct OrtModel {
//     model: Session,
// }
// impl OrtModel {
//     pub fn new(model_path: &str) -> Result<Self> {
//         // Get CUDA working
//         let model = Session::builder()?.commit_from_file(model_path)?;

//         Ok(Self { model })
//     }
// }

// impl Model for OrtModel {
//     fn run_inference(&self, game_states: &Vec<GameState>, player_index: usize) -> Result<Vec<f32>> {
//         // Example input tensor

//         let mut input: Array4<f32> = Array::zeros((game_states.len(), 3, 10, 10));

//         for (i, game) in game_states.iter().enumerate() {
//             // Probably a better way of doing this...should game.grid be an ndarray to begin with?
//             for row in 0..game.grid.len() {
//                 for col in 0..game.grid[0].len() {
//                     input[[i, 0, row, col]] = if game.grid[row][col] { 1.0 } else { 0.0 };
//                 }
//             }

//             // Assuming 2 players
//             let opponent_index = if player_index == 0 {1} else {0};

//             input[[i, 1, game.players[player_index].row, game.players[player_index].col]] = 1.0;

//             input[[i, 2, game.players[opponent_index].row, game.players[opponent_index].col]] = -1.0;
//         }

//         let outputs = self.model.run(ort::inputs![input]?)?;
//         let predictions = outputs[0].try_extract_tensor::<f32>()?;

//         let predictions_array = predictions.to_owned(); // This creates an ArrayD<f32>
//         let (predictions_vec, _) = predictions_array.into_raw_vec_and_offset();

//         Ok(predictions_vec)
//     }
// }

// pub struct TorchScriptModel {
//     model: CModule,
//     device: Device
// }
// impl TorchScriptModel {
//     pub fn new(model_path: &str, device: Device) -> Result<Self> {
//         let model = tch::CModule::load_on_device(model_path, device)?;
//         Ok(Self { model: model, device: device })
//     }
// }

// impl Model for TorchScriptModel {
//     fn run_inference(&self, game_states: &Vec<GameState>, player_index: usize) -> Result<Vec<f32>> {
//         // Example input tensor

//         let mut flattened_data: Vec<f32> = vec![0.0; game_states.len() * 3 * 10 * 10];

//         let game_offset = 10 * 10 * 3;
//         let channel_offset = 10 * 10;
//         let row_offset = 10;

//         for (i, game) in game_states.iter().enumerate(){

//             // Place walls into flattened data

//             for row in 0..game.grid.len() {
//                 for col in 0..game.grid[0].len() {

//                     if game.grid[row][col]{
//                         flattened_data[(i * game_offset) + (row_offset * row) + col] = 1.0;
//                     }
//                 }
//             }

//             // Place player positions into flattened data
//             let row = game.players[player_index].row;
//             let col = game.players[player_index].col;

//             flattened_data[(i * game_offset) + channel_offset + (row_offset * row) + col] = 1.0;

//             // Assumes 2 players
//             let opponent_index = if player_index == 0 {1} else {0};

//             let opponent_row = game.players[opponent_index].row;
//             let opponent_col = game.players[opponent_index].col;

//             flattened_data[(i * game_offset) + (channel_offset * 2) + (row_offset * opponent_row) + opponent_col] = -1.0;

//         }

//         let input_tensor = Tensor::from_slice(&flattened_data)
//             .view([game_states.len() as i64, 3, 10, 10]).to(self.device); // Reshape to [n, 3, 10, 10]

//         let output: Vec<f32> = Vec::<f32>::try_from(self.model.forward_ts(&[input_tensor])?)?;

//         Ok(output)
//     }
// }
