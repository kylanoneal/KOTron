use crate::{
    model::Model,
    tron_2d::{GameState, PovGameState},
};
use anyhow::{bail, Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use std::fs::File;
use std::io::{Cursor, Read, Seek};
use std::path::Path;

#[derive(Debug)]
pub struct QuantizedNnue {
    pub scale: i64,
    pub clamp: i64,
    pub padding_idx: usize,

    pub embed_weights: Array2<i64>,

    fc_layer_weights: Vec<Array2<i64>>,
    fc_layer_biases: Vec<Array1<i64>>,

    fc_value_weights: Array2<i64>,
    fc_value_bias: Array1<i64>,
}

impl QuantizedNnue {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open {:?}", path.as_ref()))?;

        let mut npz = NpzReader::new(file).context("Failed to read npz file")?;

        Self::from_npz_reader(&mut npz)
    }

    pub fn from_bytes(bytes: &'static [u8]) -> Result<Self> {
        let cursor = Cursor::new(bytes);
        let mut npz = NpzReader::new(cursor).context("Failed to read npz bytes")?;

        Self::from_npz_reader(&mut npz)
    }

    fn from_npz_reader<R>(npz: &mut NpzReader<R>) -> Result<Self>
    where
        R: Read + Seek,
    {
        let scale_arr: Array1<i64> = npz.by_name("scale.npy")?;
        let clamp_arr: Array1<i64> = npz.by_name("clamp.npy")?;
        let padding_idx_arr: Array1<i64> = npz.by_name("padding_idx.npy")?;
        let num_fc_layers_arr: Array1<i64> = npz.by_name("num_fc_layers.npy")?;

        let scale = scale_arr[0];
        let clamp = clamp_arr[0];
        let padding_idx = padding_idx_arr[0] as usize;
        let num_fc_layers = num_fc_layers_arr[0] as usize;

        let embed_weights: Array2<i64> = npz.by_name("embed_weights.npy")?;

        let mut fc_layer_weights = Vec::with_capacity(num_fc_layers);
        let mut fc_layer_biases = Vec::with_capacity(num_fc_layers);

        for i in 0..num_fc_layers {
            let weight_name = format!("fc_layer_{}_weights.npy", i);
            let bias_name = format!("fc_layer_{}_bias.npy", i);

            let weight: Array2<i64> = npz
                .by_name(&weight_name)
                .with_context(|| format!("Failed to load {}", weight_name))?;

            let bias: Array1<i64> = npz
                .by_name(&bias_name)
                .with_context(|| format!("Failed to load {}", bias_name))?;

            validate_linear_shapes(&weight, &bias)
                .with_context(|| format!("Invalid shapes for FC layer {}", i))?;

            fc_layer_weights.push(weight);
            fc_layer_biases.push(bias);
        }

        let fc_value_weights: Array2<i64> = npz.by_name("fc_value_weights.npy")?;
        let fc_value_bias: Array1<i64> = npz.by_name("fc_value_bias.npy")?;

        validate_linear_shapes(&fc_value_weights, &fc_value_bias)
            .context("Invalid shapes for value layer")?;

        let model = Self {
            scale,
            clamp,
            padding_idx,
            embed_weights,
            fc_layer_weights,
            fc_layer_biases,
            fc_value_weights,
            fc_value_bias,
        };

        model.validate_network_shapes()?;
        model.validate_feature_layout()?;

        Ok(model)
    }
    pub fn acc_dim(&self) -> usize {
        self.embed_weights.shape()[1]
    }

    pub fn num_features(&self) -> usize {
        self.embed_weights.shape()[0]
    }

    pub fn num_fc_layers(&self) -> usize {
        self.fc_layer_weights.len()
    }

    pub fn make_empty_accumulator(&self) -> Vec<i64> {
        vec![0; self.acc_dim()]
    }

    pub fn accumulator_from_indices(&self, indices: &[usize]) -> Result<Vec<i64>> {
        let mut acc = self.make_empty_accumulator();

        for &idx in indices {
            if idx == self.padding_idx {
                continue;
            }

            self.add_feature(&mut acc, idx)?;
        }

        Ok(acc)
    }

    pub fn add_feature(&self, acc: &mut [i64], feature_idx: usize) -> Result<()> {
        if acc.len() != self.acc_dim() {
            bail!(
                "Accumulator has wrong length. Got {}, expected {}",
                acc.len(),
                self.acc_dim()
            );
        }

        if feature_idx >= self.num_features() {
            bail!(
                "Feature index {} is out of bounds for {} features",
                feature_idx,
                self.num_features()
            );
        }

        let row = self.embed_weights.row(feature_idx);

        for i in 0..acc.len() {
            acc[i] += row[i];
        }

        Ok(())
    }

    pub fn remove_feature(&self, acc: &mut [i64], feature_idx: usize) -> Result<()> {
        if acc.len() != self.acc_dim() {
            bail!(
                "Accumulator has wrong length. Got {}, expected {}",
                acc.len(),
                self.acc_dim()
            );
        }

        if feature_idx >= self.num_features() {
            bail!(
                "Feature index {} is out of bounds for {} features",
                feature_idx,
                self.num_features()
            );
        }

        let row = self.embed_weights.row(feature_idx);

        for i in 0..acc.len() {
            acc[i] -= row[i];
        }

        Ok(())
    }

    pub fn infer_from_indices(&self, indices: &[usize]) -> Result<f32> {
        let acc = self.accumulator_from_indices(indices)?;
        self.infer_from_accumulator(&acc)
    }

    pub fn infer_from_accumulator(&self, acc: &[i64]) -> Result<f32> {
        if acc.len() != self.acc_dim() {
            bail!(
                "Accumulator has wrong length. Got {}, expected {}",
                acc.len(),
                self.acc_dim()
            );
        }

        let mut x = acc.to_vec();

        clamp_in_place(&mut x, 0, self.scale * self.clamp);

        for (layer_idx, (weight, bias)) in self
            .fc_layer_weights
            .iter()
            .zip(self.fc_layer_biases.iter())
            .enumerate()
        {
            x = linear_raw(&x, weight, bias)
                .with_context(|| format!("Failed during FC layer {}", layer_idx))?;
            let layer_clamp = self.clamp * (self.scale.pow((layer_idx + 2) as u32));
            clamp_in_place(&mut x, 0, self.scale * layer_clamp);
        }

        let out = linear_raw(&x, &self.fc_value_weights, &self.fc_value_bias)
            .context("Failed during value layer")?;

        if out.len() != 1 {
            bail!(
                "Expected value layer to produce one output, but got {}",
                out.len()
            );
        }

        let denom = (self.scale.pow(2 + self.fc_layer_weights.len() as u32)) as f32;
        Ok(out[0] as f32 / denom)
    }

    fn validate_network_shapes(&self) -> Result<()> {
        let acc_dim = self.acc_dim();

        if let Some(first_weight) = self.fc_layer_weights.first() {
            let first_in_dim = first_weight.shape()[1];

            if first_in_dim != acc_dim {
                bail!(
                    "First FC layer input dim {} does not match accumulator dim {}",
                    first_in_dim,
                    acc_dim
                );
            }
        } else {
            let value_in_dim = self.fc_value_weights.shape()[1];

            if value_in_dim != acc_dim {
                bail!(
                    "Value layer input dim {} does not match accumulator dim {}",
                    value_in_dim,
                    acc_dim
                );
            }
        }

        for i in 1..self.fc_layer_weights.len() {
            let prev_out_dim = self.fc_layer_weights[i - 1].shape()[0];
            let curr_in_dim = self.fc_layer_weights[i].shape()[1];

            if curr_in_dim != prev_out_dim {
                bail!(
                    "FC layer {} input dim {} does not match previous output dim {}",
                    i,
                    curr_in_dim,
                    prev_out_dim
                );
            }
        }

        if let Some(last_weight) = self.fc_layer_weights.last() {
            let last_out_dim = last_weight.shape()[0];
            let value_in_dim = self.fc_value_weights.shape()[1];

            if value_in_dim != last_out_dim {
                bail!(
                    "Value layer input dim {} does not match final hidden dim {}",
                    value_in_dim,
                    last_out_dim
                );
            }
        }

        Ok(())
    }

    fn validate_feature_layout(&self) -> Result<()> {
        if self.num_features() < 4 {
            bail!("Expected at least 4 features, got {}", self.num_features());
        }

        if self.padding_idx + 1 != self.num_features() {
            bail!(
                "Expected padding_idx to be the final feature index. Got padding_idx={}, num_features={}",
                self.padding_idx,
                self.num_features()
            );
        }

        let non_padding_features = self.num_features() - 1;

        if non_padding_features % 3 != 0 {
            bail!(
                "Expected non-padding feature count to be divisible by 3, got {}",
                non_padding_features
            );
        }

        Ok(())
    }
}

impl Model for QuantizedNnue {
    fn run_inference(&self, pov_game_state: &PovGameState) -> Result<f32> {
        let indices = pov_game_state_to_feature_indices(pov_game_state, self.padding_idx)?;
        self.infer_from_indices(&indices)
    }
}

pub fn pov_game_state_to_feature_indices(
    pov_game_state: &PovGameState,
    padding_idx: usize,
) -> Result<Vec<usize>> {
    let game_state = &pov_game_state.game_state;

    let num_rows = game_state.grid.len();

    if num_rows == 0 {
        bail!("Game grid has zero rows");
    }

    let num_cols = game_state.grid[0].len();

    if num_cols == 0 {
        bail!("Game grid has zero columns");
    }

    validate_rectangular_grid(game_state)?;

    let num_cells = num_rows * num_cols;

    if padding_idx != num_cells * 3 {
        bail!(
            "Expected padding_idx={} for a {}x{} board, but model has padding_idx={}",
            num_cells * 3,
            num_rows,
            num_cols,
            padding_idx
        );
    }

    if pov_game_state.hero_index >= game_state.players.len() {
        bail!(
            "hero_index {} out of bounds for {} players",
            pov_game_state.hero_index,
            game_state.players.len()
        );
    }

    if pov_game_state.opponent_index >= game_state.players.len() {
        bail!(
            "opponent_index {} out of bounds for {} players",
            pov_game_state.opponent_index,
            game_state.players.len()
        );
    }

    let mut indices = Vec::with_capacity(num_cells + 2);

    // Occupied/wall cell features.
    //
    // Feature range:
    //   [0, num_cells)
    for row in 0..num_rows {
        for col in 0..num_cols {
            if game_state.grid[row][col] {
                indices.push(cell_index(row, col, num_cols));
            }
        }
    }

    let hero = game_state.players[pov_game_state.hero_index];
    let opponent = game_state.players[pov_game_state.opponent_index];

    validate_player_position(hero.row, hero.col, num_rows, num_cols, "hero")?;
    validate_player_position(opponent.row, opponent.col, num_rows, num_cols, "opponent")?;

    let hero_cell = cell_index(hero.row, hero.col, num_cols);
    let opponent_cell = cell_index(opponent.row, opponent.col, num_cols);

    // Hero position features.
    //
    // Feature range:
    //   [num_cells, 2 * num_cells)
    indices.push(num_cells + hero_cell);

    // Opponent position features.
    //
    // Feature range:
    //   [2 * num_cells, 3 * num_cells)
    indices.push((2 * num_cells) + opponent_cell);

    Ok(indices)
}

fn validate_rectangular_grid(game_state: &GameState) -> Result<()> {
    let num_rows = game_state.grid.len();

    if num_rows == 0 {
        bail!("Game grid has zero rows");
    }

    let num_cols = game_state.grid[0].len();

    for row in 0..num_rows {
        if game_state.grid[row].len() != num_cols {
            bail!(
                "Grid is not rectangular. Row 0 has len {}, but row {} has len {}",
                num_cols,
                row,
                game_state.grid[row].len()
            );
        }
    }

    Ok(())
}

fn validate_player_position(
    row: usize,
    col: usize,
    num_rows: usize,
    num_cols: usize,
    label: &str,
) -> Result<()> {
    if row >= num_rows || col >= num_cols {
        bail!(
            "{} player position ({}, {}) is out of bounds for {}x{} board",
            label,
            row,
            col,
            num_rows,
            num_cols
        );
    }

    Ok(())
}

fn cell_index(row: usize, col: usize, num_cols: usize) -> usize {
    row * num_cols + col
}

fn validate_linear_shapes(weight: &Array2<i64>, bias: &Array1<i64>) -> Result<()> {
    let out_dim = weight.shape()[0];

    if bias.len() != out_dim {
        bail!(
            "Bias length {} does not match weight output dim {}",
            bias.len(),
            out_dim
        );
    }

    Ok(())
}

fn clamp_in_place(x: &mut [i64], min_val: i64, max_val: i64) {
    for v in x {
        *v = (*v).clamp(min_val, max_val);
    }
}

fn linear_raw(x: &[i64], weight: &Array2<i64>, bias: &Array1<i64>) -> Result<Vec<i64>> {
    let out_dim = weight.shape()[0];
    let in_dim = weight.shape()[1];

    if x.len() != in_dim {
        bail!(
            "Input length {} does not match layer input dim {}",
            x.len(),
            in_dim
        );
    }

    if bias.len() != out_dim {
        bail!(
            "Bias length {} does not match layer output dim {}",
            bias.len(),
            out_dim
        );
    }

    let mut out = vec![0i64; out_dim];

    for o in 0..out_dim {
        let mut sum = bias[o];

        for i in 0..in_dim {
            sum += weight[[o, i]] * x[i];
        }

        out[o] = sum;
    }

    Ok(out)
}
