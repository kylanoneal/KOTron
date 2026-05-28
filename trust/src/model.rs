use crate::tron_2d::PovGameState;
use anyhow::Result;

pub trait Model {
    fn run_inference(&self, pov_game_state: &PovGameState) -> Result<f32>;
}
