use im::vector;
use std::hint::black_box;
use std::time::{Duration, Instant};

use trust::{
    tron::{self, BitBoard, Direction, GameState, Player},
    tron_2d,
};

const DEFAULT_ITERATIONS: usize = 1_000_000;
const WARMUP_ITERATIONS: usize = 10_000;

fn main() {
    let iterations = std::env::var("BENCH_NEXT_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_ITERATIONS);

    println!("Benchmarking next() with {iterations} iterations");
    println!("Tip: override with BENCH_NEXT_ITERS=5000000");
    println!();

    for board_size in [5, 30] {
        let bitboard_game = make_bitboard_game(board_size);
        let grid_game = make_2d_game(board_size);
        let directions = [Direction::Right, Direction::Left];

        warmup_bitboard(&bitboard_game, &directions);
        warmup_2d(&grid_game, &directions);

        let bitboard_duration = bench_bitboard(&bitboard_game, &directions, iterations);
        let grid_duration = bench_2d(&grid_game, &directions, iterations);

        print_result("bitboard", board_size, iterations, bitboard_duration);
        print_result("2d grid ", board_size, iterations, grid_duration);
        println!();
    }
}

fn make_bitboard_game(board_size: usize) -> GameState {
    let p1_idx = cell_idx(1, 1, board_size);
    let p2_idx = cell_idx(board_size - 2, board_size - 2, board_size);

    GameState::new(
        board_size,
        board_size,
        BitBoard::from_indices([p1_idx, p2_idx]),
        vec![Player::new(p1_idx, true), Player::new(p2_idx, true)],
    )
}

fn make_2d_game(board_size: usize) -> tron_2d::GameState2D {
    tron_2d::new_game(
        vector![
            tron_2d::Player {
                row: 1,
                col: 1,
                can_move: true,
            },
            tron_2d::Player {
                row: board_size - 2,
                col: board_size - 2,
                can_move: true,
            },
        ],
        board_size,
        board_size,
    )
}

fn warmup_bitboard(game: &GameState, directions: &[Direction; 2]) {
    for _ in 0..WARMUP_ITERATIONS {
        black_box(tron::next(black_box(game), black_box(directions)));
    }
}

fn warmup_2d(game: &tron_2d::GameState2D, directions: &[Direction; 2]) {
    for _ in 0..WARMUP_ITERATIONS {
        black_box(tron_2d::next(black_box(game), black_box(directions)));
    }
}

fn bench_bitboard(game: &GameState, directions: &[Direction; 2], iterations: usize) -> Duration {
    let start = Instant::now();

    for _ in 0..iterations {
        black_box(tron::next(black_box(game), black_box(directions)));
    }

    start.elapsed()
}

fn bench_2d(
    game: &tron_2d::GameState2D,
    directions: &[Direction; 2],
    iterations: usize,
) -> Duration {
    let start = Instant::now();

    for _ in 0..iterations {
        black_box(tron_2d::next(black_box(game), black_box(directions)));
    }

    start.elapsed()
}

fn print_result(label: &str, board_size: usize, iterations: usize, duration: Duration) {
    let ns_per_call = duration.as_nanos() as f64 / iterations as f64;

    println!(
        "{label} {board_size:>2}x{board_size:<2}: {:>10.3?} total, {:>10.2} ns/call",
        duration, ns_per_call
    );
}

fn cell_idx(row: usize, col: usize, num_cols: usize) -> usize {
    row * num_cols + col
}
