use trust::run_engine;

fn main() {

    let bruh: trust::Move = run_engine();

    println!("bruh: {}, {} ", bruh.row_offset, bruh.col_offset);
}