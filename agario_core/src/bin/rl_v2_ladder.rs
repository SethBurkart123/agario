fn main() {
    if let Err(error) = agario_core::world::native_benchmark::run_cli() {
        eprintln!("ladder error: {error}");
        std::process::exit(2);
    }
}
