#[tokio::main]
async fn main() {
    if let Err(error) = agario_core::server::run().await {
        eprintln!("server error: {error}");
        std::process::exit(1);
    }
}
