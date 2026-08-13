use std::io;

#[tokio::main]
async fn main() {
    if let Err(error) = rag_provider::serve(io::stdin().lock(), io::stdout()).await {
        eprintln!("rag-provider: {error}");
        std::process::exit(1);
    }
}
