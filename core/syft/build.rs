use std::fs;

/// dynamically compiles all protos
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths = fs::read_dir("../protos").unwrap();

    for path in paths {
        tonic_build::compile_protos(path.unwrap().path())
            .unwrap_or_else(|e| panic!("Failed to compile proto {:?}", e));
    }

    Ok(())
}
