use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn pysyft(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(start))?;
    Ok(())
}

#[pyfunction]
fn start() -> String {
    return String::from("Started");
}
