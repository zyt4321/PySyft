use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{wrap_pyfunction, wrap_pymodule};
use syft::message::{Id, Message, Params};

#[pymodule]
fn pysyft(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(message))?;
    Ok(())
}

#[pymodule]
fn message(_py: Python, m: &PyModule) -> PyResult<()> {
    //m.add_wrapped(wrap_pyfunction!(run_class_method_message))?;
    Ok(())
}

// #[pyfunction]
// fn run_class_method_message(
//     path: String,
//     _self: Vec<u8>,
//     args: Vec<String>,
//     kwargs: PyDict,
//     id_at_location: String,
// ) -> PyDict {
//     ()
// }
