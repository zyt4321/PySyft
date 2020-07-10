use std::collections::HashMap;

#[allow(dead_code)]
pub struct Message {
    id: Option<Id>,
    path: Option<String>,
    params: Option<Params>,
    object: Option<Vec<u8>>,
}

#[allow(dead_code)]
pub enum Id {
    Local(String),
    Remote(String),
}

#[allow(dead_code)]
pub struct Params {
    args: Vec<String>,
    kwargs: HashMap<String, String>,
}
