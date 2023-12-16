#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Params {
    pub threads: usize,
    pub buffer_size: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FileSizeRange {
    pub min_size: usize,
    pub max_size: usize,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ParamsAndFitness {
    pub params: Params,
    pub fitness: f64,
}