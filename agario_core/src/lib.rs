mod batch;
mod config;
mod rng;
mod world;

use pyo3::prelude::*;

/// Debug wrapper so parity tests can compare the Rust RNG against
/// Python's random.Random directly.
#[pyclass]
struct CoreRng {
    inner: rng::PyMt19937,
}

#[pymethods]
impl CoreRng {
    #[new]
    fn new(seed: u64) -> Self {
        CoreRng { inner: rng::PyMt19937::new(seed) }
    }

    fn random(&mut self) -> f64 {
        self.inner.random()
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.inner.uniform(lo, hi)
    }

    fn getrandbits(&mut self, k: u32) -> u32 {
        self.inner.getrandbits(k)
    }

    fn randbelow(&mut self, n: u32) -> u32 {
        self.inner.randbelow(n)
    }
}

#[pymodule]
fn agario_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<world::CoreWorld>()?;
    m.add_class::<world::PlayerHandle>()?;
    m.add_class::<batch::BatchedArenas>()?;
    m.add_class::<CoreRng>()?;
    Ok(())
}
