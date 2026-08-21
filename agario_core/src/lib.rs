mod batch;
pub mod config;
mod rng;
pub mod server;
pub mod world;

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
fn mechanics(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let cfg = config::WorldConfig::default();
    let values = PyDict::new(py);
    values.set_item("tick_rate", config::TICK_RATE)?;
    values.set_item("world_width", cfg.world_width)?;
    values.set_item("world_height", cfg.world_height)?;
    values.set_item("eat_size_ratio", cfg.eat_size_ratio)?;
    values.set_item("eat_mass_ratio", cfg.eat_size_ratio * cfg.eat_size_ratio)?;
    values.set_item("eat_overlap_fraction", 1.0 / cfg.eat_overlap_divisor)?;
    values.set_item("virus_mass", cfg.virus_mass)?;
    values.set_item("player_start_mass", cfg.player_start_mass)?;
    values.set_item("player_min_mass", cfg.player_min_mass)?;
    values.set_item("player_min_split_mass", cfg.player_min_split_mass)?;
    values.set_item("player_min_eject_mass", cfg.player_min_eject_mass)?;
    values.set_item("max_player_blobs", cfg.max_player_blobs)?;
    values.set_item("player_split_boost", cfg.player_split_boost)?;
    values.set_item("player_split_distance", cfg.player_split_distance)?;
    values.set_item("ejected_mass", cfg.ejected_mass)?;
    values.set_item("eject_loss_mass", cfg.eject_loss_mass)?;
    values.set_item("merge_base_seconds", cfg.player_merge_seconds)?;
    Ok(values.unbind())
}

#[pymodule]
fn agario_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<world::CoreWorld>()?;
    m.add_class::<world::PlayerHandle>()?;
    m.add_class::<batch::BatchedArenas>()?;
    m.add_function(wrap_pyfunction!(mechanics, m)?)?;
    Ok(())
}
