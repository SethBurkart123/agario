use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub const TICK_RATE: f64 = 25.0;

#[derive(Clone)]
pub struct WorldConfig {
    pub world_width: f64,
    pub world_height: f64,
    pub safe_spawn_tries: usize,
    pub spawn_from_ejected_chance: f64,
    pub eat_size_ratio: f64,
    pub eat_overlap_divisor: f64,

    pub food_target_count: usize,
    pub food_min_mass: f64,
    pub food_max_mass: f64,
    pub food_grow_seconds: f64,

    pub virus_min_count: usize,
    pub virus_max_count: usize,
    pub virus_mass: f64,
    pub virus_feed_times: usize,
    pub virus_split_boost: f64,

    pub player_start_mass: f64,
    pub player_min_mass: f64,
    pub player_max_mass: f64,
    pub player_min_split_mass: f64,
    pub player_min_eject_mass: f64,
    pub max_player_blobs: usize,
    pub player_move_mult: f64,
    pub player_decay_mult: f64,
    pub player_no_collide_seconds: f64,
    pub player_no_merge_seconds: f64,
    pub player_merge_seconds: f64,
    pub player_merge_size_factor: f64,
    pub player_split_distance: f64,
    pub player_split_boost: f64,

    pub ejected_mass: f64,
    pub eject_loss_mass: f64,
    pub eject_dispersion: f64,
    pub ejected_boost: f64,
    pub action_cooldown_seconds: f64,

    pub view_width: f64,
    pub view_height: f64,
    pub view_padding: f64,
    pub max_player_name_length: usize,
    pub player_colors: Vec<String>,
    pub food_colors: Vec<String>,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            // The modern classic map is 10,000 * sqrt(2) units per side.
            world_width: 14_142.135_623_730_952,
            world_height: 14_142.135_623_730_952,
            safe_spawn_tries: 64,
            spawn_from_ejected_chance: 0.8,
            eat_size_ratio: 1.140_175_425_099_138,
            eat_overlap_divisor: 3.0,

            food_target_count: 2500,
            food_min_mass: 1.0,
            food_max_mass: 4.0,
            food_grow_seconds: 60.0,

            virus_min_count: 30,
            virus_max_count: 90,
            virus_mass: 100.0,
            virus_feed_times: 7,
            virus_split_boost: 780.0,

            player_start_mass: 10.24,
            player_min_mass: 10.24,
            player_max_mass: 22_500.0,
            player_min_split_mass: 36.0,
            player_min_eject_mass: 36.0,
            max_player_blobs: 16,
            player_move_mult: 1.0,
            player_decay_mult: 0.001,
            player_no_collide_seconds: 13.0 / TICK_RATE,
            player_no_merge_seconds: 15.0 / TICK_RATE,
            player_merge_seconds: 30.0,
            player_merge_size_factor: 0.02,
            player_split_distance: 40.0,
            player_split_boost: 780.0,

            ejected_mass: 14.44,
            eject_loss_mass: 18.49,
            eject_dispersion: 0.3,
            ejected_boost: 780.0,
            action_cooldown_seconds: 3.0 / TICK_RATE,

            view_width: 1920.0,
            view_height: 1080.0,
            view_padding: 400.0,
            max_player_name_length: 18,
            player_colors: [
                "#21B8FF", "#33FF3A", "#FF364B", "#FFBC09", "#8E31FF", "#FF8A1F", "#26E5DF",
                "#FF2CCB",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            food_colors: [
                "#FF2A40", "#1D38FF", "#22D9F0", "#59F12F", "#7D2BFF", "#FFE625", "#FF8D1F",
                "#FF1FCF",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        }
    }
}

impl WorldConfig {
    pub fn apply_overrides(&mut self, overrides: &Bound<'_, PyDict>) -> PyResult<()> {
        macro_rules! number {
            ($field:ident) => {
                if let Some(value) = overrides.get_item(stringify!($field))? {
                    self.$field = value.extract()?;
                }
            };
        }
        number!(world_width);
        number!(world_height);
        number!(food_target_count);
        number!(virus_min_count);
        number!(virus_max_count);
        number!(player_start_mass);
        for (key, _) in overrides.iter() {
            let key: &str = key.extract()?;
            if !matches!(
                key,
                "world_width"
                    | "world_height"
                    | "food_target_count"
                    | "virus_min_count"
                    | "virus_max_count"
                    | "player_start_mass"
            ) {
                return Err(PyValueError::new_err(format!(
                    "unknown world override: {key}"
                )));
            }
        }
        Ok(())
    }
}
