//! World tunables mirroring agario/config.py. Field defaults must stay in
//! sync with the Python module; parity tests will catch drift.

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Clone)]
pub struct WorldConfig {
    pub world_width: f64,
    pub world_height: f64,

    pub food_target_count: usize,
    pub food_min_mass: f64,
    pub food_max_mass: f64,
    pub food_radius_factor: f64,
    pub food_eat_range_factor: f64,

    pub virus_count: usize,
    pub virus_mass: f64,
    pub virus_radius_factor: f64,
    pub virus_bonus_mass: f64,

    pub player_start_mass: f64,
    pub player_min_split_mass: f64,
    pub player_min_eject_mass: f64,
    pub player_eject_mass: f64,
    pub max_player_blobs: usize,
    pub min_blob_mass: f64,

    pub mass_decay_start: f64,
    pub mass_decay_end: f64,
    pub mass_decay_min_rate: f64,
    pub mass_decay_max_rate: f64,
    pub mass_decay_curve: f64,

    pub blob_radius_factor: f64,
    pub player_base_speed: f64,
    pub player_min_speed: f64,
    pub speed_exponent: f64,
    pub blob_boundary_factor: f64,
    pub input_deadzone_world: f64,
    pub input_speed_ramp_world: f64,
    pub input_speed_ease_exponent: f64,

    pub split_boost_speed: f64,
    pub eject_boost_speed: f64,
    pub boost_damping: f64,
    pub softbody_min_dist_merged: f64,

    pub ejected_mass_lifetime: f64,
    pub ejected_eat_range_factor: f64,
    pub merge_delay_seconds: f64,
    pub merge_coverage_fraction: f64,
    pub split_cooldown_seconds: f64,
    pub eject_cooldown_seconds: f64,

    pub blob_eat_ratio: f64,
    pub blob_eat_overlap: f64,

    pub virus_split_min_parts: usize,
    pub virus_split_max_parts: usize,

    pub view_width: f64,
    pub view_height: f64,
    pub view_padding: f64,
    pub split_zoom_max_penalty: f64,
    pub split_zoom_max_penalty_huge: f64,
    pub split_zoom_mass_soft_cap: f64,
    pub split_zoom_mass_hard_cap: f64,
    pub split_zoom_mass_curve: f64,
    pub split_zoom_decay: f64,

    pub max_player_name_length: usize,

    pub player_colors: Vec<String>,
    pub food_colors: Vec<String>,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig {
            world_width: 6000.0,
            world_height: 6000.0,

            food_target_count: 1200,
            food_min_mass: 1.0,
            food_max_mass: 3.8,
            food_radius_factor: 4.0,
            food_eat_range_factor: 1.06,

            virus_count: 24,
            virus_mass: 144.0,
            virus_radius_factor: 4.0,
            virus_bonus_mass: 60.0,

            player_start_mass: 560.0,
            player_min_split_mass: 90.0,
            player_min_eject_mass: 28.0,
            player_eject_mass: 12.0,
            max_player_blobs: 16,
            min_blob_mass: 10.0,

            mass_decay_start: 200.0,
            mass_decay_end: 1800.0,
            mass_decay_min_rate: 0.00005,
            mass_decay_max_rate: 0.006,
            mass_decay_curve: 2.6,

            blob_radius_factor: 4.0,
            player_base_speed: 1400.0,
            player_min_speed: 140.0,
            speed_exponent: 0.45,
            blob_boundary_factor: 0.84,
            input_deadzone_world: 8.0,
            input_speed_ramp_world: 82.0,
            input_speed_ease_exponent: 0.7,

            split_boost_speed: 880.0,
            eject_boost_speed: 780.0,
            boost_damping: 3.2,
            softbody_min_dist_merged: 0.08,

            ejected_mass_lifetime: 12.0,
            ejected_eat_range_factor: 1.04,
            merge_delay_seconds: 25.0,
            merge_coverage_fraction: 0.5,
            split_cooldown_seconds: 0.12,
            eject_cooldown_seconds: 0.12,

            blob_eat_ratio: 1.12,
            blob_eat_overlap: 0.78,

            virus_split_min_parts: 4,
            virus_split_max_parts: 8,

            view_width: 1900.0,
            view_height: 1100.0,
            view_padding: 400.0,
            split_zoom_max_penalty: 0.14,
            split_zoom_max_penalty_huge: 0.26,
            split_zoom_mass_soft_cap: 600.0,
            split_zoom_mass_hard_cap: 2600.0,
            split_zoom_mass_curve: 1.4,
            split_zoom_decay: 0.25,

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
    /// Apply overrides from a Python dict whose keys match config.py names
    /// (e.g. {"WORLD_WIDTH": 2000.0, "FOOD_TARGET_COUNT": 140}).
    pub fn apply_overrides(&mut self, overrides: &Bound<'_, PyDict>) -> PyResult<()> {
        macro_rules! f64_field {
            ($key:literal, $field:ident) => {
                if let Some(v) = overrides.get_item($key)? {
                    self.$field = v.extract::<f64>()?;
                }
            };
        }
        macro_rules! usize_field {
            ($key:literal, $field:ident) => {
                if let Some(v) = overrides.get_item($key)? {
                    self.$field = v.extract::<usize>()?;
                }
            };
        }

        f64_field!("WORLD_WIDTH", world_width);
        f64_field!("WORLD_HEIGHT", world_height);
        usize_field!("FOOD_TARGET_COUNT", food_target_count);
        f64_field!("FOOD_MIN_MASS", food_min_mass);
        f64_field!("FOOD_MAX_MASS", food_max_mass);
        f64_field!("FOOD_RADIUS_FACTOR", food_radius_factor);
        f64_field!("FOOD_EAT_RANGE_FACTOR", food_eat_range_factor);
        usize_field!("VIRUS_COUNT", virus_count);
        f64_field!("VIRUS_MASS", virus_mass);
        f64_field!("VIRUS_RADIUS_FACTOR", virus_radius_factor);
        f64_field!("VIRUS_BONUS_MASS", virus_bonus_mass);
        f64_field!("PLAYER_START_MASS", player_start_mass);
        f64_field!("PLAYER_MIN_SPLIT_MASS", player_min_split_mass);
        f64_field!("PLAYER_MIN_EJECT_MASS", player_min_eject_mass);
        f64_field!("PLAYER_EJECT_MASS", player_eject_mass);
        usize_field!("MAX_PLAYER_BLOBS", max_player_blobs);
        f64_field!("MIN_BLOB_MASS", min_blob_mass);
        f64_field!("MASS_DECAY_START", mass_decay_start);
        f64_field!("MASS_DECAY_END", mass_decay_end);
        f64_field!("MASS_DECAY_MIN_RATE", mass_decay_min_rate);
        f64_field!("MASS_DECAY_MAX_RATE", mass_decay_max_rate);
        f64_field!("MASS_DECAY_CURVE", mass_decay_curve);
        f64_field!("BLOB_RADIUS_FACTOR", blob_radius_factor);
        f64_field!("PLAYER_BASE_SPEED", player_base_speed);
        f64_field!("PLAYER_MIN_SPEED", player_min_speed);
        f64_field!("SPEED_EXPONENT", speed_exponent);
        f64_field!("BLOB_BOUNDARY_FACTOR", blob_boundary_factor);
        f64_field!("INPUT_DEADZONE_WORLD", input_deadzone_world);
        f64_field!("INPUT_SPEED_RAMP_WORLD", input_speed_ramp_world);
        f64_field!("INPUT_SPEED_EASE_EXPONENT", input_speed_ease_exponent);
        f64_field!("SPLIT_BOOST_SPEED", split_boost_speed);
        f64_field!("EJECT_BOOST_SPEED", eject_boost_speed);
        f64_field!("BOOST_DAMPING", boost_damping);
        f64_field!("SOFTBODY_MIN_DIST_MERGED", softbody_min_dist_merged);
        f64_field!("EJECTED_MASS_LIFETIME", ejected_mass_lifetime);
        f64_field!("EJECTED_EAT_RANGE_FACTOR", ejected_eat_range_factor);
        f64_field!("MERGE_DELAY_SECONDS", merge_delay_seconds);
        f64_field!("MERGE_COVERAGE_FRACTION", merge_coverage_fraction);
        f64_field!("SPLIT_COOLDOWN_SECONDS", split_cooldown_seconds);
        f64_field!("EJECT_COOLDOWN_SECONDS", eject_cooldown_seconds);
        f64_field!("BLOB_EAT_RATIO", blob_eat_ratio);
        f64_field!("BLOB_EAT_OVERLAP", blob_eat_overlap);
        usize_field!("VIRUS_SPLIT_MIN_PARTS", virus_split_min_parts);
        usize_field!("VIRUS_SPLIT_MAX_PARTS", virus_split_max_parts);
        f64_field!("VIEW_WIDTH", view_width);
        f64_field!("VIEW_HEIGHT", view_height);
        f64_field!("VIEW_PADDING", view_padding);
        f64_field!("SPLIT_ZOOM_MAX_PENALTY", split_zoom_max_penalty);
        f64_field!("SPLIT_ZOOM_MAX_PENALTY_HUGE", split_zoom_max_penalty_huge);
        f64_field!("SPLIT_ZOOM_MASS_SOFT_CAP", split_zoom_mass_soft_cap);
        f64_field!("SPLIT_ZOOM_MASS_HARD_CAP", split_zoom_mass_hard_cap);
        f64_field!("SPLIT_ZOOM_MASS_CURVE", split_zoom_mass_curve);
        f64_field!("SPLIT_ZOOM_DECAY", split_zoom_decay);
        usize_field!("MAX_PLAYER_NAME_LENGTH", max_player_name_length);
        Ok(())
    }
}
