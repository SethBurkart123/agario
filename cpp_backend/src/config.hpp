#pragma once

#include <string>
#include <vector>

namespace agario::config {

struct Config {
  double world_width = 6000.0;
  double world_height = 6000.0;

  int tick_rate = 75;

  int food_target_count = 1200;
  double food_min_mass = 1.0;
  double food_max_mass = 3.8;
  double food_radius_factor = 4.0;
  double food_eat_range_factor = 1.06;

  int virus_count = 24;
  double virus_mass = 144.0;
  double virus_radius_factor = 4.0;
  double virus_bonus_mass = 60.0;

  double player_start_mass = 160.0;
  double player_min_split_mass = 36.0;
  double player_min_eject_mass = 28.0;
  double player_eject_mass = 12.0;
  int max_player_blobs = 16;
  double min_blob_mass = 10.0;

  double blob_radius_factor = 4.0;
  double player_base_speed = 1400.0;
  double player_min_speed = 140.0;
  double speed_exponent = 0.45;
  double blob_boundary_factor = 0.84;
  double input_deadzone_world = 8.0;
  double input_speed_ramp_world = 82.0;
  double input_speed_ease_exponent = 0.7;

  double split_boost_speed = 880.0;
  double eject_boost_speed = 780.0;
  double boost_damping = 3.2;
  double softbody_min_dist_unmerged = 0.72;
  double softbody_min_dist_merged = 0.08;

  double ejected_mass_lifetime = 12.0;
  double ejected_eat_range_factor = 1.04;
  double merge_delay_seconds = 14.0;
  double split_cooldown_seconds = 0.2;
  double eject_cooldown_seconds = 0.12;

  double blob_eat_ratio = 1.12;
  double blob_eat_overlap = 0.78;

  int virus_split_min_parts = 4;
  int virus_split_max_parts = 8;

  double view_width = 1900.0;
  double view_height = 1100.0;
  double view_padding = 400.0;

  int input_hz = 90;
  int max_player_name_length = 18;

  bool bots_enabled = true;
  std::vector<std::string> bot_plugin_modules;
  std::string bot_specs;
  int bot_random_seed = 1337;
  bool bot_spawn_on_eaten = true;
  int bot_spawn_per_elimination = 1;
  int bot_max_active = 40;
  int bot_threads = 0;

  std::vector<std::string> player_colors;
  std::vector<std::string> food_colors;
};

Config load_from_env();
const Config& get();

}  // namespace agario::config
