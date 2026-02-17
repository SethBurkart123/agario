#include "world.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace agario {
namespace {

double clamp(double value, double min_value, double max_value) {
  return std::min(std::max(value, min_value), max_value);
}

double distance_sq(double ax, double ay, double bx, double by) {
  double dx = ax - bx;
  double dy = ay - by;
  return dx * dx + dy * dy;
}

std::pair<double, double> unit_vec(double dx, double dy) {
  double mag_sq = dx * dx + dy * dy;
  if (mag_sq <= 1e-9) {
    return {1.0, 0.0};
  }
  double inv = 1.0 / std::sqrt(mag_sq);
  return {dx * inv, dy * inv};
}

double round_to(double value, int decimals) {
  double factor = std::pow(10.0, static_cast<double>(decimals));
  return std::round(value * factor) / factor;
}

double rand_uniform(std::mt19937& rng, double min_value, double max_value) {
  std::uniform_real_distribution<double> dist(min_value, max_value);
  return dist(rng);
}

int rand_int(std::mt19937& rng, int min_value, int max_value) {
  std::uniform_int_distribution<int> dist(min_value, max_value);
  return dist(rng);
}

std::string trim_copy(const std::string& input) {
  std::size_t start = input.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    return "";
  }
  std::size_t end = input.find_last_not_of(" \t\n\r");
  return input.substr(start, end - start + 1);
}

}  // namespace

GameWorld::GameWorld(std::optional<int> seed)
    : rng_(seed ? static_cast<std::mt19937::result_type>(*seed) : std::random_device{}()),
      food_hash_(150.0),
      blob_hash_(250.0),
      ejected_hash_(180.0) {
  spawn_initial_food();
  spawn_initial_viruses();
  rebuild_spatial_indexes();
}

GameWorld GameWorld::fast_clone() const {
  GameWorld clone(std::optional<int>{});
  clone.rng_ = rng_;
  clone.players = players;
  clone.foods = foods;
  clone.ejected = ejected;
  clone.viruses = viruses;
  clone.player_id_counter_ = player_id_counter_;
  clone.blob_id_counter_ = blob_id_counter_;
  clone.food_id_counter_ = food_id_counter_;
  clone.ejected_id_counter_ = ejected_id_counter_;
  clone.virus_id_counter_ = virus_id_counter_;
  clone.rebuild_spatial_indexes();
  return clone;
}

std::string GameWorld::next_player_id() {
  return "p" + std::to_string(player_id_counter_++);
}

std::string GameWorld::next_blob_id() {
  return "b" + std::to_string(blob_id_counter_++);
}

std::string GameWorld::next_food_id() {
  return "f" + std::to_string(food_id_counter_++);
}

std::string GameWorld::next_ejected_id() {
  return "e" + std::to_string(ejected_id_counter_++);
}

std::string GameWorld::next_virus_id() {
  return "v" + std::to_string(virus_id_counter_++);
}

Player& GameWorld::add_player(
    const std::string& player_name,
    double now,
    bool is_bot,
    const std::string& bot_plugin,
    const std::string& bot_team,
    const std::string& color) {
  const auto& cfg = config::get();
  std::string name = trim_copy(player_name.empty() ? "Cell" : player_name);
  if (static_cast<int>(name.size()) > cfg.max_player_name_length) {
    name.resize(static_cast<std::size_t>(cfg.max_player_name_length));
  }
  if (name.empty()) {
    name = "Cell";
  }

  std::string player_id = next_player_id();
  std::string player_color = color;
  if (player_color.empty()) {
    if (!cfg.player_colors.empty()) {
      std::size_t idx = players.size() % cfg.player_colors.size();
      player_color = cfg.player_colors[idx];
    } else {
      player_color = "#21B8FF";
    }
  }

  Player player;
  player.id = player_id;
  player.name = name;
  player.color = player_color;
  player.is_bot = is_bot;
  player.bot_plugin = bot_plugin;
  player.bot_team = bot_team;

  auto [spawn_x, spawn_y] = random_spawn(40.0);
  Blob blob;
  blob.id = next_blob_id();
  blob.player_id = player.id;
  blob.x = spawn_x;
  blob.y = spawn_y;
  blob.mass = cfg.player_start_mass;
  blob.can_merge_at = now + cfg.merge_delay_seconds;
  player.blobs.emplace(blob.id, blob);
  player.target_x = spawn_x;
  player.target_y = spawn_y;

  players.emplace(player.id, std::move(player));
  return players[player_id];
}

void GameWorld::remove_player(const std::string& player_id) {
  players.erase(player_id);
}

void GameWorld::set_input(
    const std::string& player_id,
    std::optional<double> target_x,
    std::optional<double> target_y,
    bool split,
    bool eject) {
  auto it = players.find(player_id);
  if (it == players.end()) {
    return;
  }
  Player& player = it->second;
  if (target_x && target_y) {
    player.target_x = *target_x;
    player.target_y = *target_y;
  }
  if (split) {
    player.split_requested = true;
  }
  if (eject) {
    player.eject_requested = true;
  }
}

void GameWorld::update(double dt, double now) {
  respawn_eliminated_players(now);
  apply_actions(now);
  move_blobs(dt, now);
  move_ejected(dt);

  rebuild_spatial_indexes();
  resolve_blob_food_collisions();
  resolve_blob_ejected_collisions();
  resolve_blob_blob_collisions(now);
  resolve_virus_blob_collisions(now);

  spawn_food_to_target();
  rebuild_spatial_indexes();
}

std::pair<double, double> GameWorld::random_spawn(double radius) {
  const auto& cfg = config::get();
  double x = rand_uniform(rng_, radius, cfg.world_width - radius);
  double y = rand_uniform(rng_, radius, cfg.world_height - radius);
  return {x, y};
}

void GameWorld::spawn_initial_food() {
  const auto& cfg = config::get();
  while (static_cast<int>(foods.size()) < cfg.food_target_count) {
    spawn_food();
  }
}

void GameWorld::spawn_food() {
  const auto& cfg = config::get();
  auto [fx, fy] = random_spawn(8.0);
  double mass = rand_uniform(rng_, cfg.food_min_mass, cfg.food_max_mass);
  Food food;
  food.id = next_food_id();
  food.x = fx;
  food.y = fy;
  food.mass = mass;
  if (!cfg.food_colors.empty()) {
    int idx = rand_int(rng_, 0, static_cast<int>(cfg.food_colors.size() - 1));
    food.color = cfg.food_colors[idx];
  } else {
    food.color = "#FF2A40";
  }
  foods.emplace(food.id, std::move(food));
}

void GameWorld::spawn_initial_viruses() {
  const auto& cfg = config::get();
  for (int i = 0; i < cfg.virus_count; ++i) {
    spawn_virus();
  }
}

void GameWorld::spawn_virus() {
  const auto& cfg = config::get();
  auto [vx, vy] = random_spawn(80.0);
  Virus virus;
  virus.id = next_virus_id();
  virus.x = vx;
  virus.y = vy;
  virus.mass = cfg.virus_mass;
  viruses.emplace(virus.id, std::move(virus));
}

void GameWorld::respawn_eliminated_players(double now) {
  const auto& cfg = config::get();
  for (auto& kv : players) {
    Player& player = kv.second;
    if (!player.blobs.empty()) {
      continue;
    }
    auto [x, y] = random_spawn(40.0);
    Blob blob;
    blob.id = next_blob_id();
    blob.player_id = player.id;
    blob.x = x;
    blob.y = y;
    blob.mass = cfg.player_start_mass;
    blob.can_merge_at = now + cfg.merge_delay_seconds;
    player.blobs.emplace(blob.id, blob);
    player.target_x = x;
    player.target_y = y;
  }
}

void GameWorld::apply_actions(double now) {
  for (auto& kv : players) {
    Player& player = kv.second;
    if (player.split_requested) {
      split_player(player, now);
      player.split_requested = false;
    }
    if (player.eject_requested) {
      eject_player_mass(player, now);
      player.eject_requested = false;
    }
  }
}

void GameWorld::split_player(Player& player, double now) {
  const auto& cfg = config::get();
  if (now - player.last_split_at < cfg.split_cooldown_seconds) {
    return;
  }
  if (static_cast<int>(player.blobs.size()) >= cfg.max_player_blobs) {
    return;
  }

  std::vector<Blob> created;
  std::vector<Blob*> snapshot;
  snapshot.reserve(player.blobs.size());
  for (auto& kv : player.blobs) {
    snapshot.push_back(&kv.second);
  }

  for (Blob* blob : snapshot) {
    if (blob->mass < cfg.player_min_split_mass) {
      continue;
    }
    if (static_cast<int>(player.blobs.size() + created.size()) >= cfg.max_player_blobs) {
      break;
    }

    double dx = player.target_x - blob->x;
    double dy = player.target_y - blob->y;
    auto [ux, uy] = unit_vec(dx, dy);

    double split_mass = blob->mass / 2.0;
    blob->mass = split_mass;
    blob->can_merge_at = now + cfg.merge_delay_seconds;

    double offset = blob->radius() + std::sqrt(split_mass) * cfg.blob_radius_factor;
    Blob new_blob;
    new_blob.id = next_blob_id();
    new_blob.player_id = player.id;
    new_blob.x = clamp(blob->x + ux * offset, 0.0, cfg.world_width);
    new_blob.y = clamp(blob->y + uy * offset, 0.0, cfg.world_height);
    new_blob.mass = split_mass;
    new_blob.vx = ux * cfg.split_boost_speed;
    new_blob.vy = uy * cfg.split_boost_speed;
    new_blob.can_merge_at = now + cfg.merge_delay_seconds;
    created.push_back(new_blob);
  }

  for (auto& blob : created) {
    player.blobs.emplace(blob.id, std::move(blob));
  }

  if (!created.empty()) {
    player.last_split_at = now;
  }
}

void GameWorld::eject_player_mass(Player& player, double now) {
  const auto& cfg = config::get();
  if (now - player.last_eject_at < cfg.eject_cooldown_seconds) {
    return;
  }

  bool spawned_any = false;
  for (auto& kv : player.blobs) {
    Blob& blob = kv.second;
    if (blob.mass <= cfg.player_min_eject_mass) {
      continue;
    }

    double remaining_mass = blob.mass - cfg.player_eject_mass;
    if (remaining_mass < cfg.min_blob_mass) {
      continue;
    }

    double dx = player.target_x - blob.x;
    double dy = player.target_y - blob.y;
    auto [ux, uy] = unit_vec(dx, dy);

    blob.mass = remaining_mass;
    double eject_x = blob.x + ux * (blob.radius() + 12.0);
    double eject_y = blob.y + uy * (blob.radius() + 12.0);

    EjectedMass mass;
    mass.id = next_ejected_id();
    mass.x = clamp(eject_x, 0.0, cfg.world_width);
    mass.y = clamp(eject_y, 0.0, cfg.world_height);
    mass.mass = cfg.player_eject_mass;
    mass.owner_id = player.id;
    mass.vx = ux * cfg.eject_boost_speed;
    mass.vy = uy * cfg.eject_boost_speed;
    mass.ttl = cfg.ejected_mass_lifetime;
    ejected.emplace(mass.id, std::move(mass));
    spawned_any = true;
  }

  if (spawned_any) {
    player.last_eject_at = now;
  }
}

void GameWorld::move_blobs(double dt, double now) {
  const auto& cfg = config::get();
  double damping = std::max(0.0, 1.0 - cfg.boost_damping * dt);

  for (auto& kv : players) {
    Player& player = kv.second;
    for (auto& bkv : player.blobs) {
      Blob& blob = bkv.second;
      double dx = player.target_x - blob.x;
      double dy = player.target_y - blob.y;
      auto [ux, uy] = unit_vec(dx, dy);
      double input_distance = std::sqrt(dx * dx + dy * dy);
      double input_excess = std::max(0.0, input_distance - cfg.input_deadzone_world);
      double ramp = std::max(1.0, cfg.input_speed_ramp_world);
      double ease_power = std::max(0.25, cfg.input_speed_ease_exponent);
      double eased_distance = std::pow(input_excess / ramp, ease_power);
      double input_scale = 1.0 - std::exp(-eased_distance);

      double max_speed = std::max(cfg.player_min_speed, cfg.player_base_speed / std::pow(blob.mass, cfg.speed_exponent));
      double speed = max_speed * input_scale;

      blob.x += (ux * speed + blob.vx) * dt;
      blob.y += (uy * speed + blob.vy) * dt;
      blob.vx *= damping;
      blob.vy *= damping;
    }

    apply_same_player_softbody(player, now);

    for (auto& bkv : player.blobs) {
      Blob& blob = bkv.second;
      double radius = blob.radius();
      double clamp_r = radius * cfg.blob_boundary_factor;
      blob.x = clamp(blob.x, clamp_r, cfg.world_width - clamp_r);
      blob.y = clamp(blob.y, clamp_r, cfg.world_height - clamp_r);
    }
  }
}

void GameWorld::apply_same_player_softbody(Player& player, double now) {
  const auto& cfg = config::get();
  if (player.blobs.size() <= 1) {
    return;
  }

  std::vector<Blob*> blobs;
  blobs.reserve(player.blobs.size());
  for (auto& kv : player.blobs) {
    blobs.push_back(&kv.second);
  }

  for (std::size_t i = 0; i < blobs.size(); ++i) {
    Blob* a = blobs[i];
    for (std::size_t j = i + 1; j < blobs.size(); ++j) {
      Blob* b = blobs[j];
      double dx = b->x - a->x;
      double dy = b->y - a->y;
      double dist_sq = dx * dx + dy * dy;
      if (dist_sq <= 1e-8) {
        double theta = rand_uniform(rng_, 0.0, 6.283185);
        dx = std::cos(theta);
        dy = std::sin(theta);
        dist_sq = 1.0;
      }
      double dist = std::sqrt(dist_sq);
      double ux = dx / dist;
      double uy = dy / dist;

      double touch = a->radius() + b->radius();
      bool ready_to_merge = now >= a->can_merge_at && now >= b->can_merge_at;
      double min_dist = touch * (ready_to_merge ? cfg.softbody_min_dist_merged : cfg.softbody_min_dist_unmerged);
      if (dist < min_dist) {
        double correction = (min_dist - dist) * 0.5;
        a->x -= ux * correction;
        a->y -= uy * correction;
        b->x += ux * correction;
        b->y += uy * correction;
      }
    }
  }
}

void GameWorld::move_ejected(double dt) {
  const auto& cfg = config::get();
  std::vector<std::string> to_remove;
  double damping = std::max(0.0, 1.0 - cfg.boost_damping * dt);

  for (auto& kv : ejected) {
    EjectedMass& mass = kv.second;
    mass.x += mass.vx * dt;
    mass.y += mass.vy * dt;
    mass.vx *= damping;
    mass.vy *= damping;
    mass.ttl -= dt;

    double radius = mass.radius();
    mass.x = clamp(mass.x, radius, cfg.world_width - radius);
    mass.y = clamp(mass.y, radius, cfg.world_height - radius);

    if (mass.ttl <= 0.0) {
      to_remove.push_back(mass.id);
    }
  }

  for (const auto& id : to_remove) {
    ejected.erase(id);
  }
}

void GameWorld::rebuild_spatial_indexes() {
  food_hash_.clear();
  blob_hash_.clear();
  ejected_hash_.clear();

  for (auto& kv : foods) {
    food_hash_.insert(kv.second.x, kv.second.y, &kv.second);
  }

  for (auto& pkv : players) {
    for (auto& bkv : pkv.second.blobs) {
      blob_hash_.insert(bkv.second.x, bkv.second.y, &bkv.second);
    }
  }

  for (auto& kv : ejected) {
    ejected_hash_.insert(kv.second.x, kv.second.y, &kv.second);
  }
}

void GameWorld::resolve_blob_food_collisions() {
  const auto& cfg = config::get();
  std::unordered_set<std::string> eaten_ids;

  for (auto& pkv : players) {
    for (auto& bkv : pkv.second.blobs) {
      Blob& blob = bkv.second;
      auto nearby = food_hash_.query_rect(
          blob.x - blob.radius(),
          blob.y - blob.radius(),
          blob.x + blob.radius(),
          blob.y + blob.radius());
      for (Food* food : nearby) {
        if (!food) {
          continue;
        }
        if (eaten_ids.find(food->id) != eaten_ids.end()) {
          continue;
        }
        double eat_dist = (blob.radius() + food->radius()) * cfg.food_eat_range_factor;
        if (distance_sq(blob.x, blob.y, food->x, food->y) <= eat_dist * eat_dist) {
          blob.mass += food->mass;
          eaten_ids.insert(food->id);
        }
      }
    }
  }

  for (const auto& id : eaten_ids) {
    foods.erase(id);
  }
}

void GameWorld::resolve_blob_ejected_collisions() {
  const auto& cfg = config::get();
  std::unordered_set<std::string> consumed;

  for (auto& pkv : players) {
    for (auto& bkv : pkv.second.blobs) {
      Blob& blob = bkv.second;
      auto nearby = ejected_hash_.query_rect(
          blob.x - blob.radius(),
          blob.y - blob.radius(),
          blob.x + blob.radius(),
          blob.y + blob.radius());
      for (EjectedMass* mass : nearby) {
        if (!mass) {
          continue;
        }
        if (consumed.find(mass->id) != consumed.end()) {
          continue;
        }
        if (mass->owner_id == blob.player_id && mass->ttl > cfg.ejected_mass_lifetime - 0.35) {
          continue;
        }
        double eat_dist = (blob.radius() + mass->radius()) * cfg.ejected_eat_range_factor;
        if (distance_sq(blob.x, blob.y, mass->x, mass->y) <= eat_dist * eat_dist) {
          blob.mass += mass->mass;
          consumed.insert(mass->id);
        }
      }
    }
  }

  for (const auto& id : consumed) {
    ejected.erase(id);
  }
}

void GameWorld::resolve_blob_blob_collisions(double now) {
  const auto& cfg = config::get();
  std::unordered_set<std::string> eaten_blob_ids;
  std::unordered_set<std::string> checked_pairs;

  std::vector<Blob*> all_blobs;
  for (auto& pkv : players) {
    for (auto& bkv : pkv.second.blobs) {
      all_blobs.push_back(&bkv.second);
    }
  }

  for (Blob* blob : all_blobs) {
    if (!blob) {
      continue;
    }
    if (eaten_blob_ids.find(blob->id) != eaten_blob_ids.end()) {
      continue;
    }

    auto nearby = blob_hash_.query_rect(
        blob->x - blob->radius() * 2.0,
        blob->y - blob->radius() * 2.0,
        blob->x + blob->radius() * 2.0,
        blob->y + blob->radius() * 2.0);

    for (Blob* other : nearby) {
      if (!other || other->id == blob->id) {
        continue;
      }
      if (eaten_blob_ids.find(other->id) != eaten_blob_ids.end()) {
        continue;
      }
      std::string a = blob->id < other->id ? blob->id : other->id;
      std::string b = blob->id < other->id ? other->id : blob->id;
      std::string pair_key = a + ":" + b;
      if (checked_pairs.find(pair_key) != checked_pairs.end()) {
        continue;
      }
      checked_pairs.insert(pair_key);

      auto owner_a = players.find(blob->player_id);
      auto owner_b = players.find(other->player_id);
      if (owner_a == players.end() || owner_b == players.end()) {
        continue;
      }
      if (owner_a->second.blobs.find(blob->id) == owner_a->second.blobs.end() ||
          owner_b->second.blobs.find(other->id) == owner_b->second.blobs.end()) {
        continue;
      }

      Blob* bigger = blob;
      Blob* smaller = other;
      if (blob->mass < other->mass) {
        bigger = other;
        smaller = blob;
      }

      double dist_sq = distance_sq(bigger->x, bigger->y, smaller->x, smaller->y);

      if (bigger->player_id == smaller->player_id) {
        if (now < bigger->can_merge_at || now < smaller->can_merge_at) {
          continue;
        }
        double max_radius = std::max(bigger->radius(), smaller->radius());
        if (dist_sq <= (max_radius * 0.35) * (max_radius * 0.35)) {
          bigger->mass += smaller->mass;
          eaten_blob_ids.insert(smaller->id);
        }
        continue;
      }

      if (bigger->mass < smaller->mass * cfg.blob_eat_ratio) {
        continue;
      }

      double eat_distance = bigger->radius() - (smaller->radius() * cfg.blob_eat_overlap);
      if (eat_distance <= 0.0) {
        continue;
      }
      if (dist_sq <= eat_distance * eat_distance) {
        bigger->mass += smaller->mass;
        eaten_blob_ids.insert(smaller->id);
      }
    }
  }

  for (auto& pkv : players) {
    Player& player = pkv.second;
    for (auto it = player.blobs.begin(); it != player.blobs.end();) {
      if (eaten_blob_ids.find(it->first) != eaten_blob_ids.end()) {
        it = player.blobs.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void GameWorld::resolve_virus_blob_collisions(double now) {
  const auto& cfg = config::get();
  std::unordered_set<std::string> consumed_viruses;

  for (auto& pkv : players) {
    Player& player = pkv.second;
    std::vector<std::string> blob_ids;
    blob_ids.reserve(player.blobs.size());
    for (const auto& bkv : player.blobs) {
      blob_ids.push_back(bkv.first);
    }

    for (const auto& blob_id : blob_ids) {
      auto it = player.blobs.find(blob_id);
      if (it == player.blobs.end()) {
        continue;
      }
      Blob& blob = it->second;
      for (auto& vkv : viruses) {
        Virus& virus = vkv.second;
        if (consumed_viruses.find(virus.id) != consumed_viruses.end()) {
          continue;
        }
        if (blob.mass < virus.mass * 1.15) {
          continue;
        }
        double trigger_distance = blob.radius() - (virus.radius() * 0.18);
        if (trigger_distance <= 0.0) {
          continue;
        }
        if (distance_sq(blob.x, blob.y, virus.x, virus.y) <= trigger_distance * trigger_distance) {
          blob.mass += cfg.virus_bonus_mass;
          explode_blob_into_player(blob, player, now);
          consumed_viruses.insert(virus.id);
          break;
        }
      }
    }
  }

  for (const auto& id : consumed_viruses) {
    viruses.erase(id);
  }

  for (std::size_t i = 0; i < consumed_viruses.size(); ++i) {
    spawn_virus();
  }
}

void GameWorld::explode_blob_into_player(Blob& blob, Player& player, double now) {
  const auto& cfg = config::get();
  if (player.blobs.find(blob.id) == player.blobs.end()) {
    return;
  }

  int available_slots = cfg.max_player_blobs - static_cast<int>(player.blobs.size()) + 1;
  if (available_slots <= 1) {
    return;
  }

  int split_parts = static_cast<int>(blob.mass / 30.0);
  split_parts = std::max(cfg.virus_split_min_parts, split_parts);
  split_parts = std::min({cfg.virus_split_max_parts, split_parts, available_slots});
  if (split_parts <= 1) {
    return;
  }

  double total_mass = blob.mass;
  double base_x = blob.x;
  double base_y = blob.y;

  player.blobs.erase(blob.id);

  double part_mass = total_mass / static_cast<double>(split_parts);
  for (int idx = 0; idx < split_parts; ++idx) {
    double angle = (static_cast<double>(idx) / split_parts) * 6.283185 + rand_uniform(rng_, -0.15, 0.15);
    double ux = std::cos(angle);
    double uy = std::sin(angle);
    Blob spawned;
    spawned.id = next_blob_id();
    spawned.player_id = player.id;
    spawned.x = clamp(base_x + ux * 14.0, 0.0, cfg.world_width);
    spawned.y = clamp(base_y + uy * 14.0, 0.0, cfg.world_height);
    spawned.mass = part_mass;
    spawned.vx = ux * cfg.split_boost_speed * rand_uniform(rng_, 0.62, 0.92);
    spawned.vy = uy * cfg.split_boost_speed * rand_uniform(rng_, 0.62, 0.92);
    spawned.can_merge_at = now + cfg.merge_delay_seconds;
    player.blobs.emplace(spawned.id, std::move(spawned));
  }
}

void GameWorld::spawn_food_to_target() {
  const auto& cfg = config::get();
  while (static_cast<int>(foods.size()) < cfg.food_target_count) {
    spawn_food();
  }
}

std::tuple<std::vector<Blob*>, std::vector<Food*>, std::vector<EjectedMass*>, std::vector<Virus*>>
GameWorld::visible_entities(double cx, double cy, double view_w, double view_h) const {
  const auto& cfg = config::get();
  double min_x = clamp(cx - view_w / 2.0, 0.0, cfg.world_width);
  double max_x = clamp(cx + view_w / 2.0, 0.0, cfg.world_width);
  double min_y = clamp(cy - view_h / 2.0, 0.0, cfg.world_height);
  double max_y = clamp(cy + view_h / 2.0, 0.0, cfg.world_height);

  auto blobs = blob_hash_.query_rect(min_x, min_y, max_x, max_y);
  auto foods_view = food_hash_.query_rect(min_x, min_y, max_x, max_y);
  auto ejected_view = ejected_hash_.query_rect(min_x, min_y, max_x, max_y);
  std::vector<Virus*> viruses_view;
  viruses_view.reserve(viruses.size());
  for (auto& kv : viruses) {
    Virus& virus = const_cast<Virus&>(kv.second);
    if (min_x <= virus.x && virus.x <= max_x && min_y <= virus.y && virus.y <= max_y) {
      viruses_view.push_back(&virus);
    }
  }

  return {blobs, foods_view, ejected_view, viruses_view};
}

boost::json::array GameWorld::leaderboard() const {
  struct Entry {
    std::string name;
    double score = 0.0;
  };

  std::vector<Entry> entries;
  entries.reserve(players.size());
  for (const auto& kv : players) {
    const Player& player = kv.second;
    double score = player.total_mass();
    if (score <= 0.0) {
      continue;
    }
    entries.push_back({player.name, std::round(score)});
  }

  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
    return a.score > b.score;
  });

  boost::json::array out;
  std::size_t limit = std::min<std::size_t>(10, entries.size());
  for (std::size_t i = 0; i < limit; ++i) {
    boost::json::object row;
    row["name"] = entries[i].name;
    row["score"] = entries[i].score;
    out.push_back(std::move(row));
  }
  return out;
}

boost::json::object GameWorld::snapshot_payload(
    const std::string& you,
    const std::string& player_name,
    double player_score,
    double camera_x,
    double camera_y,
    double camera_zoom,
    const std::vector<Blob*>& blobs,
    const std::vector<Food*>& foods_view,
    const std::vector<EjectedMass*>& ejected_view,
    const std::vector<Virus*>& viruses_view) const {
  const auto& cfg = config::get();
  boost::json::object payload;
  payload["type"] = "state";
  if (you.empty()) {
    payload["you"] = nullptr;
  } else {
    payload["you"] = you;
  }

  boost::json::object world_obj;
  world_obj["w"] = cfg.world_width;
  world_obj["h"] = cfg.world_height;
  payload["world"] = std::move(world_obj);

  boost::json::object camera;
  camera["x"] = round_to(camera_x, 2);
  camera["y"] = round_to(camera_y, 2);
  camera["zoom"] = round_to(camera_zoom, 3);
  payload["camera"] = std::move(camera);

  boost::json::object player_obj;
  player_obj["name"] = player_name;
  player_obj["score"] = std::round(player_score);
  payload["player"] = std::move(player_obj);

  payload["leaderboard"] = leaderboard();

  std::unordered_map<std::string, const Player*> player_lookup;
  player_lookup.reserve(players.size());
  for (const auto& kv : players) {
    player_lookup.emplace(kv.first, &kv.second);
  }

  boost::json::array blobs_arr;
  blobs_arr.reserve(blobs.size());
  for (const Blob* blob : blobs) {
    if (!blob) {
      continue;
    }
    const Player* owner = nullptr;
    auto it = player_lookup.find(blob->player_id);
    if (it != player_lookup.end()) {
      owner = it->second;
    }
    boost::json::object row;
    row["id"] = blob->id;
    row["playerId"] = blob->player_id;
    row["name"] = owner ? owner->name : "";
    row["color"] = owner ? owner->color : "#ddd";
    row["x"] = round_to(blob->x, 2);
    row["y"] = round_to(blob->y, 2);
    row["mass"] = round_to(blob->mass, 2);
    blobs_arr.push_back(std::move(row));
  }
  payload["blobs"] = std::move(blobs_arr);

  boost::json::array food_arr;
  food_arr.reserve(foods_view.size());
  for (const Food* food : foods_view) {
    if (!food) {
      continue;
    }
    boost::json::object row;
    row["id"] = food->id;
    row["x"] = round_to(food->x, 2);
    row["y"] = round_to(food->y, 2);
    row["mass"] = food->mass;
    row["color"] = food->color;
    food_arr.push_back(std::move(row));
  }
  payload["foods"] = std::move(food_arr);

  boost::json::array ejected_arr;
  ejected_arr.reserve(ejected_view.size());
  for (const EjectedMass* mass : ejected_view) {
    if (!mass) {
      continue;
    }
    boost::json::object row;
    row["id"] = mass->id;
    row["x"] = round_to(mass->x, 2);
    row["y"] = round_to(mass->y, 2);
    row["mass"] = mass->mass;
    ejected_arr.push_back(std::move(row));
  }
  payload["ejected"] = std::move(ejected_arr);

  boost::json::array virus_arr;
  virus_arr.reserve(viruses_view.size());
  for (const Virus* virus : viruses_view) {
    if (!virus) {
      continue;
    }
    boost::json::object row;
    row["id"] = virus->id;
    row["x"] = round_to(virus->x, 2);
    row["y"] = round_to(virus->y, 2);
    row["mass"] = virus->mass;
    virus_arr.push_back(std::move(row));
  }
  payload["viruses"] = std::move(virus_arr);

  return payload;
}

std::optional<boost::json::object> GameWorld::snapshot_for(const std::string& player_id) const {
  const auto& cfg = config::get();
  auto it = players.find(player_id);
  if (it == players.end()) {
    return std::nullopt;
  }
  const Player& player = it->second;
  auto [cx, cy] = player.center();
  double total_mass = std::max(player.total_mass(), cfg.player_start_mass);
  int split_count = static_cast<int>(player.blobs.size()) - 1;
  if (split_count < 0) {
    split_count = 0;
  }
  double base_zoom = 1.52 - std::pow(total_mass, 0.4) / 22.0;
  double split_penalty = split_count * 0.055;
  double zoom = clamp(base_zoom - split_penalty, 0.24, 1.35);
  double view_w = cfg.view_width / zoom + cfg.view_padding;
  double view_h = cfg.view_height / zoom + cfg.view_padding;

  auto [blobs, foods_view, ejected_view, viruses_view] = visible_entities(cx, cy, view_w, view_h);
  return snapshot_payload(
      player.id,
      player.name,
      player.total_mass(),
      cx,
      cy,
      zoom,
      blobs,
      foods_view,
      ejected_view,
      viruses_view);
}

boost::json::object GameWorld::snapshot_overview() const {
  const auto& cfg = config::get();
  double zoom = std::min(cfg.view_width / cfg.world_width, cfg.view_height / cfg.world_height) * 0.92;

  std::vector<Blob*> blobs;
  for (auto& pkv : players) {
    for (auto& bkv : pkv.second.blobs) {
      blobs.push_back(const_cast<Blob*>(&bkv.second));
    }
  }

  std::vector<Food*> foods_view;
  foods_view.reserve(foods.size());
  for (auto& kv : foods) {
    foods_view.push_back(const_cast<Food*>(&kv.second));
  }

  std::vector<EjectedMass*> ejected_view;
  ejected_view.reserve(ejected.size());
  for (auto& kv : ejected) {
    ejected_view.push_back(const_cast<EjectedMass*>(&kv.second));
  }

  std::vector<Virus*> viruses_view;
  viruses_view.reserve(viruses.size());
  for (auto& kv : viruses) {
    viruses_view.push_back(const_cast<Virus*>(&kv.second));
  }

  return snapshot_payload(
      "",
      "Spectator",
      0.0,
      cfg.world_width * 0.5,
      cfg.world_height * 0.5,
      clamp(zoom, 0.05, 1.35),
      blobs,
      foods_view,
      ejected_view,
      viruses_view);
}

}  // namespace agario
