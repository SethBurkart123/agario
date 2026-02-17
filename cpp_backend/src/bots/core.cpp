#include "core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_map>

#include "../config.hpp"

namespace agario::bots {
namespace {

double clamp(double value, double min_value, double max_value) {
  return std::min(std::max(value, min_value), max_value);
}

std::pair<double, double> unit(double dx, double dy) {
  double dist = std::hypot(dx, dy);
  if (dist <= 1e-9) {
    return {0.0, 0.0};
  }
  return {dx / dist, dy / dist};
}

double lerp(double a, double b, double t) {
  return a + (b - a) * t;
}

double rand_uniform(std::mt19937& rng, double min_value, double max_value) {
  std::uniform_real_distribution<double> dist(min_value, max_value);
  return dist(rng);
}

double rand_unit(std::mt19937& rng) {
  return rand_uniform(rng, 0.0, 1.0);
}

std::pair<double, double> jitter_vector(double vx, double vy, std::mt19937& rng, double amount) {
  if (amount <= 1e-6) {
    return {vx, vy};
  }
  double mag = std::hypot(vx, vy);
  if (mag <= 1e-9) {
    return {rand_uniform(rng, -amount, amount), rand_uniform(rng, -amount, amount)};
  }
  double nx = -vy / mag;
  double ny = vx / mag;
  double side = rand_uniform(rng, -amount, amount);
  double scale = 1.0 + rand_uniform(rng, -amount * 0.32, amount * 0.22);
  return {vx * scale + nx * mag * side, vy * scale + ny * mag * side};
}

std::pair<double, double> player_center(const PlayerView& player) {
  if (player.blobs.empty()) {
    return {0.0, 0.0};
  }
  double total = std::max(1.0, player.total_mass);
  double x = 0.0;
  double y = 0.0;
  for (const auto& blob : player.blobs) {
    x += blob.x * blob.mass;
    y += blob.y * blob.mass;
  }
  return {x / total, y / total};
}

const BlobView* largest_blob(const PlayerView& player) {
  if (player.blobs.empty()) {
    return nullptr;
  }
  return &*std::max_element(player.blobs.begin(), player.blobs.end(), [](const BlobView& a, const BlobView& b) {
    return a.mass < b.mass;
  });
}

const BlobView* smallest_blob(const PlayerView& player) {
  if (player.blobs.empty()) {
    return nullptr;
  }
  return &*std::min_element(player.blobs.begin(), player.blobs.end(), [](const BlobView& a, const BlobView& b) {
    return a.mass < b.mass;
  });
}

bool can_eat(double attacker_mass, double defender_mass) {
  return attacker_mass > defender_mass * config::get().blob_eat_ratio;
}

std::vector<std::pair<const PlayerView*, const BlobView*>> iter_enemy_blobs(const BotContext& ctx) {
  std::vector<std::pair<const PlayerView*, const BlobView*>> out;
  for (const auto& player : ctx.players) {
    if (player.id == ctx.me.id || player.total_mass <= 0.0) {
      continue;
    }
    for (const auto& blob : player.blobs) {
      out.emplace_back(&player, &blob);
    }
  }
  return out;
}

std::pair<double, double> closest_food_target(const BotContext& ctx, double x, double y) {
  double best_dist = std::numeric_limits<double>::infinity();
  std::optional<std::pair<double, double>> target;

  for (const auto& item : ctx.ejected) {
    double d = std::hypot(item.x - x, item.y - y);
    if (d < best_dist) {
      best_dist = d;
      target = std::pair<double, double>{item.x, item.y};
    }
  }

  for (const auto& food : ctx.foods) {
    double d = std::hypot(food.x - x, food.y - y);
    if (d < best_dist) {
      best_dist = d;
      target = std::pair<double, double>{food.x, food.y};
    }
  }

  if (target.has_value()) {
    return *target;
  }
  return {x, y};
}

std::tuple<double, double, double, double> threat_field(const BotContext& ctx) {
  const auto& cfg = config::get();
  double vec_x = 0.0;
  double vec_y = 0.0;
  double pressure = 0.0;
  double imminence = 0.0;

  auto enemies = iter_enemy_blobs(ctx);
  for (const auto& [_, enemy] : enemies) {
    for (const auto& me_blob : ctx.me.blobs) {
      if (!can_eat(enemy->mass, me_blob.mass)) {
        continue;
      }
      double dx = me_blob.x - enemy->x;
      double dy = me_blob.y - enemy->y;
      double dist = std::max(1.0, std::hypot(dx, dy));
      double safe_radius = enemy->radius + me_blob.radius * 2.6 + 95.0;
      if (dist >= safe_radius) {
        continue;
      }
      auto [ux, uy] = unit(dx, dy);
      double mass_ratio = enemy->mass / std::max(1.0, me_blob.mass);
      double local_pressure = (safe_radius - dist) / safe_radius;
      local_pressure *= 0.44 + std::min(2.8, mass_ratio * 0.56);
      vec_x += ux * local_pressure;
      vec_y += uy * local_pressure;
      pressure += local_pressure;

      double eat_reach = std::max(0.0, enemy->radius - me_blob.radius * cfg.blob_eat_overlap);
      double danger_window = std::max(20.0, enemy->radius * 0.82);
      double gap = dist - eat_reach;
      if (gap < danger_window) {
        imminence = std::max(imminence, 1.0 - clamp(gap / danger_window, 0.0, 1.0));
      }
    }
  }

  return {vec_x, vec_y, pressure, imminence};
}

std::tuple<double, double, double> wall_field(const BotContext& ctx, double x, double y, double radius) {
  double margin = std::max(170.0, radius * 2.6);
  double vx = 0.0;
  double vy = 0.0;
  double strength = 0.0;

  double left = margin - x;
  if (left > 0) {
    double p = left / margin;
    vx += p;
    strength += p;
  }

  double right = margin - (ctx.world_width - x);
  if (right > 0) {
    double p = right / margin;
    vx -= p;
    strength += p;
  }

  double top = margin - y;
  if (top > 0) {
    double p = top / margin;
    vy += p;
    strength += p;
  }

  double bottom = margin - (ctx.world_height - y);
  if (bottom > 0) {
    double p = bottom / margin;
    vy -= p;
    strength += p;
  }

  return {vx, vy, strength};
}

std::tuple<double, double, double> virus_field(const BotContext& ctx, const BlobView& me_blob) {
  const auto& cfg = config::get();
  if (me_blob.mass <= cfg.virus_mass * 1.08) {
    return {0.0, 0.0, 0.0};
  }
  double vx = 0.0;
  double vy = 0.0;
  double strength = 0.0;
  for (const auto& virus : ctx.viruses) {
    double dx = me_blob.x - virus.x;
    double dy = me_blob.y - virus.y;
    double dist = std::max(1.0, std::hypot(dx, dy));
    double avoid = me_blob.radius + virus.radius * 1.42 + 54.0;
    if (dist >= avoid) {
      continue;
    }
    auto [ux, uy] = unit(dx, dy);
    double p = (avoid - dist) / avoid;
    vx += ux * p;
    vy += uy * p;
    strength += p;
  }
  return {vx, vy, strength};
}

struct NearestTarget {
  double dist = 0.0;
  double x = 0.0;
  double y = 0.0;
};

std::tuple<double, double, std::optional<std::pair<double, double>>> food_field(const BotContext& ctx, double x, double y) {
  double vx = 0.0;
  double vy = 0.0;
  std::optional<NearestTarget> nearest;

  for (const auto& item : ctx.ejected) {
    double dx = item.x - x;
    double dy = item.y - y;
    double dist = std::max(1.0, std::hypot(dx, dy));
    if (dist > 1700.0) {
      continue;
    }
    auto [ux, uy] = unit(dx, dy);
    double weight = (item.mass * 4.0) / (dist + 32.0);
    vx += ux * weight;
    vy += uy * weight;
    if (!nearest || dist < nearest->dist) {
      nearest = NearestTarget{dist, item.x, item.y};
    }
  }

  for (const auto& food : ctx.foods) {
    double dx = food.x - x;
    double dy = food.y - y;
    double dist = std::max(1.0, std::hypot(dx, dy));
    if (dist > 1400.0) {
      continue;
    }
    auto [ux, uy] = unit(dx, dy);
    double weight = (food.mass * 1.1) / (dist + 18.0);
    vx += ux * weight;
    vy += uy * weight;
    if (!nearest || dist < nearest->dist) {
      nearest = NearestTarget{dist, food.x, food.y};
    }
  }

  if (nearest) {
    return {vx, vy, std::pair<double, double>{nearest->x, nearest->y}};
  }
  return {vx, vy, std::nullopt};
}

double local_food_density(const BotContext& ctx, double x, double y, double radius = 540.0) {
  double r2 = radius * radius;
  int food_hits = 0;
  for (const auto& food : ctx.foods) {
    double dx = food.x - x;
    double dy = food.y - y;
    if (dx * dx + dy * dy <= r2) {
      food_hits += 1;
    }
  }
  int ejected_hits = 0;
  for (const auto& item : ctx.ejected) {
    double dx = item.x - x;
    double dy = item.y - y;
    if (dx * dx + dy * dy <= r2) {
      ejected_hits += 1;
    }
  }
  return std::min(1.25, food_hits / 24.0 + ejected_hits / 10.0);
}

std::optional<std::pair<double, double>> best_food_hotspot(const BotContext& ctx, double x, double y, std::mt19937& rng) {
  if (ctx.foods.empty() && ctx.ejected.empty()) {
    return std::nullopt;
  }

  double cell_size = 220.0;
  struct PairHash {
    std::size_t operator()(const std::pair<int, int>& key) const noexcept {
      return (static_cast<std::size_t>(key.first) << 32) ^ static_cast<std::size_t>(key.second);
    }
  };
  std::unordered_map<std::pair<int, int>, double, PairHash> buckets;

  for (const auto& food : ctx.foods) {
    int gx = static_cast<int>(food.x / cell_size);
    int gy = static_cast<int>(food.y / cell_size);
    buckets[{gx, gy}] = buckets[{gx, gy}] + food.mass * 1.0;
  }

  for (const auto& item : ctx.ejected) {
    int gx = static_cast<int>(item.x / cell_size);
    int gy = static_cast<int>(item.y / cell_size);
    buckets[{gx, gy}] = buckets[{gx, gy}] + item.mass * 3.2;
  }

  double best_score = -1e9;
  std::optional<std::pair<double, double>> best;
  for (const auto& entry : buckets) {
    int gx = entry.first.first;
    int gy = entry.first.second;
    double mass_score = entry.second;
    double cx = (gx + 0.5) * cell_size;
    double cy = (gy + 0.5) * cell_size;
    double dist = std::hypot(cx - x, cy - y);
    double score = mass_score - dist * 0.018 + rand_uniform(rng, -0.08, 0.08);
    if (score > best_score) {
      best_score = score;
      best = {cx + rand_uniform(rng, -cell_size * 0.16, cell_size * 0.16),
              cy + rand_uniform(rng, -cell_size * 0.16, cell_size * 0.16)};
    }
  }

  return best;
}

double target_risk(const BotContext& ctx, double x, double y, double my_mass) {
  double risk = 0.0;
  auto enemies = iter_enemy_blobs(ctx);
  for (const auto& [_, enemy] : enemies) {
    if (!can_eat(enemy->mass, my_mass)) {
      continue;
    }
    double dist = std::hypot(enemy->x - x, enemy->y - y);
    if (dist > 900.0) {
      continue;
    }
    risk += (900.0 - dist) / 900.0 * (enemy->mass / std::max(1.0, my_mass));
  }
  return risk;
}

std::optional<std::pair<double, double>> sample_food_route(
    const BotContext& ctx,
    double x,
    double y,
    std::mt19937& rng,
    double greed) {
  if (ctx.foods.empty() && ctx.ejected.empty()) {
    return std::nullopt;
  }

  std::optional<std::pair<double, double>> best_point;
  double best_score = -1e9;
  double preferred_dist = 240.0 + greed * 320.0;

  if (!ctx.foods.empty()) {
    int samples = std::min(54, static_cast<int>(ctx.foods.size()));
    std::uniform_int_distribution<int> idx_dist(0, static_cast<int>(ctx.foods.size() - 1));
    for (int i = 0; i < samples; ++i) {
      const auto& food = ctx.foods[idx_dist(rng)];
      double dist = std::max(1.0, std::hypot(food.x - x, food.y - y));
      double value = (food.mass * 2.6) / (dist + 52.0);
      double fit = std::max(0.0, 1.0 - std::abs(dist - preferred_dist) / std::max(160.0, preferred_dist * 1.18));
      double risk = target_risk(ctx, food.x, food.y, std::max(18.0, greed * 22.0));
      double score = value * (0.55 + greed * 0.24) + fit * 0.6 - risk * 0.4 + rand_unit(rng) * 0.1;
      if (score > best_score) {
        best_score = score;
        best_point = {food.x, food.y};
      }
    }
  }

  for (const auto& item : ctx.ejected) {
    double dist = std::max(1.0, std::hypot(item.x - x, item.y - y));
    double value = (item.mass * 7.2) / (dist + 34.0);
    double fit = std::max(0.0, 1.0 - std::abs(dist - preferred_dist * 0.85) / std::max(130.0, preferred_dist));
    double risk = target_risk(ctx, item.x, item.y, std::max(18.0, greed * 22.0));
    double score = value + fit * 0.85 - risk * 0.45 + rand_unit(rng) * 0.08;
    if (score > best_score) {
      best_score = score;
      best_point = {item.x, item.y};
    }
  }

  return best_point;
}

BotContext clip_context_for_view(const BotContext& ctx, double me_x, double me_y, double view_range) {
  auto in_range = [&](double px, double py, double extra = 0.0) {
    double dx = px - me_x;
    double dy = py - me_y;
    double r = view_range + extra;
    return (dx * dx + dy * dy) <= (r * r);
  };

  std::vector<FoodView> foods;
  for (const auto& food : ctx.foods) {
    if (in_range(food.x, food.y)) {
      foods.push_back(food);
    }
  }

  std::vector<EjectedView> ejected;
  for (const auto& item : ctx.ejected) {
    if (in_range(item.x, item.y)) {
      ejected.push_back(item);
    }
  }

  std::vector<VirusView> viruses;
  for (const auto& virus : ctx.viruses) {
    if (in_range(virus.x, virus.y, virus.radius)) {
      viruses.push_back(virus);
    }
  }

  std::vector<PlayerView> players;
  players.push_back(ctx.me);
  for (const auto& player : ctx.players) {
    if (player.id == ctx.me.id) {
      continue;
    }
    std::vector<BlobView> blobs;
    for (const auto& blob : player.blobs) {
      if (in_range(blob.x, blob.y, blob.radius)) {
        blobs.push_back(blob);
      }
    }
    if (blobs.empty()) {
      continue;
    }
    PlayerView clipped = player;
    clipped.blobs = std::move(blobs);
    clipped.total_mass = 0.0;
    for (const auto& blob : clipped.blobs) {
      clipped.total_mass += blob.mass;
    }
    players.push_back(std::move(clipped));
  }

  BotContext out;
  out.now = ctx.now;
  out.dt = ctx.dt;
  out.world_width = ctx.world_width;
  out.world_height = ctx.world_height;
  out.me = ctx.me;
  out.players = std::move(players);
  out.foods = std::move(foods);
  out.ejected = std::move(ejected);
  out.viruses = std::move(viruses);
  out.team_state = ctx.team_state;
  out.memory = ctx.memory;
  return out;
}

double instant_imminence(const BotContext& ctx, const BlobView& me_blob) {
  const auto& cfg = config::get();
  double imminence = 0.0;
  auto enemies = iter_enemy_blobs(ctx);
  for (const auto& [_, enemy] : enemies) {
    if (!can_eat(enemy->mass, me_blob.mass)) {
      continue;
    }
    double dist = std::hypot(enemy->x - me_blob.x, enemy->y - me_blob.y);
    double eat_reach = std::max(0.0, enemy->radius - me_blob.radius * cfg.blob_eat_overlap);
    double danger_window = std::max(26.0, enemy->radius * 0.95);
    double gap = dist - eat_reach;
    if (gap < danger_window) {
      imminence = std::max(imminence, 1.0 - clamp(gap / danger_window, 0.0, 1.0));
    }
  }
  return imminence;
}

double crowding_penalty(const BotContext& ctx, const PlayerView& me, const BlobView& target) {
  double crowd = 0.0;
  double crowd_radius = target.radius * 3.8 + 180.0;
  for (const auto& player : ctx.players) {
    if (player.id == me.id) {
      continue;
    }
    for (const auto& blob : player.blobs) {
      double dist = std::hypot(blob.x - target.x, blob.y - target.y);
      if (dist > crowd_radius) {
        continue;
      }
      double factor = 1.0 - clamp(dist / crowd_radius, 0.0, 1.0);
      if (player.is_bot) {
        factor *= 1.25;
      }
      crowd += factor;
    }
  }
  return crowd;
}

std::tuple<const PlayerView*, const BlobView*, double, double> best_prey(
    const BotContext& ctx,
    const BlobView& me_blob,
    double x,
    double y) {
  const auto& cfg = config::get();
  const PlayerView* best_owner = nullptr;
  const BlobView* best = nullptr;
  double best_score = -1e9;
  double best_dist = std::numeric_limits<double>::infinity();

  auto enemies = iter_enemy_blobs(ctx);
  for (const auto& [player, enemy] : enemies) {
    if (!can_eat(me_blob.mass, enemy->mass)) {
      continue;
    }
    if (!player->team_id.empty() && !ctx.me.team_id.empty() && player->team_id == ctx.me.team_id) {
      continue;
    }
    double dist = std::hypot(enemy->x - x, enemy->y - y);
    double close_factor = std::max(0.0, 1.0 - dist / (me_blob.radius * 9.0 + 520.0));
    double mass_adv = (me_blob.mass / std::max(1.0, enemy->mass)) - cfg.blob_eat_ratio;
    double risk = target_risk(ctx, enemy->x, enemy->y, me_blob.mass);
    double crowd_penalty = crowding_penalty(ctx, ctx.me, *enemy);
    double score = close_factor * 1.25 + std::min(2.2, mass_adv * 0.55) + enemy->mass * 0.004 - risk * 1.08 - crowd_penalty * 0.3;
    if (score > best_score) {
      best_score = score;
      best = enemy;
      best_owner = player;
      best_dist = dist;
    }
  }

  return {best_owner, best, best_score, best_dist};
}

std::pair<int, double> edible_cluster_value(const BlobView& me_blob, const PlayerView& target_player) {
  int edible_count = 0;
  double edible_mass = 0.0;
  for (const auto& blob : target_player.blobs) {
    if (can_eat(me_blob.mass, blob.mass)) {
      edible_count += 1;
      edible_mass += blob.mass;
    }
  }
  return {edible_count, edible_mass};
}

std::tuple<double, bool, bool, double> attack_likelihood(
    const BlobView& me_blob,
    const PlayerView& target_player,
    const BlobView& target_blob,
    double dist,
    bool split_ready,
    double aggression,
    double local_food_density,
    bool on_cooldown) {
  const auto& cfg = config::get();
  double mass_ratio = me_blob.mass / std::max(1.0, target_blob.mass);
  bool close_kill = dist < (me_blob.radius * 2.0 + target_blob.radius * 1.2 + 72.0) && mass_ratio > 1.9;
  double chase_range = me_blob.radius * (4.35 + aggression * 1.1) + 360.0;

  double post_split_mass = me_blob.mass * 0.5;
  bool split_can_eat = post_split_mass > target_blob.mass * (cfg.blob_eat_ratio + 0.03);
  double split_reach = me_blob.radius * 2.75 + target_blob.radius * 1.45 + 90.0;
  bool split_window = split_ready && split_can_eat && dist < split_reach;

  double distance_factor = clamp(1.0 - dist / std::max(1.0, chase_range), 0.0, 1.0);
  double size_factor = clamp((mass_ratio - cfg.blob_eat_ratio) / 1.65, 0.0, 1.0);
  auto [edible_count, edible_mass] = edible_cluster_value(me_blob, target_player);
  double cluster_count_factor = clamp((edible_count - 1) / 4.0, 0.0, 1.0);
  double cluster_mass_factor = clamp(edible_mass / std::max(1.0, me_blob.mass * 0.95), 0.0, 1.0);
  double cluster_factor = cluster_count_factor * 0.6 + cluster_mass_factor * 0.4;

  double prob = 0.08 + aggression * 0.1;
  prob += distance_factor * 0.36;
  prob += size_factor * 0.31;
  prob += cluster_factor * 0.23;
  if (split_window) {
    prob += 0.24;
  }
  if (on_cooldown && !close_kill) {
    prob -= 0.34;
  }
  if (dist > chase_range && !split_window) {
    prob -= 0.32;
  }
  prob -= local_food_density * 0.2;

  return {clamp(prob, 0.02, 0.98), close_kill, split_window, chase_range};
}

std::tuple<double, double, double> crowd_field(const BotContext& ctx, const BlobView& me_blob) {
  double vx = 0.0;
  double vy = 0.0;
  double density = 0.0;
  double comfort = me_blob.radius * 2.45 + 140.0;

  auto enemies = iter_enemy_blobs(ctx);
  for (const auto& [player, blob] : enemies) {
    double dx = me_blob.x - blob->x;
    double dy = me_blob.y - blob->y;
    double dist = std::max(1.0, std::hypot(dx, dy));
    if (dist >= comfort) {
      continue;
    }
    auto [ux, uy] = unit(dx, dy);
    double p = (comfort - dist) / comfort;
    if (player->is_bot) {
      p *= 1.24;
    }
    if (can_eat(me_blob.mass, blob->mass)) {
      p *= 0.55;
    }
    vx += ux * p;
    vy += uy * p;
    density += p;
  }

  return {vx, vy, density};
}

class SoloSmartBrain : public BotBrain {
 public:
  explicit SoloSmartBrain(const BotInitContext& init_ctx)
      : rng_(init_ctx.rng) {
    aggression_ = rand_uniform(rng_, 0.84, 1.22);
    caution_ = rand_uniform(rng_, 0.82, 1.2);
    greed_ = rand_uniform(rng_, 0.84, 1.3);
    strafe_dir_ = rand_unit(rng_) < 0.5 ? -1.0 : 1.0;
    reaction_min_ = rand_uniform(rng_, 0.065, 0.12);
    reaction_max_ = rand_uniform(rng_, 0.14, 0.27);
    view_scale_ = rand_uniform(rng_, 0.8, 1.03);
    steer_rate_ = rand_uniform(rng_, 6.4, 10.8);
    dodge_skill_ = rand_uniform(rng_, 0.52, 0.86);
    threat_miss_chance_ = rand_uniform(rng_, 0.08, 0.26);
    threat_jitter_ = rand_uniform(rng_, 0.08, 0.22);
    panic_split_chance_ = rand_uniform(rng_, 0.35, 0.7);
  }

  BotAction decide(BotContext& ctx) override {
    const auto& cfg = config::get();
    const BlobView* me_blob = largest_blob(ctx.me);
    const BlobView* me_small = smallest_blob(ctx.me);
    if (!me_blob || !me_small) {
      return {ctx.world_width * 0.5, ctx.world_height * 0.5, false, false};
    }

    auto [me_x, me_y] = player_center(ctx.me);
    double base_view = 860.0 + me_blob->radius * 8.8 + ctx.me.blobs.size() * 52.0;
    double view_range = base_view * view_scale_;
    BotContext perceived_ctx = clip_context_for_view(ctx, me_x, me_y, view_range);
    double instant_danger = instant_imminence(perceived_ctx, *me_small);

    double next_think_at = ctx.memory->get_or<double>("next_think_at", 0.0);
    double danger_override = 0.94 + (1.0 - dodge_skill_) * 0.06;
    if (ctx.now < next_think_at && instant_danger < danger_override) {
      auto cached = cached_action(ctx);
      if (cached.has_value()) {
        return *cached;
      }
    }

    int max_split_blobs = std::min(cfg.max_player_blobs, 8);
    bool split_ready = ctx.me.blobs.size() < static_cast<std::size_t>(max_split_blobs) &&
                       me_blob->mass >= cfg.player_min_split_mass &&
                       ctx.now >= ctx.memory->get_or<double>("next_split_at", 0.0);

    auto [threat_x, threat_y, threat_pressure, imminence] = threat_field(perceived_ctx);
    if (rand_unit(rng_) < threat_miss_chance_) {
      double miss_scale = rand_uniform(rng_, 0.22, 0.68);
      threat_x *= miss_scale;
      threat_y *= miss_scale;
      threat_pressure *= miss_scale;
      imminence *= miss_scale;
    }
    auto jittered = jitter_vector(threat_x, threat_y, rng_, threat_jitter_ * (1.14 - dodge_skill_));
    threat_x = jittered.first;
    threat_y = jittered.second;

    auto [wall_x, wall_y, wall_pressure] = wall_field(perceived_ctx, me_x, me_y, me_blob->radius);
    auto [virus_x, virus_y, virus_pressure] = virus_field(perceived_ctx, *me_blob);
    auto [crowd_x, crowd_y, crowd_pressure] = crowd_field(perceived_ctx, *me_blob);
    auto [prey_owner, prey, attack_score, prey_dist] = best_prey(perceived_ctx, *me_blob, me_x, me_y);
    auto [food_x, food_y, nearest_food] = food_field(perceived_ctx, me_x, me_y);
    imminence = std::max(imminence, instant_danger);
    double local_density = local_food_density(perceived_ctx, me_x, me_y);

    double effective_threat = threat_pressure + wall_pressure * 0.35 + virus_pressure * 0.65 + crowd_pressure * 0.22;
    effective_threat *= 0.74 + dodge_skill_ * 0.4;
    double flee_threshold = 1.03 / caution_;
    double imminent_threshold = 0.76 / caution_;
    double attack_threshold = 0.24 / aggression_;
    double attack_cooldown_until = ctx.memory->get_or<double>("attack_cooldown_until", 0.0);
    std::string desired_mode = "farm";

    if (effective_threat > flee_threshold || imminence > imminent_threshold) {
      desired_mode = "flee";
    } else if (prey && prey_owner) {
      bool on_cooldown = ctx.now < attack_cooldown_until;
      auto [engage_prob, close_kill, split_window, chase_range] = attack_likelihood(
          *me_blob, *prey_owner, *prey, prey_dist, split_ready, aggression_, local_density, on_cooldown);

      if (ctx.now >= ctx.memory->get_or<double>("next_engage_roll_at", 0.0)) {
        ctx.memory->set("engage_roll", rand_unit(rng_));
        ctx.memory->set("next_engage_roll_at", ctx.now + rand_uniform(rng_, 0.28, 0.68));
      }
      double engage_roll = ctx.memory->get_or<double>("engage_roll", 1.0);
      bool willing = close_kill || split_window || engage_roll < engage_prob;
      bool can_attack = close_kill || split_window || prey_dist <= chase_range;

      double attack_need = attack_threshold + local_density * 0.2;
      if (prey_dist > chase_range * 0.72) {
        attack_need += 0.16;
      }
      if (split_window) {
        attack_need -= 0.12;
      }

      if (can_attack && willing && attack_score > attack_need && effective_threat < 0.94 * caution_) {
        desired_mode = "attack";
      }
    }

    if (desired_mode == "farm" && prey && rand_unit(rng_) < 0.03 * aggression_) {
      if (attack_score > (attack_threshold + local_density * 0.12) * 0.88 && effective_threat < 0.8 * caution_) {
        desired_mode = "attack";
      }
    }
    if (desired_mode == "attack" && rand_unit(rng_) < 0.02 * caution_ && effective_threat > 0.75) {
      desired_mode = "farm";
    }

    std::string prev_mode = ctx.memory->get_or<std::string>("mode", "farm");
    double hold_until = ctx.memory->get_or<double>("mode_hold_until", 0.0);
    std::string mode = desired_mode;

    if (ctx.now < hold_until) {
      if (prev_mode == "flee") {
        if (desired_mode != "flee" && effective_threat < 0.3 / caution_ && imminence < 0.2) {
          mode = desired_mode;
        } else {
          mode = "flee";
        }
      } else if (prev_mode == "attack") {
        if (desired_mode == "flee") {
          mode = "flee";
        } else if (ctx.now < ctx.memory->get_or<double>("attack_commit_until", 0.0) && prey) {
          mode = "attack";
        } else if (prey && effective_threat < 1.05 * caution_) {
          mode = "attack";
        } else {
          mode = desired_mode;
        }
      } else {
        mode = desired_mode == "flee" ? "flee" : "farm";
      }
    }

    if (mode != prev_mode) {
      if (mode == "flee") {
        ctx.memory->set("mode_hold_until", ctx.now + rand_uniform(rng_, 0.62, 1.08));
      } else if (mode == "attack") {
        ctx.memory->set("mode_hold_until", ctx.now + rand_uniform(rng_, 0.55, 0.95));
        ctx.memory->set("attack_commit_until", ctx.now + rand_uniform(rng_, 0.45, 0.92));
      } else {
        ctx.memory->set("mode_hold_until", ctx.now + rand_uniform(rng_, 0.38, 0.75));
        if (prev_mode == "attack") {
          ctx.memory->set("attack_cooldown_until", ctx.now + rand_uniform(rng_, 0.55, 1.5));
        }
      }
    }
    ctx.memory->set("mode", mode);

    if (mode == "flee") {
      double flee_gain = 0.76 + dodge_skill_ * 0.56;
      double vx = threat_x * (1.05 * flee_gain) + wall_x * 1.05 + virus_x * 1.15 + crowd_x * 0.86;
      double vy = threat_y * (1.05 * flee_gain) + wall_y * 1.05 + virus_y * 1.15 + crowd_y * 0.86;
      auto jittered2 = jitter_vector(vx, vy, rng_, threat_jitter_ * (1.2 - dodge_skill_));
      vx = jittered2.first;
      vy = jittered2.second;
      auto [ux, uy] = unit(vx, vy);
      if (ux == 0.0 && uy == 0.0) {
        auto fallback = unit(ctx.world_width * 0.5 - me_x, ctx.world_height * 0.5 - me_y);
        ux = fallback.first;
        uy = fallback.second;
      }
      double flee_dist = 780.0 + rand_uniform(rng_, -140.0, 170.0);
      double target_x = me_x + ux * flee_dist;
      double target_y = me_y + uy * flee_dist;

      bool split_escape = split_ready &&
                          imminence > 0.84 / caution_ &&
                          effective_threat > 1.06 / caution_ &&
                          me_small->mass > cfg.min_blob_mass * 1.25 &&
                          rand_unit(rng_) < (0.22 + imminence * 0.28) * panic_split_chance_;
      if (split_escape) {
        ctx.memory->set("next_split_at", ctx.now + rand_uniform(rng_, 0.78, 1.05));
      }
      bool emergency_trigger = imminence > (0.95 + (1.0 - dodge_skill_) * 0.04) &&
                               rand_unit(rng_) < (0.34 + dodge_skill_ * 0.48);

      return remember_action(ctx,
                             {clamp(target_x, 0.0, ctx.world_width), clamp(target_y, 0.0, ctx.world_height),
                              split_escape, false},
                             "flee", emergency_trigger);
    }

    if (mode == "attack" && prey) {
      double to_target_x = prey->x - me_x;
      double to_target_y = prey->y - me_y;
      double dist = std::max(1.0, std::hypot(to_target_x, to_target_y));
      auto [ux, uy] = unit(to_target_x, to_target_y);
      bool near_finish = dist < (me_blob->radius + prey->radius) * 1.12;

      if (ctx.now >= ctx.memory->get_or<double>("next_strafe_at", 0.0)) {
        strafe_dir_ *= rand_unit(rng_) < 0.78 ? -1.0 : 1.0;
        ctx.memory->set("next_strafe_at", ctx.now + rand_uniform(rng_, 0.4, 0.95));
      }

      double push_through = std::max(185.0, std::min(360.0, me_blob->radius * 1.34 + prey->radius * 0.8));
      double strafe_amp = near_finish ? 0.0 : std::min(120.0, dist * 0.34) * strafe_dir_;
      double px = -uy;
      double py = ux;

      double target_x = prey->x + ux * push_through + px * strafe_amp;
      double target_y = prey->y + uy * push_through + py * strafe_amp;
      target_x += wall_x * 120.0 + virus_x * 105.0 - threat_x * 38.0 + crowd_x * 84.0;
      target_y += wall_y * 120.0 + virus_y * 105.0 - threat_y * 38.0 + crowd_y * 84.0;

      double post_split_mass = me_blob->mass * 0.5;
      bool split_can_eat = post_split_mass > prey->mass * (cfg.blob_eat_ratio + 0.03);
      double split_reach = me_blob->radius * 2.75 + prey->radius * 1.45 + 90.0;

      bool split_kill = split_ready &&
                        split_can_eat &&
                        prey_dist < split_reach &&
                        effective_threat < 0.58 * caution_ &&
                        attack_score > std::max(0.2, attack_threshold * 0.82) &&
                        rand_unit(rng_) < (0.52 + std::min(0.32, attack_score * 0.22));
      if (split_kill) {
        ctx.memory->set("next_split_at", ctx.now + rand_uniform(rng_, 0.78, 1.02));
      }

      return remember_action(ctx,
                             {clamp(target_x, 0.0, ctx.world_width), clamp(target_y, 0.0, ctx.world_height),
                              split_kill, false},
                             "attack", false);
    }

    auto route_target = ctx.memory->get_or<std::array<double, 2>>("route_target", {std::numeric_limits<double>::quiet_NaN(),
                                                                                     std::numeric_limits<double>::quiet_NaN()});
    bool has_route = !std::isnan(route_target[0]) && !std::isnan(route_target[1]);
    double route_dist = std::numeric_limits<double>::infinity();
    if (has_route) {
      route_dist = std::hypot(route_target[0] - me_x, route_target[1] - me_y);
    }

    if (ctx.now >= ctx.memory->get_or<double>("next_route_at", 0.0) || !has_route ||
        route_dist < std::max(85.0, me_blob->radius * 1.2)) {
      auto hotspot = best_food_hotspot(perceived_ctx, me_x, me_y, rng_);
      auto picked = hotspot ? hotspot : sample_food_route(perceived_ctx, me_x, me_y, rng_, greed_);
      if (picked) {
        route_target = {picked->first, picked->second};
        ctx.memory->set("route_target", route_target);
        has_route = true;
      }
      ctx.memory->set("next_route_at", ctx.now + rand_uniform(rng_, 0.9, 2.1));
    }

    double route_x = 0.0;
    double route_y = 0.0;
    if (has_route) {
      auto [rux, ruy] = unit(route_target[0] - me_x, route_target[1] - me_y);
      route_x = rux;
      route_y = ruy;
    }

    double lane_dx = ctx.memory->get_or<double>("lane_dx", 0.0);
    double lane_dy = ctx.memory->get_or<double>("lane_dy", 0.0);
    double lane_until = ctx.memory->get_or<double>("lane_until", 0.0);
    if (ctx.now >= lane_until || (lane_dx == 0.0 && lane_dy == 0.0)) {
      std::optional<std::pair<double, double>> anchor;
      if (has_route) {
        anchor = {route_target[0], route_target[1]};
      } else if (nearest_food) {
        anchor = *nearest_food;
      }
      if (anchor) {
        auto [ux, uy] = unit(anchor->first - me_x, anchor->second - me_y);
        lane_dx = ux;
        lane_dy = uy;
      }
      if (lane_dx == 0.0 && lane_dy == 0.0) {
        auto [ux, uy] = unit(rand_uniform(rng_, -1.0, 1.0), rand_uniform(rng_, -1.0, 1.0));
        lane_dx = ux;
        lane_dy = uy;
      }
      ctx.memory->set("lane_until", ctx.now + rand_uniform(rng_, 0.95, 2.4));
    }

    if (has_route) {
      double steer_x = lerp(lane_dx, route_x, 0.16);
      double steer_y = lerp(lane_dy, route_y, 0.16);
      auto [ux, uy] = unit(steer_x, steer_y);
      lane_dx = ux;
      lane_dy = uy;
    }

    ctx.memory->set("lane_dx", lane_dx);
    ctx.memory->set("lane_dy", lane_dy);

    auto [food_dir_x, food_dir_y] = unit(food_x, food_y);
    double vx = lane_dx * (1.08 + greed_ * 0.28) + route_x * 0.58 + food_dir_x * 0.48 + wall_x * 0.72 +
                virus_x * 0.82 + crowd_x * 0.95 - threat_x * 0.2;
    double vy = lane_dy * (1.08 + greed_ * 0.28) + route_y * 0.58 + food_dir_y * 0.48 + wall_y * 0.72 +
                virus_y * 0.82 + crowd_y * 0.95 - threat_y * 0.2;
    auto [ux, uy] = unit(vx, vy);
    if (ux != 0.0 || uy != 0.0) {
      double step = 760.0 + rand_uniform(rng_, -70.0, 110.0);
      double target_x = me_x + ux * step;
      double target_y = me_y + uy * step;
      return remember_action(ctx,
                             {clamp(target_x, 0.0, ctx.world_width), clamp(target_y, 0.0, ctx.world_height), false, false},
                             "farm", false);
    }

    if (nearest_food) {
      return remember_action(ctx, {nearest_food->first, nearest_food->second, false, false}, "farm", false);
    }

    if (ctx.now >= ctx.memory->get_or<double>("next_wander_at", 0.0)) {
      double jitter = 420.0;
      ctx.memory->set("wander_x", clamp(me_x + rand_uniform(rng_, -jitter, jitter), 0.0, ctx.world_width));
      ctx.memory->set("wander_y", clamp(me_y + rand_uniform(rng_, -jitter, jitter), 0.0, ctx.world_height));
      ctx.memory->set("next_wander_at", ctx.now + rand_uniform(rng_, 0.55, 1.25));
    }

    double wander_x = ctx.memory->get_or<double>("wander_x", me_x);
    double wander_y = ctx.memory->get_or<double>("wander_y", me_y);
    return remember_action(ctx, {wander_x, wander_y, false, false}, "farm", false);
  }

 private:
  std::optional<BotAction> cached_action(const BotContext& ctx) const {
    if (!ctx.memory) {
      return std::nullopt;
    }
    auto cached = ctx.memory->get_or<std::array<double, 4>>("last_action", {std::numeric_limits<double>::quiet_NaN(), 0, 0, 0});
    if (std::isnan(cached[0])) {
      return std::nullopt;
    }
    BotAction action;
    action.target_x = cached[0];
    action.target_y = cached[1];
    action.split = cached[2] > 0.5;
    action.eject = cached[3] > 0.5;
    return action;
  }

  void schedule_next_think(BotContext& ctx, const std::string& mode, bool emergency) {
    double min_delay = reaction_min_;
    double max_delay = reaction_max_;
    if (mode == "flee" || emergency) {
      min_delay *= 0.74;
      max_delay *= 0.95;
    } else if (mode == "attack") {
      min_delay *= 0.82;
      max_delay *= 0.95;
    }
    ctx.memory->set("next_think_at", ctx.now + rand_uniform(rng_, min_delay, max_delay));
  }

  BotAction remember_action(BotContext& ctx, const BotAction& action, const std::string& mode, bool emergency) {
    double prev_x = ctx.memory->get_or<double>("aim_x", action.target_x);
    double prev_y = ctx.memory->get_or<double>("aim_y", action.target_y);
    double dt = std::max(1.0 / 120.0, ctx.dt);
    double rate = steer_rate_;
    if (mode == "flee" || emergency) {
      rate *= 0.95 + dodge_skill_ * 0.55;
    } else if (mode == "attack") {
      rate *= 1.02 + aggression_ * 0.22;
    }
    double alpha = 1.0 - std::exp(-rate * dt);
    alpha = clamp(alpha, 0.14, 0.92);

    double target_x = lerp(prev_x, action.target_x, alpha);
    double target_y = lerp(prev_y, action.target_y, alpha);
    ctx.memory->set("aim_x", target_x);
    ctx.memory->set("aim_y", target_y);

    BotAction final_action{target_x, target_y, action.split, action.eject};
    ctx.memory->set("last_action", std::array<double, 4>{final_action.target_x, final_action.target_y,
                                                         final_action.split ? 1.0 : 0.0,
                                                         final_action.eject ? 1.0 : 0.0});
    schedule_next_think(ctx, mode, emergency);
    return final_action;
  }

  std::mt19937 rng_;
  double aggression_ = 1.0;
  double caution_ = 1.0;
  double greed_ = 1.0;
  double strafe_dir_ = 1.0;
  double reaction_min_ = 0.08;
  double reaction_max_ = 0.2;
  double view_scale_ = 1.0;
  double steer_rate_ = 8.0;
  double dodge_skill_ = 0.7;
  double threat_miss_chance_ = 0.1;
  double threat_jitter_ = 0.1;
  double panic_split_chance_ = 0.5;
};

class ForagerBrain : public BotBrain {
 public:
  explicit ForagerBrain(const BotInitContext& init_ctx) : inner_(init_ctx) {}

  BotAction decide(BotContext& ctx) override {
    return inner_.decide(ctx);
  }

 private:
  SoloSmartBrain inner_;
};

class PredatorBrain : public BotBrain {
 public:
  explicit PredatorBrain(const BotInitContext& init_ctx) : inner_(init_ctx) {}

  BotAction decide(BotContext& ctx) override {
    return inner_.decide(ctx);
  }

 private:
  SoloSmartBrain inner_;
};

class TeamSwarmBrain : public BotBrain {
 public:
  explicit TeamSwarmBrain(const BotInitContext& init_ctx)
      : rng_(init_ctx.rng), team_id_(init_ctx.team_id.empty() ? "swarm" : init_ctx.team_id) {}

  BotAction decide(BotContext& ctx) override {
    const BlobView* me_blob = largest_blob(ctx.me);
    if (!me_blob) {
      return {ctx.world_width * 0.5, ctx.world_height * 0.5, false, false};
    }

    std::vector<PlayerView> teammates;
    for (const auto& player : ctx.players) {
      if (player.id == ctx.me.id) {
        continue;
      }
      if (player.team_id == team_id_ && player.plugin_name == ctx.me.plugin_name && player.total_mass > 0) {
        teammates.push_back(player);
      }
    }
    double team_mass = ctx.me.total_mass;
    for (const auto& player : teammates) {
      team_mass += player.total_mass;
    }

    std::string focus_id = ctx.team_state->get_or<std::string>("focus_id", "");
    double focus_until = ctx.team_state->get_or<double>("focus_until", 0.0);
    const PlayerView* focus = nullptr;
    if (!focus_id.empty() && ctx.now <= focus_until) {
      for (const auto& player : ctx.players) {
        if (player.id == focus_id && player.total_mass > 0) {
          focus = &player;
          break;
        }
      }
    }

    if (!focus) {
      std::vector<const PlayerView*> candidates;
      for (const auto& player : ctx.players) {
        if (player.id == ctx.me.id) {
          continue;
        }
        if (player.total_mass <= 0) {
          continue;
        }
        if (!player.team_id.empty() && player.team_id == team_id_) {
          continue;
        }
        candidates.push_back(&player);
      }
      double best_score = std::numeric_limits<double>::infinity();
      for (const auto* enemy : candidates) {
        auto [ex, ey] = player_center(*enemy);
        double d = std::hypot(ex - me_blob->x, ey - me_blob->y);
        double strength = enemy->total_mass / std::max(1.0, team_mass);
        double score = d * (0.55 + strength);
        if (score < best_score) {
          best_score = score;
          focus = enemy;
        }
      }
      if (focus) {
        ctx.team_state->set("focus_id", focus->id);
        ctx.team_state->set("focus_until", ctx.now + 1.1);
      }
    }

    if (!focus) {
      auto target = closest_food_target(ctx, me_blob->x, me_blob->y);
      return {target.first, target.second, false, false};
    }

    auto [fx, fy] = player_center(*focus);
    std::vector<std::string> members = ctx.team_state->get_or<std::vector<std::string>>("members", {});
    int rank = 0;
    auto it = std::find(members.begin(), members.end(), ctx.me.id);
    if (it != members.end()) {
      rank = static_cast<int>(std::distance(members.begin(), it));
    }

    double spacing = 55.0 + (rank / 2) * 26.0;
    double side = (rank % 2 == 0) ? -1.0 : 1.0;

    double to_enemy_x = fx - me_blob->x;
    double to_enemy_y = fy - me_blob->y;
    double dist = std::max(1.0, std::hypot(to_enemy_x, to_enemy_y));
    double nx = to_enemy_x / dist;
    double ny = to_enemy_y / dist;
    double px = -ny;
    double py = nx;

    double target_x = fx - nx * std::min(25.0, focus->total_mass * 0.02) + px * side * spacing;
    double target_y = fy - ny * std::min(25.0, focus->total_mass * 0.02) + py * side * spacing;

    bool should_split = ctx.me.blobs.size() < 8 &&
                        me_blob->mass > focus->total_mass * 0.62 &&
                        dist < me_blob->radius * 2.2 &&
                        ctx.now >= ctx.memory->get_or<double>("next_split_at", 0.0);
    if (should_split) {
      ctx.memory->set("next_split_at", ctx.now + 1.0);
    }

    bool should_eject = false;
    if (!teammates.empty() && ctx.me.total_mass > 110 && ctx.now >= ctx.memory->get_or<double>("next_eject_at", 0.0)) {
      const PlayerView* heavier = nullptr;
      for (const auto& mate : teammates) {
        if (!heavier || mate.total_mass > heavier->total_mass) {
          heavier = &mate;
        }
      }
      if (heavier && heavier->total_mass > ctx.me.total_mass * 1.45) {
        auto [mx, my] = player_center(*heavier);
        double mate_dist = std::hypot(mx - me_blob->x, my - me_blob->y);
        if (mate_dist < me_blob->radius * 6.5) {
          target_x = mx;
          target_y = my;
          should_eject = true;
          ctx.memory->set("next_eject_at", ctx.now + 0.75);
        }
      }
    }

    return {clamp(target_x, 0.0, ctx.world_width), clamp(target_y, 0.0, ctx.world_height), should_split, should_eject};
  }

 private:
  std::mt19937 rng_;
  std::string team_id_;
};

}  // namespace

void register_core_plugins(BotRegistry& registry) {
  registry.register_factory("solo_smart", [](const BotInitContext& ctx) {
    return std::make_unique<SoloSmartBrain>(ctx);
  });
  registry.register_factory("solo", [](const BotInitContext& ctx) {
    return std::make_unique<SoloSmartBrain>(ctx);
  });
  registry.register_factory("forager", [](const BotInitContext& ctx) {
    return std::make_unique<ForagerBrain>(ctx);
  });
  registry.register_factory("predator", [](const BotInitContext& ctx) {
    return std::make_unique<PredatorBrain>(ctx);
  });
  registry.register_factory("team_swarm", [](const BotInitContext& ctx) {
    return std::make_unique<TeamSwarmBrain>(ctx);
  });
  registry.register_factory("swarm", [](const BotInitContext& ctx) {
    return std::make_unique<TeamSwarmBrain>(ctx);
  });
}

}  // namespace agario::bots
