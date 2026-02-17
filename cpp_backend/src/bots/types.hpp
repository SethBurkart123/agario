#pragma once

#include <array>
#include <random>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace agario::bots {

struct Memory {
  using Value = std::variant<std::monostate, double, bool, std::string, std::array<double, 2>, std::array<double, 4>,
                             std::vector<std::string>>;

  std::unordered_map<std::string, Value> data;

  bool has(const std::string& key) const {
    return data.find(key) != data.end();
  }

  template <typename T>
  T get_or(const std::string& key, const T& fallback) const {
    auto it = data.find(key);
    if (it == data.end()) {
      return fallback;
    }
    if (const auto* value = std::get_if<T>(&it->second)) {
      return *value;
    }
    return fallback;
  }

  template <typename T>
  void set(const std::string& key, T value) {
    data[key] = std::move(value);
  }
};

struct BlobView {
  std::string id;
  std::string player_id;
  double x = 0.0;
  double y = 0.0;
  double mass = 0.0;
  double radius = 0.0;
};

struct PlayerView {
  std::string id;
  std::string name;
  std::string color;
  bool is_bot = false;
  std::string plugin_name;
  std::string team_id;
  double total_mass = 0.0;
  std::vector<BlobView> blobs;
};

struct FoodView {
  std::string id;
  double x = 0.0;
  double y = 0.0;
  double mass = 0.0;
  double radius = 0.0;
  std::string color;
};

struct EjectedView {
  std::string id;
  double x = 0.0;
  double y = 0.0;
  double mass = 0.0;
  double radius = 0.0;
  std::string owner_id;
  double ttl = 0.0;
};

struct VirusView {
  std::string id;
  double x = 0.0;
  double y = 0.0;
  double mass = 0.0;
  double radius = 0.0;
};

struct BotAction {
  double target_x = 0.0;
  double target_y = 0.0;
  bool split = false;
  bool eject = false;
};

struct BotSpec {
  std::string plugin_name;
  int count = 1;
  std::string team_id;
  std::string name_prefix;
};

struct BotContext {
  double now = 0.0;
  double dt = 0.0;
  double world_width = 0.0;
  double world_height = 0.0;
  PlayerView me;
  std::vector<PlayerView> players;
  std::vector<FoodView> foods;
  std::vector<EjectedView> ejected;
  std::vector<VirusView> viruses;
  Memory* team_state = nullptr;
  Memory* memory = nullptr;
};

struct BotInitContext {
  std::string plugin_name;
  std::string bot_name;
  std::string team_id;
  int bot_index = 0;
  std::mt19937 rng;
};

class BotBrain {
 public:
  virtual ~BotBrain() = default;
  virtual BotAction decide(BotContext& ctx) = 0;
};

}  // namespace agario::bots
