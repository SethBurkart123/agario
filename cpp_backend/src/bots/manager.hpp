#pragma once

#include <boost/json.hpp>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../config.hpp"
#include "../world.hpp"
#include "core.hpp"
#include "registry.hpp"
#include "types.hpp"

namespace agario::bots {

std::vector<BotSpec> parse_bot_specs(const std::string& raw);

class BotManager {
 public:
  BotManager(GameWorld& world,
             bool enabled,
             std::vector<std::string> plugin_modules,
             std::vector<BotSpec> bot_specs,
             int seed,
             int bot_threads);

  static BotManager from_config(GameWorld& world);

  boost::json::object describe() const;

  void ensure_started(double now);
  void tick(double dt, double now);

 private:
  struct BotAgent {
    std::string player_id;
    std::string plugin_name;
    std::string team_id;
    std::string name_prefix;
    std::unique_ptr<BotBrain> brain;
    Memory memory;
  };

  struct SpawnRequest {
    std::string plugin_name;
    std::string team_id;
    std::string name_prefix;
  };

  struct TeamData {
    Memory state;
    std::mutex mutex;
  };

  GameWorld& world_;
  bool enabled_ = true;
  std::vector<std::string> plugin_modules_;
  std::vector<BotSpec> bot_specs_;
  bool started_ = false;
  BotRegistry registry_;
  std::mt19937 rng_;
  int bot_threads_ = 0;
  std::unordered_map<std::string, BotAgent> agents_;
  std::unordered_map<std::string, std::shared_ptr<TeamData>> team_state_;
  int spawn_index_ = 0;

  struct ViewPack {
    std::unordered_map<std::string, PlayerView> players_by_id;
    std::vector<PlayerView> players;
    std::vector<FoodView> foods;
    std::vector<EjectedView> ejected;
    std::vector<VirusView> viruses;
  };

  ViewPack build_views() const;
  BotAction fallback_action(const PlayerView& me) const;
  std::string build_name(const BotSpec& spec, int index) const;
  std::string spawn_bot(const std::string& plugin_name,
                        const std::string& team_id,
                        const std::string& name_prefix,
                        double now,
                        const std::string& explicit_name);
  void spawn_extra_on_elimination(const SpawnRequest& eliminated, double now);
  std::string team_key(const std::string& plugin_name, const std::string& team_id, const std::string& player_id) const;
  static double clamp(double value, double min_value, double max_value);
};

}  // namespace agario::bots
