#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace agario::config {
namespace {

bool env_bool(const char* name, bool fallback) {
  const char* raw = std::getenv(name);
  if (!raw) {
    return fallback;
  }
  std::string value(raw);
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
    return !std::isspace(ch);
  }));
  value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch) {
    return !std::isspace(ch);
  }).base(), value.end());
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

int env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (!raw) {
    return fallback;
  }
  try {
    return std::stoi(raw);
  } catch (...) {
    return fallback;
  }
}

std::string env_string(const char* name, const char* fallback) {
  const char* raw = std::getenv(name);
  if (!raw) {
    return std::string(fallback ? fallback : "");
  }
  return std::string(raw);
}

std::vector<std::string> env_csv(const char* name, const char* fallback) {
  std::string raw = env_string(name, fallback);
  std::vector<std::string> out;
  std::string current;
  for (char ch : raw) {
    if (ch == ',') {
      if (!current.empty()) {
        out.push_back(current);
      }
      current.clear();
    } else if (!std::isspace(static_cast<unsigned char>(ch))) {
      current.push_back(ch);
    }
  }
  if (!current.empty()) {
    out.push_back(current);
  }
  return out;
}

}  // namespace

Config load_from_env() {
  Config cfg;
  cfg.bots_enabled = env_bool("AGARIO_BOTS_ENABLED", true);
  cfg.bot_plugin_modules = env_csv("AGARIO_BOT_PLUGIN_MODULES", "agario.bot_plugins.core");
  cfg.bot_specs = env_string("AGARIO_BOT_SPECS", "solo_smart:16");
  cfg.bot_random_seed = env_int("AGARIO_BOT_RANDOM_SEED", 1337);
  cfg.bot_spawn_on_eaten = env_bool("AGARIO_BOT_SPAWN_ON_EATEN", true);
  cfg.bot_spawn_per_elimination = std::max(0, env_int("AGARIO_BOT_SPAWN_PER_ELIMINATION", 1));
  cfg.bot_max_active = std::max(1, env_int("AGARIO_BOT_MAX_ACTIVE", 40));
  cfg.bot_threads = std::max(0, env_int("AGARIO_BOT_THREADS", 0));

  cfg.player_colors = {
      "#21B8FF",
      "#33FF3A",
      "#FF364B",
      "#FFBC09",
      "#8E31FF",
      "#FF8A1F",
      "#26E5DF",
      "#FF2CCB",
  };

  cfg.food_colors = {
      "#FF2A40",
      "#1D38FF",
      "#22D9F0",
      "#59F12F",
      "#7D2BFF",
      "#FFE625",
      "#FF8D1F",
      "#FF1FCF",
  };

  return cfg;
}

const Config& get() {
  static Config cfg = load_from_env();
  return cfg;
}

}  // namespace agario::config
