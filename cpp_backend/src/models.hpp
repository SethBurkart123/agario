#pragma once

#include <cmath>
#include <string>
#include <unordered_map>

#include "config.hpp"

namespace agario {

struct Blob {
  std::string id;
  std::string player_id;
  double x = 0.0;
  double y = 0.0;
  double mass = 0.0;
  double vx = 0.0;
  double vy = 0.0;
  double can_merge_at = 0.0;

  double radius() const {
    return std::sqrt(std::max(0.0, mass)) * config::get().blob_radius_factor;
  }
};

struct Player {
  std::string id;
  std::string name;
  std::string color;
  bool is_bot = false;
  std::string bot_plugin;
  std::string bot_team;
  std::unordered_map<std::string, Blob> blobs;
  double target_x = 0.0;
  double target_y = 0.0;
  bool split_requested = false;
  bool eject_requested = false;
  double last_split_at = -1e9;
  double last_eject_at = -1e9;

  double total_mass() const {
    double total = 0.0;
    for (const auto& kv : blobs) {
      total += kv.second.mass;
    }
    return total;
  }

  bool is_alive() const {
    return !blobs.empty();
  }

  std::pair<double, double> center() const {
    if (blobs.empty()) {
      return {0.0, 0.0};
    }
    double total = total_mass();
    if (total <= 0.0) {
      const auto& blob = blobs.begin()->second;
      return {blob.x, blob.y};
    }
    double cx = 0.0;
    double cy = 0.0;
    for (const auto& kv : blobs) {
      cx += kv.second.x * kv.second.mass;
      cy += kv.second.y * kv.second.mass;
    }
    return {cx / total, cy / total};
  }
};

struct Food {
  std::string id;
  double x = 0.0;
  double y = 0.0;
  double mass = 0.0;
  std::string color;

  double radius() const {
    return std::sqrt(std::max(0.0, mass)) * config::get().food_radius_factor;
  }
};

struct EjectedMass {
  std::string id;
  double x = 0.0;
  double y = 0.0;
  double mass = 0.0;
  std::string owner_id;
  double vx = 0.0;
  double vy = 0.0;
  double ttl = 0.0;

  double radius() const {
    return std::sqrt(std::max(0.0, mass)) * config::get().blob_radius_factor;
  }
};

struct Virus {
  std::string id;
  double x = 0.0;
  double y = 0.0;
  double mass = 0.0;

  double radius() const {
    return std::sqrt(std::max(0.0, mass)) * config::get().virus_radius_factor;
  }
};

}  // namespace agario
