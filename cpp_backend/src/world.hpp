#pragma once

#include <boost/json.hpp>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "config.hpp"
#include "models.hpp"
#include "spatial_hash.hpp"

namespace agario {

class GameWorld {
 public:
  explicit GameWorld(std::optional<int> seed = std::nullopt);

  Player& add_player(
      const std::string& player_name,
      double now,
      bool is_bot = false,
      const std::string& bot_plugin = {},
      const std::string& bot_team = {},
      const std::string& color = {});

  void remove_player(const std::string& player_id);

  void set_input(
      const std::string& player_id,
      std::optional<double> target_x,
      std::optional<double> target_y,
      bool split,
      bool eject);

  void update(double dt, double now);

  std::optional<boost::json::object> snapshot_for(const std::string& player_id) const;
  boost::json::object snapshot_overview() const;

  GameWorld fast_clone() const;

  std::unordered_map<std::string, Player> players;
  std::unordered_map<std::string, Food> foods;
  std::unordered_map<std::string, EjectedMass> ejected;
  std::unordered_map<std::string, Virus> viruses;

 private:
  std::mt19937 rng_;

  int player_id_counter_ = 1;
  int blob_id_counter_ = 1;
  int food_id_counter_ = 1;
  int ejected_id_counter_ = 1;
  int virus_id_counter_ = 1;

  SpatialHash<Food> food_hash_;
  SpatialHash<Blob> blob_hash_;
  SpatialHash<EjectedMass> ejected_hash_;

  std::string next_player_id();
  std::string next_blob_id();
  std::string next_food_id();
  std::string next_ejected_id();
  std::string next_virus_id();

  std::pair<double, double> random_spawn(double radius);
  void spawn_initial_food();
  void spawn_food();
  void spawn_initial_viruses();
  void spawn_virus();

  void respawn_eliminated_players(double now);
  void apply_actions(double now);
  void split_player(Player& player, double now);
  void eject_player_mass(Player& player, double now);
  void move_blobs(double dt, double now);
  void apply_same_player_softbody(Player& player, double now);
  void move_ejected(double dt);
  void rebuild_spatial_indexes();
  void resolve_blob_food_collisions();
  void resolve_blob_ejected_collisions();
  void resolve_blob_blob_collisions(double now);
  void resolve_virus_blob_collisions(double now);
  void explode_blob_into_player(Blob& blob, Player& player, double now);
  void spawn_food_to_target();

  std::tuple<std::vector<Blob*>, std::vector<Food*>, std::vector<EjectedMass*>, std::vector<Virus*>>
  visible_entities(double cx, double cy, double view_w, double view_h) const;

  boost::json::array leaderboard() const;
  boost::json::object snapshot_payload(
      const std::string& you,
      const std::string& player_name,
      double player_score,
      double camera_x,
      double camera_y,
      double camera_zoom,
      const std::vector<Blob*>& blobs,
      const std::vector<Food*>& foods,
      const std::vector<EjectedMass*>& ejected,
      const std::vector<Virus*>& viruses) const;
};

}  // namespace agario
