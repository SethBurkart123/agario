#include "manager.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace agario::bots {
namespace {

std::string trim_copy(const std::string& input) {
  std::size_t start = input.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    return "";
  }
  std::size_t end = input.find_last_not_of(" \t\n\r");
  return input.substr(start, end - start + 1);
}

std::string lower_copy(const std::string& input) {
  std::string out = input;
  for (char& ch : out) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return out;
}

std::string title_case(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  bool start_word = true;
  for (char ch : input) {
    if (ch == '_') {
      out.push_back(' ');
      start_word = true;
      continue;
    }
    if (start_word) {
      out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
      start_word = false;
    } else {
      out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
  }
  return out;
}

}  // namespace

std::vector<BotSpec> parse_bot_specs(const std::string& raw) {
  std::vector<BotSpec> specs;
  std::string cleaned = trim_copy(raw);
  if (cleaned.empty()) {
    return specs;
  }

  std::stringstream list_stream(cleaned);
  std::string chunk;
  while (std::getline(list_stream, chunk, ',')) {
    std::string part = trim_copy(chunk);
    if (part.empty()) {
      continue;
    }
    std::stringstream spec_stream(part);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(spec_stream, token, ':')) {
      tokens.push_back(trim_copy(token));
    }
    if (tokens.empty() || tokens[0].empty()) {
      throw std::invalid_argument("Invalid bot spec '" + part + "': plugin name is required");
    }
    BotSpec spec;
    spec.plugin_name = lower_copy(tokens[0]);
    spec.count = 1;
    if (tokens.size() >= 2 && !tokens[1].empty()) {
      spec.count = std::stoi(tokens[1]);
    }
    if (tokens.size() >= 3 && !tokens[2].empty() && tokens[2] != "-") {
      spec.team_id = tokens[2];
    }
    if (tokens.size() >= 4 && !tokens[3].empty() && tokens[3] != "-") {
      spec.name_prefix = tokens[3];
    }
    if (spec.count <= 0) {
      throw std::invalid_argument("Invalid bot spec '" + part + "': count must be > 0");
    }
    specs.push_back(std::move(spec));
  }
  return specs;
}

BotManager::BotManager(GameWorld& world,
                       bool enabled,
                       std::vector<std::string> plugin_modules,
                       std::vector<BotSpec> bot_specs,
                       int seed,
                       int bot_threads)
    : world_(world),
      enabled_(enabled),
      plugin_modules_(std::move(plugin_modules)),
      bot_specs_(std::move(bot_specs)),
      rng_(static_cast<std::mt19937::result_type>(seed)),
      bot_threads_(bot_threads) {}

BotManager BotManager::from_config(GameWorld& world) {
  const auto& cfg = config::get();
  return BotManager(
      world,
      cfg.bots_enabled,
      cfg.bot_plugin_modules,
      parse_bot_specs(cfg.bot_specs),
      cfg.bot_random_seed,
      cfg.bot_threads);
}

boost::json::object BotManager::describe() const {
  const auto& cfg = config::get();
  boost::json::object payload;
  payload["enabled"] = enabled_;
  payload["started"] = started_;

  boost::json::array plugin_modules;
  for (const auto& mod : plugin_modules_) {
    plugin_modules.push_back(boost::json::value(mod));
  }
  payload["pluginModules"] = std::move(plugin_modules);

  boost::json::array registered;
  for (const auto& name : registry_.names()) {
    registered.push_back(boost::json::value(name));
  }
  payload["registeredPlugins"] = std::move(registered);

  boost::json::array specs;
  for (const auto& spec : bot_specs_) {
    boost::json::object row;
    row["plugin"] = spec.plugin_name;
    row["count"] = spec.count;
    row["team"] = spec.team_id.empty() ? boost::json::value(nullptr) : boost::json::value(spec.team_id);
    row["namePrefix"] = spec.name_prefix.empty() ? boost::json::value(nullptr) : boost::json::value(spec.name_prefix);
    specs.push_back(std::move(row));
  }
  payload["botSpecs"] = std::move(specs);
  payload["activeBots"] = static_cast<int>(agents_.size());
  payload["spawnOnEaten"] = cfg.bot_spawn_on_eaten;
  payload["spawnPerElimination"] = cfg.bot_spawn_per_elimination;
  payload["maxActiveBots"] = cfg.bot_max_active;
  return payload;
}

void BotManager::ensure_started(double now) {
  if (started_ || !enabled_) {
    return;
  }
  register_core_plugins(registry_);
  for (const auto& spec : bot_specs_) {
    for (int i = 0; i < spec.count; ++i) {
      std::string name_prefix = !spec.name_prefix.empty() ? spec.name_prefix : title_case(spec.plugin_name);
      std::string player_name = build_name(spec, i);
      spawn_bot(spec.plugin_name, spec.team_id, name_prefix, now, player_name);
    }
  }
  started_ = true;
}

void BotManager::tick(double dt, double now) {
  if (!started_ || agents_.empty()) {
    return;
  }

  auto views = build_views();
  if (views.players_by_id.empty()) {
    return;
  }

  std::vector<SpawnRequest> pending_spawns;

  std::unordered_map<std::string, std::vector<std::string>> team_members;
  for (const auto& entry : agents_) {
    const auto& agent = entry.second;
    std::string key = team_key(agent.plugin_name, agent.team_id, agent.player_id);
    team_members[key].push_back(agent.player_id);
  }
  for (auto& entry : team_members) {
    auto& members = entry.second;
    members.erase(std::remove_if(members.begin(), members.end(), [&](const std::string& pid) {
      return views.players_by_id.find(pid) == views.players_by_id.end();
    }), members.end());
    std::sort(members.begin(), members.end());
    auto& team_ptr = team_state_[entry.first];
    if (!team_ptr) {
      team_ptr = std::make_shared<TeamData>();
    }
    auto& team = *team_ptr;
    std::lock_guard<std::mutex> guard(team.mutex);
    team.state.set("members", members);
  }

  std::vector<BotAgent*> active_agents;
  std::vector<PlayerView> active_views;
  std::vector<TeamData*> team_refs;
  active_agents.reserve(agents_.size());
  active_views.reserve(agents_.size());
  team_refs.reserve(agents_.size());

  for (auto it = agents_.begin(); it != agents_.end();) {
    auto& agent = it->second;
    auto player_it = views.players_by_id.find(agent.player_id);
    if (player_it == views.players_by_id.end()) {
      it = agents_.erase(it);
      continue;
    }
    const PlayerView& me = player_it->second;

    bool was_alive = agent.memory.get_or<bool>("_was_alive", !me.blobs.empty());
    bool is_alive = !me.blobs.empty();
    if (!is_alive && was_alive) {
      pending_spawns.push_back({agent.plugin_name, agent.team_id, agent.name_prefix});
    }
    agent.memory.set("_was_alive", is_alive);

    active_agents.push_back(&agent);
    active_views.push_back(me);
    std::string key = team_key(agent.plugin_name, agent.team_id, agent.player_id);
    auto team_it = team_state_.find(key);
    if (team_it == team_state_.end() || !team_it->second) {
      auto& team_ptr = team_state_[key];
      if (!team_ptr) {
        team_ptr = std::make_shared<TeamData>();
      }
      team_refs.push_back(team_ptr.get());
    } else {
      team_refs.push_back(team_it->second.get());
    }
    ++it;
  }

  struct ActionResult {
    std::string player_id;
    BotAction action;
  };

  std::vector<ActionResult> results(active_agents.size());
  int thread_count = bot_threads_ > 0 ? bot_threads_ : static_cast<int>(std::thread::hardware_concurrency());
  if (thread_count <= 0) {
    thread_count = 1;
  }
  if (thread_count > static_cast<int>(active_agents.size())) {
    thread_count = static_cast<int>(active_agents.size());
  }
  if (thread_count <= 1) {
    thread_count = 1;
  }

  auto worker = [&](std::size_t start, std::size_t end) {
    for (std::size_t idx = start; idx < end; ++idx) {
      BotAgent* agent = active_agents[idx];
      const PlayerView& me = active_views[idx];
      TeamData* team = team_refs[idx];
      if (!team) {
        results[idx] = {agent->player_id, fallback_action(me)};
        continue;
      }

      BotContext ctx;
      ctx.now = now;
      ctx.dt = dt;
      ctx.world_width = config::get().world_width;
      ctx.world_height = config::get().world_height;
      ctx.me = me;
      ctx.players = views.players;
      ctx.foods = views.foods;
      ctx.ejected = views.ejected;
      ctx.viruses = views.viruses;
      ctx.team_state = &team->state;
      ctx.memory = &agent->memory;

      BotAction action;
      try {
        std::lock_guard<std::mutex> guard(team->mutex);
        action = agent->brain->decide(ctx);
      } catch (...) {
        action = fallback_action(me);
      }

      results[idx] = {agent->player_id, action};
    }
  };

  if (thread_count == 1) {
    worker(0, active_agents.size());
  } else {
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(thread_count));
    std::size_t total = active_agents.size();
    std::size_t chunk = (total + static_cast<std::size_t>(thread_count) - 1) / static_cast<std::size_t>(thread_count);
    std::size_t start = 0;
    for (int i = 0; i < thread_count; ++i) {
      std::size_t end = std::min(total, start + chunk);
      if (start >= end) {
        break;
      }
      threads.emplace_back(worker, start, end);
      start = end;
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  for (const auto& result : results) {
    world_.set_input(
        result.player_id,
        clamp(result.action.target_x, 0.0, config::get().world_width),
        clamp(result.action.target_y, 0.0, config::get().world_height),
        result.action.split,
        result.action.eject);
  }

  for (const auto& request : pending_spawns) {
    spawn_extra_on_elimination(request, now);
  }
}

BotManager::ViewPack BotManager::build_views() const {
  ViewPack pack;
  for (const auto& pkv : world_.players) {
    const Player& player = pkv.second;
    PlayerView view;
    view.id = player.id;
    view.name = player.name;
    view.color = player.color;
    view.is_bot = player.is_bot;
    view.plugin_name = player.bot_plugin;
    view.team_id = player.bot_team;
    view.total_mass = 0.0;
    for (const auto& bkv : player.blobs) {
      const Blob& blob = bkv.second;
      BlobView b;
      b.id = blob.id;
      b.player_id = blob.player_id;
      b.x = blob.x;
      b.y = blob.y;
      b.mass = blob.mass;
      b.radius = blob.radius();
      view.total_mass += b.mass;
      view.blobs.push_back(std::move(b));
    }
    pack.players.push_back(std::move(view));
  }
  std::sort(pack.players.begin(), pack.players.end(), [](const PlayerView& a, const PlayerView& b) {
    return a.id < b.id;
  });
  for (const auto& player : pack.players) {
    pack.players_by_id[player.id] = player;
  }

  for (const auto& fkv : world_.foods) {
    const Food& food = fkv.second;
    FoodView fv;
    fv.id = food.id;
    fv.x = food.x;
    fv.y = food.y;
    fv.mass = food.mass;
    fv.radius = food.radius();
    fv.color = food.color;
    pack.foods.push_back(std::move(fv));
  }

  for (const auto& ekv : world_.ejected) {
    const EjectedMass& item = ekv.second;
    EjectedView ev;
    ev.id = item.id;
    ev.x = item.x;
    ev.y = item.y;
    ev.mass = item.mass;
    ev.radius = item.radius();
    ev.owner_id = item.owner_id;
    ev.ttl = item.ttl;
    pack.ejected.push_back(std::move(ev));
  }

  for (const auto& vkv : world_.viruses) {
    const Virus& virus = vkv.second;
    VirusView vv;
    vv.id = virus.id;
    vv.x = virus.x;
    vv.y = virus.y;
    vv.mass = virus.mass;
    vv.radius = virus.radius();
    pack.viruses.push_back(std::move(vv));
  }

  return pack;
}

BotAction BotManager::fallback_action(const PlayerView& me) const {
  if (!me.blobs.empty()) {
    const auto& first = me.blobs.front();
    return {first.x, first.y, false, false};
  }
  return {config::get().world_width * 0.5, config::get().world_height * 0.5, false, false};
}

std::string BotManager::build_name(const BotSpec& spec, int index) const {
  std::string prefix = !spec.name_prefix.empty() ? spec.name_prefix : title_case(spec.plugin_name);
  return prefix + "-" + std::to_string(index + 1);
}

std::string BotManager::spawn_bot(const std::string& plugin_name,
                                  const std::string& team_id,
                                  const std::string& name_prefix,
                                  double now,
                                  const std::string& explicit_name) {
  spawn_index_ += 1;
  int sequence = spawn_index_;
  std::string bot_name = !explicit_name.empty() ? explicit_name : name_prefix + "-" + std::to_string(sequence);
  Player& player = world_.add_player(bot_name, now, true, plugin_name, team_id);

  std::uniform_int_distribution<int> seed_dist(0, 2'000'000'000);
  BotInitContext init_ctx;
  init_ctx.plugin_name = plugin_name;
  init_ctx.bot_name = player.name;
  init_ctx.team_id = team_id;
  init_ctx.bot_index = sequence;
  init_ctx.rng = std::mt19937(static_cast<std::mt19937::result_type>(seed_dist(rng_)));

  std::unique_ptr<BotBrain> brain;
  try {
    brain = registry_.create(plugin_name, init_ctx);
  } catch (...) {
    world_.remove_player(player.id);
    throw;
  }
  BotAgent agent;
  agent.player_id = player.id;
  agent.plugin_name = plugin_name;
  agent.team_id = team_id;
  agent.name_prefix = name_prefix;
  agent.brain = std::move(brain);
  agent.memory.set("_was_alive", true);
  agents_[player.id] = std::move(agent);
  return player.id;
}

void BotManager::spawn_extra_on_elimination(const SpawnRequest& eliminated, double now) {
  const auto& cfg = config::get();
  if (!cfg.bot_spawn_on_eaten || cfg.bot_spawn_per_elimination <= 0) {
    return;
  }
  int max_active = std::max(1, cfg.bot_max_active);
  int spawned = 0;
  for (int i = 0; i < cfg.bot_spawn_per_elimination; ++i) {
    if (static_cast<int>(agents_.size()) >= max_active) {
      break;
    }
    try {
      spawn_bot(eliminated.plugin_name, eliminated.team_id, eliminated.name_prefix, now, "");
      spawned += 1;
    } catch (...) {
      break;
    }
  }
  (void)spawned;
}

std::string BotManager::team_key(const std::string& plugin_name, const std::string& team_id, const std::string& player_id) const {
  if (team_id.empty()) {
    return plugin_name + ":" + player_id;
  }
  return plugin_name + ":" + team_id;
}

double BotManager::clamp(double value, double min_value, double max_value) {
  return std::min(std::max(value, min_value), max_value);
}

}  // namespace agario::bots
