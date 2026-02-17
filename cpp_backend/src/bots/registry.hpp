#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "types.hpp"

namespace agario::bots {

using BotFactory = std::function<std::unique_ptr<BotBrain>(const BotInitContext&)>;

class BotRegistry {
 public:
  void register_factory(const std::string& name, BotFactory factory);
  std::unique_ptr<BotBrain> create(const std::string& name, const BotInitContext& init_ctx) const;
  std::vector<std::string> names() const;

 private:
  std::unordered_map<std::string, BotFactory> factories_;
};

}  // namespace agario::bots
