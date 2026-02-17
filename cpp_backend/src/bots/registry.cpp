#include "registry.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace agario::bots {
namespace {

std::string normalize(const std::string& name) {
  std::string out;
  out.reserve(name.size());
  for (char ch : name) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
      continue;
    }
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return out;
}

}  // namespace

void BotRegistry::register_factory(const std::string& name, BotFactory factory) {
  std::string key = normalize(name);
  if (key.empty()) {
    throw std::invalid_argument("Bot plugin name cannot be empty");
  }
  if (factories_.find(key) != factories_.end()) {
    throw std::invalid_argument("Duplicate bot plugin registration: " + key);
  }
  factories_[key] = std::move(factory);
}

std::unique_ptr<BotBrain> BotRegistry::create(const std::string& name, const BotInitContext& init_ctx) const {
  std::string key = normalize(name);
  auto it = factories_.find(key);
  if (it == factories_.end()) {
    std::string available;
    for (const auto& entry : factories_) {
      if (!available.empty()) {
        available += ", ";
      }
      available += entry.first;
    }
    if (available.empty()) {
      available = "<none>";
    }
    throw std::invalid_argument("Unknown bot plugin '" + key + "'. Available: " + available);
  }
  return it->second(init_ctx);
}

std::vector<std::string> BotRegistry::names() const {
  std::vector<std::string> out;
  out.reserve(factories_.size());
  for (const auto& entry : factories_) {
    out.push_back(entry.first);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace agario::bots
