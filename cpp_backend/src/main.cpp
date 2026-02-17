#include "server.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage() {
  std::cout << "Usage: agario_server [--host HOST] [--port PORT] [--static-dir PATH] [--threads N]"
               " [--bots N] [--bot-specs SPEC] [--bot-threads N]\n";
}

void set_env(const char* key, const std::string& value) {
  setenv(key, value.c_str(), 1);
}

void set_env(const char* key, const char* value) {
  setenv(key, value, 1);
}

}  // namespace

int main(int argc, char** argv) {
  agario::server::ServerConfig config;
  bool bot_specs_set = false;
  bool bot_enabled_set = false;
  int bot_count = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    }
    if (arg == "--host" && i + 1 < argc) {
      config.host = argv[++i];
      continue;
    }
    if (arg == "--port" && i + 1 < argc) {
      config.port = std::atoi(argv[++i]);
      continue;
    }
    if (arg == "--static-dir" && i + 1 < argc) {
      config.static_dir = argv[++i];
      continue;
    }
    if (arg == "--threads" && i + 1 < argc) {
      config.threads = std::max(1, std::atoi(argv[++i]));
      continue;
    }
    if (arg == "--bots" && i + 1 < argc) {
      bot_count = std::atoi(argv[++i]);
      bot_specs_set = false;
      bot_enabled_set = true;
      continue;
    }
    if (arg == "--bot-specs" && i + 1 < argc) {
      set_env("AGARIO_BOT_SPECS", argv[++i]);
      bot_specs_set = true;
      continue;
    }
    if (arg == "--bot-threads" && i + 1 < argc) {
      set_env("AGARIO_BOT_THREADS", argv[++i]);
      continue;
    }
  }

  if (bot_enabled_set) {
    if (bot_count <= 0) {
      set_env("AGARIO_BOTS_ENABLED", "0");
    } else if (!bot_specs_set) {
      set_env("AGARIO_BOT_SPECS", "solo_smart:" + std::to_string(bot_count));
    }
  }

  return agario::server::run(config);
}
