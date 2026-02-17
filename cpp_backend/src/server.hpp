#pragma once

#include <string>

namespace agario::server {

struct ServerConfig {
  std::string host = "0.0.0.0";
  int port = 8099;
  std::string static_dir = "static";
  int threads = 1;
};

int run(const ServerConfig& config);

}  // namespace agario::server
