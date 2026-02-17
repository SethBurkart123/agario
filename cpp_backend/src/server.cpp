#include "server.hpp"

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/json.hpp>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "bots/manager.hpp"
#include "config.hpp"
#include "world.hpp"

namespace agario::server {
namespace {

namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
using tcp = net::ip::tcp;

std::string mime_type(const std::string& path) {
  std::string ext;
  auto dot = path.find_last_of('.');
  if (dot != std::string::npos) {
    ext = path.substr(dot);
  }
  if (ext == ".htm" || ext == ".html") return "text/html";
  if (ext == ".css") return "text/css";
  if (ext == ".js") return "application/javascript";
  if (ext == ".json") return "application/json";
  if (ext == ".png") return "image/png";
  if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
  if (ext == ".svg") return "image/svg+xml";
  return "application/octet-stream";
}

bool read_file(const std::string& path, std::string& out) {
  std::ifstream file(path.c_str(), std::ios::binary);
  if (!file) {
    return false;
  }
  file.seekg(0, std::ios::end);
  std::streampos size = file.tellg();
  if (size > 0) {
    out.resize(static_cast<std::size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(out.data(), static_cast<std::streamsize>(size));
  }
  return true;
}

std::optional<double> as_double(const boost::json::value& value) {
  if (value.is_double()) {
    return value.as_double();
  }
  if (value.is_int64()) {
    return static_cast<double>(value.as_int64());
  }
  if (value.is_uint64()) {
    return static_cast<double>(value.as_uint64());
  }
  return std::nullopt;
}

struct ConnectResult {
  boost::json::object welcome;
  std::string player_id;
  std::string spectator_id;
};

class RealtimeServer;

class WebSocketSession : public std::enable_shared_from_this<WebSocketSession> {
 public:
  WebSocketSession(tcp::socket&& socket, std::shared_ptr<RealtimeServer> server);

  void run(http::request<http::string_body> req);
  void send_json(const boost::json::value& value);

 private:
  void do_read();
  void on_read(beast::error_code ec, std::size_t bytes);
  void do_write();
  void on_write(beast::error_code ec);
  void close();
  void on_close();

  websocket::stream<beast::tcp_stream> ws_;
  std::shared_ptr<RealtimeServer> server_;
  net::strand<net::io_context::executor_type> strand_;
  beast::flat_buffer buffer_;
  std::deque<std::string> send_queue_;
  bool joined_ = false;
  bool closed_ = false;
  std::string player_id_;
  std::string spectator_id_;
};

class RealtimeServer : public std::enable_shared_from_this<RealtimeServer> {
 public:
  RealtimeServer(net::io_context& ioc, std::string static_dir)
      : ioc_(ioc),
        timer_(ioc),
        static_dir_(std::move(static_dir)),
        world_(),
        bot_manager_(bots::BotManager::from_config(world_)) {
    start_time_ = std::chrono::steady_clock::now();
  }

  void start() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (running_) {
      return;
    }
    running_ = true;
    last_tick_ = std::chrono::steady_clock::now();
    bot_manager_.ensure_started(now_seconds());
    schedule_tick();
  }

  void stop() {
    std::lock_guard<std::mutex> guard(mutex_);
    running_ = false;
    timer_.cancel();
  }

  const std::string& static_dir() const {
    return static_dir_;
  }

  ConnectResult connect(const std::shared_ptr<WebSocketSession>& session, const std::string& name, bool spectator) {
    std::lock_guard<std::mutex> guard(mutex_);
    ConnectResult result;
    if (spectator) {
      std::string spectator_id = "s" + std::to_string(++spectator_id_counter_);
      spectators_[spectator_id] = session;
      result.spectator_id = spectator_id;
      result.welcome["type"] = "welcome";
      result.welcome["spectator"] = true;
      result.welcome["spectatorId"] = spectator_id;
      result.welcome["tickRate"] = config::get().tick_rate;
      result.welcome["inputHz"] = config::get().input_hz;
      boost::json::object world;
      world["w"] = config::get().world_width;
      world["h"] = config::get().world_height;
      result.welcome["world"] = std::move(world);
      return result;
    }

    Player& player = world_.add_player(name, now_seconds());
    connections_[player.id] = session;
    connection_overview_mode_[player.id] = false;

    result.player_id = player.id;
    result.welcome["type"] = "welcome";
    result.welcome["playerId"] = player.id;
    result.welcome["name"] = player.name;
    result.welcome["spectator"] = false;
    result.welcome["tickRate"] = config::get().tick_rate;
    result.welcome["inputHz"] = config::get().input_hz;
    boost::json::object world;
    world["w"] = config::get().world_width;
    world["h"] = config::get().world_height;
    result.welcome["world"] = std::move(world);
    return result;
  }

  void disconnect(const std::string& player_id, const std::string& spectator_id) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!player_id.empty()) {
      connections_.erase(player_id);
      connection_overview_mode_.erase(player_id);
      world_.remove_player(player_id);
    }
    if (!spectator_id.empty()) {
      spectators_.erase(spectator_id);
    }
  }

  void handle_input(const std::string& player_id, const boost::json::object& payload) {
    std::optional<double> tx;
    std::optional<double> ty;
    auto target_it = payload.find("target");
    if (target_it != payload.end() && target_it->value().is_object()) {
      const auto& target = target_it->value().as_object();
      auto x_it = target.find("x");
      auto y_it = target.find("y");
      if (x_it != target.end()) {
        tx = as_double(x_it->value());
      }
      if (y_it != target.end()) {
        ty = as_double(y_it->value());
      }
    }

    bool split = false;
    bool eject = false;
    auto split_it = payload.find("split");
    if (split_it != payload.end() && split_it->value().is_bool()) {
      split = split_it->value().as_bool();
    }
    auto eject_it = payload.find("eject");
    if (eject_it != payload.end() && eject_it->value().is_bool()) {
      eject = eject_it->value().as_bool();
    }

    std::lock_guard<std::mutex> guard(mutex_);
    world_.set_input(player_id, tx, ty, split, eject);
  }

  void set_view_mode(const std::string& player_id, bool overview) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (connections_.find(player_id) != connections_.end()) {
      connection_overview_mode_[player_id] = overview;
    }
  }

  boost::json::object bot_status() {
    std::lock_guard<std::mutex> guard(mutex_);
    return bot_manager_.describe();
  }

 private:
  struct SnapshotTarget {
    std::shared_ptr<WebSocketSession> session;
    boost::json::object payload;
    std::string player_id;
    std::string spectator_id;
  };

  void schedule_tick() {
    const double interval = 1.0 / config::get().tick_rate;
    auto wait = std::chrono::duration_cast<net::steady_timer::duration>(std::chrono::duration<double>(interval));
    timer_.expires_after(wait);
    timer_.async_wait([self = shared_from_this()](beast::error_code ec) {
      if (!ec) {
        self->on_tick();
      }
    });
  }

  void on_tick() {
    auto tick_start = std::chrono::steady_clock::now();
    double dt = std::min(0.1, std::chrono::duration<double>(tick_start - last_tick_).count());
    last_tick_ = tick_start;
    double now = now_seconds();

    std::vector<SnapshotTarget> snapshots;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      if (!running_) {
        return;
      }
      bot_manager_.tick(dt, now);
      world_.update(dt, now);

      std::optional<boost::json::object> overview_snapshot;

      for (auto it = connections_.begin(); it != connections_.end();) {
        auto session = it->second.lock();
        if (!session) {
          connection_overview_mode_.erase(it->first);
          it = connections_.erase(it);
          continue;
        }

        bool overview = false;
        auto mode_it = connection_overview_mode_.find(it->first);
        if (mode_it != connection_overview_mode_.end()) {
          overview = mode_it->second;
        }

        boost::json::object payload;
        if (overview) {
          if (!overview_snapshot) {
            overview_snapshot = world_.snapshot_overview();
          }
          payload = *overview_snapshot;
        } else {
          auto snapshot = world_.snapshot_for(it->first);
          if (!snapshot) {
            ++it;
            continue;
          }
          payload = *snapshot;
        }

        snapshots.push_back({session, payload, it->first, ""});
        ++it;
      }

      for (auto it = spectators_.begin(); it != spectators_.end();) {
        auto session = it->second.lock();
        if (!session) {
          it = spectators_.erase(it);
          continue;
        }
        if (!overview_snapshot) {
          overview_snapshot = world_.snapshot_overview();
        }
        snapshots.push_back({session, *overview_snapshot, "", it->first});
        ++it;
      }
    }

    for (auto& target : snapshots) {
      if (target.session) {
        target.session->send_json(target.payload);
      }
    }

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - tick_start).count();
    double interval = 1.0 / config::get().tick_rate;
    double sleep = std::max(0.0, interval - elapsed);
    auto wait = std::chrono::duration_cast<net::steady_timer::duration>(std::chrono::duration<double>(sleep));
    timer_.expires_after(wait);
    timer_.async_wait([self = shared_from_this()](beast::error_code ec) {
      if (!ec) {
        self->on_tick();
      }
    });
  }

  double now_seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time_).count();
  }

  net::io_context& ioc_;
  net::steady_timer timer_;
  std::string static_dir_;
  GameWorld world_;
  bots::BotManager bot_manager_;
  std::unordered_map<std::string, std::weak_ptr<WebSocketSession>> connections_;
  std::unordered_map<std::string, bool> connection_overview_mode_;
  std::unordered_map<std::string, std::weak_ptr<WebSocketSession>> spectators_;
  int spectator_id_counter_ = 0;
  std::mutex mutex_;
  bool running_ = false;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_tick_ = std::chrono::steady_clock::now();
};

WebSocketSession::WebSocketSession(tcp::socket&& socket, std::shared_ptr<RealtimeServer> server)
    : ws_(std::move(socket)),
      server_(std::move(server)),
      strand_(net::make_strand(static_cast<net::io_context&>(ws_.get_executor().context()))) {}

void WebSocketSession::run(http::request<http::string_body> req) {
  ws_.set_option(websocket::stream_base::timeout::suggested(beast::role_type::server));
  ws_.set_option(websocket::stream_base::decorator([](websocket::response_type& res) {
    res.set(http::field::server, "agario-cpp");
  }));
  ws_.async_accept(req, net::bind_executor(strand_,
                                          [self = shared_from_this()](beast::error_code ec) {
                                            if (ec) {
                                              return;
                                            }
                                            self->do_read();
                                          }));
}

void WebSocketSession::send_json(const boost::json::value& value) {
  auto message = boost::json::serialize(value);
  net::post(strand_, [self = shared_from_this(), msg = std::move(message)]() mutable {
    bool writing = !self->send_queue_.empty();
    self->send_queue_.push_back(std::move(msg));
    if (!writing) {
      self->do_write();
    }
  });
}

void WebSocketSession::do_read() {
  ws_.async_read(buffer_, net::bind_executor(strand_,
                                            [self = shared_from_this()](beast::error_code ec, std::size_t bytes) {
                                              self->on_read(ec, bytes);
                                            }));
}

void WebSocketSession::on_read(beast::error_code ec, std::size_t) {
  if (ec == websocket::error::closed) {
    on_close();
    return;
  }
  if (ec) {
    on_close();
    return;
  }

  std::string data = beast::buffers_to_string(buffer_.data());
  buffer_.consume(buffer_.size());

  boost::system::error_code jec;
  boost::json::value value = boost::json::parse(data, jec);
  if (jec || !value.is_object()) {
    close();
    return;
  }
  auto payload = value.as_object();
  auto type_it = payload.find("type");
  if (type_it == payload.end() || !type_it->value().is_string()) {
    close();
    return;
  }
  std::string type = type_it->value().as_string().c_str();

  if (!joined_) {
    if (type != "join") {
      close();
      return;
    }
    std::string name = "Cell";
    auto name_it = payload.find("name");
    if (name_it != payload.end() && name_it->value().is_string()) {
      name = name_it->value().as_string().c_str();
    }
    bool spectator = false;
    auto spec_it = payload.find("spectator");
    if (spec_it != payload.end() && spec_it->value().is_bool()) {
      spectator = spec_it->value().as_bool();
    }
    auto result = server_->connect(shared_from_this(), name, spectator);
    player_id_ = result.player_id;
    spectator_id_ = result.spectator_id;
    joined_ = true;
    send_json(result.welcome);
    do_read();
    return;
  }

  if (type == "ping") {
    boost::json::object pong;
    pong["type"] = "pong";
    auto ts_it = payload.find("ts");
    if (ts_it != payload.end()) {
      pong["ts"] = ts_it->value();
    }
    send_json(pong);
    do_read();
    return;
  }

  if (type == "view_mode" && !player_id_.empty()) {
    bool overview = false;
    auto ov_it = payload.find("overview");
    if (ov_it != payload.end() && ov_it->value().is_bool()) {
      overview = ov_it->value().as_bool();
    }
    server_->set_view_mode(player_id_, overview);
    do_read();
    return;
  }

  if (type == "input" && !player_id_.empty()) {
    server_->handle_input(player_id_, payload);
    do_read();
    return;
  }

  do_read();
}

void WebSocketSession::do_write() {
  ws_.async_write(net::buffer(send_queue_.front()), net::bind_executor(strand_,
                                                                     [self = shared_from_this()](beast::error_code ec,
                                                                                                 std::size_t) {
                                                                       self->on_write(ec);
                                                                     }));
}

void WebSocketSession::on_write(beast::error_code ec) {
  if (ec) {
    on_close();
    return;
  }
  send_queue_.pop_front();
  if (!send_queue_.empty()) {
    do_write();
  }
}

void WebSocketSession::close() {
  beast::error_code ec;
  ws_.close(websocket::close_code::normal, ec);
  on_close();
}

void WebSocketSession::on_close() {
  if (closed_) {
    return;
  }
  closed_ = true;
  if (server_) {
    server_->disconnect(player_id_, spectator_id_);
  }
}

class HttpSession : public std::enable_shared_from_this<HttpSession> {
 public:
  HttpSession(tcp::socket&& socket, std::shared_ptr<RealtimeServer> server)
      : stream_(std::move(socket)),
        server_(std::move(server)) {}

  void run() {
    do_read();
  }

 private:
  void do_read() {
    req_ = {};
    http::async_read(stream_, buffer_, req_,
                     [self = shared_from_this()](beast::error_code ec, std::size_t bytes) {
                       self->on_read(ec, bytes);
                     });
  }

  void on_read(beast::error_code ec, std::size_t) {
    if (ec == http::error::end_of_stream) {
      return do_close();
    }
    if (ec) {
      return;
    }

    if (websocket::is_upgrade(req_) && req_.target() == "/ws") {
      auto session = std::make_shared<WebSocketSession>(stream_.release_socket(), server_);
      return session->run(std::move(req_));
    }

    handle_request();
  }

  void handle_request() {
    if (req_.method() != http::verb::get) {
      send_response(http::status::method_not_allowed, "Method not allowed", "text/plain");
      return;
    }

    std::string target = std::string(req_.target());
    if (target == "/" || target == "/overview") {
      serve_static_file("index.html");
      return;
    }
    if (target == "/api/bots") {
      boost::json::object payload = server_->bot_status();
      send_response(http::status::ok, boost::json::serialize(payload), "application/json");
      return;
    }
    if (target.rfind("/static/", 0) == 0) {
      std::string rel = target.substr(std::string("/static/").size());
      serve_static_file(rel);
      return;
    }

    send_response(http::status::not_found, "Not found", "text/plain");
  }

  void serve_static_file(const std::string& rel_path) {
    if (rel_path.find("..") != std::string::npos) {
      send_response(http::status::bad_request, "Invalid path", "text/plain");
      return;
    }
    std::filesystem::path base(server_->static_dir().c_str());
    std::filesystem::path file = base / rel_path;
    std::string content;
    if (!read_file(file.string(), content)) {
      send_response(http::status::not_found, "Not found", "text/plain");
      return;
    }
    send_response(http::status::ok, content, mime_type(file.string()));
  }

  void send_response(http::status status, const std::string& body, const std::string& content_type) {
    auto res = std::make_shared<http::response<http::string_body>>(status, req_.version());
    res->set(http::field::server, "agario-cpp");
    res->set(http::field::content_type, content_type);
    res->keep_alive(req_.keep_alive());
    res->body() = body;
    res->prepare_payload();
    http::async_write(stream_, *res,
                      [self = shared_from_this(), res](beast::error_code ec, std::size_t) {
                        if (ec) {
                          return;
                        }
                        if (!self->req_.keep_alive()) {
                          self->do_close();
                        } else {
                          self->do_read();
                        }
                      });
  }

  void do_close() {
    beast::error_code ec;
    stream_.socket().shutdown(tcp::socket::shutdown_send, ec);
  }

  beast::tcp_stream stream_;
  beast::flat_buffer buffer_;
  http::request<http::string_body> req_;
  std::shared_ptr<RealtimeServer> server_;
};

class Listener : public std::enable_shared_from_this<Listener> {
 public:
  Listener(net::io_context& ioc, tcp::endpoint endpoint, std::shared_ptr<RealtimeServer> server)
      : acceptor_(ioc),
        socket_(ioc),
        server_(std::move(server)) {
    beast::error_code ec;
    acceptor_.open(endpoint.protocol(), ec);
    acceptor_.set_option(net::socket_base::reuse_address(true), ec);
    acceptor_.bind(endpoint, ec);
    acceptor_.listen(net::socket_base::max_listen_connections, ec);
  }

  void run() {
    do_accept();
  }

 private:
  void do_accept() {
    acceptor_.async_accept(socket_, [self = shared_from_this()](beast::error_code ec) {
      if (!ec) {
        std::make_shared<HttpSession>(std::move(self->socket_), self->server_)->run();
      }
      self->do_accept();
    });
  }

  tcp::acceptor acceptor_;
  tcp::socket socket_;
  std::shared_ptr<RealtimeServer> server_;
};

}  // namespace

int run(const ServerConfig& config) {
  net::io_context ioc{config.threads};
  auto server = std::make_shared<RealtimeServer>(ioc, config.static_dir);
  server->start();

  beast::error_code ec;
  auto address = net::ip::make_address(config.host, ec);
  if (ec) {
    return 1;
  }
  tcp::endpoint endpoint{address, static_cast<unsigned short>(config.port)};
  std::make_shared<Listener>(ioc, endpoint, server)->run();

  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(config.threads > 0 ? config.threads - 1 : 0));
  for (int i = 1; i < config.threads; ++i) {
    threads.emplace_back([&ioc]() { ioc.run(); });
  }
  ioc.run();
  for (auto& t : threads) {
    t.join();
  }
  return 0;
}

}  // namespace agario::server
