#pragma once

#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

namespace agario {

template <typename T>
class SpatialHash {
 public:
  explicit SpatialHash(double cell_size) : cell_size_(cell_size) {}

  void clear() {
    buckets_.clear();
  }

  void insert(double x, double y, T* item) {
    buckets_[key(x, y)].push_back(item);
  }

  std::vector<T*> query_rect(double min_x, double min_y, double max_x, double max_y) const {
    int min_cx = static_cast<int>(std::floor(min_x / cell_size_));
    int max_cx = static_cast<int>(std::floor(max_x / cell_size_));
    int min_cy = static_cast<int>(std::floor(min_y / cell_size_));
    int max_cy = static_cast<int>(std::floor(max_y / cell_size_));

    std::vector<T*> hits;
    for (int cx = min_cx; cx <= max_cx; ++cx) {
      for (int cy = min_cy; cy <= max_cy; ++cy) {
        auto it = buckets_.find({cx, cy});
        if (it == buckets_.end()) {
          continue;
        }
        hits.insert(hits.end(), it->second.begin(), it->second.end());
      }
    }
    return hits;
  }

 private:
  using Key = std::pair<int, int>;

  struct KeyHash {
    std::size_t operator()(const Key& key) const noexcept {
      return static_cast<std::size_t>(key.first) * 1315423911u + static_cast<std::size_t>(key.second);
    }
  };

  Key key(double x, double y) const {
    return {static_cast<int>(std::floor(x / cell_size_)), static_cast<int>(std::floor(y / cell_size_))};
  }

  double cell_size_;
  std::unordered_map<Key, std::vector<T*>, KeyHash> buckets_;
};

}  // namespace agario
