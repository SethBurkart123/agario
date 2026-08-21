use rayon::prelude::*;

use super::{clamp, distance_sq, size_from_mass, unit_vec, CoreWorld};

#[derive(Clone, Copy)]
struct Action {
    player_index: usize,
    x: f64,
    y: f64,
    split: bool,
}

#[derive(Clone, Copy)]
struct Traits {
    aggression: f64,
    caution: f64,
    greed: f64,
    edge_bias: f64,
}

const TRAITS: [Traits; 4] = [
    Traits {
        aggression: 0.90,
        caution: 1.18,
        greed: 1.02,
        edge_bias: 1.00,
    },
    Traits {
        aggression: 1.26,
        caution: 0.88,
        greed: 0.90,
        edge_bias: 0.24,
    },
    Traits {
        aggression: 1.10,
        caution: 1.04,
        greed: 0.96,
        edge_bias: 0.52,
    },
    Traits {
        aggression: 0.90,
        caution: 1.08,
        greed: 1.28,
        edge_bias: 0.82,
    },
];

impl CoreWorld {
    /// Thinks for one third of the native bots per simulation tick. Targets
    /// persist between decisions, giving every bot an 8.3 Hz reaction rate
    /// while Rayon distributes that independent work across CPU cores.
    pub fn tick_native_bots(&mut self, now: f64) {
        let phase = ((now * 25.0) as u64) % 3;
        let actions: Vec<Action> = self
            .players
            .par_iter()
            .enumerate()
            .filter(|(_, player)| {
                player.is_bot && player.bot_plugin.is_some() && player.id % 3 == phase
            })
            .filter_map(|(player_index, player)| {
                let v2 = player.bot_plugin.as_deref() == Some("solo_smart_v2");
                self.native_bot_action(player_index, now, v2)
            })
            .collect();

        for action in actions {
            let player = &mut self.players[action.player_index];
            player.target_x = action.x;
            player.target_y = action.y;
            if action.split {
                player.split_requested = true;
            }
        }
    }

    fn native_bot_action(&self, player_index: usize, now: f64, v2: bool) -> Option<Action> {
        let player = &self.players[player_index];
        let largest = player
            .blobs
            .iter()
            .max_by(|a, b| a.mass.total_cmp(&b.mass))?;
        let smallest = player
            .blobs
            .iter()
            .min_by(|a, b| a.mass.total_cmp(&b.mass))?;
        let (cx, cy) = player.center();
        let traits = if v2 {
            TRAITS[(player.id as usize - 1) % TRAITS.len()]
        } else {
            Traits {
                aggression: 0.96,
                caution: 1.0,
                greed: 1.0,
                edge_bias: 0.55,
            }
        };
        let eat_ratio = self.cfg.eat_size_ratio * self.cfg.eat_size_ratio;

        let mut flee_x = 0.0;
        let mut flee_y = 0.0;
        let mut threat_pressure = 0.0;
        let mut closest_gap = f64::INFINITY;
        let mut prey: Option<(f64, f64, f64, u64)> = None;
        let mut prey_score = 0.0;

        for enemy in &self.players {
            if enemy.id == player.id {
                continue;
            }
            for blob in &enemy.blobs {
                let dx = cx - blob.x;
                let dy = cy - blob.y;
                let dist_sq = dx * dx + dy * dy;
                if blob.mass > smallest.mass * eat_ratio {
                    let range = 920.0 + blob.size() + smallest.size() * traits.caution;
                    if dist_sq < range * range {
                        let dist = dist_sq.sqrt().max(1.0);
                        let gap = dist - (blob.size() - smallest.size() / 3.0).max(0.0);
                        let pressure =
                            (1.0 - dist / range).max(0.0) * (blob.mass / smallest.mass).min(4.0);
                        flee_x += dx / dist * pressure;
                        flee_y += dy / dist * pressure;
                        threat_pressure += pressure;
                        closest_gap = closest_gap.min(gap);
                    }
                } else if largest.mass > blob.mass * eat_ratio {
                    let dist = dist_sq.sqrt();
                    if dist < 1500.0 {
                        let score = blob.mass.powf(0.78) * traits.aggression / (dist + 120.0);
                        if score > prey_score {
                            prey = Some((blob.x, blob.y, blob.mass, blob.id));
                            prey_score = score;
                        }
                    }
                }
            }
        }

        let poppable = largest.mass > self.cfg.virus_mass * eat_ratio
            && player.blobs.len() < self.cfg.max_player_blobs;
        if poppable {
            for virus in &self.viruses {
                let dx = cx - virus.x;
                let dy = cy - virus.y;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                let range = largest.size() + size_from_mass(virus.mass) + 260.0;
                if dist < range {
                    let pressure = (1.0 - dist / range) * 2.2 * traits.caution;
                    flee_x += dx / dist * pressure;
                    flee_y += dy / dist * pressure;
                    threat_pressure += pressure;
                }
            }
        }

        let edge = 520.0;
        if cx < edge {
            flee_x += (edge - cx) / edge * (1.2 - traits.edge_bias * 0.55);
        } else if self.cfg.world_width - cx < edge {
            flee_x -= (edge - (self.cfg.world_width - cx)) / edge * (1.2 - traits.edge_bias * 0.55);
        }
        if cy < edge {
            flee_y += (edge - cy) / edge * (1.2 - traits.edge_bias * 0.55);
        } else if self.cfg.world_height - cy < edge {
            flee_y -=
                (edge - (self.cfg.world_height - cy)) / edge * (1.2 - traits.edge_bias * 0.55);
        }

        if threat_pressure > 0.12 * traits.caution {
            let (ux, uy) = unit_vec(flee_x, flee_y);
            return Some(Action {
                player_index,
                x: clamp(cx + ux * 1000.0, 0.0, self.cfg.world_width),
                y: clamp(cy + uy * 1000.0, 0.0, self.cfg.world_height),
                split: false,
            });
        }

        if let Some((px, py, prey_mass, prey_id)) = prey {
            let distance = distance_sq(cx, cy, px, py).sqrt();
            let child_mass = largest.mass * 0.5;
            let child_radius = size_from_mass(child_mass);
            let split_reach =
                self.cfg.player_split_distance + self.cfg.player_split_boost + child_radius
                    - size_from_mass(prey_mass) / self.cfg.eat_overlap_divisor;
            let reward = prey_mass / largest.mass;
            let landing_distance = distance.min(650.0);
            let (ux, uy) = unit_vec(px - largest.x, py - largest.y);
            let landing_x = largest.x + ux * landing_distance;
            let landing_y = largest.y + uy * landing_distance;
            let intercepted = self.players.iter().any(|enemy| {
                enemy.id != player.id
                    && enemy.blobs.iter().any(|blob| {
                        blob.id != prey_id
                            && blob.mass > child_mass * eat_ratio
                            && distance_sq(blob.x, blob.y, landing_x, landing_y)
                                < (blob.size() + 220.0).powi(2)
                    })
            });
            let split = player.blobs.len() <= 4
                && largest.mass >= self.cfg.player_min_split_mass
                && child_mass > prey_mass * (eat_ratio + 0.04)
                && distance < split_reach
                && reward > 0.17 + traits.caution * 0.02 - traits.aggression * 0.025
                && now - player.last_split_at > 1.4
                && !intercepted;
            if reward > 0.035 / traits.aggression || distance < 440.0 {
                return Some(Action {
                    player_index,
                    x: px,
                    y: py,
                    split,
                });
            }
        }

        let view = 1300.0;
        let mut food_hits = Vec::new();
        self.food_grid
            .query_rect(cx - view, cy - view, cx + view, cy + view, &mut food_hits);
        let mut bins = [(0.0_f64, 0.0_f64, 0.0_f64); 16];
        let mut nearest: Option<(f64, f64, f64)> = None;
        for index in food_hits {
            let food = &self.foods[index];
            let dx = food.x - cx;
            let dy = food.y - cy;
            let distance = (dx * dx + dy * dy).sqrt();
            if distance > view || distance <= 1e-6 {
                continue;
            }
            let angle = dy.atan2(dx);
            let bin =
                (((angle + std::f64::consts::PI) / std::f64::consts::TAU * 16.0) as usize).min(15);
            let value = food.mass * traits.greed / (45.0 + distance * 0.12);
            bins[bin].0 += value;
            bins[bin].1 += food.x * value;
            bins[bin].2 += food.y * value;
            if nearest.is_none_or(|row| distance < row.0) {
                nearest = Some((distance, food.x, food.y));
            }
        }
        for item in &self.ejected {
            let dx = item.x - cx;
            let dy = item.y - cy;
            let distance = (dx * dx + dy * dy).sqrt();
            if distance > view || distance <= 1e-6 {
                continue;
            }
            let angle = dy.atan2(dx);
            let bin =
                (((angle + std::f64::consts::PI) / std::f64::consts::TAU * 16.0) as usize).min(15);
            let value = item.mass * 1.8 / (45.0 + distance * 0.12);
            bins[bin].0 += value;
            bins[bin].1 += item.x * value;
            bins[bin].2 += item.y * value;
        }
        if let Some(best) = bins
            .iter()
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .filter(|row| row.0 > 0.0)
        {
            let target_x = best.1 / best.0;
            let target_y = best.2 / best.0;
            let (ux, uy) = unit_vec(target_x - cx, target_y - cy);
            let corridor_mass: f64 = self
                .foods
                .iter()
                .filter_map(|food| {
                    let dx = food.x - cx;
                    let dy = food.y - cy;
                    let forward = dx * ux + dy * uy;
                    (forward > 0.0
                        && forward < 900.0
                        && (dx * uy - dy * ux).abs() < 120.0 + forward * 0.14)
                        .then_some(food.mass)
                })
                .sum();
            let split = v2
                && player.total_mass() < 340.0
                && player.blobs.len() < 4
                && largest.mass >= 52.0
                && corridor_mass > 9.0 / traits.greed
                && closest_gap.is_infinite()
                && now - player.last_split_at > 1.5;
            return Some(Action {
                player_index,
                x: clamp(cx + ux * 950.0, 0.0, self.cfg.world_width),
                y: clamp(cy + uy * 950.0, 0.0, self.cfg.world_height),
                split,
            });
        }

        let angle = player.id as f64 * 2.399_963 + (now * 0.12).floor() * 0.7;
        Some(Action {
            player_index,
            x: clamp(cx + angle.cos() * 900.0, 0.0, self.cfg.world_width),
            y: clamp(cy + angle.sin() * 900.0, 0.0, self.cfg.world_height),
            split: false,
        })
    }
}
