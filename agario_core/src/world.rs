//! Authoritative Agar.io-style world simulation shared by live play and RL.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::config::{WorldConfig, TICK_RATE};
use crate::rng::PyMt19937;

fn clamp(value: f64, min_value: f64, max_value: f64) -> f64 {
    value.max(min_value).min(max_value)
}

fn distance_sq(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    dx * dx + dy * dy
}

fn unit_vec(dx: f64, dy: f64) -> (f64, f64) {
    let mag_sq = dx * dx + dy * dy;
    if mag_sq <= 1e-9 {
        return (1.0, 0.0);
    }
    let inv_mag = mag_sq.powf(-0.5);
    (dx * inv_mag, dy * inv_mag)
}

fn size_from_mass(mass: f64) -> f64 {
    (mass * 100.0).sqrt()
}

fn mass_from_size(size: f64) -> f64 {
    size * size / 100.0
}

fn boost_step(remaining: &mut f64, dt: f64) -> f64 {
    if *remaining < 1.0 || dt <= 0.0 {
        *remaining = 0.0;
        return 0.0;
    }
    let fraction = (TICK_RATE * dt / 9.0).min(1.0);
    let distance = *remaining * fraction;
    *remaining = (*remaining - distance).max(0.0);
    distance
}

/// Python's round(x, 2) (banker's rounding).
fn round2(x: f64) -> f64 {
    (x * 100.0).round_ties_even() / 100.0
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round_ties_even() / 1000.0
}

fn round0(x: f64) -> i64 {
    x.round_ties_even() as i64
}

#[derive(Clone)]
struct Blob {
    id: u64,
    player_id: u64,
    x: f64,
    y: f64,
    mass: f64,
    boost_dx: f64,
    boost_dy: f64,
    boost_distance: f64,
    born_at: f64,
    // Actual observed velocity (total position delta per second over the last
    // tick, including steering, boosts, softbody pushes). Not part of the
    // physics — provided for observers (bots/RL) only.
    obs_vx: f64,
    obs_vy: f64,
}

impl Blob {
    fn size(&self) -> f64 {
        size_from_mass(self.mass)
    }
}

#[derive(Clone)]
struct Player {
    id: u64,
    name: String,
    color: String,
    is_bot: bool,
    bot_plugin: Option<String>,
    bot_team: Option<String>,
    blobs: Vec<Blob>,
    target_x: f64,
    target_y: f64,
    split_requested: bool,
    eject_requested: bool,
    last_split_at: f64,
    last_eject_at: f64,
    deaths: u64,
    kills: u64,
}

impl Player {
    fn total_mass(&self) -> f64 {
        let mut total = 0.0;
        for blob in &self.blobs {
            total += blob.mass;
        }
        total
    }

    fn center(&self) -> (f64, f64) {
        if self.blobs.is_empty() {
            return (0.0, 0.0);
        }
        let total = self.total_mass();
        if total <= 0.0 {
            let blob = &self.blobs[0];
            return (blob.x, blob.y);
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        for blob in &self.blobs {
            cx += blob.x * blob.mass;
            cy += blob.y * blob.mass;
        }
        (cx / total, cy / total)
    }

    fn camera_center(&self) -> (f64, f64) {
        let count = self.blobs.len() as f64;
        if count == 0.0 {
            return (0.0, 0.0);
        }
        self.blobs.iter().fold((0.0, 0.0), |(x, y), blob| {
            (x + blob.x / count, y + blob.y / count)
        })
    }

    fn camera_zoom(&self) -> f64 {
        let total_size: f64 = self.blobs.iter().map(Blob::size).sum();
        (64.0 / total_size.max(1.0)).min(1.0).powf(0.4)
    }
}

#[derive(Clone)]
struct Food {
    id: u64,
    x: f64,
    y: f64,
    mass: f64,
    color: usize, // index into cfg.food_colors
    grow_elapsed: f64,
}

#[derive(Clone)]
struct Ejected {
    id: u64,
    x: f64,
    y: f64,
    mass: f64,
    owner_id: u64,
    boost_dx: f64,
    boost_dy: f64,
    boost_distance: f64,
}

#[derive(Clone)]
struct Virus {
    id: u64,
    x: f64,
    y: f64,
    mass: f64,
    fed: usize,
    boost_dx: f64,
    boost_dy: f64,
    boost_distance: f64,
}

/// Spatial grid matching agario/spatial.py semantics: point insertion, rect
/// queries scanning cells row-major (cx outer, cy inner), bucket contents in
/// insertion order. Backed by a dense array over the world bounds instead of
/// a hash map — entities are always inside the world, so out-of-range query
/// cells are simply clamped away (they would be empty buckets anyway).
#[derive(Clone, Default)]
struct Grid {
    cell: f64,
    cols: i64,
    rows: i64,
    buckets: Vec<Vec<usize>>,
}

impl Grid {
    fn new(cell: f64, world_w: f64, world_h: f64) -> Self {
        let cols = (world_w / cell).floor() as i64 + 1;
        let rows = (world_h / cell).floor() as i64 + 1;
        Grid {
            cell,
            cols,
            rows,
            buckets: vec![Vec::new(); (cols * rows) as usize],
        }
    }

    fn clear(&mut self) {
        for bucket in self.buckets.iter_mut() {
            bucket.clear();
        }
    }

    fn insert(&mut self, x: f64, y: f64, idx: usize) {
        let cx = ((x / self.cell).floor() as i64).clamp(0, self.cols - 1);
        let cy = ((y / self.cell).floor() as i64).clamp(0, self.rows - 1);
        self.buckets[(cx * self.rows + cy) as usize].push(idx);
    }

    fn query_rect(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64, out: &mut Vec<usize>) {
        out.clear();
        let min_cx = ((min_x / self.cell).floor() as i64).max(0);
        let max_cx = ((max_x / self.cell).floor() as i64).min(self.cols - 1);
        let min_cy = ((min_y / self.cell).floor() as i64).max(0);
        let max_cy = ((max_y / self.cell).floor() as i64).min(self.rows - 1);
        for cx in min_cx..=max_cx {
            let row_base = cx * self.rows;
            for cy in min_cy..=max_cy {
                out.extend_from_slice(&self.buckets[(row_base + cy) as usize]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RL observation encoding used by both training and the live neural plugin.
// ---------------------------------------------------------------------------
const OBS_VIEW: f64 = 1400.0;
const OBS_SPEED_NORM: f64 = 400.0;
const OBS_K_OWN: usize = 16;
const OBS_K_ENEMY: usize = 12;
const OBS_K_VIRUS: usize = 3;
const OBS_N_SECTORS: usize = 8;
const OBS_K_FOOD: usize = 6;
const OBS_FOOD_F: usize = 4;
const OBS_N_TURNS: usize = 8;
const OBS_N_OPS: usize = 3;
const OBS_N_DIRECTIONS: usize = 16;
const OBS_SELF_DIM: usize = 28;
const OBS_OWN_F: usize = 9;
const OBS_ENEMY_F: usize = 15;
pub const OBS_DIM: usize = OBS_SELF_DIM
    + OBS_K_OWN * OBS_OWN_F
    + OBS_K_ENEMY * OBS_ENEMY_F
    + OBS_N_SECTORS * 2
    + OBS_K_FOOD * OBS_FOOD_F
    + OBS_K_VIRUS * 4;
const TWO_PI: f64 = std::f64::consts::PI * 2.0;
type EnemyObservation = (f64, f64, f64, f64, f64, f64, f64, f64, usize, f64);

fn fclamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

#[pyclass]
pub struct PlayerHandle {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub name: String,
}

#[pyclass]
#[derive(Clone)]
pub struct CoreWorld {
    cfg: WorldConfig,
    rng: PyMt19937,

    players: Vec<Player>,
    foods: Vec<Food>,
    ejected: Vec<Ejected>,
    viruses: Vec<Virus>,

    next_player_id: u64,
    next_blob_id: u64,
    next_food_id: u64,
    next_ejected_id: u64,
    next_virus_id: u64,

    food_grid: Grid,
    blob_grid: Grid,
    ejected_grid: Grid,
    // Flat (player_idx, blob_idx) list in the same order blobs were inserted
    // into blob_grid; rebuilt together with the grids.
    blob_index: Vec<(usize, usize)>,
}

impl CoreWorld {
    /// GIL-free constructor used by the batched-arena layer.
    pub(crate) fn new_internal(seed: u64, cfg: WorldConfig) -> CoreWorld {
        let mut world = CoreWorld {
            cfg,
            rng: PyMt19937::new(seed),
            players: Vec::new(),
            foods: Vec::new(),
            ejected: Vec::new(),
            viruses: Vec::new(),
            next_player_id: 1,
            next_blob_id: 1,
            next_food_id: 1,
            next_ejected_id: 1,
            next_virus_id: 1,
            food_grid: Grid::default(),
            blob_grid: Grid::default(),
            ejected_grid: Grid::default(),
            blob_index: Vec::new(),
        };
        world.food_grid = Grid::new(150.0, world.cfg.world_width, world.cfg.world_height);
        world.blob_grid = Grid::new(250.0, world.cfg.world_width, world.cfg.world_height);
        world.ejected_grid = Grid::new(180.0, world.cfg.world_width, world.cfg.world_height);
        while world.foods.len() < world.cfg.food_target_count {
            world.spawn_food();
        }
        for _ in 0..world.cfg.virus_min_count {
            world.spawn_virus();
        }
        world.rebuild_spatial_indexes();
        world
    }

    /// GIL-free update used by the batched-arena layer (same body as the
    /// pymethod `update`; kept as the single implementation).
    pub(crate) fn update_internal(&mut self, dt: f64, now: f64) {
        self.update(dt, now);
    }

    pub(crate) fn add_player_internal(&mut self, name: &str, now: f64) -> u64 {
        let handle = self.add_player(name, now, true, None, None, None);
        handle.id.strip_prefix('p').unwrap().parse().unwrap()
    }

    pub(crate) fn set_input_raw(&mut self, id: u64, tx: f64, ty: f64, split: bool, eject: bool) {
        if let Some(pos) = self.player_pos(id) {
            let player = &mut self.players[pos];
            player.target_x = tx;
            player.target_y = ty;
            if split {
                player.split_requested = true;
            }
            if eject {
                player.eject_requested = true;
            }
        }
    }

    /// (total_mass, kills, deaths, alive) for one player.
    pub(crate) fn stats_of(&self, id: u64) -> (f64, u64, u64, bool) {
        match self.player_pos(id) {
            Some(pos) => {
                let p = &self.players[pos];
                (p.total_mass(), p.kills, p.deaths, !p.blobs.is_empty())
            }
            None => (0.0, 0, 0, false),
        }
    }

    pub(crate) fn center_of(&self, id: u64) -> Option<(f64, f64)> {
        let pos = self.player_pos(id)?;
        let p = &self.players[pos];
        if p.blobs.is_empty() {
            return None;
        }
        Some(p.center())
    }

    pub(crate) fn biggest_blob_mass_of(&self, id: u64) -> f64 {
        match self.player_pos(id) {
            Some(pos) => self.players[pos]
                .blobs
                .iter()
                .map(|b| b.mass)
                .fold(0.0, f64::max),
            None => 0.0,
        }
    }

    /// Nearest threatening enemy blob (mass > my_mass * ratio) within range.
    pub(crate) fn nearest_threat(
        &self,
        id: u64,
        cx: f64,
        cy: f64,
        my_mass: f64,
        ratio: f64,
        range: f64,
    ) -> Option<(f64, f64)> {
        let mut best: Option<(f64, f64, f64)> = None;
        for p in &self.players {
            if p.id == id {
                continue;
            }
            for b in &p.blobs {
                if b.mass > my_mass * ratio {
                    let d = distance_sq(cx, cy, b.x, b.y);
                    if d < range * range && best.is_none_or(|(_, _, bd)| d < bd) {
                        best = Some((b.x, b.y, d));
                    }
                }
            }
        }
        best.map(|(x, y, _)| (x, y))
    }

    pub(crate) fn nearest_food_to(&self, cx: f64, cy: f64) -> Option<(f64, f64)> {
        let mut best: Option<(f64, f64, f64)> = None;
        for f in &self.foods {
            let d = distance_sq(cx, cy, f.x, f.y);
            if best.is_none_or(|(_, _, bd)| d < bd) {
                best = Some((f.x, f.y, d));
            }
        }
        best.map(|(x, y, _)| (x, y))
    }

    pub(crate) fn scale_mass_internal(&mut self, id: u64, total_mass: f64) {
        if let Some(pos) = self.player_pos(id) {
            let current = self.players[pos].total_mass();
            if current > 0.0 {
                let factor = total_mass / current;
                let min_mass = self.cfg.player_min_mass;
                for blob in self.players[pos].blobs.iter_mut() {
                    blob.mass = (blob.mass * factor).max(min_mass);
                }
            }
        }
    }

    pub(crate) fn clone_internal(&self) -> CoreWorld {
        self.clone()
    }

    /// GIL-free observation encode by numeric player id.
    pub(crate) fn observe_by_id(
        &self,
        id: u64,
        now: f64,
        control: (usize, usize, usize, f64),
        out: &mut [f32],
    ) {
        if let Some(pos) = self.player_pos(id) {
            self.observe_one(pos, now, control, out);
        } else {
            out.fill(0.0);
        }
    }

    fn parse_player_id(&self, raw: &str) -> Option<u64> {
        raw.strip_prefix('p')?.parse::<u64>().ok()
    }

    fn player_pos(&self, id: u64) -> Option<usize> {
        self.players.iter().position(|p| p.id == id)
    }

    fn random_spawn(&mut self, size: f64) -> (f64, f64) {
        let r = size * 0.5;
        let x = self.rng.uniform(r, self.cfg.world_width - r);
        let y = self.rng.uniform(r, self.cfg.world_height - r);
        (x, y)
    }

    fn safe_spawn(&mut self, size: f64) -> (f64, f64) {
        for _ in 0..self.cfg.safe_spawn_tries {
            let pos = self.random_spawn(size);
            if self.spawn_is_safe(pos.0, pos.1, size) {
                return pos;
            }
        }
        self.random_spawn(size)
    }

    fn spawn_is_safe(&self, x: f64, y: f64, size: f64) -> bool {
        self.players
            .iter()
            .flat_map(|p| &p.blobs)
            .all(|b| distance_sq(x, y, b.x, b.y) >= (size + b.size()).powi(2))
            && self
                .viruses
                .iter()
                .all(|v| distance_sq(x, y, v.x, v.y) >= (size + size_from_mass(v.mass)).powi(2))
    }

    fn player_spawn(&mut self, size: f64) -> (f64, f64) {
        if !self.ejected.is_empty() && self.rng.random() < self.cfg.spawn_from_ejected_chance {
            for _ in 0..self.cfg.safe_spawn_tries {
                let index = self.rng.choice_index(self.ejected.len());
                let item = &self.ejected[index];
                if self.spawn_is_safe(item.x, item.y, size) {
                    let pos = (item.x, item.y);
                    self.ejected.swap_remove(index);
                    return pos;
                }
            }
        }
        self.safe_spawn(size)
    }

    fn merge_ready_at(&self, blob: &Blob) -> f64 {
        blob.born_at
            + self.cfg.player_no_merge_seconds.max(
                self.cfg.player_merge_seconds + blob.size() * self.cfg.player_merge_size_factor,
            )
    }

    fn spawn_food(&mut self) {
        let (fx, fy) = self.safe_spawn(size_from_mass(self.cfg.food_min_mass));
        let mass = self.cfg.food_min_mass;
        let color = self.rng.choice_index(self.cfg.food_colors.len());
        let id = self.next_food_id;
        self.next_food_id += 1;
        self.foods.push(Food {
            id,
            x: fx,
            y: fy,
            mass,
            color,
            grow_elapsed: 0.0,
        });
    }

    fn spawn_virus(&mut self) {
        let (vx, vy) = self.safe_spawn(size_from_mass(self.cfg.virus_mass));
        let id = self.next_virus_id;
        self.next_virus_id += 1;
        let mass = self.cfg.virus_mass;
        self.viruses.push(Virus {
            id,
            x: vx,
            y: vy,
            mass,
            fed: 0,
            boost_dx: 0.0,
            boost_dy: 0.0,
            boost_distance: 0.0,
        });
    }

    fn respawn_eliminated_players(&mut self, now: f64) {
        for pi in 0..self.players.len() {
            if !self.players[pi].blobs.is_empty() {
                continue;
            }
            let (x, y) = self.player_spawn(size_from_mass(self.cfg.player_start_mass));
            let blob_id = self.next_blob_id;
            self.next_blob_id += 1;
            let player = &mut self.players[pi];
            player.deaths += 1;
            player.blobs.push(Blob {
                id: blob_id,
                player_id: player.id,
                x,
                y,
                mass: self.cfg.player_start_mass,
                boost_dx: 0.0,
                boost_dy: 0.0,
                boost_distance: 0.0,
                born_at: now,
                obs_vx: 0.0,
                obs_vy: 0.0,
            });
            player.target_x = x;
            player.target_y = y;
        }
    }

    fn apply_actions(&mut self, now: f64) {
        for pi in 0..self.players.len() {
            if self.players[pi].split_requested {
                self.split_player(pi, now);
                self.players[pi].split_requested = false;
            }
            if self.players[pi].eject_requested {
                self.eject_player_mass(pi, now);
                self.players[pi].eject_requested = false;
            }
        }
    }

    fn split_player(&mut self, pi: usize, now: f64) {
        let cfg = &self.cfg;
        let player = &self.players[pi];
        if now - player.last_split_at < cfg.action_cooldown_seconds {
            return;
        }
        if player.blobs.len() >= cfg.max_player_blobs {
            return;
        }

        let target_x = player.target_x;
        let target_y = player.target_y;
        let snapshot_len = player.blobs.len();
        let mut created: Vec<Blob> = Vec::new();

        for bi in 0..snapshot_len {
            let player = &self.players[pi];
            let blob = &player.blobs[bi];
            if blob.mass < self.cfg.player_min_split_mass {
                continue;
            }
            if player.blobs.len() + created.len() >= self.cfg.max_player_blobs {
                break;
            }

            let dx = target_x - blob.x;
            let dy = target_y - blob.y;
            let (ux, uy) = unit_vec(dx, dy);

            let split_mass = blob.mass / 2.0;
            let blob = &mut self.players[pi].blobs[bi];
            blob.mass = split_mass;

            let offset = self.cfg.player_split_distance;
            let bx = blob.x;
            let by = blob.y;
            let blob_id = self.next_blob_id;
            self.next_blob_id += 1;
            created.push(Blob {
                id: blob_id,
                player_id: self.players[pi].id,
                x: clamp(bx + ux * offset, 0.0, self.cfg.world_width),
                y: clamp(by + uy * offset, 0.0, self.cfg.world_height),
                mass: split_mass,
                boost_dx: ux,
                boost_dy: uy,
                boost_distance: self.cfg.player_split_boost,
                born_at: now,
                obs_vx: 0.0,
                obs_vy: 0.0,
            });
        }

        let any_created = !created.is_empty();
        self.players[pi].blobs.append(&mut created);
        if any_created {
            self.players[pi].last_split_at = now;
        }
    }

    fn eject_player_mass(&mut self, pi: usize, now: f64) {
        if now - self.players[pi].last_eject_at < self.cfg.action_cooldown_seconds {
            return;
        }

        let target_x = self.players[pi].target_x;
        let target_y = self.players[pi].target_y;
        let owner_id = self.players[pi].id;
        let mut spawned_any = false;

        for bi in 0..self.players[pi].blobs.len() {
            let blob = &self.players[pi].blobs[bi];
            if blob.mass < self.cfg.player_min_eject_mass {
                continue;
            }
            let remaining_mass = blob.mass - self.cfg.eject_loss_mass;
            if remaining_mass < self.cfg.player_min_mass {
                continue;
            }

            let dx = target_x - blob.x;
            let dy = target_y - blob.y;
            let (ux, uy) = unit_vec(dx, dy);
            let size = blob.size();

            let angle = uy.atan2(ux)
                + self
                    .rng
                    .uniform(-self.cfg.eject_dispersion, self.cfg.eject_dispersion);
            let (eject_dx, eject_dy) = (angle.cos(), angle.sin());
            let blob = &mut self.players[pi].blobs[bi];
            blob.mass = remaining_mass;
            let eject_x = blob.x + ux * size;
            let eject_y = blob.y + uy * size;

            let id = self.next_ejected_id;
            self.next_ejected_id += 1;
            self.ejected.push(Ejected {
                id,
                x: clamp(eject_x, 0.0, self.cfg.world_width),
                y: clamp(eject_y, 0.0, self.cfg.world_height),
                mass: self.cfg.ejected_mass,
                owner_id,
                boost_dx: eject_dx,
                boost_dy: eject_dy,
                boost_distance: self.cfg.ejected_boost,
            });
            spawned_any = true;
        }

        if spawned_any {
            self.players[pi].last_eject_at = now;
        }
    }

    fn move_blobs(&mut self, dt: f64, now: f64) {
        for pi in 0..self.players.len() {
            {
                let cfg = &self.cfg;
                let player = &mut self.players[pi];
                let target_x = player.target_x;
                let target_y = player.target_y;
                for blob in player.blobs.iter_mut() {
                    let dx = target_x - blob.x;
                    let dy = target_y - blob.y;
                    let distance = (dx * dx + dy * dy).sqrt();
                    let (ux, uy) = unit_vec(dx, dy);
                    let boost = boost_step(&mut blob.boost_distance, dt);
                    blob.x += blob.boost_dx * boost;
                    blob.y += blob.boost_dy * boost;
                    if distance >= 1.0 {
                        let movement = (88.0
                            * blob.size().powf(-0.439_675_4)
                            * cfg.player_move_mult
                            * TICK_RATE
                            * dt
                            * (distance / 32.0).min(1.0))
                        .min(distance);
                        blob.x += ux * movement;
                        blob.y += uy * movement;
                    }
                }
            }

            self.apply_same_player_softbody(pi, now);

            let cfg = &self.cfg;
            let player = &mut self.players[pi];
            for blob in player.blobs.iter_mut() {
                let r = blob.size() * 0.5;
                if blob.x < r || blob.x > cfg.world_width - r {
                    blob.boost_dx = -blob.boost_dx;
                }
                if blob.y < r || blob.y > cfg.world_height - r {
                    blob.boost_dy = -blob.boost_dy;
                }
                blob.x = clamp(blob.x, r, cfg.world_width - r);
                blob.y = clamp(blob.y, r, cfg.world_height - r);
            }
        }
    }

    fn apply_same_player_softbody(&mut self, pi: usize, now: f64) {
        let n = self.players[pi].blobs.len();
        if n <= 1 {
            return;
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let (a_x, a_y, a_r, a_born, a_merge) = {
                    let a = &self.players[pi].blobs[i];
                    (a.x, a.y, a.size(), a.born_at, self.merge_ready_at(a))
                };
                let (b_x, b_y, b_r, b_born, b_merge) = {
                    let b = &self.players[pi].blobs[j];
                    (b.x, b.y, b.size(), b.born_at, self.merge_ready_at(b))
                };

                if now - a_born < self.cfg.player_no_collide_seconds
                    || now - b_born < self.cfg.player_no_collide_seconds
                    || (now >= a_merge && now >= b_merge)
                {
                    continue;
                }

                let mut dx = b_x - a_x;
                let mut dy = b_y - a_y;
                let mut dist_sq = dx * dx + dy * dy;
                if dist_sq <= 1e-8 {
                    let theta = self.rng.random() * TWO_PI;
                    dx = theta.cos();
                    dy = theta.sin();
                    dist_sq = 1.0;
                }

                let dist = dist_sq.powf(0.5);
                let ux = dx / dist;
                let uy = dy / dist;

                let touch = a_r + b_r;
                if dist < touch {
                    let overlap = touch - dist;
                    let total_mass =
                        self.players[pi].blobs[i].mass + self.players[pi].blobs[j].mass;
                    let a_share = self.players[pi].blobs[j].mass / total_mass;
                    let b_share = self.players[pi].blobs[i].mass / total_mass;
                    let blobs = &mut self.players[pi].blobs;
                    blobs[i].x -= ux * overlap * a_share;
                    blobs[i].y -= uy * overlap * a_share;
                    blobs[j].x += ux * overlap * b_share;
                    blobs[j].y += uy * overlap * b_share;
                }
            }
        }
    }

    fn apply_mass_decay(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        for pi in 0..self.players.len() {
            if self.players[pi].blobs.is_empty() {
                continue;
            }
            let min_mass = self.cfg.player_min_mass;
            for blob in self.players[pi].blobs.iter_mut() {
                if blob.mass <= min_mass {
                    continue;
                }
                let size = blob.size();
                let next = size - size * self.cfg.player_decay_mult / 50.0 * TICK_RATE * dt;
                blob.mass = mass_from_size(next).max(min_mass);
            }
        }
    }

    fn move_ejected(&mut self, dt: f64) {
        let cfg = &self.cfg;

        for ejected in self.ejected.iter_mut() {
            let movement = boost_step(&mut ejected.boost_distance, dt);
            ejected.x += ejected.boost_dx * movement;
            ejected.y += ejected.boost_dy * movement;
            let r = size_from_mass(ejected.mass) * 0.5;
            if ejected.x < r || ejected.x > cfg.world_width - r {
                ejected.boost_dx = -ejected.boost_dx;
            }
            if ejected.y < r || ejected.y > cfg.world_height - r {
                ejected.boost_dy = -ejected.boost_dy;
            }
            ejected.x = clamp(ejected.x, r, cfg.world_width - r);
            ejected.y = clamp(ejected.y, r, cfg.world_height - r);
        }
    }

    fn resolve_ejected_collisions(&mut self) {
        for i in 0..self.ejected.len() {
            for j in (i + 1)..self.ejected.len() {
                let (left, right) = self.ejected.split_at_mut(j);
                let a = &mut left[i];
                let b = &mut right[0];
                let mut dx = b.x - a.x;
                let mut dy = b.y - a.y;
                let mut distance = dx.hypot(dy);
                let overlap = size_from_mass(a.mass) + size_from_mass(b.mass) - distance;
                if overlap <= 0.0 {
                    continue;
                }
                if distance <= 1e-9 {
                    dx = 1.0;
                    dy = 0.0;
                    distance = 1.0;
                }
                let total = a.mass + b.mass;
                let (ux, uy) = (dx / distance, dy / distance);
                a.x -= ux * overlap * b.mass / total;
                a.y -= uy * overlap * b.mass / total;
                b.x += ux * overlap * a.mass / total;
                b.y += uy * overlap * a.mass / total;
            }
        }
        for item in &mut self.ejected {
            let r = size_from_mass(item.mass) * 0.5;
            item.x = clamp(item.x, r, self.cfg.world_width - r);
            item.y = clamp(item.y, r, self.cfg.world_height - r);
        }
    }

    fn rebuild_spatial_indexes(&mut self) {
        self.food_grid.clear();
        self.blob_grid.clear();
        self.ejected_grid.clear();
        self.blob_index.clear();

        for (fi, food) in self.foods.iter().enumerate() {
            self.food_grid.insert(food.x, food.y, fi);
        }

        for (pi, player) in self.players.iter().enumerate() {
            for (bi, blob) in player.blobs.iter().enumerate() {
                let flat = self.blob_index.len();
                self.blob_index.push((pi, bi));
                self.blob_grid.insert(blob.x, blob.y, flat);
            }
        }

        for (ei, ejected) in self.ejected.iter().enumerate() {
            self.ejected_grid.insert(ejected.x, ejected.y, ei);
        }
    }

    fn resolve_blob_food_collisions(&mut self) {
        let mut eaten: HashSet<usize> = HashSet::new();
        let mut nearby: Vec<usize> = Vec::new();

        for pi in 0..self.players.len() {
            for bi in 0..self.players[pi].blobs.len() {
                let (bx, by, radius) = {
                    let blob = &self.players[pi].blobs[bi];
                    (blob.x, blob.y, blob.size())
                };
                self.food_grid.query_rect(
                    bx - radius,
                    by - radius,
                    bx + radius,
                    by + radius,
                    &mut nearby,
                );
                for &fi in nearby.iter() {
                    if eaten.contains(&fi) {
                        continue;
                    }
                    let food = &self.foods[fi];
                    let blob = &self.players[pi].blobs[bi];
                    let blob_radius = blob.size();
                    let food_radius = size_from_mass(food.mass);
                    let eat_dist = blob_radius - food_radius / self.cfg.eat_overlap_divisor;
                    if distance_sq(blob.x, blob.y, food.x, food.y) <= eat_dist * eat_dist {
                        let food_mass = food.mass;
                        self.players[pi].blobs[bi].mass += food_mass;
                        eaten.insert(fi);
                    }
                }
            }
        }

        if !eaten.is_empty() {
            let mut idx = 0usize;
            self.foods.retain(|_| {
                let keep = !eaten.contains(&idx);
                idx += 1;
                keep
            });
        }
    }

    fn resolve_blob_ejected_collisions(&mut self) {
        let mut consumed: HashSet<usize> = HashSet::new();
        let mut nearby: Vec<usize> = Vec::new();

        for pi in 0..self.players.len() {
            for bi in 0..self.players[pi].blobs.len() {
                let (bx, by, radius) = {
                    let blob = &self.players[pi].blobs[bi];
                    (blob.x, blob.y, blob.size())
                };
                self.ejected_grid.query_rect(
                    bx - radius,
                    by - radius,
                    bx + radius,
                    by + radius,
                    &mut nearby,
                );
                for &ei in nearby.iter() {
                    if consumed.contains(&ei) {
                        continue;
                    }
                    let ejected = &self.ejected[ei];
                    let blob = &self.players[pi].blobs[bi];
                    let blob_radius = blob.size();
                    let ejected_radius = size_from_mass(ejected.mass);
                    if blob_radius < ejected_radius * self.cfg.eat_size_ratio {
                        continue;
                    }
                    let eat_dist = blob_radius - ejected_radius / self.cfg.eat_overlap_divisor;
                    if distance_sq(blob.x, blob.y, ejected.x, ejected.y) <= eat_dist * eat_dist {
                        let ejected_mass = ejected.mass;
                        self.players[pi].blobs[bi].mass += ejected_mass;
                        consumed.insert(ei);
                    }
                }
            }
        }

        if !consumed.is_empty() {
            let mut idx = 0usize;
            self.ejected.retain(|_| {
                let keep = !consumed.contains(&idx);
                idx += 1;
                keep
            });
        }
    }

    fn resolve_blob_blob_collisions(&mut self, now: f64) {
        // Work on a flat copy so cross-player pairs can be mutated freely;
        // masses are written back and eaten blobs removed at the end —
        // matching the Python version, which defers deletions to the end of
        // the phase while mutating masses live.
        let mut flat: Vec<Blob> = Vec::with_capacity(self.blob_index.len());
        for &(pi, bi) in &self.blob_index {
            flat.push(self.players[pi].blobs[bi].clone());
        }

        let mut eaten: HashSet<u64> = HashSet::new();
        let mut checked: HashSet<(u64, u64)> = HashSet::new();
        let mut nearby: Vec<usize> = Vec::new();
        let mut kill_credit: Vec<(u64, u64)> = Vec::new(); // (attacker player, victim player)

        for i in 0..flat.len() {
            if eaten.contains(&flat[i].id) {
                continue;
            }

            let (bx, by, radius) = {
                let blob = &flat[i];
                (blob.x, blob.y, blob.size())
            };
            self.blob_grid.query_rect(
                bx - radius * 2.0,
                by - radius * 2.0,
                bx + radius * 2.0,
                by + radius * 2.0,
                &mut nearby,
            );
            let candidates = nearby.clone();

            for &j in candidates.iter() {
                if flat[j].id == flat[i].id || eaten.contains(&flat[j].id) {
                    continue;
                }
                let pair = if flat[i].id < flat[j].id {
                    (flat[i].id, flat[j].id)
                } else {
                    (flat[j].id, flat[i].id)
                };
                if !checked.insert(pair) {
                    continue;
                }

                let (big, small) = if flat[i].mass >= flat[j].mass {
                    (i, j)
                } else {
                    (j, i)
                };
                let dist_sq = distance_sq(flat[big].x, flat[big].y, flat[small].x, flat[small].y);
                let big_radius = flat[big].size();
                let small_radius = flat[small].size();

                if flat[big].player_id == flat[small].player_id {
                    if now < self.merge_ready_at(&flat[big])
                        || now < self.merge_ready_at(&flat[small])
                    {
                        continue;
                    }
                    let merge_distance = big_radius - small_radius / self.cfg.eat_overlap_divisor;
                    if merge_distance <= 0.0 {
                        continue;
                    }
                    if dist_sq <= merge_distance * merge_distance {
                        let small_mass = flat[small].mass;
                        flat[big].mass += small_mass;
                        eaten.insert(flat[small].id);
                    }
                    continue;
                }

                if big_radius < small_radius * self.cfg.eat_size_ratio {
                    continue;
                }

                let eat_distance = big_radius - small_radius / self.cfg.eat_overlap_divisor;
                if eat_distance <= 0.0 {
                    continue;
                }
                if dist_sq <= eat_distance * eat_distance {
                    let small_mass = flat[small].mass;
                    flat[big].mass += small_mass;
                    eaten.insert(flat[small].id);
                    kill_credit.push((flat[big].player_id, flat[small].player_id));
                }
            }
        }

        for (attacker, _victim) in &kill_credit {
            if let Some(pos) = self.players.iter().position(|p| p.id == *attacker) {
                self.players[pos].kills += 1;
            }
        }

        // Write back masses, then drop eaten blobs.
        for (flat_idx, &(pi, bi)) in self.blob_index.iter().enumerate() {
            self.players[pi].blobs[bi].mass = flat[flat_idx].mass;
        }
        if !eaten.is_empty() {
            for player in self.players.iter_mut() {
                player.blobs.retain(|b| !eaten.contains(&b.id));
            }
        }
    }

    fn move_viruses(&mut self, dt: f64) {
        for virus in &mut self.viruses {
            let movement = boost_step(&mut virus.boost_distance, dt);
            virus.x += virus.boost_dx * movement;
            virus.y += virus.boost_dy * movement;
            let r = size_from_mass(virus.mass) * 0.5;
            if virus.x < r || virus.x > self.cfg.world_width - r {
                virus.boost_dx = -virus.boost_dx;
            }
            if virus.y < r || virus.y > self.cfg.world_height - r {
                virus.boost_dy = -virus.boost_dy;
            }
            virus.x = clamp(virus.x, r, self.cfg.world_width - r);
            virus.y = clamp(virus.y, r, self.cfg.world_height - r);
        }
    }

    fn resolve_virus_ejected_collisions(&mut self) {
        if self.viruses.len() >= self.cfg.virus_max_count {
            return;
        }
        let mut eaten = HashSet::new();
        let mut children = Vec::new();
        for vi in 0..self.viruses.len() {
            for ei in 0..self.ejected.len() {
                if eaten.contains(&ei)
                    || self.viruses.len() + children.len() >= self.cfg.virus_max_count
                {
                    continue;
                }
                let vsize = size_from_mass(self.viruses[vi].mass);
                let esize = size_from_mass(self.ejected[ei].mass);
                let reach = vsize - esize / self.cfg.eat_overlap_divisor;
                if distance_sq(
                    self.viruses[vi].x,
                    self.viruses[vi].y,
                    self.ejected[ei].x,
                    self.ejected[ei].y,
                ) > reach * reach
                {
                    continue;
                }
                let direction = unit_vec(self.ejected[ei].boost_dx, self.ejected[ei].boost_dy);
                eaten.insert(ei);
                self.viruses[vi].fed += 1;
                if self.viruses[vi].fed >= self.cfg.virus_feed_times {
                    self.viruses[vi].fed = 0;
                    self.viruses[vi].mass = self.cfg.virus_mass;
                    children.push((self.viruses[vi].x, self.viruses[vi].y, direction));
                } else {
                    self.viruses[vi].mass += self.ejected[ei].mass;
                }
            }
        }
        if !eaten.is_empty() {
            let mut index = 0;
            self.ejected.retain(|_| {
                let keep = !eaten.contains(&index);
                index += 1;
                keep
            });
        }
        for (x, y, (dx, dy)) in children {
            let id = self.next_virus_id;
            self.next_virus_id += 1;
            self.viruses.push(Virus {
                id,
                x,
                y,
                mass: self.cfg.virus_mass,
                fed: 0,
                boost_dx: dx,
                boost_dy: dy,
                boost_distance: self.cfg.virus_split_boost,
            });
        }
    }

    fn resolve_virus_blob_collisions(&mut self, now: f64) {
        let mut consumed = HashSet::new();
        for pi in 0..self.players.len() {
            let blob_ids: Vec<u64> = self.players[pi].blobs.iter().map(|b| b.id).collect();
            for blob_id in blob_ids {
                let Some(bi) = self.players[pi].blobs.iter().position(|b| b.id == blob_id) else {
                    continue;
                };
                for vi in 0..self.viruses.len() {
                    if consumed.contains(&self.viruses[vi].id) {
                        continue;
                    }
                    let bsize = self.players[pi].blobs[bi].size();
                    let vsize = size_from_mass(self.viruses[vi].mass);
                    if bsize < vsize * self.cfg.eat_size_ratio {
                        continue;
                    }
                    let reach = bsize - vsize / self.cfg.eat_overlap_divisor;
                    if distance_sq(
                        self.players[pi].blobs[bi].x,
                        self.players[pi].blobs[bi].y,
                        self.viruses[vi].x,
                        self.viruses[vi].y,
                    ) <= reach * reach
                    {
                        self.players[pi].blobs[bi].mass += self.viruses[vi].mass;
                        self.explode_blob_into_player(pi, bi, now);
                        consumed.insert(self.viruses[vi].id);
                        break;
                    }
                }
            }
        }
        self.viruses.retain(|v| !consumed.contains(&v.id));
    }

    fn virus_pop_masses(&self, cell_mass: f64, cells_left: usize) -> Vec<f64> {
        if cells_left == 0 {
            return Vec::new();
        }
        let split_min = self.cfg.player_min_split_mass;
        if cell_mass / (cells_left as f64) < split_min {
            let mut amount = 2usize;
            while cell_mass / (amount + 1) as f64 >= split_min && amount * 2 <= cells_left {
                amount *= 2;
            }
            return vec![cell_mass / (amount + 1) as f64; amount];
        }

        let mut splits = Vec::new();
        let mut next_mass = cell_mass / 2.0;
        let mut mass_left = cell_mass / 2.0;
        let mut slots = cells_left;
        while slots > 0 {
            if next_mass / (slots as f64) < split_min {
                break;
            }
            while next_mass >= mass_left && slots > 1 {
                next_mass /= 2.0;
            }
            splits.push(next_mass);
            mass_left -= next_mass;
            slots -= 1;
        }
        if slots > 0 {
            splits.extend(vec![mass_left / slots as f64; slots]);
        }
        splits
    }

    fn explode_blob_into_player(&mut self, pi: usize, bi: usize, now: f64) {
        let cells_left = self
            .cfg
            .max_player_blobs
            .saturating_sub(self.players[pi].blobs.len());
        let splits = self.virus_pop_masses(self.players[pi].blobs[bi].mass, cells_left);
        for mass in splits {
            if mass <= 0.0 || mass >= self.players[pi].blobs[bi].mass {
                continue;
            }
            let angle = self.rng.uniform(0.0, TWO_PI);
            let (dx, dy) = (angle.sin(), angle.cos());
            let (x, y, player_id) = {
                let parent = &mut self.players[pi].blobs[bi];
                parent.mass -= mass;
                (parent.x, parent.y, parent.player_id)
            };
            let id = self.next_blob_id;
            self.next_blob_id += 1;
            self.players[pi].blobs.push(Blob {
                id,
                player_id,
                x: clamp(
                    x + dx * self.cfg.player_split_distance,
                    0.0,
                    self.cfg.world_width,
                ),
                y: clamp(
                    y + dy * self.cfg.player_split_distance,
                    0.0,
                    self.cfg.world_height,
                ),
                mass,
                boost_dx: dx,
                boost_dy: dy,
                boost_distance: self.cfg.player_split_boost,
                born_at: now,
                obs_vx: 0.0,
                obs_vy: 0.0,
            });
        }
    }

    fn maintain_world_entities(&mut self, dt: f64) {
        for food in &mut self.foods {
            food.grow_elapsed += dt;
            while food.mass < self.cfg.food_max_mass
                && food.grow_elapsed > self.cfg.food_grow_seconds
            {
                food.mass = (food.mass + 1.0).min(self.cfg.food_max_mass);
                food.grow_elapsed -= self.cfg.food_grow_seconds;
            }
        }
        while self.foods.len() < self.cfg.food_target_count {
            self.spawn_food();
        }
        while self.viruses.len() < self.cfg.virus_min_count {
            self.spawn_virus();
        }
    }

    fn autosplit_players(&mut self, now: f64) {
        for pi in 0..self.players.len() {
            let ids: Vec<u64> = self.players[pi].blobs.iter().map(|b| b.id).collect();
            for id in ids {
                let Some(bi) = self.players[pi].blobs.iter().position(|b| b.id == id) else {
                    continue;
                };
                let cells_left = 1 + self.cfg.max_player_blobs - self.players[pi].blobs.len();
                let overflow =
                    (self.players[pi].blobs[bi].mass / self.cfg.player_max_mass).ceil() as usize;
                if overflow <= 1 || cells_left == 0 {
                    continue;
                }
                let pieces = overflow.min(cells_left);
                let split_mass =
                    (self.players[pi].blobs[bi].mass / pieces as f64).min(self.cfg.player_max_mass);
                for _ in 1..pieces {
                    let angle = self.rng.uniform(0.0, TWO_PI);
                    let (dx, dy) = (angle.sin(), angle.cos());
                    let (x, y, player_id) = {
                        let parent = &mut self.players[pi].blobs[bi];
                        parent.mass -= split_mass;
                        (parent.x, parent.y, parent.player_id)
                    };
                    let blob_id = self.next_blob_id;
                    self.next_blob_id += 1;
                    self.players[pi].blobs.push(Blob {
                        id: blob_id,
                        player_id,
                        x: clamp(
                            x + dx * self.cfg.player_split_distance,
                            0.0,
                            self.cfg.world_width,
                        ),
                        y: clamp(
                            y + dy * self.cfg.player_split_distance,
                            0.0,
                            self.cfg.world_height,
                        ),
                        mass: split_mass,
                        boost_dx: dx,
                        boost_dy: dy,
                        boost_distance: self.cfg.player_split_boost,
                        born_at: now,
                        obs_vx: 0.0,
                        obs_vy: 0.0,
                    });
                }
            }
        }
    }

    /// Encode one agent's observation into `out` (length OBS_DIM), mirroring
    /// bot_solutions/rl_v1/obs.py::encode line-for-line. `control` is the agent's own
    /// control state: (heading, prev_turn, prev_op, prev_speed).
    fn observe_one(
        &self,
        pi: usize,
        now: f64,
        control: (usize, usize, usize, f64),
        out: &mut [f32],
    ) {
        let world_w = self.cfg.world_width;
        let world_h = self.cfg.world_height;
        let me = &self.players[pi];
        if me.blobs.is_empty() {
            return;
        }

        let total_mass: f64 = me.blobs.iter().map(|b| b.mass).sum();
        let cx: f64 = me.blobs.iter().map(|b| b.x * b.mass).sum::<f64>() / total_mass;
        let cy: f64 = me.blobs.iter().map(|b| b.y * b.mass).sum::<f64>() / total_mass;
        let my_vx: f64 = me.blobs.iter().map(|b| b.obs_vx * b.mass).sum::<f64>() / total_mass;
        let my_vy: f64 = me.blobs.iter().map(|b| b.obs_vy * b.mass).sum::<f64>() / total_mass;

        let mut own_sorted: Vec<&Blob> = me.blobs.iter().collect();
        own_sorted.sort_by(|a, b| b.mass.partial_cmp(&a.mass).unwrap());
        let largest_mass = own_sorted[0].mass;
        let largest_radius = own_sorted[0].size();
        let max_merge_in = me
            .blobs
            .iter()
            .map(|b| (self.merge_ready_at(b) - now).max(0.0))
            .fold(f64::NEG_INFINITY, f64::max);
        let can_split = largest_mass >= self.cfg.player_min_split_mass
            && me.blobs.len() < self.cfg.max_player_blobs;
        let my_split_reach = self.cfg.player_split_distance
            + self.cfg.player_split_boost
            + largest_radius / 2.0_f64.sqrt();
        let my_half_mass = largest_mass / 2.0;

        let mut i = 0usize;
        let put = |out: &mut [f32], i: &mut usize, v: f64| {
            out[*i] = v as f32;
            *i += 1;
        };

        put(
            out,
            &mut i,
            fclamp(
                (total_mass / self.cfg.player_start_mass).ln() / 3.0,
                -1.5,
                1.5,
            ),
        );
        put(
            out,
            &mut i,
            me.blobs.len() as f64 / self.cfg.max_player_blobs as f64,
        );
        put(out, &mut i, (largest_radius / 250.0).min(2.0));
        put(out, &mut i, cx / world_w * 2.0 - 1.0);
        put(out, &mut i, cy / world_h * 2.0 - 1.0);
        put(out, &mut i, (cx.min(world_w - cx) / OBS_VIEW).min(1.0));
        put(out, &mut i, (cy.min(world_h - cy) / OBS_VIEW).min(1.0));
        put(out, &mut i, if can_split { 1.0 } else { 0.0 });
        put(
            out,
            &mut i,
            (88.0 * largest_radius.powf(-0.439_675_4) * TICK_RATE / 400.0).min(1.5),
        );
        put(
            out,
            &mut i,
            if largest_radius > size_from_mass(self.cfg.virus_mass) * self.cfg.eat_size_ratio {
                1.0
            } else {
                0.0
            },
        );
        put(out, &mut i, fclamp(my_vx / OBS_SPEED_NORM, -2.0, 2.0));
        put(out, &mut i, fclamp(my_vy / OBS_SPEED_NORM, -2.0, 2.0));
        put(out, &mut i, (my_split_reach / OBS_VIEW).min(1.0));
        put(out, &mut i, (max_merge_in / 60.0).min(1.0));
        let (heading, prev_turn, prev_op, prev_speed) = control;
        let heading_angle = (heading as f64 / OBS_N_DIRECTIONS as f64) * TWO_PI;
        put(out, &mut i, heading_angle.sin());
        put(out, &mut i, heading_angle.cos());
        out[i + prev_turn.min(OBS_N_TURNS - 1)] = 1.0;
        i += OBS_N_TURNS;
        out[i + prev_op.min(OBS_N_OPS - 1)] = 1.0;
        i += OBS_N_OPS;
        put(out, &mut i, fclamp(prev_speed, 0.0, 1.0));
        debug_assert_eq!(i, OBS_SELF_DIM);

        for blob in own_sorted.iter().take(OBS_K_OWN) {
            let dx = blob.x - cx;
            let dy = blob.y - cy;
            let dist = dx.hypot(dy);
            let (sin_b, cos_b) = if dist > 1e-6 {
                (dy / dist, dx / dist)
            } else {
                (0.0, 1.0)
            };
            let merge_in = (self.merge_ready_at(blob) - now).max(0.0);
            put(out, &mut i, (dist / OBS_VIEW).min(1.5));
            put(out, &mut i, sin_b);
            put(out, &mut i, cos_b);
            put(out, &mut i, blob.mass / total_mass);
            put(out, &mut i, (blob.size() / 250.0).min(2.0));
            put(
                out,
                &mut i,
                if blob.mass >= self.cfg.player_min_split_mass {
                    1.0
                } else {
                    0.0
                },
            );
            put(
                out,
                &mut i,
                fclamp((blob.obs_vx - my_vx) / OBS_SPEED_NORM, -2.5, 2.5),
            );
            put(
                out,
                &mut i,
                fclamp((blob.obs_vy - my_vy) / OBS_SPEED_NORM, -2.5, 2.5),
            );
            put(out, &mut i, (merge_in / 60.0).min(1.0));
        }
        i = OBS_SELF_DIM + OBS_K_OWN * OBS_OWN_F;

        // (dist, dx, dy, mass, radius, vx, vy, owner_mass, owner_cells, merge_in)
        let mut enemies: Vec<EnemyObservation> = Vec::new();
        for (pj, other) in self.players.iter().enumerate() {
            if pj == pi {
                continue;
            }
            let o_mass: f64 = other.blobs.iter().map(|b| b.mass).sum();
            let o_cells = other.blobs.len();
            for blob in &other.blobs {
                let dx = blob.x - cx;
                let dy = blob.y - cy;
                let dist = dx.hypot(dy);
                if dist <= OBS_VIEW * 1.5 {
                    enemies.push((
                        dist,
                        dx,
                        dy,
                        blob.mass,
                        blob.size(),
                        blob.obs_vx,
                        blob.obs_vy,
                        o_mass,
                        o_cells,
                        (self.merge_ready_at(blob) - now).max(0.0),
                    ));
                }
            }
        }
        enemies.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for &(dist, dx, dy, em, er, evx, evy, o_mass, o_cells, merge_in) in
            enemies.iter().take(OBS_K_ENEMY)
        {
            let (ux, uy) = if dist > 1e-6 {
                (dx / dist, dy / dist)
            } else {
                (1.0, 0.0)
            };
            let rel_vx = evx - my_vx;
            let rel_vy = evy - my_vy;
            let closing = -(rel_vx * ux + rel_vy * uy);
            let tangential = rel_vx * -uy + rel_vy * ux;

            put(out, &mut i, (dist / OBS_VIEW).min(1.5));
            put(out, &mut i, uy);
            put(out, &mut i, ux);
            put(out, &mut i, fclamp(closing / OBS_SPEED_NORM, -2.5, 2.5));
            put(out, &mut i, fclamp(tangential / OBS_SPEED_NORM, -2.5, 2.5));
            put(
                out,
                &mut i,
                fclamp((em / largest_mass).ln() / 2.0, -2.0, 2.0),
            );
            put(
                out,
                &mut i,
                if largest_radius >= er * self.cfg.eat_size_ratio {
                    1.0
                } else {
                    0.0
                },
            );
            put(
                out,
                &mut i,
                if er >= largest_radius * self.cfg.eat_size_ratio {
                    1.0
                } else {
                    0.0
                },
            );
            put(
                out,
                &mut i,
                if can_split && size_from_mass(my_half_mass) >= er * self.cfg.eat_size_ratio {
                    1.0
                } else {
                    0.0
                },
            );
            put(
                out,
                &mut i,
                fclamp(
                    dist / (my_split_reach - er / self.cfg.eat_overlap_divisor).max(1.0),
                    0.0,
                    3.0,
                ),
            );
            let their_reach =
                self.cfg.player_split_distance + self.cfg.player_split_boost + er / 2.0_f64.sqrt();
            put(
                out,
                &mut i,
                if size_from_mass(em / 2.0) >= largest_radius * self.cfg.eat_size_ratio
                    && em >= self.cfg.player_min_split_mass
                {
                    1.0
                } else {
                    0.0
                },
            );
            put(
                out,
                &mut i,
                fclamp(
                    dist / (their_reach - largest_radius / self.cfg.eat_overlap_divisor).max(1.0),
                    0.0,
                    3.0,
                ),
            );
            put(
                out,
                &mut i,
                fclamp((o_mass.max(1.0) / total_mass).ln() / 2.0, -2.0, 2.0),
            );
            put(
                out,
                &mut i,
                o_cells as f64 / self.cfg.max_player_blobs as f64,
            );
            put(out, &mut i, (merge_in / 60.0).min(1.0));
        }
        i = OBS_SELF_DIM + OBS_K_OWN * OBS_OWN_F + OBS_K_ENEMY * OBS_ENEMY_F;

        let mut sector_mass = [0.0f64; OBS_N_SECTORS];
        let mut sector_near = [OBS_VIEW; OBS_N_SECTORS];
        for food in &self.foods {
            let dx = food.x - cx;
            let dy = food.y - cy;
            let dist = dx.hypot(dy);
            if dist > OBS_VIEW {
                continue;
            }
            let s = (((dy.atan2(dx) + std::f64::consts::PI) / TWO_PI * OBS_N_SECTORS as f64)
                as usize)
                % OBS_N_SECTORS;
            sector_mass[s] += food.mass;
            if dist < sector_near[s] {
                sector_near[s] = dist;
            }
        }
        for e in &self.ejected {
            let dx = e.x - cx;
            let dy = e.y - cy;
            let dist = dx.hypot(dy);
            if dist > OBS_VIEW {
                continue;
            }
            let s = (((dy.atan2(dx) + std::f64::consts::PI) / TWO_PI * OBS_N_SECTORS as f64)
                as usize)
                % OBS_N_SECTORS;
            sector_mass[s] += e.mass * 2.0;
            if dist < sector_near[s] {
                sector_near[s] = dist;
            }
        }
        for s in 0..OBS_N_SECTORS {
            put(out, &mut i, (sector_mass[s] / 40.0).min(2.0));
            put(out, &mut i, sector_near[s] / OBS_VIEW);
        }

        // Nearest pellets individually (foods then ejected, stable by dist).
        let mut pellets: Vec<(f64, f64, f64, f64)> = Vec::new();
        for food in &self.foods {
            let dx = food.x - cx;
            let dy = food.y - cy;
            let dist = dx.hypot(dy);
            if dist <= OBS_VIEW {
                pellets.push((dist, dx, dy, food.mass));
            }
        }
        for e in &self.ejected {
            let dx = e.x - cx;
            let dy = e.y - cy;
            let dist = dx.hypot(dy);
            if dist <= OBS_VIEW {
                pellets.push((dist, dx, dy, e.mass));
            }
        }
        pellets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for &(dist, dx, dy, fm) in pellets.iter().take(OBS_K_FOOD) {
            let (ux, uy) = if dist > 1e-6 {
                (dx / dist, dy / dist)
            } else {
                (1.0, 0.0)
            };
            put(out, &mut i, (dist / OBS_VIEW).min(1.0));
            put(out, &mut i, uy);
            put(out, &mut i, ux);
            put(out, &mut i, (fm / 12.0).min(2.0));
        }
        i = OBS_SELF_DIM
            + OBS_K_OWN * OBS_OWN_F
            + OBS_K_ENEMY * OBS_ENEMY_F
            + OBS_N_SECTORS * 2
            + OBS_K_FOOD * OBS_FOOD_F;

        let mut viruses: Vec<(f64, f64, f64)> = Vec::new();
        for v in &self.viruses {
            let dx = v.x - cx;
            let dy = v.y - cy;
            let dist = dx.hypot(dy);
            if dist <= OBS_VIEW * 1.5 {
                viruses.push((dist, dx, dy));
            }
        }
        viruses.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for &(dist, dx, dy) in viruses.iter().take(OBS_K_VIRUS) {
            let (ux, uy) = if dist > 1e-6 {
                (dx / dist, dy / dist)
            } else {
                (1.0, 0.0)
            };
            put(out, &mut i, (dist / OBS_VIEW).min(1.5));
            put(out, &mut i, uy);
            put(out, &mut i, ux);
            put(
                out,
                &mut i,
                if largest_radius > size_from_mass(self.cfg.virus_mass) * self.cfg.eat_size_ratio {
                    1.0
                } else {
                    0.0
                },
            );
        }
    }

    fn leaderboard(&self, py: Python<'_>, you: Option<&str>) -> PyResult<Py<PyList>> {
        let mut rows: Vec<(u64, String, i64)> = self
            .players
            .iter()
            .filter(|p| p.total_mass() > 0.0)
            .map(|p| (p.id, p.name.clone(), round0(p.total_mass())))
            .collect();
        rows.sort_by(|a, b| b.2.cmp(&a.2));

        let local_id = you.and_then(|id| id.strip_prefix('p')?.parse::<u64>().ok());
        let local_rank = local_id.and_then(|id| rows.iter().position(|row| row.0 == id));
        let mut visible: Vec<usize> = (0..rows.len().min(10)).collect();
        if let Some(rank) = local_rank.filter(|rank| *rank >= 10) {
            visible.push(rank);
        }

        let list = PyList::empty(py);
        for rank in visible {
            let (id, name, score) = &rows[rank];
            let row = PyDict::new(py);
            row.set_item("name", name)?;
            row.set_item("score", score)?;
            row.set_item("rank", rank + 1)?;
            row.set_item("you", local_id == Some(*id))?;
            list.append(row)?;
        }
        Ok(list.unbind())
    }

    #[allow(clippy::too_many_arguments)]
    fn snapshot_payload(
        &self,
        py: Python<'_>,
        you: Option<&str>,
        player_name: &str,
        player_score: f64,
        camera_x: f64,
        camera_y: f64,
        camera_zoom: f64,
        blobs: &[(usize, usize)],
        food_indices: &[usize],
        ejected_indices: &[usize],
        virus_indices: &[usize],
    ) -> PyResult<Py<PyDict>> {
        let payload = PyDict::new(py);
        payload.set_item("type", "state")?;
        payload.set_item("you", you)?;

        let world = PyDict::new(py);
        world.set_item("w", self.cfg.world_width)?;
        world.set_item("h", self.cfg.world_height)?;
        payload.set_item("world", world)?;

        let camera = PyDict::new(py);
        camera.set_item("x", round2(camera_x))?;
        camera.set_item("y", round2(camera_y))?;
        camera.set_item("zoom", round3(camera_zoom))?;
        payload.set_item("camera", camera)?;

        let player = PyDict::new(py);
        player.set_item("name", player_name)?;
        player.set_item("score", round0(player_score))?;
        payload.set_item("player", player)?;

        payload.set_item("leaderboard", self.leaderboard(py, you)?)?;

        let blob_list = PyList::empty(py);
        for &(pi, bi) in blobs {
            let owner = &self.players[pi];
            let blob = &owner.blobs[bi];
            let entry = PyDict::new(py);
            entry.set_item("id", format!("b{}", blob.id))?;
            entry.set_item("playerId", format!("p{}", blob.player_id))?;
            entry.set_item("name", owner.name.as_str())?;
            entry.set_item("color", owner.color.as_str())?;
            entry.set_item("x", round2(blob.x))?;
            entry.set_item("y", round2(blob.y))?;
            entry.set_item("mass", round2(blob.mass))?;
            blob_list.append(entry)?;
        }
        payload.set_item("blobs", blob_list)?;

        let food_list = PyList::empty(py);
        for &fi in food_indices {
            let food = &self.foods[fi];
            let entry = PyDict::new(py);
            entry.set_item("id", format!("f{}", food.id))?;
            entry.set_item("x", round2(food.x))?;
            entry.set_item("y", round2(food.y))?;
            entry.set_item("mass", food.mass)?;
            entry.set_item("color", self.cfg.food_colors[food.color].as_str())?;
            food_list.append(entry)?;
        }
        payload.set_item("foods", food_list)?;

        let ejected_list = PyList::empty(py);
        for &ei in ejected_indices {
            let e = &self.ejected[ei];
            let entry = PyDict::new(py);
            entry.set_item("id", format!("e{}", e.id))?;
            entry.set_item("x", round2(e.x))?;
            entry.set_item("y", round2(e.y))?;
            entry.set_item("mass", e.mass)?;
            ejected_list.append(entry)?;
        }
        payload.set_item("ejected", ejected_list)?;

        let virus_list = PyList::empty(py);
        for &vi in virus_indices {
            let v = &self.viruses[vi];
            let entry = PyDict::new(py);
            entry.set_item("id", format!("v{}", v.id))?;
            entry.set_item("x", round2(v.x))?;
            entry.set_item("y", round2(v.y))?;
            entry.set_item("mass", v.mass)?;
            virus_list.append(entry)?;
        }
        payload.set_item("viruses", virus_list)?;

        Ok(payload.unbind())
    }
}

#[cfg(test)]
mod mechanics_checks {
    use super::*;

    fn empty_world() -> CoreWorld {
        let cfg = WorldConfig {
            food_target_count: 0,
            virus_min_count: 0,
            ..WorldConfig::default()
        };
        CoreWorld::new_internal(7, cfg)
    }

    #[test]
    fn start_cell_moves_at_the_reference_rate() {
        let mut world = empty_world();
        let id = world.add_player_internal("probe", 0.0);
        let player = world.player_pos(id).unwrap();
        world.players[player].blobs[0].x = 7_000.0;
        world.players[player].blobs[0].y = 7_000.0;
        world.set_input_raw(id, 8_000.0, 7_000.0, false, false);

        let size = world.players[player].blobs[0].size();
        let expected = 88.0 * size.powf(-0.439_675_4);
        world.move_blobs(1.0 / TICK_RATE, 0.0);

        let moved = world.players[player].blobs[0].x - 7_000.0;
        assert!((moved - expected).abs() < 1e-9);
    }

    #[test]
    fn pellet_pickup_uses_the_reference_overlap_boundary() {
        let check = |offset: f64| {
            let mut world = empty_world();
            let id = world.add_player_internal("probe", 0.0);
            let player = world.player_pos(id).unwrap();
            let blob = &mut world.players[player].blobs[0];
            blob.x = 1_000.0;
            blob.y = 1_000.0;
            let reach = blob.size() - size_from_mass(1.0) / world.cfg.eat_overlap_divisor;
            world.foods = vec![Food {
                id: 1,
                x: 1_000.0 + reach + offset,
                y: 1_000.0,
                mass: 1.0,
                color: 0,
                grow_elapsed: 0.0,
            }];
            world.rebuild_spatial_indexes();
            world.resolve_blob_food_collisions();
            world.foods.is_empty()
        };

        assert!(check(-0.01));
        assert!(!check(0.01));
    }
}

#[pymethods]
impl CoreWorld {
    #[new]
    #[pyo3(signature = (seed=None, config=None))]
    fn py_new(seed: Option<u64>, config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut cfg = WorldConfig::default();
        if let Some(overrides) = config {
            cfg.apply_overrides(overrides)?;
        }
        let seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x5eed)
        });
        Ok(CoreWorld::new_internal(seed, cfg))
    }

    #[pyo3(signature = (player_name, now, *, is_bot=false, bot_plugin=None, bot_team=None, color=None))]
    fn add_player(
        &mut self,
        player_name: &str,
        now: f64,
        is_bot: bool,
        bot_plugin: Option<String>,
        bot_team: Option<String>,
        color: Option<String>,
    ) -> PlayerHandle {
        let mut name: String = player_name
            .trim()
            .chars()
            .take(self.cfg.max_player_name_length)
            .collect();
        if name.is_empty() {
            name = "Cell".to_string();
        }

        let player_id = self.next_player_id;
        self.next_player_id += 1;
        let player_color = color.unwrap_or_else(|| {
            self.cfg.player_colors[self.players.len() % self.cfg.player_colors.len()].clone()
        });

        let (spawn_x, spawn_y) = self.player_spawn(size_from_mass(self.cfg.player_start_mass));
        let blob_id = self.next_blob_id;
        self.next_blob_id += 1;

        let player = Player {
            id: player_id,
            name: name.clone(),
            color: player_color,
            is_bot,
            bot_plugin,
            bot_team,
            blobs: vec![Blob {
                id: blob_id,
                player_id,
                x: spawn_x,
                y: spawn_y,
                mass: self.cfg.player_start_mass,
                boost_dx: 0.0,
                boost_dy: 0.0,
                boost_distance: 0.0,
                born_at: now,
                obs_vx: 0.0,
                obs_vy: 0.0,
            }],
            target_x: spawn_x,
            target_y: spawn_y,
            split_requested: false,
            eject_requested: false,
            last_split_at: -1e9,
            last_eject_at: -1e9,
            deaths: 0,
            kills: 0,
        };
        self.players.push(player);

        PlayerHandle {
            id: format!("p{player_id}"),
            name,
        }
    }

    fn remove_player(&mut self, player_id: &str) {
        if let Some(id) = self.parse_player_id(player_id) {
            self.players.retain(|p| p.id != id);
        }
    }

    #[pyo3(signature = (player_id, target_x, target_y, split=false, eject=false))]
    fn set_input(
        &mut self,
        player_id: &str,
        target_x: Option<&Bound<'_, PyAny>>,
        target_y: Option<&Bound<'_, PyAny>>,
        split: bool,
        eject: bool,
    ) {
        let Some(id) = self.parse_player_id(player_id) else {
            return;
        };
        let Some(pos) = self.player_pos(id) else {
            return;
        };
        let player = &mut self.players[pos];

        if let (Some(tx), Some(ty)) = (target_x, target_y) {
            if let (Ok(x), Ok(y)) = (tx.extract::<f64>(), ty.extract::<f64>()) {
                player.target_x = x;
                player.target_y = y;
            }
        }
        if split {
            player.split_requested = true;
        }
        if eject {
            player.eject_requested = true;
        }
    }

    #[pyo3(signature = (dt, now))]
    fn update(&mut self, dt: f64, now: f64) {
        // Observed-velocity bookkeeping (observer-only; no physics effect).
        let mut prev_pos: HashMap<u64, (f64, f64)> = HashMap::new();
        for player in &self.players {
            for blob in &player.blobs {
                prev_pos.insert(blob.id, (blob.x, blob.y));
            }
        }

        self.respawn_eliminated_players(now);
        self.apply_actions(now);
        self.move_viruses(dt);
        self.move_blobs(dt, now);
        self.move_ejected(dt);
        self.resolve_ejected_collisions();

        self.rebuild_spatial_indexes();
        self.resolve_virus_ejected_collisions();
        self.rebuild_spatial_indexes();
        self.resolve_blob_food_collisions();
        self.resolve_blob_ejected_collisions();
        self.resolve_blob_blob_collisions(now);
        self.resolve_virus_blob_collisions(now);
        self.apply_mass_decay(dt);
        self.autosplit_players(now);

        self.maintain_world_entities(dt);
        self.rebuild_spatial_indexes();

        if dt > 0.0 {
            let inv_dt = 1.0 / dt;
            for player in self.players.iter_mut() {
                for blob in player.blobs.iter_mut() {
                    if let Some((px, py)) = prev_pos.get(&blob.id) {
                        blob.obs_vx = (blob.x - px) * inv_dt;
                        blob.obs_vy = (blob.y - py) * inv_dt;
                    }
                }
            }
        }
    }

    fn fast_clone(&self) -> CoreWorld {
        self.clone()
    }

    /// Compact per-player state for bot view building:
    /// [(id, name, color, is_bot, plugin, team, target_x, target_y,
    ///   [(blob_id, x, y, mass, radius, vx, vy, can_merge_at), ...]), ...]
    /// vx/vy are observed velocities (true position deltas per second);
    /// can_merge_at is the absolute sim time when this cell may re-merge.
    #[allow(clippy::type_complexity)]
    fn players_compact(
        &self,
    ) -> Vec<(
        String,
        String,
        String,
        bool,
        Option<String>,
        Option<String>,
        f64,
        f64,
        Vec<(String, f64, f64, f64, f64, f64, f64, f64)>,
    )> {
        self.players
            .iter()
            .map(|p| {
                (
                    format!("p{}", p.id),
                    p.name.clone(),
                    p.color.clone(),
                    p.is_bot,
                    p.bot_plugin.clone(),
                    p.bot_team.clone(),
                    p.target_x,
                    p.target_y,
                    p.blobs
                        .iter()
                        .map(|b| {
                            (
                                format!("b{}", b.id),
                                b.x,
                                b.y,
                                b.mass,
                                b.size(),
                                b.obs_vx,
                                b.obs_vy,
                                self.merge_ready_at(b),
                            )
                        })
                        .collect(),
                )
            })
            .collect()
    }

    /// Compact food/ejected/virus state for bot view building.
    #[allow(clippy::type_complexity)]
    fn entities_compact(
        &self,
    ) -> (
        Vec<(String, f64, f64, f64, f64, String)>,
        Vec<(String, f64, f64, f64, f64, String, f64)>,
        Vec<(String, f64, f64, f64, f64)>,
    ) {
        let foods = self
            .foods
            .iter()
            .map(|f| {
                (
                    format!("f{}", f.id),
                    f.x,
                    f.y,
                    f.mass,
                    size_from_mass(f.mass),
                    self.cfg.food_colors[f.color].clone(),
                )
            })
            .collect();
        let ejected = self
            .ejected
            .iter()
            .map(|e| {
                (
                    format!("e{}", e.id),
                    e.x,
                    e.y,
                    e.mass,
                    size_from_mass(e.mass),
                    format!("p{}", e.owner_id),
                    -1.0,
                )
            })
            .collect();
        let viruses = self
            .viruses
            .iter()
            .map(|v| {
                (
                    format!("v{}", v.id),
                    v.x,
                    v.y,
                    v.mass,
                    size_from_mass(v.mass),
                )
            })
            .collect();
        (foods, ejected, viruses)
    }

    /// Cumulative respawn counts per player: [(player_id, deaths)].
    /// A respawn is counted when a player with zero blobs is revived at the
    /// start of an update; the initial spawn is not counted.
    fn death_counts(&self) -> Vec<(String, u64)> {
        self.players
            .iter()
            .map(|p| (format!("p{}", p.id), p.deaths))
            .collect()
    }

    /// Scenario seeding (training only; never called by the live server):
    /// blow a player apart into `parts` scattered fragments with full merge
    /// delay — produces the "I'm fragmented, how do I regroup" states that
    /// organic play rarely visits.
    #[pyo3(signature = (player_id, parts, now))]
    fn scatter_player(&mut self, player_id: &str, parts: usize, now: f64) {
        let Some(id) = self.parse_player_id(player_id) else {
            return;
        };
        let Some(pi) = self.player_pos(id) else {
            return;
        };
        if self.players[pi].blobs.is_empty() {
            return;
        }

        let total_mass = self.players[pi].total_mass();
        let max_parts = ((total_mass / self.cfg.player_min_mass).floor() as usize)
            .min(self.cfg.max_player_blobs);
        let parts = parts.clamp(1, max_parts.max(1));
        let (cx, cy) = self.players[pi].center();
        self.players[pi].blobs.clear();

        let part_mass = total_mass / parts as f64;
        for idx in 0..parts {
            let angle = (idx as f64 / parts as f64) * TWO_PI + self.rng.uniform(-0.3, 0.3);
            let (ux, uy) = (angle.cos(), angle.sin());
            let offset = self.rng.uniform(20.0, 160.0);
            let boost = self.cfg.player_split_boost * self.rng.uniform(0.3, 0.9);
            let blob_id = self.next_blob_id;
            self.next_blob_id += 1;
            let blob = Blob {
                id: blob_id,
                player_id: id,
                x: clamp(cx + ux * offset, 0.0, self.cfg.world_width),
                y: clamp(cy + uy * offset, 0.0, self.cfg.world_height),
                mass: part_mass,
                boost_dx: ux,
                boost_dy: uy,
                boost_distance: boost,
                born_at: now,
                obs_vx: 0.0,
                obs_vy: 0.0,
            };
            self.players[pi].blobs.push(blob);
        }
    }

    /// Scenario seeding: rescale a player's total mass (proportionally across
    /// blobs) — creates big-vs-small matchups on demand.
    #[pyo3(signature = (player_id, total_mass))]
    fn set_player_mass(&mut self, player_id: &str, total_mass: f64) {
        let Some(id) = self.parse_player_id(player_id) else {
            return;
        };
        let Some(pi) = self.player_pos(id) else {
            return;
        };
        let current = self.players[pi].total_mass();
        if current <= 0.0 {
            return;
        }
        let factor = total_mass / current;
        let min_mass = self.cfg.player_min_mass;
        for blob in self.players[pi].blobs.iter_mut() {
            blob.mass = (blob.mass * factor).max(min_mass);
        }
    }

    /// Encode RL observations for the given players at sim time `now`.
    /// `control` per player: (heading, prev_turn, prev_op, prev_speed).
    /// Returns raw little-endian f32 bytes, shape (len(player_ids), OBS_DIM) —
    /// decode with np.frombuffer(buf, dtype=np.float32).reshape(n, OBS_DIM).
    #[pyo3(signature = (player_ids, now, control))]
    fn observe(
        &self,
        py: Python<'_>,
        player_ids: Vec<String>,
        now: f64,
        control: Vec<(usize, usize, usize, f64)>,
    ) -> Py<PyBytes> {
        let mut buf = vec![0f32; player_ids.len() * OBS_DIM];
        for (k, pid) in player_ids.iter().enumerate() {
            if let Some(id) = self.parse_player_id(pid) {
                if let Some(pos) = self.player_pos(id) {
                    let ctl = control.get(k).copied().unwrap_or((0, 0, 0, 1.0));
                    self.observe_one(pos, now, ctl, &mut buf[k * OBS_DIM..(k + 1) * OBS_DIM]);
                }
            }
        }
        let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) };
        PyBytes::new(py, bytes).unbind()
    }

    /// Observation vector length per agent (kept in sync with obs.py).
    #[staticmethod]
    fn obs_dim() -> usize {
        OBS_DIM
    }

    /// Cumulative enemy-blob-eat counts per player: [(player_id, kills)].
    fn kill_counts(&self) -> Vec<(String, u64)> {
        self.players
            .iter()
            .map(|p| (format!("p{}", p.id), p.kills))
            .collect()
    }

    /// Quick (total_mass, blob_count) per player without building views.
    fn player_masses(&self) -> Vec<(String, f64, usize)> {
        self.players
            .iter()
            .map(|p| (format!("p{}", p.id), p.total_mass(), p.blobs.len()))
            .collect()
    }

    /// Full raw state for mechanics diagnostics.
    fn debug_state(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let state = PyDict::new(py);

        let players = PyList::empty(py);
        for p in &self.players {
            let entry = PyDict::new(py);
            entry.set_item("id", format!("p{}", p.id))?;
            entry.set_item("name", p.name.as_str())?;
            entry.set_item("target", (p.target_x, p.target_y))?;
            entry.set_item("last_split_at", p.last_split_at)?;
            entry.set_item("last_eject_at", p.last_eject_at)?;
            let blobs = PyList::empty(py);
            for b in &p.blobs {
                blobs.append((
                    format!("b{}", b.id),
                    b.x,
                    b.y,
                    b.mass,
                    b.boost_dx,
                    b.boost_dy,
                    self.merge_ready_at(b),
                ))?;
            }
            entry.set_item("blobs", blobs)?;
            players.append(entry)?;
        }
        state.set_item("players", players)?;

        let foods = PyList::empty(py);
        for f in &self.foods {
            foods.append((
                format!("f{}", f.id),
                f.x,
                f.y,
                f.mass,
                self.cfg.food_colors[f.color].as_str(),
            ))?;
        }
        state.set_item("foods", foods)?;

        let ejected = PyList::empty(py);
        for e in &self.ejected {
            ejected.append((
                format!("e{}", e.id),
                e.x,
                e.y,
                e.mass,
                format!("p{}", e.owner_id),
                e.boost_dx,
                e.boost_dy,
                e.boost_distance,
            ))?;
        }
        state.set_item("ejected", ejected)?;

        let viruses = PyList::empty(py);
        for v in &self.viruses {
            viruses.append((format!("v{}", v.id), v.x, v.y, v.mass))?;
        }
        state.set_item("viruses", viruses)?;

        Ok(state.unbind())
    }

    fn snapshot_for(&self, py: Python<'_>, player_id: &str) -> PyResult<Option<Py<PyDict>>> {
        let Some(id) = self.parse_player_id(player_id) else {
            return Ok(None);
        };
        let Some(pos) = self.player_pos(id) else {
            return Ok(None);
        };
        let player = &self.players[pos];
        let cfg = &self.cfg;

        let (cx, cy) = player.camera_center();
        let zoom = player.camera_zoom();
        let view_w = cfg.view_width / zoom + cfg.view_padding;
        let view_h = cfg.view_height / zoom + cfg.view_padding;

        let min_x = clamp(cx - view_w / 2.0, 0.0, cfg.world_width);
        let max_x = clamp(cx + view_w / 2.0, 0.0, cfg.world_width);
        let min_y = clamp(cy - view_h / 2.0, 0.0, cfg.world_height);
        let max_y = clamp(cy + view_h / 2.0, 0.0, cfg.world_height);

        let mut blob_hits: Vec<usize> = Vec::new();
        self.blob_grid
            .query_rect(min_x, min_y, max_x, max_y, &mut blob_hits);
        let blobs: Vec<(usize, usize)> = blob_hits
            .iter()
            .map(|&flat| self.blob_index[flat])
            .collect();

        let mut food_hits: Vec<usize> = Vec::new();
        self.food_grid
            .query_rect(min_x, min_y, max_x, max_y, &mut food_hits);
        let mut ejected_hits: Vec<usize> = Vec::new();
        self.ejected_grid
            .query_rect(min_x, min_y, max_x, max_y, &mut ejected_hits);

        let virus_indices: Vec<usize> = self
            .viruses
            .iter()
            .enumerate()
            .filter(|(_, v)| min_x <= v.x && v.x <= max_x && min_y <= v.y && v.y <= max_y)
            .map(|(i, _)| i)
            .collect();

        let payload = self.snapshot_payload(
            py,
            Some(player_id),
            &player.name,
            player.total_mass(),
            cx,
            cy,
            zoom,
            &blobs,
            &food_hits,
            &ejected_hits,
            &virus_indices,
        )?;
        Ok(Some(payload))
    }

    fn snapshot_overview(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let cfg = &self.cfg;
        let zoom =
            (cfg.view_width / cfg.world_width).min(cfg.view_height / cfg.world_height) * 0.92;

        let blobs: Vec<(usize, usize)> = self
            .players
            .iter()
            .enumerate()
            .flat_map(|(pi, p)| (0..p.blobs.len()).map(move |bi| (pi, bi)))
            .collect();
        let food_indices: Vec<usize> = (0..self.foods.len()).collect();
        let ejected_indices: Vec<usize> = (0..self.ejected.len()).collect();
        let virus_indices: Vec<usize> = (0..self.viruses.len()).collect();

        self.snapshot_payload(
            py,
            None,
            "Spectator",
            0.0,
            cfg.world_width * 0.5,
            cfg.world_height * 0.5,
            clamp(zoom, 0.05, 1.35),
            &blobs,
            &food_indices,
            &ejected_indices,
            &virus_indices,
        )
    }
}
