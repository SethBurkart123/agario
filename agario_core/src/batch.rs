//! Batched arenas: the entire RL env loop — apply actions, tick physics,
//! shape rewards, auto-reset, encode observations — for N arenas in one FFI
//! call, rayon-parallel across cores. Python does zero per-step work.
//!
//! Semantics mirror bot_solutions/rl_v1/env.py's ArenaEnv: turn-relative actions with
//! continuous speed, frame-skip ticking, continuing-task rewards
//! (sqrt-mass delta * scale + kill_bonus * kills, death replaced by
//! -death_penalty), truncation-only episodes with auto-reset, spawn-mass
//! jitter. Anchors are driven by a built-in flee-or-feed mini-brain so prey
//! pressure exists without any Python in the loop.

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

use crate::config::{WorldConfig, TICK_RATE};
use crate::world::{CoreWorld, OBS_DIM};

const DT: f64 = 1.0 / TICK_RATE;
const TURN_OFFSETS: [i64; 8] = [0, 1, -1, 2, -2, 4, -4, 8];
const N_DIRECTIONS: i64 = 16;
const TWO_PI: f64 = std::f64::consts::PI * 2.0;
const ACTION_TARGET_DISTANCE: f64 = 600.0;

fn distance_from_speed(speed: f64) -> f64 {
    speed.clamp(0.0, 1.0) * ACTION_TARGET_DISTANCE
}

/// splitmix64 — cheap deterministic per-arena RNG for scenario sampling.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64
}

struct AgentCtl {
    heading: i64,
    prev_turn: usize,
    prev_op: usize,
    prev_speed: f64,
    prev_sqrt: f64,
    prev_kills: u64,
    prev_deaths: u64,
}

struct Arena {
    world: CoreWorld,
    now: f64,
    size: f64,
    decision: u32,
    learners: Vec<u64>,
    anchors: Vec<u64>,
    ctl: Vec<AgentCtl>,
}

struct EpisodeStat {
    mass: f64,
    kills: f64,
    deaths: f64,
}

/// One imagined future: a cloned arena world in which ALL learners are
/// driven by externally supplied (batched GPU) policy actions, anchors by
/// the built-in brain, and one focal agent's shaped reward is scored.
struct Rollout {
    world: CoreWorld,
    now: f64,
    size: f64,
    root: usize,   // index into the search roots
    cand: usize,   // which candidate first-move
    sample: usize, // which averaging sample this clone is
    focal: u64,
    drivers: Vec<u64>,                  // learner ids driven by external actions
    ctl: Vec<(i64, usize, usize, f64)>, // (heading, prev_turn, prev_op, prev_speed)
    anchors: Vec<u64>,
    score: f64,
    discount: f64,
    prev_sqrt: f64,
    prev_kills: u64,
    prev_deaths: u64,
    step: u32,
}

struct SearchState {
    rollouts: Vec<Rollout>,
    n_roots: usize,
    k_cand: usize,
    m_samples: usize,
}

// Candidate first-moves are supplied per root by the caller (Gumbel-top-k
// sampled from the policy's joint logits, per Gumbel MuZero).

#[pyclass]
pub struct BatchedArenas {
    arenas: Vec<Arena>,
    n_learners: usize,
    n_anchors: usize,
    frame_skip: usize,
    episode_decisions: u32,
    anchor_every: u32,
    kill_bonus: f64,
    death_penalty: f64,
    mass_scale: f64,
    spawn_jitter: f64,
    size_lo: f64,
    size_hi: f64,
    seed_state: u64,
    pending_stats: Vec<EpisodeStat>,
    search: Option<SearchState>,
}

impl BatchedArenas {
    fn make_arena(&mut self) -> Arena {
        let seed = splitmix(&mut self.seed_state);
        let mut rng = seed ^ 0xA5A5_5A5A;
        let size = uniform(&mut rng, self.size_lo, self.size_hi);
        let area_scale = (size * size) / (2500.0 * 2500.0);

        let mut cfg = WorldConfig::default();
        cfg.world_width = size;
        cfg.world_height = size;
        cfg.food_target_count = ((220.0 * area_scale) as usize).max(40);
        cfg.virus_min_count = ((6.0 * area_scale) as usize).max(2);
        cfg.virus_max_count = (cfg.virus_min_count * 3).max(cfg.virus_min_count);
        let start_mass = cfg.player_start_mass;

        let mut world = CoreWorld::new_internal(seed, cfg);
        let learners: Vec<u64> = (0..self.n_learners)
            .map(|i| world.add_player_internal(&format!("learner-{i}"), 0.0))
            .collect();
        let anchors: Vec<u64> = (0..self.n_anchors)
            .map(|i| world.add_player_internal(&format!("anchor-{i}"), 0.0))
            .collect();

        if self.spawn_jitter > 0.0 {
            let lo = 1.0 / (1.0 + self.spawn_jitter);
            let hi = 1.0 + self.spawn_jitter;
            for id in learners.iter().chain(anchors.iter()) {
                let m = start_mass * uniform(&mut rng, lo, hi);
                world.scale_mass_internal(*id, m);
            }
        }

        let ctl = learners
            .iter()
            .map(|id| {
                let (mass, kills, deaths, _) = world.stats_of(*id);
                AgentCtl {
                    heading: (splitmix(&mut rng) % 16) as i64,
                    prev_turn: 0,
                    prev_op: 0,
                    prev_speed: 1.0,
                    prev_sqrt: mass.max(1.0).sqrt(),
                    prev_kills: kills,
                    prev_deaths: deaths,
                }
            })
            .collect();

        Arena {
            world,
            now: 0.0,
            size,
            decision: 0,
            learners,
            anchors,
            ctl,
        }
    }
}

fn drive(world: &mut CoreWorld, id: u64, heading: i64, speed: f64, op: usize, size: f64) {
    let Some((cx, cy)) = world.center_of(id) else {
        return;
    };
    let angle = (heading as f64 / N_DIRECTIONS as f64) * TWO_PI;
    let dist = distance_from_speed(speed);
    let tx = (cx + angle.cos() * dist).clamp(0.0, size);
    let ty = (cy + angle.sin() * dist).clamp(0.0, size);
    world.set_input_raw(id, tx, ty, op == 1, op == 2);
}

/// Built-in anchor brain: flee anything that can eat you, otherwise eat.
fn drive_anchor_world(world: &mut CoreWorld, id: u64, size: f64) {
    let Some((cx, cy)) = world.center_of(id) else {
        return;
    };
    let my_big = world.biggest_blob_mass_of(id);
    let (tx, ty) = match world.nearest_threat(id, cx, cy, my_big, 1.2, 600.0) {
        Some((bx, by)) => {
            let dx = cx - bx;
            let dy = cy - by;
            let mag = (dx * dx + dy * dy).sqrt().max(1e-6);
            (cx + dx / mag * 600.0, cy + dy / mag * 600.0)
        }
        None => world
            .nearest_food_to(cx, cy)
            .unwrap_or((size * 0.5, size * 0.5)),
    };
    world.set_input_raw(id, tx.clamp(0.0, size), ty.clamp(0.0, size), false, false);
}

fn drive_anchor(arena: &mut Arena, id: u64) {
    let size = arena.size;
    drive_anchor_world(&mut arena.world, id, size);
}

/// Steps one arena for one decision. Returns (rewards, truncated, stats).
fn step_arena(
    arena: &mut Arena,
    actions: &[f32],
    obs_out: &mut [f32],
    p: &Params,
) -> (Vec<f32>, bool, Option<EpisodeStat>) {
    let n_l = arena.learners.len();

    for (i, id) in arena.learners.clone().into_iter().enumerate() {
        let turn = (actions[i * 3] as usize).min(TURN_OFFSETS.len() - 1);
        let op = (actions[i * 3 + 1] as usize).min(2);
        let speed = (actions[i * 3 + 2] as f64).clamp(0.0, 1.0);
        let ctl = &mut arena.ctl[i];
        ctl.heading = (ctl.heading + TURN_OFFSETS[turn]).rem_euclid(N_DIRECTIONS);
        ctl.prev_turn = turn;
        ctl.prev_op = op;
        ctl.prev_speed = speed;
        let heading = ctl.heading;
        drive(&mut arena.world, id, heading, speed, op, arena.size);
    }
    if arena.decision % p.anchor_every == 0 {
        for id in arena.anchors.clone() {
            drive_anchor(arena, id);
        }
    }

    for _ in 0..p.frame_skip {
        arena.now += DT;
        arena.world.update_internal(DT, arena.now);
    }

    let mut rewards = vec![0f32; n_l];
    for (i, id) in arena.learners.iter().enumerate() {
        let (mass, kills, deaths, _) = arena.world.stats_of(*id);
        let new_sqrt = mass.max(1.0).sqrt();
        let ctl = &mut arena.ctl[i];
        let mut r = if deaths > ctl.prev_deaths {
            -p.death_penalty
        } else {
            (new_sqrt - ctl.prev_sqrt) * p.mass_scale
        };
        r += p.kill_bonus * (kills - ctl.prev_kills) as f64;
        rewards[i] = r as f32;
        ctl.prev_sqrt = new_sqrt;
        ctl.prev_kills = kills;
        ctl.prev_deaths = deaths;
    }

    arena.decision += 1;
    let truncated = arena.decision >= p.episode_decisions;
    let stat = if truncated {
        let mut mass_sum = 0.0;
        let mut kills_sum = 0.0;
        let mut deaths_sum = 0.0;
        for id in &arena.learners {
            let (mass, kills, deaths, _) = arena.world.stats_of(*id);
            mass_sum += mass;
            kills_sum += kills as f64;
            deaths_sum += deaths as f64;
        }
        Some(EpisodeStat {
            mass: mass_sum / n_l as f64,
            kills: kills_sum / n_l as f64,
            deaths: deaths_sum / n_l as f64,
        })
    } else {
        None
    };

    if !truncated {
        for (i, id) in arena.learners.iter().enumerate() {
            let ctl = &arena.ctl[i];
            arena.world.observe_by_id(
                *id,
                arena.now,
                (
                    ctl.heading.rem_euclid(16) as usize,
                    ctl.prev_turn,
                    ctl.prev_op,
                    ctl.prev_speed,
                ),
                &mut obs_out[i * OBS_DIM..(i + 1) * OBS_DIM],
            );
        }
    }
    (rewards, truncated, stat)
}

#[derive(Clone, Copy)]
struct Params {
    frame_skip: usize,
    episode_decisions: u32,
    anchor_every: u32,
    kill_bonus: f64,
    death_penalty: f64,
    mass_scale: f64,
}

#[pymethods]
impl BatchedArenas {
    #[new]
    #[pyo3(signature = (n_arenas, n_learners, n_anchors=2, seed=0,
        size_lo=2000.0, size_hi=4200.0, frame_skip=2, episode_decisions=1100,
        anchor_every=4, kill_bonus=1.0, death_penalty=2.0, mass_scale=0.1,
        spawn_jitter=1.2))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_arenas: usize,
        n_learners: usize,
        n_anchors: usize,
        seed: u64,
        size_lo: f64,
        size_hi: f64,
        frame_skip: usize,
        episode_decisions: u32,
        anchor_every: u32,
        kill_bonus: f64,
        death_penalty: f64,
        mass_scale: f64,
        spawn_jitter: f64,
    ) -> Self {
        let mut me = BatchedArenas {
            arenas: Vec::new(),
            n_learners,
            n_anchors,
            frame_skip,
            episode_decisions,
            anchor_every: anchor_every.max(1),
            kill_bonus,
            death_penalty,
            mass_scale,
            spawn_jitter,
            size_lo,
            size_hi,
            seed_state: seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x1234_5678),
            pending_stats: Vec::new(),
            search: None,
        };
        for _ in 0..n_arenas {
            let arena = me.make_arena();
            me.arenas.push(arena);
        }
        me
    }

    #[staticmethod]
    fn obs_dim() -> usize {
        OBS_DIM
    }

    fn total_agents(&self) -> usize {
        self.arenas.len() * self.n_learners
    }

    fn reset(&mut self, py: Python<'_>) -> Py<PyBytes> {
        let n_l = self.n_learners;
        let mut obs = vec![0f32; self.arenas.len() * n_l * OBS_DIM];
        py.allow_threads(|| {
            self.arenas
                .par_iter()
                .zip(obs.par_chunks_mut(n_l * OBS_DIM))
                .for_each(|(arena, chunk)| {
                    for (i, id) in arena.learners.iter().enumerate() {
                        let ctl = &arena.ctl[i];
                        arena.world.observe_by_id(
                            *id,
                            arena.now,
                            (
                                ctl.heading.rem_euclid(16) as usize,
                                ctl.prev_turn,
                                ctl.prev_op,
                                ctl.prev_speed,
                            ),
                            &mut chunk[i * OBS_DIM..(i + 1) * OBS_DIM],
                        );
                    }
                });
        });
        let bytes = unsafe { std::slice::from_raw_parts(obs.as_ptr() as *const u8, obs.len() * 4) };
        PyBytes::new(py, bytes).unbind()
    }

    /// actions: flat f32 [n_arenas * n_learners * 3] of (turn, op, speed).
    /// Returns (obs_bytes, rewards_bytes f32, truncations_bytes u8).
    fn step(
        &mut self,
        py: Python<'_>,
        actions: Vec<f32>,
    ) -> (Py<PyBytes>, Py<PyBytes>, Py<PyBytes>) {
        let n_l = self.n_learners;
        let n_total = self.arenas.len() * n_l;
        assert_eq!(actions.len(), n_total * 3, "actions length mismatch");

        let params = Params {
            frame_skip: self.frame_skip,
            episode_decisions: self.episode_decisions,
            anchor_every: self.anchor_every,
            kill_bonus: self.kill_bonus,
            death_penalty: self.death_penalty,
            mass_scale: self.mass_scale,
        };

        let mut obs = vec![0f32; n_total * OBS_DIM];
        let mut rewards = vec![0f32; n_total];
        let mut truncs = vec![0u8; n_total];

        let results: Vec<(Vec<f32>, bool, Option<EpisodeStat>)> = py.allow_threads(|| {
            self.arenas
                .par_iter_mut()
                .zip(actions.par_chunks(n_l * 3))
                .zip(obs.par_chunks_mut(n_l * OBS_DIM))
                .map(|((arena, act_chunk), obs_chunk)| {
                    step_arena(arena, act_chunk, obs_chunk, &params)
                })
                .collect()
        });

        // Sequential epilogue: auto-reset truncated arenas, fill their obs.
        for (a, (arena_rewards, truncated, stat)) in results.into_iter().enumerate() {
            for (i, r) in arena_rewards.into_iter().enumerate() {
                rewards[a * n_l + i] = r;
            }
            if truncated {
                if let Some(s) = stat {
                    self.pending_stats.push(s);
                }
                let fresh = self.make_arena();
                self.arenas[a] = fresh;
                let arena = &self.arenas[a];
                for (i, id) in arena.learners.iter().enumerate() {
                    let ctl = &arena.ctl[i];
                    arena.world.observe_by_id(
                        *id,
                        arena.now,
                        (
                            ctl.heading.rem_euclid(16) as usize,
                            ctl.prev_turn,
                            ctl.prev_op,
                            ctl.prev_speed,
                        ),
                        &mut obs[(a * n_l + i) * OBS_DIM..(a * n_l + i + 1) * OBS_DIM],
                    );
                }
                for i in 0..n_l {
                    truncs[a * n_l + i] = 1;
                }
            }
        }

        let ob = unsafe { std::slice::from_raw_parts(obs.as_ptr() as *const u8, obs.len() * 4) };
        let rb =
            unsafe { std::slice::from_raw_parts(rewards.as_ptr() as *const u8, rewards.len() * 4) };
        (
            PyBytes::new(py, ob).unbind(),
            PyBytes::new(py, rb).unbind(),
            PyBytes::new(py, &truncs).unbind(),
        )
    }

    /// Per-arena, per-learner (mass, kills, deaths) — for gate comparisons.
    fn learner_stats(&self) -> Vec<Vec<(f64, u64, u64)>> {
        self.arenas
            .iter()
            .map(|a| {
                a.learners
                    .iter()
                    .map(|id| {
                        let (m, k, d, _) = a.world.stats_of(*id);
                        (m, k, d)
                    })
                    .collect()
            })
            .collect()
    }

    /// Drain accumulated per-episode stats: [(mass, kills, deaths), ...].
    fn take_episode_stats(&mut self) -> Vec<(f64, f64, f64)> {
        self.pending_stats
            .drain(..)
            .map(|s| (s.mass, s.kills, s.deaths))
            .collect()
    }

    // -----------------------------------------------------------------
    // Search session: imagined futures for AlphaZero-style targets.
    // -----------------------------------------------------------------

    /// Begin a search over `roots` = [(arena_idx, learner_idx), ...].
    /// Candidate first-moves are explicit per root: cand_turns/cand_ops are
    /// flat (n_roots * k) arrays; speed comes from root_speeds. m_samples
    /// clones per candidate are averaged.
    /// Returns the number of externally-driven rows per observe/apply call.
    fn search_begin(
        &mut self,
        py: Python<'_>,
        roots: Vec<(usize, usize)>,
        root_speeds: Vec<f64>,
        m_samples: usize,
        cand_turns: Vec<usize>,
        cand_ops: Vec<usize>,
    ) -> usize {
        let n_l = self.n_learners;
        assert_eq!(cand_turns.len(), cand_ops.len());
        assert_eq!(cand_turns.len() % roots.len().max(1), 0);
        let k = cand_turns.len() / roots.len().max(1);
        let mut rollouts: Vec<Rollout> = Vec::with_capacity(roots.len() * k * m_samples);
        py.allow_threads(|| {
            for (r, &(a_idx, agent_idx)) in roots.iter().enumerate() {
                let arena = &self.arenas[a_idx];
                let focal = arena.learners[agent_idx];
                for c in 0..k {
                    for s in 0..m_samples {
                        let mut world = arena.world.clone_internal();
                        let (turn, op, speed) = (
                            cand_turns[r * k + c].min(TURN_OFFSETS.len() - 1),
                            cand_ops[r * k + c].min(2),
                            root_speeds[r],
                        );
                        // Seed every learner's control state from the live arena.
                        let mut ctl: Vec<(i64, usize, usize, f64)> = arena
                            .ctl
                            .iter()
                            .map(|x| (x.heading, x.prev_turn, x.prev_op, x.prev_speed))
                            .collect();
                        // Apply the candidate first move to the focal agent.
                        let heading =
                            (ctl[agent_idx].0 + TURN_OFFSETS[turn]).rem_euclid(N_DIRECTIONS);
                        ctl[agent_idx] = (heading, turn, op, speed);
                        drive(&mut world, focal, heading, speed, op, arena.size);
                        let (mass, kills, deaths, _) = world.stats_of(focal);
                        rollouts.push(Rollout {
                            world,
                            now: arena.now,
                            size: arena.size,
                            root: r,
                            cand: c,
                            sample: s,
                            focal,
                            drivers: arena.learners.clone(),
                            ctl,
                            anchors: arena.anchors.clone(),
                            score: 0.0,
                            discount: 1.0,
                            prev_sqrt: mass.max(1.0).sqrt(),
                            prev_kills: kills,
                            prev_deaths: deaths,
                            step: 0,
                        });
                    }
                }
            }
        });
        let n = rollouts.len() * n_l;
        self.search = Some(SearchState {
            rollouts,
            n_roots: roots.len(),
            k_cand: k,
            m_samples,
        });
        n
    }

    /// Observations for every externally-driven player in every rollout,
    /// rollout-major then learner-minor. Shape: (n_rollouts * n_learners,
    /// OBS_DIM) as f32 bytes.
    fn search_obs(&self, py: Python<'_>) -> Py<PyBytes> {
        let st = self.search.as_ref().expect("no active search");
        let n_l = self.n_learners;
        let mut obs = vec![0f32; st.rollouts.len() * n_l * OBS_DIM];
        py.allow_threads(|| {
            st.rollouts
                .par_iter()
                .zip(obs.par_chunks_mut(n_l * OBS_DIM))
                .for_each(|(ro, chunk)| {
                    for (i, id) in ro.drivers.iter().enumerate() {
                        let c = &ro.ctl[i];
                        ro.world.observe_by_id(
                            *id,
                            ro.now,
                            (c.0.rem_euclid(16) as usize, c.1, c.2, c.3),
                            &mut chunk[i * OBS_DIM..(i + 1) * OBS_DIM],
                        );
                    }
                });
        });
        let bytes = unsafe { std::slice::from_raw_parts(obs.as_ptr() as *const u8, obs.len() * 4) };
        PyBytes::new(py, bytes).unbind()
    }

    /// Apply batched policy actions (turn, op, speed per driven row, same
    /// layout as search_obs) to every rollout, then tick frame_skip. The
    /// focal agent's shaped reward accrues with the same continuing-death
    /// semantics as training. Sample index jitters NON-focal first moves so
    /// the m clones cover branching futures instead of replaying one.
    fn search_apply_tick(&mut self, py: Python<'_>, actions: Vec<f32>) {
        let n_l = self.n_learners;
        let frame_skip = self.frame_skip;
        let kill_bonus = self.kill_bonus;
        let death_penalty = self.death_penalty;
        let mass_scale = self.mass_scale;
        let anchor_every = self.anchor_every;
        let st = self.search.as_mut().expect("no active search");
        assert_eq!(actions.len(), st.rollouts.len() * n_l * 3);

        py.allow_threads(|| {
            st.rollouts
                .par_iter_mut()
                .zip(actions.par_chunks(n_l * 3))
                .for_each(|(ro, acts)| {
                    let focal_idx = ro.drivers.iter().position(|d| *d == ro.focal).unwrap();
                    for (i, id) in ro.drivers.clone().into_iter().enumerate() {
                        let mut turn = (acts[i * 3] as usize).min(TURN_OFFSETS.len() - 1);
                        let op = (acts[i * 3 + 1] as usize).min(2);
                        let speed = (acts[i * 3 + 2] as f64).clamp(0.0, 1.0);
                        if ro.step == 0 && i != focal_idx && ro.sample > 0 {
                            turn = (turn + ro.sample) % TURN_OFFSETS.len();
                        }
                        let c = &mut ro.ctl[i];
                        c.0 = (c.0 + TURN_OFFSETS[turn]).rem_euclid(N_DIRECTIONS);
                        c.1 = turn;
                        c.2 = op;
                        c.3 = speed;
                        let heading = c.0;
                        drive(&mut ro.world, id, heading, speed, op, ro.size);
                    }
                    if ro.step % anchor_every == 0 {
                        for id in ro.anchors.clone() {
                            drive_anchor_world(&mut ro.world, id, ro.size);
                        }
                    }
                    for _ in 0..frame_skip {
                        ro.now += DT;
                        ro.world.update_internal(DT, ro.now);
                    }
                    let (mass, kills, deaths, _) = ro.world.stats_of(ro.focal);
                    let new_sqrt = mass.max(1.0).sqrt();
                    let mut r = if deaths > ro.prev_deaths {
                        -death_penalty
                    } else {
                        (new_sqrt - ro.prev_sqrt) * mass_scale
                    };
                    r += kill_bonus * (kills - ro.prev_kills) as f64;
                    ro.score += ro.discount * r;
                    ro.prev_sqrt = new_sqrt;
                    ro.prev_kills = kills;
                    ro.prev_deaths = deaths;
                    ro.discount *= 0.995;
                    ro.step += 1;
                });
        });
    }

    /// Focal-agent observations for the terminal value bootstrap:
    /// (n_rollouts, OBS_DIM) f32 bytes.
    fn search_focal_obs(&self, py: Python<'_>) -> Py<PyBytes> {
        let st = self.search.as_ref().expect("no active search");
        let mut obs = vec![0f32; st.rollouts.len() * OBS_DIM];
        py.allow_threads(|| {
            st.rollouts
                .par_iter()
                .zip(obs.par_chunks_mut(OBS_DIM))
                .for_each(|(ro, chunk)| {
                    let i = ro.drivers.iter().position(|d| *d == ro.focal).unwrap();
                    let c = &ro.ctl[i];
                    ro.world.observe_by_id(
                        ro.focal,
                        ro.now,
                        (c.0.rem_euclid(16) as usize, c.1, c.2, c.3),
                        chunk,
                    );
                });
        });
        let bytes = unsafe { std::slice::from_raw_parts(obs.as_ptr() as *const u8, obs.len() * 4) };
        PyBytes::new(py, bytes).unbind()
    }

    /// Close the search: fold in discounted terminal values, average across
    /// samples. Returns per-root candidate Q values, shape (n_roots *
    /// K_CANDIDATES) row-major.
    fn search_finish(&mut self, values: Vec<f64>) -> Vec<f64> {
        let st = self.search.take().expect("no active search");
        assert_eq!(values.len(), st.rollouts.len());
        let mut q = vec![0f64; st.n_roots * st.k_cand];
        let m = st.m_samples as f64;
        for (ro, v) in st.rollouts.iter().zip(values) {
            q[ro.root * st.k_cand + ro.cand] += (ro.score + ro.discount * v) / m;
        }
        q
    }
}
