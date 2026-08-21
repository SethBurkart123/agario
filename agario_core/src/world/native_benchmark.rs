use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::CoreWorld;
use crate::config::{WorldConfig, TICK_RATE};

const RATING_START: f64 = 1000.0;
const RATING_K: f64 = 24.0;

#[derive(Clone, Default)]
struct PlayerSample {
    mass_sum: f64,
    sqrt_mass_sum: f64,
    cells_sum: usize,
    samples: usize,
    peak_mass: f64,
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Metrics {
    pub mean_mass: f64,
    pub final_mass: f64,
    pub peak_mass: f64,
    pub mean_cells: f64,
    pub kills: f64,
    pub deaths: f64,
    pub splits: f64,
    pub ejects: f64,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchResult {
    pub seed: u64,
    pub reversed: bool,
    pub score: f64,
    pub candidate: Metrics,
    pub opponent: Metrics,
    pub realtime_factor: f64,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunResult {
    pub id: u64,
    pub candidate: String,
    pub opponent: String,
    pub games: usize,
    pub seconds: u64,
    pub players_per_policy: usize,
    pub mean_score: f64,
    pub candidate_metrics: Metrics,
    pub opponent_metrics: Metrics,
    #[serde(default, skip_serializing)]
    pub matches: Vec<MatchResult>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Standing {
    pub policy: String,
    pub rating: f64,
    pub games: usize,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Ladder {
    pub updated_at: u64,
    #[serde(default)]
    pub standings: Vec<Standing>,
    #[serde(default)]
    pub history: Vec<RunResult>,
}

pub struct Options {
    candidate: String,
    opponent: String,
    games: usize,
    seconds: u64,
    players: usize,
    seed: u64,
    out: PathBuf,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            candidate: "rl_v2_h1".into(),
            opponent: "solo_smart".into(),
            games: 6,
            seconds: 180,
            players: 16,
            seed: 20_260_821,
            out: "bot_solutions/rl_v2/results/ladder.json".into(),
        }
    }
}

pub fn run_cli() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_args()?;
    let mut ladder = load_ladder(&options.out);
    let mut ratings: HashMap<String, (f64, usize)> = ladder
        .standings
        .drain(..)
        .map(|row| (row.policy, (row.rating, row.games)))
        .collect();
    ratings
        .entry(options.candidate.clone())
        .or_insert((RATING_START, 0));
    ratings
        .entry(options.opponent.clone())
        .or_insert((RATING_START, 0));

    let mut matches = Vec::with_capacity(options.games);
    for game in 0..options.games {
        let seed = options.seed + (game / 2) as u64;
        let reversed = game % 2 == 1;
        let result = run_match(&options, seed, reversed);
        apply_rating(
            &mut ratings,
            &options.candidate,
            &options.opponent,
            result.score,
        );
        println!(
            "game {:>2}/{} seed {}{}: score {:.3}, mass {:.1} vs {:.1}, {:.0}x realtime",
            game + 1,
            options.games,
            seed,
            if reversed { " reversed" } else { "" },
            result.score,
            result.candidate.mean_mass,
            result.opponent.mean_mass,
            result.realtime_factor,
        );
        matches.push(result);
    }

    let run = RunResult {
        id: unix_time(),
        candidate: options.candidate.clone(),
        opponent: options.opponent.clone(),
        games: options.games,
        seconds: options.seconds,
        players_per_policy: options.players,
        mean_score: matches.iter().map(|row| row.score).sum::<f64>() / matches.len() as f64,
        candidate_metrics: mean_metrics(matches.iter().map(|row| &row.candidate)),
        opponent_metrics: mean_metrics(matches.iter().map(|row| &row.opponent)),
        matches,
    };
    println!(
        "result: {} {:.1}% vs {} | mass {:.1} vs {:.1} | kills {:.2} vs {:.2}",
        run.candidate,
        run.mean_score * 100.0,
        run.opponent,
        run.candidate_metrics.mean_mass,
        run.opponent_metrics.mean_mass,
        run.candidate_metrics.kills,
        run.opponent_metrics.kills,
    );

    ladder.updated_at = unix_time();
    ladder.standings = ratings
        .into_iter()
        .map(|(policy, (rating, games))| Standing {
            policy,
            rating,
            games,
        })
        .collect();
    ladder
        .standings
        .sort_by(|a, b| b.rating.total_cmp(&a.rating));
    ladder.history.push(run);
    if ladder.history.len() > 50 {
        ladder.history.drain(..ladder.history.len() - 50);
    }
    write_ladder(&options.out, &ladder)?;
    println!("wrote {}", options.out.display());
    Ok(())
}

fn parse_args() -> Result<Options, String> {
    let mut options = Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("missing value after {flag}"))
        };
        match flag.as_str() {
            "--candidate" => options.candidate = value()?,
            "--opponent" => options.opponent = value()?,
            "--games" => options.games = value()?.parse().map_err(|_| "invalid --games")?,
            "--seconds" => options.seconds = value()?.parse().map_err(|_| "invalid --seconds")?,
            "--players" => options.players = value()?.parse().map_err(|_| "invalid --players")?,
            "--seed" => options.seed = value()?.parse().map_err(|_| "invalid --seed")?,
            "--out" => options.out = value()?.into(),
            "--help" | "-h" => {
                return Err("usage: rl_v2_ladder [--candidate POLICY] [--opponent POLICY] [--games N] [--seconds N] [--players N] [--seed N] [--out PATH]".into());
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    for policy in [&options.candidate, &options.opponent] {
        if !matches!(
            policy.as_str(),
            "solo_smart" | "solo_smart_v2" | "rl_v2_fast" | "rl_v2_h1"
        ) {
            return Err(format!("unsupported native policy: {policy}"));
        }
    }
    if options.games == 0 || options.players == 0 || options.seconds == 0 {
        return Err("games, players, and seconds must be positive".into());
    }
    Ok(options)
}

fn run_match(options: &Options, seed: u64, reversed: bool) -> MatchResult {
    let total_players = options.players * 2;
    let mut cfg = WorldConfig::default();
    let area_scale = (total_players as f64 / 64.0).max(1.0);
    let area = cfg.world_width * cfg.world_height * area_scale;
    cfg.world_width = (area * 4.0 / 3.0).sqrt();
    cfg.world_height = cfg.world_width * 0.75;
    cfg.food_target_count = (cfg.food_target_count as f64 * area_scale.sqrt()) as usize;
    cfg.virus_min_count = (cfg.virus_min_count as f64 * area_scale.sqrt()) as usize;
    cfg.virus_max_count = cfg.virus_min_count * 3;

    let mut world = CoreWorld::new_internal(seed, cfg);
    let order = if reversed {
        [&options.opponent, &options.candidate]
    } else {
        [&options.candidate, &options.opponent]
    };
    let mut groups: HashMap<String, Vec<u64>> = HashMap::new();
    for policy in order {
        for index in 0..options.players {
            let (id, _) =
                world.add_native_player(&format!("{}-{}", policy, index + 1), 0.0, Some(policy));
            groups.entry(policy.clone()).or_default().push(id);
        }
    }

    let mut samples: HashMap<u64, PlayerSample> = groups
        .values()
        .flatten()
        .map(|id| (*id, PlayerSample::default()))
        .collect();
    let ticks = options.seconds * TICK_RATE as u64;
    let started = Instant::now();
    for tick in 0..ticks {
        let now = tick as f64 / TICK_RATE;
        world.tick_native_bots(now);
        world.step(1.0 / TICK_RATE, now);
        if tick % TICK_RATE as u64 == 0 {
            for player in &world.players {
                if let Some(sample) = samples.get_mut(&player.id) {
                    let mass = player.total_mass();
                    sample.mass_sum += mass;
                    sample.sqrt_mass_sum += mass.sqrt();
                    sample.cells_sum += player.blobs.len();
                    sample.samples += 1;
                    sample.peak_mass = sample.peak_mass.max(mass);
                }
            }
        }
    }
    let wall_seconds = started.elapsed().as_secs_f64().max(1e-6);

    let candidate = policy_metrics(&world, &groups[&options.candidate], &samples);
    let opponent = policy_metrics(&world, &groups[&options.opponent], &samples);
    let candidate_utility = utilities(&world, &groups[&options.candidate], &samples);
    let opponent_utility = utilities(&world, &groups[&options.opponent], &samples);
    let mut wins = 0.0;
    let mut comparisons = 0.0;
    for a in &candidate_utility {
        for b in &opponent_utility {
            wins += if a > b {
                1.0
            } else if (a - b).abs() < 1e-9 {
                0.5
            } else {
                0.0
            };
            comparisons += 1.0;
        }
    }

    MatchResult {
        seed,
        reversed,
        score: wins / comparisons,
        candidate,
        opponent,
        realtime_factor: options.seconds as f64 / wall_seconds,
    }
}

fn policy_metrics(world: &CoreWorld, ids: &[u64], samples: &HashMap<u64, PlayerSample>) -> Metrics {
    let mut result = Metrics::default();
    for id in ids {
        let player = &world.players[world.player_pos(*id).unwrap()];
        let sample = &samples[id];
        let n = sample.samples.max(1) as f64;
        result.mean_mass += sample.mass_sum / n;
        result.final_mass += player.total_mass();
        result.peak_mass += sample.peak_mass;
        result.mean_cells += sample.cells_sum as f64 / n;
        result.kills += player.kills as f64;
        result.deaths += player.deaths as f64;
        result.splits += player.splits as f64;
        result.ejects += player.ejects as f64;
    }
    scale_metrics(&mut result, 1.0 / ids.len() as f64);
    result
}

fn utilities(world: &CoreWorld, ids: &[u64], samples: &HashMap<u64, PlayerSample>) -> Vec<f64> {
    ids.iter()
        .map(|id| {
            let player = &world.players[world.player_pos(*id).unwrap()];
            let sample = &samples[id];
            sample.sqrt_mass_sum / sample.samples.max(1) as f64 + player.kills as f64
                - player.deaths as f64 * 0.5
        })
        .collect()
}

fn mean_metrics<'a>(rows: impl Iterator<Item = &'a Metrics>) -> Metrics {
    let rows: Vec<_> = rows.collect();
    let mut result = Metrics::default();
    for row in &rows {
        result.mean_mass += row.mean_mass;
        result.final_mass += row.final_mass;
        result.peak_mass += row.peak_mass;
        result.mean_cells += row.mean_cells;
        result.kills += row.kills;
        result.deaths += row.deaths;
        result.splits += row.splits;
        result.ejects += row.ejects;
    }
    scale_metrics(&mut result, 1.0 / rows.len() as f64);
    result
}

fn scale_metrics(metrics: &mut Metrics, factor: f64) {
    metrics.mean_mass *= factor;
    metrics.final_mass *= factor;
    metrics.peak_mass *= factor;
    metrics.mean_cells *= factor;
    metrics.kills *= factor;
    metrics.deaths *= factor;
    metrics.splits *= factor;
    metrics.ejects *= factor;
}

fn apply_rating(
    ratings: &mut HashMap<String, (f64, usize)>,
    candidate: &str,
    opponent: &str,
    score: f64,
) {
    let a = ratings[candidate].0;
    let b = ratings[opponent].0;
    let expected = 1.0 / (1.0 + 10.0_f64.powf((b - a) / 400.0));
    let delta = RATING_K * (score - expected);
    let a_games = ratings[candidate].1 + 1;
    let b_games = ratings[opponent].1 + 1;
    ratings.insert(candidate.into(), (a + delta, a_games));
    ratings.insert(opponent.into(), (b - delta, b_games));
}

fn load_ladder(path: &Path) -> Ladder {
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

fn write_ladder(path: &Path, ladder: &Ladder) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension("json.tmp");
    fs::write(&temp, serde_json::to_vec_pretty(ladder)?)?;
    fs::rename(temp, path)?;
    Ok(())
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
