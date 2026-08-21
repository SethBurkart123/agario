use std::collections::{HashMap, HashSet};
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::http::{header, HeaderValue};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot, watch};

use crate::config::{WorldConfig, TICK_RATE};
use crate::world::CoreWorld;

const CLIENT_PROTOCOL: u32 = 4;
const INPUT_HZ: u32 = 90;
const BOTS_PER_CLASSIC_AREA: f64 = 512.0;
const INDEX_HTML: &str = include_str!("../../static/index.html");
const STYLES_CSS: &str = include_str!("../../static/styles.css");
const CLIENT_JS: &str = include_str!("../../static/client.js");

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BotSpec {
    plugin: String,
    count: usize,
    team: Option<String>,
    name_prefix: Option<String>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BotStatus {
    enabled: bool,
    runtime: &'static str,
    bot_specs: Vec<BotSpec>,
    active_bots: usize,
    worker_threads: usize,
}

#[derive(Default)]
struct Frames {
    revision: u64,
    players: HashMap<u64, Arc<str>>,
    overview: Option<Arc<str>>,
}

enum Command {
    Join {
        name: String,
        spectator: bool,
        reply: oneshot::Sender<JoinResult>,
    },
    Input {
        player_id: u64,
        x: f64,
        y: f64,
        split: bool,
        eject: bool,
    },
    Disconnect {
        player_id: Option<u64>,
        spectator_id: Option<u64>,
    },
    SetFullSpeed {
        enabled: bool,
        reply: oneshot::Sender<bool>,
    },
}

struct JoinResult {
    player_id: Option<u64>,
    spectator_id: Option<u64>,
    name: String,
    world_width: f64,
    world_height: f64,
    full_speed: bool,
}

#[derive(Clone)]
struct AppState {
    commands: mpsc::UnboundedSender<Command>,
    frames: watch::Receiver<Arc<Frames>>,
    bots: Arc<BotStatus>,
}

#[derive(Deserialize)]
struct ClientMessage {
    #[serde(rename = "type")]
    kind: String,
    name: Option<String>,
    spectator: Option<bool>,
    #[serde(rename = "clientProtocol")]
    client_protocol: Option<u32>,
    target: Option<Target>,
    split: Option<bool>,
    eject: Option<bool>,
    #[serde(rename = "fullSpeed")]
    full_speed: Option<bool>,
    ts: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct Target {
    x: f64,
    y: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WorldSize {
    w: f64,
    h: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Welcome<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    player_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    spectator_id: Option<String>,
    name: &'a str,
    spectator: bool,
    tick_rate: f64,
    input_hz: u32,
    full_speed: bool,
    world: WorldSize,
}

pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let bot_specs =
        parse_bot_specs(&env::var("AGARIO_BOT_SPECS").unwrap_or_else(|_| "solo_smart:16".into()))?;
    let (commands, frames) = spawn_engine(bot_specs.clone());
    let active_bots = bot_specs.iter().map(|spec| spec.count).sum();
    let state = AppState {
        commands,
        frames,
        bots: Arc::new(BotStatus {
            enabled: active_bots > 0,
            runtime: "rust",
            bot_specs,
            active_bots,
            worker_threads: rayon::current_num_threads(),
        }),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/overview", get(index))
        .route("/static/styles.css", get(styles))
        .route("/static/client.js", get(client_script))
        .route("/api/bots", get(bot_status))
        .route("/ws", get(websocket_upgrade))
        .with_state(state);

    let host = env::var("AGARIO_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port = env::var("AGARIO_PORT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8099);
    let address: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    println!("Agar.io Rust server listening on http://{address}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn parse_bot_specs(raw: &str) -> Result<Vec<BotSpec>, String> {
    raw.split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            let fields: Vec<_> = part.split(':').map(str::trim).collect();
            let plugin = fields.first().copied().unwrap_or_default();
            if !matches!(plugin, "solo_smart" | "solo_smart_v2") {
                return Err(format!("unsupported native bot policy: {plugin}"));
            }
            let count = fields
                .get(1)
                .filter(|value| !value.is_empty())
                .map(|value| value.parse::<usize>())
                .transpose()
                .map_err(|_| format!("invalid bot count in '{part}'"))?
                .unwrap_or(1);
            if count == 0 {
                return Err(format!("bot count must be positive in '{part}'"));
            }
            Ok(BotSpec {
                plugin: plugin.to_owned(),
                count,
                team: fields
                    .get(2)
                    .filter(|value| !value.is_empty() && **value != "-")
                    .map(|value| (*value).to_owned()),
                name_prefix: fields
                    .get(3)
                    .filter(|value| !value.is_empty() && **value != "-")
                    .map(|value| (*value).to_owned()),
            })
        })
        .collect()
}

fn spawn_engine(
    bot_specs: Vec<BotSpec>,
) -> (mpsc::UnboundedSender<Command>, watch::Receiver<Arc<Frames>>) {
    let (command_tx, mut command_rx) = mpsc::unbounded_channel();
    let (frame_tx, frame_rx) = watch::channel(Arc::new(Frames::default()));

    std::thread::Builder::new()
        .name("agario-engine".into())
        .spawn(move || {
            let seed = env::var("AGARIO_BOT_RANDOM_SEED")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(1337);
            let bot_count = bot_specs.iter().map(|spec| spec.count).sum::<usize>();
            let world_config = live_world_config(bot_count);
            println!(
                "arena: {:.0}x{:.0}, {} food, {}-{} viruses",
                world_config.world_width,
                world_config.world_height,
                world_config.food_target_count,
                world_config.virus_min_count,
                world_config.virus_max_count,
            );
            let mut world = CoreWorld::new_internal(seed, world_config);
            for spec in &bot_specs {
                let prefix = spec
                    .name_prefix
                    .clone()
                    .unwrap_or_else(|| title_case(&spec.plugin));
                for index in 0..spec.count {
                    world.add_native_player(
                        &format!("{prefix}-{}", index + 1),
                        0.0,
                        Some(&spec.plugin),
                    );
                }
            }

            let interval = Duration::from_secs_f64(1.0 / TICK_RATE);
            let mut human_players = HashSet::new();
            let mut spectators = HashSet::new();
            let mut next_spectator_id = 1_u64;
            let mut revision = 0_u64;
            let mut tick = 0_u64;
            let mut report_started = Instant::now();
            let mut report_busy = Duration::ZERO;
            let mut report_ticks = 0_u64;
            let mut next_tick = Instant::now();
            let mut next_frame = Instant::now();
            let mut full_speed = false;

            loop {
                let started = Instant::now();
                while let Ok(command) = command_rx.try_recv() {
                    match command {
                        Command::Join {
                            name,
                            spectator,
                            reply,
                        } => {
                            let (world_width, world_height) = world.dimensions();
                            if spectator {
                                let id = next_spectator_id;
                                next_spectator_id += 1;
                                spectators.insert(id);
                                let _ = reply.send(JoinResult {
                                    player_id: None,
                                    spectator_id: Some(id),
                                    name: "Spectator".into(),
                                    world_width,
                                    world_height,
                                    full_speed,
                                });
                            } else {
                                let (id, safe_name) =
                                    world.add_native_player(&name, tick as f64 / TICK_RATE, None);
                                human_players.insert(id);
                                let _ = reply.send(JoinResult {
                                    player_id: Some(id),
                                    spectator_id: None,
                                    name: safe_name,
                                    world_width,
                                    world_height,
                                    full_speed,
                                });
                            }
                        }
                        Command::Input {
                            player_id,
                            x,
                            y,
                            split,
                            eject,
                        } => {
                            world.set_native_input(player_id, x, y, split, eject);
                        }
                        Command::Disconnect {
                            player_id,
                            spectator_id,
                        } => {
                            if let Some(id) = player_id {
                                human_players.remove(&id);
                                world.remove_native_player(id);
                            }
                            if let Some(id) = spectator_id {
                                spectators.remove(&id);
                                if spectators.is_empty() {
                                    full_speed = false;
                                    next_tick = Instant::now();
                                }
                            }
                        }
                        Command::SetFullSpeed { enabled, reply } => {
                            full_speed = enabled;
                            next_tick = Instant::now();
                            let _ = reply.send(full_speed);
                        }
                    }
                }

                let now = tick as f64 / TICK_RATE;
                world.tick_native_bots(now);
                world.step(1.0 / TICK_RATE, now);
                tick += 1;
                revision += 1;

                if started >= next_frame && (!human_players.is_empty() || !spectators.is_empty()) {
                    let players = human_players
                        .iter()
                        .filter_map(|id| {
                            world
                                .snapshot_json_for(*id)
                                .map(|json| (*id, Arc::from(json)))
                        })
                        .collect();
                    let overview =
                        (!spectators.is_empty()).then(|| Arc::from(world.overview_json()));
                    frame_tx.send_replace(Arc::new(Frames {
                        revision,
                        players,
                        overview,
                    }));
                    next_frame += interval;
                    if started.saturating_duration_since(next_frame) > interval {
                        next_frame = started + interval;
                    }
                }

                let busy = started.elapsed();
                report_busy += busy;
                report_ticks += 1;
                if report_started.elapsed() >= Duration::from_secs(5) {
                    let average_ms = report_busy.as_secs_f64() * 1000.0 / report_ticks as f64;
                    println!(
                        "engine: {:.1} TPS, {:.2} ms/tick, {} bots, {} clients",
                        report_ticks as f64 / report_started.elapsed().as_secs_f64(),
                        average_ms,
                        bot_specs.iter().map(|spec| spec.count).sum::<usize>(),
                        human_players.len() + spectators.len(),
                    );
                    report_started = Instant::now();
                    report_busy = Duration::ZERO;
                    report_ticks = 0;
                }
                if full_speed {
                    next_tick = Instant::now();
                } else {
                    next_tick += interval;
                    let now = Instant::now();
                    if now < next_tick {
                        std::thread::sleep(next_tick - now);
                    } else if now.duration_since(next_tick) > interval {
                        next_tick = now;
                    }
                }
            }
        })
        .expect("failed to start simulation thread");

    (command_tx, frame_rx)
}

fn live_world_config(bot_count: usize) -> WorldConfig {
    let mut config = WorldConfig::default();
    let area_scale = (bot_count as f64 / BOTS_PER_CLASSIC_AREA).max(1.0);
    let area = config.world_width * config.world_height * area_scale;
    config.world_width = (area * 4.0 / 3.0).sqrt();
    config.world_height = config.world_width * 3.0 / 4.0;

    let entity_scale = area_scale.sqrt();
    config.food_target_count =
        ((config.food_target_count as f64 * entity_scale) as usize).min(10_000);
    config.virus_min_count = (config.virus_min_count as f64 * entity_scale) as usize;
    config.virus_max_count = config.virus_min_count * 3;
    config
}

fn title_case(plugin: &str) -> String {
    plugin
        .split('_')
        .map(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().collect::<String>() + chars.as_str())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

async fn websocket_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| websocket(socket, state))
}

async fn websocket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    let Some(Ok(Message::Text(first))) = receiver.next().await else {
        return;
    };
    let Ok(join) = serde_json::from_str::<ClientMessage>(&first) else {
        return;
    };
    if join.kind != "join" {
        return;
    }
    if join.client_protocol != Some(CLIENT_PROTOCOL) {
        let reload = format!(r#"{{"type":"reload","clientProtocol":{CLIENT_PROTOCOL}}}"#);
        let _ = sender.send(Message::Text(reload.into())).await;
        return;
    }

    let (reply_tx, reply_rx) = oneshot::channel();
    if state
        .commands
        .send(Command::Join {
            name: join.name.unwrap_or_else(|| "Cell".into()),
            spectator: join.spectator.unwrap_or(false),
            reply: reply_tx,
        })
        .is_err()
    {
        return;
    }
    let Ok(joined) = reply_rx.await else { return };
    let welcome = Welcome {
        kind: "welcome",
        player_id: joined.player_id.map(|id| format!("p{id}")),
        spectator_id: joined.spectator_id.map(|id| format!("s{id}")),
        name: &joined.name,
        spectator: joined.spectator_id.is_some(),
        tick_rate: TICK_RATE,
        input_hz: INPUT_HZ,
        full_speed: joined.full_speed,
        world: WorldSize {
            w: joined.world_width,
            h: joined.world_height,
        },
    };
    if sender
        .send(Message::Text(
            serde_json::to_string(&welcome).unwrap().into(),
        ))
        .await
        .is_err()
    {
        return;
    }

    let mut frames = state.frames.clone();
    let mut last_revision = 0;
    loop {
        tokio::select! {
            changed = frames.changed() => {
                if changed.is_err() { break; }
                let frame = frames.borrow_and_update().clone();
                if frame.revision == last_revision { continue; }
                last_revision = frame.revision;
                let payload = joined.player_id
                    .and_then(|id| frame.players.get(&id).cloned())
                    .or_else(|| joined.spectator_id.and_then(|_| frame.overview.clone()));
                if let Some(payload) = payload {
                    if sender.send(Message::Text(payload.to_string().into())).await.is_err() { break; }
                }
            }
            message = receiver.next() => {
                let Some(Ok(message)) = message else { break };
                match message {
                    Message::Text(text) => {
                        let Ok(input) = serde_json::from_str::<ClientMessage>(&text) else { continue };
                        if input.kind == "ping" {
                            let pong = serde_json::json!({"type": "pong", "ts": input.ts});
                            if sender.send(Message::Text(pong.to_string().into())).await.is_err() { break; }
                        } else if input.kind == "input" {
                            if let (Some(player_id), Some(target)) = (joined.player_id, input.target) {
                                let _ = state.commands.send(Command::Input {
                                    player_id,
                                    x: target.x,
                                    y: target.y,
                                    split: input.split.unwrap_or(false),
                                    eject: input.eject.unwrap_or(false),
                                });
                            }
                        } else if input.kind == "speed" && joined.spectator_id.is_some() {
                            let (reply_tx, reply_rx) = oneshot::channel();
                            if state.commands.send(Command::SetFullSpeed {
                                enabled: input.full_speed.unwrap_or(false),
                                reply: reply_tx,
                            }).is_err() {
                                break;
                            }
                            let Ok(full_speed) = reply_rx.await else { break };
                            let response = serde_json::json!({
                                "type": "speed",
                                "fullSpeed": full_speed,
                            });
                            if sender.send(Message::Text(response.to_string().into())).await.is_err() {
                                break;
                            }
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        }
    }
    let _ = state.commands.send(Command::Disconnect {
        player_id: joined.player_id,
        spectator_id: joined.spectator_id,
    });
}

async fn bot_status(State(state): State<AppState>) -> Json<BotStatus> {
    Json((*state.bots).clone())
}

async fn index() -> Response {
    static_response(INDEX_HTML, "text/html; charset=utf-8")
}

async fn styles() -> Response {
    static_response(STYLES_CSS, "text/css; charset=utf-8")
}

async fn client_script() -> Response {
    static_response(CLIENT_JS, "text/javascript; charset=utf-8")
}

fn static_response(body: &'static str, content_type: &'static str) -> Response {
    let mut response = body.into_response();
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}
