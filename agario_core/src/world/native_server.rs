use serde::Serialize;

use super::{clamp, round0, round2, round3, CoreWorld};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WorldSize {
    w: f64,
    h: f64,
}

#[derive(Serialize)]
struct Camera {
    x: f64,
    y: f64,
    zoom: f64,
}

#[derive(Serialize)]
struct LocalPlayer<'a> {
    name: &'a str,
    score: i64,
}

#[derive(Serialize)]
struct LeaderboardRow<'a> {
    name: &'a str,
    score: i64,
    rank: usize,
    you: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BlobRow<'a> {
    id: String,
    player_id: String,
    name: &'a str,
    color: &'a str,
    x: f64,
    y: f64,
    mass: f64,
}

#[derive(Serialize)]
struct FoodRow<'a> {
    id: String,
    x: f64,
    y: f64,
    mass: f64,
    color: &'a str,
}

#[derive(Serialize)]
struct MassRow {
    id: String,
    x: f64,
    y: f64,
    mass: f64,
}

#[derive(Serialize)]
struct Snapshot<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    you: Option<String>,
    world: WorldSize,
    camera: Camera,
    player: LocalPlayer<'a>,
    leaderboard: Vec<LeaderboardRow<'a>>,
    blobs: Vec<BlobRow<'a>>,
    foods: Vec<FoodRow<'a>>,
    ejected: Vec<MassRow>,
    viruses: Vec<MassRow>,
}

struct SnapshotView<'a> {
    local_id: Option<u64>,
    player_name: &'a str,
    player_score: f64,
    camera: (f64, f64, f64),
    bounds: (f64, f64, f64, f64),
}

impl CoreWorld {
    pub fn snapshot_json_for(&self, player_id: u64) -> Option<String> {
        let player = self.players.iter().find(|player| player.id == player_id)?;
        let (cx, cy) = player.camera_center();
        let zoom = player.camera_zoom();
        let view_w = self.cfg.view_width / zoom + self.cfg.view_padding * 2.0;
        let view_h = self.cfg.view_height / zoom + self.cfg.view_padding * 2.0;
        let min_x = clamp(cx - view_w / 2.0, 0.0, self.cfg.world_width);
        let max_x = clamp(cx + view_w / 2.0, 0.0, self.cfg.world_width);
        let min_y = clamp(cy - view_h / 2.0, 0.0, self.cfg.world_height);
        let max_y = clamp(cy + view_h / 2.0, 0.0, self.cfg.world_height);

        Some(self.snapshot_json(SnapshotView {
            local_id: Some(player_id),
            player_name: &player.name,
            player_score: player.total_mass(),
            camera: (cx, cy, zoom),
            bounds: (min_x, min_y, max_x, max_y),
        }))
    }

    pub fn overview_json(&self) -> String {
        let zoom = (self.cfg.view_width / self.cfg.world_width)
            .min(self.cfg.view_height / self.cfg.world_height)
            * 0.92;
        self.snapshot_json(SnapshotView {
            local_id: None,
            player_name: "Spectator",
            player_score: 0.0,
            camera: (
                self.cfg.world_width * 0.5,
                self.cfg.world_height * 0.5,
                clamp(zoom, 0.001, 1.35),
            ),
            bounds: (0.0, 0.0, self.cfg.world_width, self.cfg.world_height),
        })
    }

    fn snapshot_json(&self, view: SnapshotView<'_>) -> String {
        let (min_x, min_y, max_x, max_y) = view.bounds;
        let (camera_x, camera_y, camera_zoom) = view.camera;
        let mut rankings: Vec<_> = self
            .players
            .iter()
            .filter(|player| player.total_mass() > 0.0)
            .map(|player| (player, round0(player.total_mass())))
            .collect();
        rankings.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        let local_rank = view
            .local_id
            .and_then(|id| rankings.iter().position(|(player, _)| player.id == id));
        let mut visible_ranks: Vec<usize> = (0..rankings.len().min(10)).collect();
        if let Some(rank) = local_rank.filter(|rank| *rank >= 10) {
            visible_ranks.push(rank);
        }
        let leaderboard = visible_ranks
            .into_iter()
            .map(|rank| LeaderboardRow {
                name: rankings[rank].0.name.as_str(),
                score: rankings[rank].1,
                rank: rank + 1,
                you: view.local_id == Some(rankings[rank].0.id),
            })
            .collect();

        let blobs = self
            .players
            .iter()
            .flat_map(|owner| {
                owner
                    .blobs
                    .iter()
                    .filter(move |blob| {
                        min_x <= blob.x && blob.x <= max_x && min_y <= blob.y && blob.y <= max_y
                    })
                    .map(move |blob| BlobRow {
                        id: format!("b{}", blob.id),
                        player_id: format!("p{}", blob.player_id),
                        name: owner.name.as_str(),
                        color: owner.color.as_str(),
                        x: round2(blob.x),
                        y: round2(blob.y),
                        mass: round2(blob.mass),
                    })
            })
            .collect();
        let foods = self
            .foods
            .iter()
            .filter(|food| min_x <= food.x && food.x <= max_x && min_y <= food.y && food.y <= max_y)
            .map(|food| FoodRow {
                id: format!("f{}", food.id),
                x: round2(food.x),
                y: round2(food.y),
                mass: food.mass,
                color: self.cfg.food_colors[food.color].as_str(),
            })
            .collect();
        let ejected = self
            .ejected
            .iter()
            .filter(|item| min_x <= item.x && item.x <= max_x && min_y <= item.y && item.y <= max_y)
            .map(|item| MassRow {
                id: format!("e{}", item.id),
                x: round2(item.x),
                y: round2(item.y),
                mass: item.mass,
            })
            .collect();
        let viruses = self
            .viruses
            .iter()
            .filter(|virus| {
                min_x <= virus.x && virus.x <= max_x && min_y <= virus.y && virus.y <= max_y
            })
            .map(|virus| MassRow {
                id: format!("v{}", virus.id),
                x: round2(virus.x),
                y: round2(virus.y),
                mass: virus.mass,
            })
            .collect();

        serde_json::to_string(&Snapshot {
            kind: "state",
            you: view.local_id.map(|id| format!("p{id}")),
            world: WorldSize {
                w: self.cfg.world_width,
                h: self.cfg.world_height,
            },
            camera: Camera {
                x: round2(camera_x),
                y: round2(camera_y),
                zoom: round3(camera_zoom),
            },
            player: LocalPlayer {
                name: view.player_name,
                score: round0(view.player_score),
            },
            leaderboard,
            blobs,
            foods,
            ejected,
            viruses,
        })
        .expect("snapshot serialization cannot fail")
    }
}
