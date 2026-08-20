const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const scoreEl = document.getElementById("score");
const leaderboardEl = document.getElementById("leaderboard");

const BG_COLOR = "#F4FBFF";
const GRID_COLOR = "#CDD4D7";
const VIRUS_FILL = "#34FF32";
const VIRUS_BORDER = "#2EE52C";
const PLAYER_BORDER_WORLD = 10;
const VIRUS_BORDER_WORLD = 10;
const EJECTED_BORDER_WORLD = 2.2;
const VIRUS_SPIKE_SPACING_WORLD = 6.0;
const SNAPSHOT_BLEND_MS = 120;
const CLIENT_PROTOCOL = 3;

let ws;
let world = { w: 14142.135623730952, h: 14142.135623730952 };
let state = null;
let playerId = null;
let spectatorMode = false;

let splitQueued = false;
let ejectQueued = false;

let serverInputHz = 60;
let nextInputAt = 0;

const mouse = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
const camera = { x: world.w / 2, y: world.h / 2, zoom: 1 };
const cameraTarget = { x: world.w / 2, y: world.h / 2, zoom: 1 };
let cameraReady = false;

let lastFrameAt = performance.now();

const blobVisuals = new Map();
const consumeFx = new Map();
const visibleFoods = new Map();
const visibleEjected = new Map();
const locallyConsumedUntil = new Map();

const nameKey = "agar-clone-name";
const randomName = () => `Cell-${Math.floor(Math.random() * 900 + 100)}`;
const playerName = localStorage.getItem(nameKey) || randomName();
localStorage.setItem(nameKey, playerName);
const pathIsOverview = window.location.pathname === "/overview";
const queryOverview = new URLSearchParams(window.location.search).get("overview") === "1";
const startInOverview = pathIsOverview || queryOverview;
spectatorMode = startInOverview;

function hashString(value) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
  }
  return hash / 4294967295;
}

function clamp(value, minValue, maxValue) {
  return Math.min(maxValue, Math.max(minValue, value));
}

function worldStroke(worldUnits, minPx = 0.75, maxPx = 56) {
  return clamp(worldUnits * camera.zoom, minPx, maxPx);
}

function autoBorderColor(hex) {
  const match = /^#([0-9a-fA-F]{6})$/.exec(hex);
  if (!match) return hex;
  const raw = match[1];
  const darken = (part) => {
    let value = Math.floor(parseInt(part, 16) * 0.9);
    if (value < 20) value = Math.max(0, value - 2);
    return value;
  };
  const r = darken(raw.slice(0, 2));
  const g = darken(raw.slice(2, 4));
  const b = darken(raw.slice(4, 6));
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

function resize() {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
  canvas.style.width = `${window.innerWidth}px`;
  canvas.style.height = `${window.innerHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

window.addEventListener("resize", resize);
window.addEventListener("mousemove", (evt) => {
  mouse.x = evt.clientX;
  mouse.y = evt.clientY;
});

window.addEventListener("keydown", (evt) => {
  if (spectatorMode) return;
  if (evt.code === "Space") {
    if (!evt.repeat) splitQueued = true;
    evt.preventDefault();
  }
  if (evt.code === "KeyW") {
    if (!evt.repeat) ejectQueued = true;
    evt.preventDefault();
  }
});

function connect() {
  if (
    ws &&
    (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)
  ) {
    return;
  }
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws`);
  ws = socket;

  socket.addEventListener("open", () => {
    statusEl.textContent = `Connected as ${playerName}`;
    socket.send(
      JSON.stringify({
        type: "join",
        name: playerName,
        spectator: startInOverview,
        clientProtocol: CLIENT_PROTOCOL,
      }),
    );
  });

  socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "reload") {
      location.reload();
      return;
    }

    if (data.type === "welcome") {
      spectatorMode = Boolean(data.spectator);
      playerId = data.playerId || null;
      world = data.world;
      serverInputHz = Math.max(20, Number(data.inputHz || 60));
      if (spectatorMode) {
        statusEl.textContent = "Connected as spectator";
        scoreEl.textContent = "Spectator";
      } else {
        statusEl.textContent = "";
      }
      return;
    }

    if (data.type === "state") {
      state = data;
      cameraTarget.x = state.camera.x;
      cameraTarget.y = state.camera.y;
      cameraTarget.zoom = state.camera.zoom;
      if (!spectatorMode) {
        scoreEl.textContent = `Score: ${state.player.score}`;
      }
      drawLeaderboard(state.leaderboard);
      syncBlobVisuals(state.blobs, performance.now());
      syncConsumedEffects(state);
      return;
    }
  });

  socket.addEventListener("close", () => {
    if (ws !== socket) return;
    ws = null;
    statusEl.textContent = "Disconnected. Reconnecting...";
    playerId = null;
    spectatorMode = false;
    cameraReady = false;
    state = null;
    blobVisuals.clear();
    consumeFx.clear();
    visibleFoods.clear();
    visibleEjected.clear();
    locallyConsumedUntil.clear();
    setTimeout(connect, 1000);
  });

  socket.addEventListener("error", () => {
    if (ws !== socket) return;
    statusEl.textContent = "Connection error";
    socket.close();
  });
}

function drawLeaderboard(rows) {
  leaderboardEl.innerHTML = "";
  for (const row of rows) {
    const li = document.createElement("li");
    li.textContent = row.name;
    li.value = row.rank;
    if (row.you) li.classList.add("you");
    leaderboardEl.append(li);
  }
}

function syncBlobVisuals(blobs, receivedAt) {
  const seen = new Set();

  for (const blob of blobs) {
    seen.add(blob.id);
    const existing = blobVisuals.get(blob.id);
    if (!existing) {
      blobVisuals.set(blob.id, {
        id: blob.id,
        playerId: blob.playerId,
        name: blob.name,
        color: blob.color,
        x: blob.x,
        y: blob.y,
        sx: blob.x,
        sy: blob.y,
        tx: blob.x,
        ty: blob.y,
        mass: blob.mass,
        smass: blob.mass,
        tmass: blob.mass,
        sampleAt: receivedAt - SNAPSHOT_BLEND_MS,
        vx: 0,
        vy: 0,
        seed: hashString(blob.id),
      });
      continue;
    }

    existing.playerId = blob.playerId;
    existing.name = blob.name;
    existing.color = blob.color;
    existing.sx = existing.x;
    existing.sy = existing.y;
    existing.smass = existing.mass;
    existing.tx = blob.x;
    existing.ty = blob.y;
    existing.tmass = blob.mass;
    existing.sampleAt = receivedAt;
  }

  for (const id of blobVisuals.keys()) {
    if (!seen.has(id)) blobVisuals.delete(id);
  }
}

function sendInput() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !playerId || spectatorMode) {
    return;
  }

  const worldX = camera.x + (mouse.x - window.innerWidth / 2) / camera.zoom;
  const worldY = camera.y + (mouse.y - window.innerHeight / 2) / camera.zoom;

  ws.send(
    JSON.stringify({
      type: "input",
      target: {
        x: clamp(worldX, 0, world.w),
        y: clamp(worldY, 0, world.h),
      },
      split: splitQueued,
      eject: ejectQueued,
    }),
  );

  splitQueued = false;
  ejectQueued = false;
}

function maybeSendInput(nowMs) {
  const intervalMs = 1000 / serverInputHz;
  if (nowMs >= nextInputAt || splitQueued || ejectQueued) {
    sendInput();
    nextInputAt = nowMs + intervalMs;
  }
}

function viewRange() {
  return Math.max(window.innerHeight / 1080, window.innerWidth / 1920);
}

function updateCamera(dt) {
  const owned = spectatorMode
    ? []
    : [...blobVisuals.values()].filter((blob) => blob.playerId === playerId);

  if (owned.length > 0) {
    cameraTarget.x = owned.reduce((sum, blob) => sum + blob.x, 0) / owned.length;
    cameraTarget.y = owned.reduce((sum, blob) => sum + blob.y, 0) / owned.length;
    const totalSize = owned.reduce((sum, blob) => sum + worldRadius(blob), 0);
    cameraTarget.zoom = Math.pow(Math.min(64 / Math.max(1, totalSize), 1), 0.4);
  }

  const targetZoom = cameraTarget.zoom * viewRange();
  if (!cameraReady) {
    camera.x = cameraTarget.x;
    camera.y = cameraTarget.y;
    camera.zoom = targetZoom;
    cameraReady = true;
    return;
  }

  const posSmooth = 1 - Math.pow(0.5, dt * 60);
  const zoomSmooth = 1 - Math.pow(0.9, dt * 60);
  camera.x += (cameraTarget.x - camera.x) * posSmooth;
  camera.y += (cameraTarget.y - camera.y) * posSmooth;
  camera.zoom += (targetZoom - camera.zoom) * zoomSmooth;
}

function updateBlobVisuals(nowMs, dt) {

  for (const blob of blobVisuals.values()) {
    const prevX = blob.x;
    const prevY = blob.y;

    const blend = clamp((nowMs - blob.sampleAt) / SNAPSHOT_BLEND_MS, 0, 1);
    blob.x = blob.sx + (blob.tx - blob.sx) * blend;
    blob.y = blob.sy + (blob.ty - blob.sy) * blend;
    blob.mass = blob.smass + (blob.tmass - blob.smass) * blend;

    const safeDt = Math.max(dt, 1 / 240);
    blob.vx = (blob.x - prevX) / safeDt;
    blob.vy = (blob.y - prevY) / safeDt;
  }
}

function toScreen(x, y) {
  return {
    x: (x - camera.x) * camera.zoom + window.innerWidth / 2,
    y: (y - camera.y) * camera.zoom + window.innerHeight / 2,
  };
}

function drawGrid() {
  const cell = 50 * camera.zoom;
  const detail = clamp((camera.zoom - 0.3) / 0.7, 0, 1);

  const xMinor = ((-camera.x * camera.zoom + window.innerWidth / 2) % cell + cell) % cell;
  const yMinor = ((-camera.y * camera.zoom + window.innerHeight / 2) % cell + cell) % cell;

  ctx.strokeStyle = "#000000";
  ctx.globalAlpha = 0.12 + detail * 0.05;
  ctx.lineWidth = 0.4 + detail * 0.6;
  ctx.filter = "blur(0.45px)";
  for (let x = xMinor; x <= window.innerWidth; x += cell) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, window.innerHeight);
    ctx.stroke();
  }
  for (let y = yMinor; y <= window.innerHeight; y += cell) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(window.innerWidth, y);
    ctx.stroke();
  }
  ctx.filter = "none";
  ctx.globalAlpha = 1;
}

function drawWorldBounds() {
  const topLeft = toScreen(0, 0);
  const bottomRight = toScreen(world.w, world.h);

  ctx.strokeStyle = GRID_COLOR;
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = 1;
  ctx.strokeRect(
    topLeft.x,
    topLeft.y,
    bottomRight.x - topLeft.x,
    bottomRight.y - topLeft.y,
  );
  ctx.globalAlpha = 1;
}

function consumeSuppressed(key, nowMs) {
  const until = locallyConsumedUntil.get(key);
  if (!until) return false;
  if (nowMs >= until) {
    locallyConsumedUntil.delete(key);
    return false;
  }
  return true;
}

function traceWobblingDisk(x, y, radius, key, timeSec, roughness = 0.045, points = 14) {
  const phase = hashString(String(key)) * Math.PI * 2;
  ctx.beginPath();
  for (let i = 0; i <= points; i += 1) {
    const angle = (i / points) * Math.PI * 2;
    const wave =
      Math.sin(timeSec * 3.1 + phase + i * 1.73) * 0.62 +
      Math.sin(timeSec * 4.7 - phase * 0.7 - i * 1.11) * 0.38;
    const localRadius = radius * (1 + wave * roughness);
    const px = x + Math.cos(angle) * localRadius;
    const py = y + Math.sin(angle) * localRadius;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
}

function drawFood(nowMs) {
  if (!state) return;
  const blobs = [...blobVisuals.values()];

  for (const food of state.foods) {
    const key = `f:${food.id}`;
    if (consumeFx.has(key) || consumeSuppressed(key, nowMs)) {
      continue;
    }

    const target = findNearbyConsumer(
      food.x,
      food.y,
      worldParticleRadius(food.mass),
      1.03,
      blobs,
    );
    if (target) {
      beginConsumeFx(key, "food", food.x, food.y, food.mass, food.color, target.blob.id);
      continue;
    }

    const p = toScreen(food.x, food.y);
    const radius = Math.max(5, Math.sqrt(food.mass * 100) * camera.zoom);

    traceWobblingDisk(p.x, p.y, radius, food.id, nowMs / 1000);
    ctx.fillStyle = food.color;
    ctx.fill();
  }
}

function drawEjected(nowMs) {
  if (!state) return;
  const blobs = [...blobVisuals.values()];

  for (const item of state.ejected) {
    const key = `e:${item.id}`;
    if (consumeFx.has(key) || consumeSuppressed(key, nowMs)) {
      continue;
    }

    const target = findNearbyConsumer(
      item.x,
      item.y,
      worldParticleRadius(item.mass),
      1.02,
      blobs,
    );
    if (target) {
      beginConsumeFx(key, "ejected", item.x, item.y, item.mass, "#6FE85A", target.blob.id);
      continue;
    }

    const p = toScreen(item.x, item.y);
    const radius = Math.max(3.5, Math.sqrt(item.mass * 100) * camera.zoom);

    traceWobblingDisk(p.x, p.y, radius, item.id, nowMs / 1000, 0.03, 20);
    ctx.fillStyle = "#6FE85A";
    ctx.fill();
    ctx.strokeStyle = "#4BC443";
    ctx.lineWidth = worldStroke(EJECTED_BORDER_WORLD);
    ctx.stroke();
  }
}

function worldRadius(blobLike) {
  return Math.sqrt(blobLike.mass * 100);
}

function worldParticleRadius(mass) {
  return Math.sqrt(mass * 100);
}

function findNearbyConsumer(x, y, particleRadius, rangeFactor, blobs) {
  let best = null;

  for (const blob of blobs) {
    const dx = blob.x - x;
    const dy = blob.y - y;
    const dist = Math.hypot(dx, dy) || 0.0001;
    const touch = (worldRadius(blob) + particleRadius) * rangeFactor;
    if (dist >= touch) continue;

    if (!best || dist < best.dist) {
      best = { blob, dx, dy, dist, touch };
    }
  }

  return best;
}

function beginConsumeFx(key, kind, x, y, mass, color, targetId) {
  if (consumeFx.has(key)) return;
  locallyConsumedUntil.set(key, performance.now() + 260);
  consumeFx.set(key, {
    key,
    kind,
    x,
    y,
    mass,
    color,
    targetId,
    t: 0,
    startDist: null,
    duration: kind === "food" ? 0.16 : 0.2,
  });
}

function drawConsumeFx(dt) {
  if (consumeFx.size === 0) return;

  const blobs = [...blobVisuals.values()];
  for (const [key, fx] of consumeFx.entries()) {
    fx.t += dt / fx.duration;
    const progress = clamp(fx.t, 0, 1);

    let targetBlob = fx.targetId ? blobVisuals.get(fx.targetId) : null;
    if (!targetBlob) {
      const target = findNearbyConsumer(fx.x, fx.y, worldParticleRadius(fx.mass), 1.65, blobs);
      targetBlob = target ? target.blob : null;
    }

    let alpha = 1 - progress * 0.4;
    let reachedUnderBlob = false;

    if (targetBlob) {
      const dx = targetBlob.x - fx.x;
      const dy = targetBlob.y - fx.y;
      const dist = Math.hypot(dx, dy) || 0.0001;
      if (fx.startDist === null) fx.startDist = dist;
      else fx.startDist = Math.max(fx.startDist, dist);

      const pull = 1 - Math.exp(-(fx.kind === "food" ? 30 : 24) * dt);
      fx.x += dx * pull;
      fx.y += dy * pull;

      const remainDx = targetBlob.x - fx.x;
      const remainDy = targetBlob.y - fx.y;
      const remainDist = Math.hypot(remainDx, remainDy);
      const targetRadius = worldRadius(targetBlob);
      const underDist = Math.max(worldParticleRadius(fx.mass) * 0.6, targetRadius * 0.92);
      const start = Math.max(underDist + 1, fx.startDist ?? remainDist + 1);
      const alphaProgress = clamp((start - remainDist) / Math.max(1, start - underDist), 0, 1);
      alpha = 1 - alphaProgress * 0.4;
      reachedUnderBlob = remainDist <= underDist;
    }

    const p = toScreen(fx.x, fx.y);
    const radiusBase =
      fx.kind === "food"
        ? Math.max(5, Math.sqrt(fx.mass * 100) * camera.zoom)
        : Math.max(3.5, Math.sqrt(fx.mass * 100) * camera.zoom);
    const radius = radiusBase;
    ctx.globalAlpha = alpha;
    traceWobblingDisk(p.x, p.y, Math.max(0.1, radius), fx.key, performance.now() / 1000);
    ctx.fillStyle = fx.color;
    ctx.fill();
    if (fx.kind === "ejected") {
      ctx.strokeStyle = "#4BC443";
      ctx.lineWidth = worldStroke(EJECTED_BORDER_WORLD);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    if (reachedUnderBlob || progress >= 1) consumeFx.delete(key);
  }
}

function findAbsorberId(x, y, particleRadius) {
  let best = null;
  for (const blob of blobVisuals.values()) {
    const dx = blob.x - x;
    const dy = blob.y - y;
    const dist = Math.hypot(dx, dy) || 0.0001;
    const touch = (worldRadius(blob) + particleRadius) * 1.5;
    if (dist > touch) continue;
    if (!best || dist < best.dist) {
      best = { id: blob.id, dist };
    }
  }
  return best ? best.id : null;
}

function syncConsumedEffects(nextState) {
  const nextFoods = new Map();
  for (const food of nextState.foods || []) {
    nextFoods.set(food.id, food);
  }
  for (const [id, oldFood] of visibleFoods.entries()) {
    if (nextFoods.has(id)) continue;
    const targetId = findAbsorberId(oldFood.x, oldFood.y, worldParticleRadius(oldFood.mass));
    if (targetId) {
      beginConsumeFx(`f:${id}`, "food", oldFood.x, oldFood.y, oldFood.mass, oldFood.color, targetId);
    }
  }
  visibleFoods.clear();
  for (const [id, food] of nextFoods.entries()) {
    visibleFoods.set(id, food);
  }

  const nextEjected = new Map();
  for (const item of nextState.ejected || []) {
    nextEjected.set(item.id, item);
  }
  for (const [id, oldItem] of visibleEjected.entries()) {
    if (nextEjected.has(id)) continue;
    const targetId = findAbsorberId(oldItem.x, oldItem.y, worldParticleRadius(oldItem.mass));
    if (targetId) {
      beginConsumeFx(`e:${id}`, "ejected", oldItem.x, oldItem.y, oldItem.mass, "#6FE85A", targetId);
    }
  }
  visibleEjected.clear();
  for (const [id, item] of nextEjected.entries()) {
    visibleEjected.set(id, item);
  }
}

function pushInteractor(list, dx, dy, strength) {
  if (strength <= 0.001) return;
  const dist = Math.hypot(dx, dy) || 0.0001;
  list.push({
    dx: dx / dist,
    dy: dy / dist,
    strength,
  });
}

function addWallInteractors(list, x, y, radius) {
  const range = radius * 1.38;

  const left = range - x;
  if (left > 0) list.push({ dx: -1, dy: 0, strength: clamp((left / range) * 1.35, 0, 1.6) });

  const right = range - (world.w - x);
  if (right > 0) list.push({ dx: 1, dy: 0, strength: clamp((right / range) * 1.35, 0, 1.6) });

  const top = range - y;
  if (top > 0) list.push({ dx: 0, dy: -1, strength: clamp((top / range) * 1.35, 0, 1.6) });

  const bottom = range - (world.h - y);
  if (bottom > 0) list.push({ dx: 0, dy: 1, strength: clamp((bottom / range) * 1.35, 0, 1.6) });
}

function pruneInteractors(list, maxCount = 16) {
  if (list.length > maxCount) {
    list.sort((a, b) => b.strength - a.strength);
    list.length = maxCount;
  }
  return list;
}

function buildBlobInteractors(blob, allBlobs, viruses) {
  const list = [];
  const r = worldRadius(blob);
  const largeBlobDamp = 1 - clamp((r - 30) / 180, 0, 0.3);

  for (const other of allBlobs) {
    if (other.id === blob.id) continue;

    const r2 = worldRadius(other);
    const dx = other.x - blob.x;
    const dy = other.y - blob.y;
    const dist = Math.hypot(dx, dy) || 0.0001;
    const touch = r + r2;
    const near = touch * 1.14;
    if (dist >= near) continue;

    const closeness = clamp((near - dist) / near, 0, 1);
    const overlap = clamp((touch - dist) / touch, 0, 1);
    const samePlayer = other.playerId === blob.playerId ? 1.36 : 0.82;
    const relative = r2 / Math.max(1, r);
    const sizeInfluence =
      relative < 1
        ? clamp(Math.pow(relative, 1.25), 0.14, 1.0)
        : clamp(1 + (relative - 1) * 0.22, 1.0, 1.24);
    const strength = (closeness * 0.58 + overlap * 1.72) * samePlayer * sizeInfluence * largeBlobDamp;
    pushInteractor(list, dx, dy, strength);
  }

  for (const virus of viruses) {
    const rv = worldRadius(virus);
    const dx = virus.x - blob.x;
    const dy = virus.y - blob.y;
    const dist = Math.hypot(dx, dy) || 0.0001;
    const touch = r + rv;
    const near = touch * 1.16;
    if (dist >= near) continue;

    const closeness = clamp((near - dist) / near, 0, 1);
    const overlap = clamp((touch - dist) / touch, 0, 1);
    const relativeVirus = rv / Math.max(1, r);
    const virusSizeInfluence = clamp(Math.pow(relativeVirus, 0.95), 0.22, 1.0);
    const strength = (closeness * 0.7 + overlap * 1.26) * virusSizeInfluence * largeBlobDamp;
    pushInteractor(list, dx, dy, strength);
  }

  addWallInteractors(list, blob.x, blob.y, r);
  return pruneInteractors(list);
}

function buildVirusInteractors(virus, blobs) {
  const list = [];
  const r = worldRadius(virus);

  for (const blob of blobs) {
    const rb = worldRadius(blob);
    const dx = blob.x - virus.x;
    const dy = blob.y - virus.y;
    const dist = Math.hypot(dx, dy) || 0.0001;
    const touch = r + rb;
    const near = touch * 1.12;
    if (dist >= near) continue;

    const closeness = clamp((near - dist) / near, 0, 1);
    const overlap = clamp((touch - dist) / touch, 0, 1);
    const strength = closeness * 0.6 + overlap * 1.22;
    pushInteractor(list, dx, dy, strength);
  }

  addWallInteractors(list, virus.x, virus.y, r);
  return pruneInteractors(list, 12);
}

function sampleContactPressure(nx, ny, interactors) {
  let indent = 0;
  let bulge = 0;

  for (const it of interactors) {
    const align = nx * it.dx + ny * it.dy;
    if (align > 0) {
      indent += it.strength * align * align * align;
    } else {
      const inv = -align;
      bulge += it.strength * inv * inv;
    }
  }

  return {
    indent: Math.min(1.6, indent),
    bulge: Math.min(1.6, bulge),
  };
}

function buildBlobIngestors(blob, foods, ejected) {
  const list = [];
  const r = worldRadius(blob);

  for (const food of foods) {
    const dx = food.x - blob.x;
    const dy = food.y - blob.y;
    const dist = Math.hypot(dx, dy) || 0.0001;
    const range = (r + worldParticleRadius(food.mass)) * 1.15;
    if (dist >= range) continue;

    const pull = clamp((range - dist) / range, 0, 1);
    pushInteractor(list, dx, dy, pull * 0.95);
  }

  for (const mass of ejected) {
    const dx = mass.x - blob.x;
    const dy = mass.y - blob.y;
    const dist = Math.hypot(dx, dy) || 0.0001;
    const range = (r + worldParticleRadius(mass.mass)) * 1.12;
    if (dist >= range) continue;

    const pull = clamp((range - dist) / range, 0, 1);
    pushInteractor(list, dx, dy, pull * 0.7);
  }

  return pruneInteractors(list, 10);
}

function sampleIngestBulge(nx, ny, ingestors) {
  let value = 0;
  for (const it of ingestors) {
    const align = nx * it.dx + ny * it.dy;
    if (align > 0) {
      value += it.strength * align * align * align * align;
    }
  }
  return Math.min(1.5, value);
}

function drawBlobShape(blob, timeSec, allBlobs, viruses, foods, ejected) {
  const center = toScreen(blob.x, blob.y);
  const worldR = worldRadius(blob);
  const radius = worldR * camera.zoom;
  const interactors = buildBlobInteractors(blob, allBlobs, viruses);
  const ingestors = buildBlobIngestors(blob, foods, ejected);
  const deformScale = 1 - clamp((worldR - 24) / 170, 0, 0.28);

  const speed = Math.hypot(blob.vx, blob.vy);
  const speedNorm = clamp(speed / 520, 0, 1);

  let moveUx = 1;
  let moveUy = 0;
  if (speed > 0.001) {
    moveUx = blob.vx / speed;
    moveUy = blob.vy / speed;
  }

  const points = Math.round(clamp(radius, 18, 128));

  ctx.beginPath();
  for (let i = 0; i <= points; i += 1) {
    const angle = (i / points) * Math.PI * 2;
    const nx = Math.cos(angle);
    const ny = Math.sin(angle);

    const moveDot = nx * moveUx + ny * moveUy;
    const pressure = sampleContactPressure(nx, ny, interactors);
    const ingest = sampleIngestBulge(nx, ny, ingestors);

    const stretch =
      1 +
      moveDot * speedNorm * (0.028 * (0.8 + deformScale * 0.2)) -
      pressure.indent * (0.14 * deformScale) +
      pressure.bulge * (0.035 * (0.86 + deformScale * 0.14)) +
      ingest * (0.09 * (0.7 + deformScale * 0.3));
    const wobbleAmp =
      radius * (0.004 + speedNorm * 0.004 + Math.min(1.0, pressure.indent) * 0.006) * (0.7 + deformScale * 0.3);
    const wobbleA = Math.sin(timeSec * 7.2 + i * 0.92 + blob.seed * 11.7);
    const wobbleB = Math.sin(timeSec * 11.2 - i * 1.21 + blob.seed * 5.4);
    const wobble = (wobbleA * 0.62 + wobbleB * 0.38) * wobbleAmp;

    const localRadius = Math.max(radius * 0.76, radius * stretch + wobble);
    const x = center.x + nx * localRadius;
    const y = center.y + ny * localRadius;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

function drawVirusShape(virus, timeSec, allBlobs) {
  const center = toScreen(virus.x, virus.y);
  const worldR = worldRadius(virus);
  const radius = worldR * camera.zoom;
  const interactors = buildVirusInteractors(virus, allBlobs);
  let spikes = Math.max(34, Math.round((Math.PI * 2 * worldR) / VIRUS_SPIKE_SPACING_WORLD));
  if (spikes % 2 === 1) spikes += 1;

  ctx.beginPath();
  for (let i = 0; i <= spikes; i += 1) {
    const angle = (i / spikes) * Math.PI * 2;
    const nx = Math.cos(angle);
    const ny = Math.sin(angle);
    const pressure = sampleContactPressure(nx, ny, interactors);
    const pulse = Math.sin(timeSec * 7.4 + i * 1.17 + virus.id.length * 0.8) * radius * 0.0024;
    const spike = i % 2 === 0 ? 1 + 5 / worldR : 1;
    const deform = spike - pressure.indent * 0.075 + pressure.bulge * 0.025;
    const localRadius = Math.max(radius * 0.78, radius * deform + pulse);
    const x = center.x + nx * localRadius;
    const y = center.y + ny * localRadius;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.closePath();

  ctx.fillStyle = VIRUS_FILL;
  ctx.fill();
  ctx.strokeStyle = VIRUS_BORDER;
  ctx.lineWidth = worldStroke(VIRUS_BORDER_WORLD);
  ctx.stroke();
}

function drawBlobEntity(blob, timeSec, allBlobs, viruses, foods, ejected) {
  drawBlobShape(blob, timeSec, allBlobs, viruses, foods, ejected);

  ctx.fillStyle = blob.color;
  ctx.fill();

  const p = toScreen(blob.x, blob.y);
  const radius = worldRadius(blob) * camera.zoom;
  ctx.strokeStyle = autoBorderColor(blob.color);
  ctx.lineWidth = worldStroke(PLAYER_BORDER_WORLD);
  ctx.stroke();

  const label = blob.name || "Cell";
  const textSize = Math.max(8, Math.max(24, worldRadius(blob) * 0.3) * camera.zoom);

  ctx.font = `700 ${textSize}px "Ubuntu", Arial, sans-serif`;
  ctx.strokeStyle = "rgba(36, 39, 44, 0.92)";
  ctx.lineWidth = Math.max(1.5, textSize * 0.1);
  ctx.strokeText(label, p.x, p.y);
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, p.x, p.y);
}

function drawActors(timeSec) {
  if (!state) return;

  const blobs = [...blobVisuals.values()];
  const viruses = state.viruses || [];
  const foods = state.foods || [];
  const ejected = state.ejected || [];

  const actors = [];
  for (const blob of blobs) {
    actors.push({ type: "blob", radius: worldRadius(blob), data: blob });
  }
  for (const virus of viruses) {
    actors.push({ type: "virus", radius: worldRadius(virus), data: virus });
  }

  actors.sort((a, b) => a.radius - b.radius);

  for (const actor of actors) {
    if (actor.type === "virus") {
      drawVirusShape(actor.data, timeSec, blobs);
    } else {
      drawBlobEntity(actor.data, timeSec, blobs, viruses, foods, ejected);
    }
  }
}

function render(nowMs) {
  const dt = Math.min(0.05, Math.max(0.001, (nowMs - lastFrameAt) / 1000));
  lastFrameAt = nowMs;

  maybeSendInput(nowMs);
  updateBlobVisuals(nowMs, dt);
  updateCamera(dt);

  ctx.fillStyle = BG_COLOR;
  ctx.fillRect(0, 0, window.innerWidth, window.innerHeight);

  drawGrid();
  drawFood(nowMs);
  drawEjected(nowMs);
  drawConsumeFx(dt);
  drawActors(nowMs / 1000);

  requestAnimationFrame(render);
}

resize();
connect();
setInterval(() => {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "ping", ts: Date.now() }));
  }
}, 2500);
requestAnimationFrame(render);
