# Agar.io fidelity QA

Status: passed

## Reference and capture

- Source visual truth: `/Users/sethburkart/.t3/userdata/attachments/a8485ca2-fdb8-47cd-b24c-a64d8bc481a9-7877471a-e714-45d8-89fd-8974693f0c42.png` (live Agar.io, score 21).
- Reported broken capture: `/Users/sethburkart/.t3/userdata/attachments/a8485ca2-fdb8-47cd-b24c-a64d8bc481a9-29f24888-6359-4192-aca7-b68d97f9aeab.png` (stale clone client, score 37).
- Browser-rendered implementation: `/Users/sethburkart/.t3/userdata/browser-artifacts/agario-current-matched.png` (fresh client, score 21).
- Combined comparison: `/Users/sethburkart/.t3/userdata/browser-artifacts/agario-qa-side-by-side.png`.
- Normalized viewport: 1528 x 1017 CSS px at device scale factor 2. Source and implementation content are both 3056 x 2034 physical px before the side-by-side downsample.
- State: single unsplit player at score 21, light theme, normal play. The source and implementation are compared at equal mass.

## Findings and comparison history

| Severity | Finding | Evidence | Fix and post-fix result |
| --- | --- | --- | --- |
| P0 | The browser was running an old cached renderer against the new Rust physics. | The reported capture still contains the removed `Overview: Off` control and connection label. Its renderer uses `4 * sqrt(mass)` while Rust uses `10 * sqrt(mass)`, making the visible radius 40% of the collision radius. | Static responses now use `Cache-Control: no-store`, asset URLs are versioned, and the WebSocket handshake rejects mismatched client protocols. Fresh capture uses one revision end to end. |
| P0 | Grid/camera scale was quantitatively wrong in the reported capture. | Live Agar.io has 94 physical px between 50-unit grid lines at this viewport. The stale client has 60 px because it renders the old 30-unit grid. | Fresh implementation measures 94 px exactly. |
| P0 | The stale size mismatch made correct world movement look far too fast relative to the visible cell. | A mass-22 live probe moves 405.28 world units/s; the reference formula predicts 405.17, a 0.03% error. The stale renderer made that body much smaller on screen. | Fresh rendering and physics share the same `size = sqrt(100 * mass)` conversion. |
| P1 | The clone spawned at anonymous mass 10.24 while every supplied live reference uses the signed-in score-21 state. | The camera intentionally clamps its zoom for both masses, so mass 10.24 appeared 31% smaller and moved 17% faster. Measured in body lengths, that looked about 1.7x too fast. | Default spawn mass is now 21. The authentic movement and camera equations remain unchanged. |
| P1 | Pickup looked detached from the visible body. | Rust correctly consumes a mass-1 pellet when its center reaches `player_size - pellet_size / 3`, but the stale player drawing was less than half the physics diameter. | The fresh visible radius now matches the collision radius; a boundary check covers both sides of the exact overlap threshold. |

## Quantitative visual checks

| Metric | Live Agar.io | Reported stale client | Fresh score-21 client | Result |
| --- | ---: | ---: | ---: | --- |
| Grid spacing at matched viewport | 94 px | 60 px | 94 px | exact |
| Player body | 201 x 190 px at mass 21 | 109 px fill at mass 37 | 191 x 190 px at mass 21 | same height; width within 5% of animated source edge |
| Camera scale | 94 px per 50 world units | 60 px per 30 world units | 94 px per 50 world units | exact |
| Start-cell movement | reference formula: 409.33 world units/s at mass 21 | visually exaggerated by undersized body | 409.33 world units/s at mass 21 | exact formula |
| Live movement probe | reference formula: 405.17 world units/s at sampled mass 22 | not measured | 405.28 world units/s | 0.03% error |
| Pellet pickup reach at mass 10.24 | `32 - 10/3 = 28.667` world units | invisible body did not agree | boundary check passes at +/- 0.01 unit | exact |

## Required fidelity surfaces

- Fonts and typography: Ubuntu 700, outlined cell names, score badge, and leaderboard hierarchy are retained. No stale overview control or persistent connection copy appears in the fresh capture.
- Spacing and layout: camera crop and 50-unit grid cadence match after viewport and density normalization.
- Colors and tokens: light field, grid contrast, player colors, and leaderboard overlay remain aligned with the source.
- Image quality: all entities are canvas geometry at native device density; the comparison shows no scaling blur or raster placeholder.
- Copy and content: normal HUD copy and ranked local leaderboard row are present.

## Primary interactions and diagnostics

- Mouse steering was exercised through the live WebSocket and measured over 26 server snapshots.
- Camera zoom, camera smoothing, and movement use the reverse-engineered vanilla client equations; only the reference-state spawn mass needed correction.
- Split and eject keyboard handlers remain wired; the dedicated `/overview` spectator route remains separate.
- Rust mechanics checks pass for reference movement and pellet pickup boundary.
- No page-origin console errors appeared in the fresh Helium capture; only Helium extension/runtime warnings were emitted outside the page.

## Remaining P3

- The live animated score-21 outline is about 5% wider than the deterministic formula predicts. Its irregular membrane edge and screenshot timing prevent a more exact static comparison; this does not affect camera, collision, or movement geometry.

final result: passed
