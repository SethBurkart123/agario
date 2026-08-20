# Agar.io fidelity QA

Status: passed

## Reference and capture

- Primary single-cell reference: `a8485ca2-fdb8-47cd-b24c-a64d8bc481a9-7877471a-e714-45d8-89fd-8974693f0c42.png` at score 21.
- Primary split reference: `a8485ca2-fdb8-47cd-b24c-a64d8bc481a9-ef94d377-e9cf-4030-828e-218f326a8861.png` at score 381 and 16 cells.
- The implementation was captured in Helium at the corresponding logical viewport and mass. Side-by-side comparison passes used the cropped source and live canvas capture together.

## Findings resolved

| Severity | Surface | Mismatch | Resolution |
| --- | --- | --- | --- |
| P0 | Camera | A hand-tuned mass/count penalty kept small cells too zoomed in and did not reproduce split states. | Camera zoom now uses the classic client formula based on the sum of owned cell radii. Camera position uses the equal-weight centre of owned cells. |
| P0 | Scale and motion | The old zoom made correct world velocities, entity radii, viruses, and grid spacing look wrong. | Restored the 64-size camera baseline, 50-unit grid, close-cursor slowdown, and 120 ms snapshot interpolation. |
| P1 | Rendering | Food was static, cells were overly deformed at contact, and virus geometry was oversized. | Added subtle seeded pellet motion, reduced membrane distortion, made point density scale with apparent radius, and matched the 100-size virus with 5-unit spikes. |
| P1 | Typography | Cell labels and HUD used Trebuchet and did not match the source. | Switched to Ubuntu 700, matched radius-relative label sizing and outline weight, and aligned leaderboard row sizing. |
| P1 | Density | The larger modern map made the initial 1,000-pellet field visibly sparse against the supplied live FFA captures. | Raised the field target to 2,500 while preserving 10-to-20 size growth and normal replacement behavior. |
| P2 | Chrome | The normal-play leaderboard included a clone-only overview button and connection label. | Kept full-map spectating at `/overview`; removed the button and hide the status after connection. |
| P2 | Leaderboard | The local player disappeared when ranked outside the top ten. | Append the local row with its real rank, matching references such as `17. me`. |

## Verification passes

- Score 21: player diameter, 50-unit grid cadence, label scale, border, background, HUD, and 100-size virus align with the source at the matched viewport.
- Score 381, 16 cells: calculated zoom is `0.368`; the live capture matches the reference's apparent grid, fragment, pellet, and virus scale.
- Split camera assertions verify 16 cells, equal-weight camera centre, and the radius-sum zoom calculation.
- Keyboard split/eject input remains intact; `/overview` remains the dedicated spectator view.
- A normal 16-bot live run shows the top ten plus the local `17.` row and no new client errors.
- No console syntax errors; Rust formatting, build, tests, Clippy, Python compilation, and whitespace checks pass.
