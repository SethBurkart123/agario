# Agar.io fidelity QA

Status: passed

## Evidence

- Small-state visual truth: `/Users/sethburkart/.t3/userdata/attachments/a8485ca2-fdb8-47cd-b24c-a64d8bc481a9-7877471a-e714-45d8-89fd-8974693f0c42.png` (live Agar.io, score 21).
- Large-state visual truth: `/Users/sethburkart/.t3/userdata/attachments/a8485ca2-fdb8-47cd-b24c-a64d8bc481a9-b6d98ebe-06bd-4333-82c2-a64a0a5aa6ec.png` (live Agar.io, score 675, 16 cells).
- Small implementation capture: `/Users/sethburkart/.t3/userdata/browser-artifacts/agario-current-matched.png`.
- Large implementation capture: `/Users/sethburkart/.t3/userdata/browser-artifacts/agario-large-after.png`.
- Full large-state comparison: `/Users/sethburkart/.t3/userdata/browser-artifacts/agario-large-qa-side-by-side.png`.
- Focused grid/pellet comparison: `/Users/sethburkart/.t3/userdata/browser-artifacts/agario-large-qa-focus.png`.
- Small comparison: `/Users/sethburkart/.t3/userdata/browser-artifacts/agario-qa-side-by-side.png`.
- Focused grid-softness comparison: `/Users/sethburkart/.t3/userdata/browser-artifacts/agario-grid-softness-final.png`.
- Normalized large viewport: 1526 x 1002 CSS px at device scale factor 2. Source and implementation content are both 3052 x 2004 physical px.
- Normalized small viewport: 1528 x 1017 CSS px at device scale factor 2. Source and implementation content are both 3056 x 2034 physical px.
- The large QA world spawned at mass 675 and reached 16 cells. Its centered input leaves the pieces overlapped, so entity positions and cluster silhouette are not used as renderer measurements.

## Findings and comparison history

| Severity | Earlier finding | Evidence | Fix and final evidence |
| --- | --- | --- | --- |
| P1 | Food collapsed into pinpoints when zoomed out. | At the same approximately 30-31 px physical grid cadence, the reported clone's median food diameter was 11.51 px while live Agar.io measured 19.45 px. | Food now has a 5 CSS px minimum radius. Final median is 19.58 px, a 0.7% difference. Consume animation uses the same clamp. |
| P1 | The dense zoomed-out grid dominated the scene. | Before the fix, 17.47% of neutral-region pixels exceeded the visible-line darkness threshold; live Agar.io measured 7.76%. | Grid opacity and width now soften with zoom. Final share is 7.39%; mean line-region darkness is 4.90 versus 4.73 in the source. |
| P2 | The field color was slightly too cyan. | Implementation used `#f2fbff`; sampled live Agar.io background is `#f4fbff`. | Canvas and CSS field colors now use `#f4fbff`. |
| P2 | A global soft-grid fix made the small state too faint. | The first iteration used the zoomed-out 0.12 alpha and 0.5 px width at every scale. | Grid detail now interpolates from 0.12 alpha / 0.4 px width when zoomed out to 0.17 alpha / 1 px width at normal scale. Small-state line-core color is `[203, 209, 213]` versus source `[204, 209, 212]`. |
| P2 | Grid contrast matched in aggregate, but its raster edge still looked harder than live Agar.io. | The supplied close-up showed the line energy falling off across adjacent pixels rather than ending at the antialiased stroke edge. | A grid-only 0.45 CSS px blur now spreads the stroke without blurring food, cells, names, or the HUD. At normalized CSS resolution, line-spread sigma is 0.602 px versus 0.603 px in live Agar.io. |
| — | Food density looked inconsistent between individual frames. | A single zoomed-out pair differed heavily, but local consumption and random placement make one frame unreliable. | Across all supplied full screenshots, live Agar.io ranges from 1.47 to 6.17 food pellets per 100 grid cells and the simulator ranges from 1.78 to 5.93. The averages differ by about 5%, so the global food target remains unchanged. |

## Quantitative checks

| Metric | Live Agar.io | Before | Final | Result |
| --- | ---: | ---: | ---: | --- |
| Large-state grid cadence | 31 physical px | 30 px | 30 px | camera scale retained |
| Large-state food median diameter | 19.45 px | 11.51 px | 19.58 px | 0.7% difference |
| Large-state grid mean darkness | 4.73 | 19.60 | 4.90 | 3.7% difference |
| Large-state visibly dark grid share | 7.76% | 17.47% | 7.39% | 0.37 percentage-point difference |
| Small-state grid cadence | 94 physical px | 94 px | 94 px | exact |
| Small-state grid line core RGB | `[204, 209, 212]` | `[193, 200, 204]` | `[203, 209, 213]` | within 1 channel value |
| Normalized grid line-spread sigma | 0.603 px | 0.638 px | 0.602 px | 0.2% difference |
| Field RGB | `[244, 251, 255]` | `[242, 251, 255]` | `[244, 251, 255]` | exact |

## Required fidelity surfaces

- Fonts and typography: Ubuntu 700, outlined cell names, score badge, and leaderboard hierarchy remain unchanged and readable at both scales.
- Spacing and layout: the camera and 50-world-unit grid cadence were already correct. The fix changes only zoom-dependent grid treatment and food's screen-space minimum.
- Colors and tokens: the field now matches the sampled source color; grid contrast matches the focused source region within the measured tolerance.
- Image quality: entities remain native-density canvas geometry with no raster placeholders or scaling blur.
- Copy and content: score, player names, and leaderboard copy remain intact.

## Interaction and diagnostics

- Normal join, movement input, and server snapshots were exercised on the running server.
- The large QA state exercised four split inputs and the server snapshot confirmed 16 owned cells.
- Helium rendered both target sizes at device scale factor 2. No page-origin errors appeared; the command-line capture emitted only Helium extension/runtime warnings.
- Random pellet positions, player colors, leaderboard contents, and local food density differ between live worlds and are not treated as renderer drift.

final result: passed
