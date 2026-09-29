# Algorithm comparison

The runtime starts with the portable tree hybrid. The choice follows the safety
priority stated by the organizers: minimize false alarms first, then compare
coverage, shape transfer, and latency.

| Algorithm | Normal alarms | Exact boxes: recall / specificity | Official positive coverage* | Unseen shapes | p95 |
|---|---:|---:|---:|---:|---:|
| Portable 25/75 hybrid | 33 / 2,287 (1.44%) | 40.40% / 100% | 1 / 7 | 6.53% | 92.41 ms |
| OOD-guarded G4 | 75 / 2,287 (3.28%) | 38.14% / 98.0% | 1 / 7 | incomparable | 76.46 ms |
| Linear G4 | 81 / 2,287 (3.54%) | 40.49% / 97.09% | 1 / 7 | 16.38% | 74.53 ms |
| Geometry | 1,842 / 2,287 (80.54%) | 67.98% / 89.46% | 6 / 7 | 90.60% | 79.65 ms |

\* The official bag has no point masks or time intervals. Its figures use the
published object order, approximate 100 m spacing, LiDAR odometry, and spatial
matching. They are diagnostic pseudo-labels, not ground truth.

## What each method contributes

**Geometry** estimates the rail-relative corridor and proposes components that
enter the 2.1 × 3.0 m clearance. It transfers best to unseen silhouettes and
finds six of seven positive official-style scenarios, but tunnel hardware also
enters the corridor, producing too many false alarms for deployment. It remains
the diagnostic high-recall mode.

**Linear G4** scores geometric components from size, density, height, residual,
range support, and lateral clearance. It removes most infrastructure alarms and
transfers better to unseen shapes than the tree model. Its complete-bag false
alarm rate is still 3.54%.

**Portable 25/75 hybrid** blends 25% linear score with 75% from 100 shallow
Extra Trees. It produces the fewest false alarms, perfectly rejects the paired
outside/above exact-box negatives, stays below 100 ms p95, and has no sklearn
runtime dependency. Its weakness is shape transfer: a ranker trained on boxes
rejects many unseen silhouettes. This is the default because the hidden test is
described as synthetic and similar to the organizer sample, while false alarms
have explicit priority.

**OOD-guarded G4** vetoes detections in unfamiliar tunnel context unless the G4
margin is strong. The earlier sparse-candidate audit showed zero alarms, but the
new every-frame, whole-bag audit found 75 alarm frames. That full audit replaces
the optimistic sparse result for runtime selection. The composite cache lacks
the context patches needed for a fair shape comparison, so that field remains
incomparable.

**Three-of-five confirmation** works on the G4 stream but fails when attached
to the selected portable hybrid. In the direct EXP-068 comparison, both range
and world tracking reduce 33 alarm frames and 23 episodes to 9 and 7. They also
lose two of the 17 exact-mask events found by the per-frame hybrid. World
tracking reaches 118.25 ms p95 on the full 2,287-frame run. The submission
therefore keeps the per-frame portable hybrid and does not hide its detections
behind temporal voting.

All four standalone paths were exercised on both delivered formats: 16-byte
XYZI and 26-byte XYZIRT. Full machine-readable folds, per-bag counts, pseudo-label
scenarios, timings, and layout checks are in
`lidar_geometry/artifacts/competition_scorecard_exp067.json`. The paired
temporal comparison is in
`lidar_geometry/artifacts/portable_temporal_exp068.json`.
