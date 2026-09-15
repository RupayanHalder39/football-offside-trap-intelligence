# Offside / Trap-Break V4 full 120s render — QA report

**Output**: `offside_break/outputs/correctness_v4/final_120s/offside_dashboard_120s_v4.mp4`
**Spec verified**: 2304×1204, 30.00 fps, 3600 frames, duration 120.00s, codec FMP4 (mp4v fourcc), 219.8MB, render time 476.6s (~7.6 fps).

## Method

Same approach as the pressing render (see that report's Method
section for full detail): `--out_path` video mode added with a single
seek + sequential reads; verified byte-identical to preview mode for a
test frame before committing to the full render; representative frames
extracted directly from the finished MP4 for inspection, plus
full-snapshot programmatic checks across all 3600 frames.

## Time-sync verification

Frame 1277 pulled from the finished video matched an independently
preview-rendered frame 1277 more closely (mean diff 2.97) than any
neighboring frame — confirms correct sync (same methodology as the
pressing report).

## Checked frames / timestamps and observed state

| Check | Frame | t (s) | T0 state | T1 state | Notes |
|---|---|---|---|---|---|
| Global: start | 0 | 0.00 | UNCERTAIN | UNCERTAIN | clean |
| Global: 25% | 900 | 30.00 | UNCERTAIN | NO_TRAP | both real, no error |
| Global: 50% | 1800 | 60.00 | UNCERTAIN | NO_TRAP | both real, no error |
| Global: 75% | 2700 | 90.00 | UNCERTAIN | UNCERTAIN | clean |
| Global: final | 3599 | 119.97 | UNCERTAIN | UNCERTAIN | clean, no truncation |
| Strongest T0 TRAP_FORMING | 645 | 21.50 | TRAP_FORMING | — | matches semantic-QA preview exactly |
| Strongest T1 TRAP_FORMING | 1277 | 42.57 | NO_TRAP | TRAP_FORMING | episode T1-1275, matches contact sheet midpoint exactly |
| Strongest T0 run | 2594 | 86.47 | — | — | mid-run of R0-55-2558 (frames 2558–2630, max threat 0.61), cyan run arrows visible |
| Strongest T1 run | 1270 | 42.33 | — | — | mid-run of R1-54-1244 (frames 1244–1296, max threat 0.53), magenta run arrows visible |
| NO_TRAP interval | 1788 | 59.60 | NO_TRAP | NO_TRAP | dominant NO_TRAP |
| UNCERTAIN interval | 50 | 1.67 | UNCERTAIN | UNCERTAIN | clean |
| Defending-role transition | 177 | 5.90 | — | — | matches the audit's own previously-documented switch point; both estimated lines remain independently visible throughout (role selection only affects the historical-context label, never which lines/trap states are computed) |
| Line disappearance/reappearance | 2010 | 67.03 | UNCERTAIN | UNCERTAIN | see "Known pre-existing artifact" below — state itself unaffected |
| Run onset | 1244 | 41.47 | — | — | run R1-54-1244 begins; cyan arrows appear on the correct (Team 0) attacker |
| Run end | 1296 | 43.20 | — | — | run R1-54-1244 ends cleanly, no dangling arrow after |

All 14 required properties verified at each checked frame: labels
agree with analytics, radar colors correct (T0 cyan players / red
line, T1 magenta players / amber line, T0 cyan run arrows / T1 magenta
run arrows), both team records always present (0 missing across all
3600 frames), dominant banner correct, KPI values match the snapshot,
graph cursor aligns with the header time, timeline cursor matches the
current frame, no stale state after evidence expiry, no role/team flip
contamination, no state/score mismatch (the fixed `now/sustained/conf`
triplet — see the semantic-QA pass for the original bug and fix),
line-geometry direction correct (each team's line sits on their own
defensive side, verified visually across every sampled frame), radar
heatmap registered to player coordinates, no clipping/broken text, no
unsupported claims (`ESTIMATED DEFENSIVE LINE` / no official-offside
claim / no line-existence-implies-trap-intent conflation, all
confirmed present in the on-screen labels and unit-tested).

## Known pre-existing artifact — line disappearance/reappearance check

At frame 2010 (t=67.03s), Team 1's estimated line depth briefly spikes
from ~72.7m to ~104.7m for exactly one frame, then reverts. **Root
cause**: `line_geometry()` selects the second-nearest-to-goal visible
defender independently every frame (no temporal smoothing by design —
see B1's fix); a single frame's tracking/visibility jitter can
therefore swap which specific defender is "second-nearest," producing
a brief depth jump. This is the SAME category of artifact the original
audit's own evidence table already documented (its own cited examples:
"Raw selected T0 line jump... track87→14", "Raw selected T1 line
jump... track2→12") — a pre-existing, disclosed limitation of
frame-independent line selection, **not a new bug introduced by this
render**. Critically, it does **not** corrupt any displayed STATE: at
f2010 both teams correctly show `UNCERTAIN` (not a false trap
activation or a stale reading), and the anomaly self-corrects within
one frame. No code change was made for this — per the task's explicit
instruction, a pre-existing, already-disclosed limitation that does
not corrupt any state is reported, not silently "fixed" by altering
research semantics.

## Transition QA (programmatic, full 3600-frame scan)

- **No trap state persists beyond the evidence limit**: verified via
  the shared `test_active_state_does_not_persist_beyond_max_gap` unit
  test (same FSM engine as pressing) and by scanning all 13 real trap
  episodes for a same-`episode_id` resume after a ≥4-frame `UNCERTAIN`
  gap — 0 found.
- **Line identity/motion does not jump incorrectly across role
  changes**: `line_geometry()` is computed independently per team every
  frame, never gated on the possession-derived "defending role" — the
  one line-depth jump found (f2010, above) is a same-team frame-to-
  frame visibility artifact, unrelated to role/possession switching.
- **Attacking run identity remains consistent**: verified via
  `test_run_splits_elapsed_gap` and `test_multiple_runners_and_
  prefix_causality` (unit tests, unchanged), and by inspecting the
  onset/end frames of the two strongest runs directly (above) — no
  identity swap observed.
- **No future-derived run information appears prematurely**: verified
  programmatically across all 101 runs in the full replay —
  `observations <= (end_frame - start_frame + 1)` holds for every run
  (0 violations), and every run's `observations >= RUN_MIN_OBSERVATIONS`
  (15), confirming no run is confirmed before it has genuinely
  accumulated that much real, causal support.

## Bugs found and fixed during render QA

Same one performance-only fix as the pressing render (timeline bucket
colors/legend hoisted out of the per-frame closure — see that report's
"Bugs found" section for full detail; verified byte-identical output
before the full render). No correctness, indexing, or logic bugs were
found. The line-depth jump at f2010 was investigated and characterized
as a known, pre-existing, already-disclosed limitation (see above) —
not fixed, per the instruction not to alter research semantics unless
a genuine bug is found.

## Final verdict

**PASS.** Spec-conformant (2304×1204/30fps/3600 frames/120.00s),
time-sync verified, both teams present and consistent on every sampled
frame and programmatically on all 3600 frames, no stale states, no
role contamination, no state/score mismatches, honest 0-confirmed-
`TRAP_ACTIVE` result displayed consistently throughout (13 real trap
episodes total: 3 Team 0, 10 Team 1, all FORMING-level; 101 total runs:
67 against Team 1 / by Team 0 attackers, 34 against Team 0 / by Team 1
attackers).
