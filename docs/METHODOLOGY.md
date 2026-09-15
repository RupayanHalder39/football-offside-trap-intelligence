# Offside-Trap Break — Methodology

Separate research prototype, built alongside the existing PhD Problem 1
pipeline. Does not modify, retrain, or reinterpret any PhD conclusion.

## Research question

"Where is the defensive/offside line, how are attackers moving relative
to it, and which movement patterns create legal space behind the line?"

## Input data

`outputs/tracking/testVideo1_120s/tracking.parquet` (RAW, read-only,
never modified) + the EXISTING, CANONICAL `outputs/analytics/
testVideo1_120s_v3/{ball_trajectory,homography_transformers}` files
(read-only). No new model inference anywhere in this project.

## Assumptions

1. Each team's defending end was MEASURED (not assumed) from its
   goalkeeper's average pitch position across the clip: team 0 defends
   x=12000, team 1 defends x=0 (see `offside_summary.md`).
2. The "offside line" is the SECOND-LAST defender by depth (including
   the goalkeeper in the depth ordering, per Law 11's own wording) --
   computed fresh per frame from whichever defending-team players are
   currently visible, smoothed with a 5-frame rolling median for
   DISPLAY stability only (the underlying per-frame value is never
   altered, only the on-screen line position is smoothed).
3. A frame's line is `valid=False` (never guessed) when fewer than 2
   defenders are visible that frame.
4. This is a POSITIONAL estimate only. **It is explicitly NOT an
   official offside decision** -- that requires the exact frame of the
   ball leaving the passer's foot, all players' positions at that exact
   instant, and active-play/interference rules this prototype does not
   model.

## Shared cleaned analysis view

Uses the SAME `pressing_structure.analytics.cleaned_tracking_view.
build_cleaned_view()` as Project 1 (imported, not duplicated) -- see
that project's METHODOLOGY.md for the full description of the
short-gap-fill + impossible-jump-rejection + per-segment smoothing
pipeline. The same limitation applies here: average cleaned-segment
length is ~2s, which caps how much confidence a run's velocity/angle
can carry over a longer span.

## Exact feature definitions

| Feature | Definition |
|---|---|
| offside_line (per frame) | second-last defender's depth-from-own-goal, among currently visible defenders |
| distance_to_line_cm | attacker's depth-from-defending-goal MINUS the line's depth (positive = ahead of the line / offside-position side; negative = onside) |
| onside | `distance_to_line_cm < 0` |
| relative_velocity_to_line_cm_s | attacker's velocity component pointing TOWARD the defending goal (fixed axis, measured from the defending goal's own x-position, not the attacker's) |
| run_angle_deg | angle of the attacker's velocity vector relative to that same forward axis (0=straight forward, ±90=lateral) |
| run_type | straight (\|angle\|<=20°), diagonal (20-70°), lateral (>=70°), unknown (no velocity available) -- ONLY these 3 classes are attempted, see Known Limitations |
| nearest_defender_distance_cm | min Euclidean distance, this attacker to any visible defender |
| passing_lane_open_from_carrier | same geometric point-to-segment proxy as Project 1's `_passing_lane_open` (REUSED, not duplicated) -- only computed when a confident ball carrier (possession_state==CONTROLLED) exists this frame, else `n/a` |
| time_to_enter_behind_line_sec | (line depth − attacker depth) / relative_velocity_to_line, only when onside AND closing (positive relative velocity); else `n/a` -- never a guessed ETA |
| break_potential | LOW/MEDIUM/HIGH/omitted -- see thresholds below |
| space_behind_line (dashboard only) | pitch region between the line and the defending goal's byline, shaded on the radar panel -- a fixed geometric region, not a modeled "danger" score |
| potential_receiving_space (dashboard only) | the EXISTING PhD `analytics.pitch_control` continuous surface (REUSED, unmodified), shown as the radar's background wash |

## Break Potential — exact, disclosed thresholds (no fabricated probability)

```
if not onside:                                    LOW   (already ahead of the line)
elif closing_speed >= 200 cm/s and nearest_defender_distance >= 300 cm:  HIGH
elif closing_speed >=  80 cm/s and nearest_defender_distance >= 150 cm:  MEDIUM
else:                                              LOW
```
Omitted entirely (not even LOW) when velocity or nearest-defender data
is unavailable that frame.

## Uncertainty handling

No line -> no attacker assessment that frame (empty, not fabricated).
No velocity -> run_type "unknown", not forced into a class. No
confident ball carrier -> passing-lane feature is `n/a`, not guessed.

## What is analytics vs. what is learned

Everything here is rule-based geometry over already-tracked positions
plus reused PhD modules (`possession`, `pitch_control`). No model was
trained. "HIGH break potential episodes" (409, both directions, full
clip) are geometric threshold crossings, NOT validated real-world
outcomes -- there is no ground-truth label for "this run actually beat
the offside trap and led to a chance" anywhere in this dataset.

## Known limitations

1. Not an official offside decision -- positional estimate only, no
   pass-release-instant modeling, no active-play/interference rules.
2. Run-type classification covers only 3 of the 7 classes named in the
   original brief (straight/diagonal/lateral implemented; curved,
   check-run, third-man, inside-to-outside/outside-to-inside are
   explicitly NOT implemented in this pass -- would need multi-frame
   trajectory-shape analysis not yet built).
3. "HIGH break potential" has no outcome validation (see above) --
   unlike Project 1's pass/turnover-anchored SUCCESS/FAILED, there is
   nothing here to check a break-potential call against.
4. Space-behind-line and pitch-control-behind-line are simple, disclosed
   geometric/model reuses, not bespoke danger models.
5. Same cleaned-view segment fragmentation (~2s average) as Project 1
   limits longer-run velocity confidence.
6. Only a 25-second window (matching Project 1's, for comparability)
   was rendered to video; the full 120s was analyzed but not rendered.

## What the dashboard shows / does NOT prove yet

Shows: a specific, geometry-grounded, inspectable read of where the
defending line currently sits, which attackers are ahead of or behind
it, how fast they're closing, and a disclosed-threshold Break Potential
label for each visible attacker.

Does NOT prove: that any specific run pattern is "the best" way to beat
an offside trap in general, that a HIGH break-potential moment actually
led to a chance, or that any single frame's line position would match
an official assistant referee's call at the actual moment of a pass.
