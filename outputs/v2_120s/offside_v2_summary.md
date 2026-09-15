# Offside-Break V2 Summary (full 120s clip)

**Scope**: rule-based analytics only. No predictive/learned model is
used anywhere in this document or its underlying code -- see
`offside_break/docs/METHODOLOGY.md` for the RULE-BASED vs LEARNED
separation policy. This is a V2 refinement of the v1 prototype
(`offside_break/outputs/offside_dashboard.mp4`, 25s), which remains
untouched as the baseline.

## IMPORTANT: a bug found in v1, left unfixed (by explicit decision)

While building V2's independent estimated-offside-position logic and
cross-checking it against v1's `onside` flag on real data, a **sign
inversion** was found in `offside_break/analytics/offside.py:162`
(`compute_attacker_features`):

```python
onside = distance_to_line_cm < 0  # behind (on their own side of) the line
```

Verified directly against tracking data at frame 3 (defending_team=0,
goal at x=12000cm): the defensive line sits at depth 6212.9cm (x
approx. 5787). An attacker at depth 4396cm (x approx. 7604) -- clearly
*closer* to the goal than the line, i.e. in the pocket beyond the last
defender -- is labeled `onside=True` by this code. A teammate at depth
6761cm (x approx. 5239), clearly farther from goal than the line
(i.e. on the normal, onside side), is labeled `onside=False`. Both are
backwards. This propagates into v1's on-screen "ONSIDE (behind line)" /
"AHEAD OF LINE (offside position)" label, `compute_break_potential()`,
and `time_to_enter_behind_line_sec`.

**By the user's explicit decision, v1 (`offside.py`, its dashboard, and
its outputs) is left completely untouched.** All V2 logic in
`offside_v2.py` is independent and was verified correct against the
same real frame (see `compute_line_relation_cm`'s docstring). Anyone
reusing v1's `onside` field, `compute_break_potential`, or the v1
dashboard's line-relation label should be aware it reads backwards.

## Part A: terminology fix

v1 displayed "ONSIDE (behind line)" / "AHEAD OF LINE (offside
position)" -- overstating a pure line-relation check as an offside
"position" claim. V2 replaces this everywhere with:

- **AHEAD OF DEFENSIVE LINE** / **BEHIND DEFENSIVE LINE** -- pure
  geometry (`line_relation_label`, fed by `compute_line_relation_cm`,
  positive = ahead of the line), no legality claim.
- A **separate** estimated-position state (see Part B) is only ever
  shown alongside, never merged into, the line-relation label.

## Part B: estimated offside-position logic

`compute_estimated_offside_position()` requires ALL THREE of:
1. `ahead_of_line` -- closer to the defending goal than the second-last defender.
2. `ahead_of_ball` -- closer to the defending goal than the ball.
3. `in_opponent_half` -- past the halfway line (6000cm from either goal).

Returns `ESTIMATED_OFFSIDE_POSITION` only if all three hold, else
`ESTIMATED_ONSIDE_POSITION`. If the ball's pitch position is missing OR
its detection confidence is below `BALL_CONFIDENCE_MIN_FOR_OFFSIDE =
0.4` (a disclosed threshold -- on this clip 990/1642 ball-track frames
have any pitch position at all, confidence ranging 0.25-0.86 mean 0.63;
0.4 drops only the least-confident ~9% of those), the function returns
`LINE_RELATION_ONLY` and refuses to guess. This is **never** presented
as an official offside decision: exact pass-release timing and
active-involvement/interference are not modeled -- see
`offside_break/docs/`-level caveats carried into this module's
docstrings.

## Part C: offside-trap candidate detector

A defensive line existing is not a trap. `compute_trap_synchrony()`
computes a `trap_synchrony_score` in [0,1] as the **mean of three
documented, equally-weighted components** (no fitted weights):

| Component | Definition |
|---|---|
| `fraction_stepping_forward` | fraction of visible defenders with forward velocity >= 50cm/s |
| `depth_dispersion_score` | `1 - std(defender depths) / 500cm`, clipped to [0,1] -- how tightly the line holds together |
| `velocity_alignment_score` | `1 - std(defender forward velocities) / 200cm/s`, clipped to [0,1] -- how synchronized their movement is |

States: `NO_TRAP` (score < 0.4), `TRAP_FORMING` (0.4 <= score < 0.65,
or score >= 0.65 but not yet sustained), `TRAP_ACTIVE` (score >= 0.65
sustained for >= 15 consecutive frames / 0.5s), `UNCERTAIN` (< 3
defenders visible).

**Measured finding**: on this clip, the score reaches a maximum of
0.66 (team-0-defending) / 0.67 (team-1-defending) only in brief,
isolated spikes (see `graphs/03_trap_synchrony_over_time.png`) -- it
**never sustains >= 0.65 for 15 consecutive frames, for either
defending team**. Result: **0 TRAP_ACTIVE frames on this entire 120s
clip.** `TRAP_FORMING` occurs on 899/3600 frames (25.0%) when team 0
defends, 377/3600 (10.5%) when team 1 defends. This is reported as a
real, disclosed finding -- the thresholds were not lowered to
manufacture a TRAP_ACTIVE event; they are the same simple thresholds
used throughout.

## Part D: temporal run detection

Runs are detected **within a single `cleaned_segment_id`** (never
bridging a rejected impossible-jump boundary) per `(track_id,
cleaned_segment_id)`, requiring sustained forward speed (>=150cm/s
toward the defending goal) while within 30m of the current defensive
line depth, for >=0.5s (15 frames); gaps in the qualifying condition
are bridged only if <=5 frames.

Run-type taxonomy (8 classes: straight, diagonal, lateral, curved,
check-run, inside-to-outside, outside-to-inside, unknown). Complex
classes (curved/check-run/in-out) are only attempted when a run has
**>=45 frames (1.5s)** of data -- raised from an initial 20-frame
(0.67s) cutoff during development after measuring that the shorter
cutoff pushed roughly half of all qualifying runs into "curved" purely
from frame-to-frame velocity-angle noise on short segments; 45 frames
produces a materially more plausible distribution (see below).
Third-man-run classification is **not implemented** -- it needs
teammate/ball-interaction evidence this detector does not have access
to, and is never forced.

**Run counts (full clip, both directions)**:

| Direction | Runs | straight | diagonal | lateral | curved | check-run | in->out | out->in |
|---|---|---|---|---|---|---|---|---|
| team 1 attacks (team 0 defends) | 132 | 58 | 41 | 1 | 25 | 3 | 1 | 3 |
| team 0 attacks (team 1 defends) | 242 | 107 | 88 | 0 | 40 | 0 | 4 | 3 |

**374 total run episodes** across both directions.

## Part E: offside-break candidate detector

A run only becomes a trap-break **candidate** if TRAP_ACTIVE held at
some point during the run AND the run threatens/crosses the line
(`distance_to_line_end_cm >= -200cm`, i.e. ahead of the line or within
2m of it). Since TRAP_ACTIVE never occurs on this clip (Part C), **0
of the 374 run episodes qualify as trap-break candidates, in either
direction.** No outcome labels (SUCCESSFUL_BREAK / OFFSIDE_OR_EARLY /
PASS_NOT_PLAYED / DEFENDER_RECOVERS / UNCERTAIN) were assigned because
no candidate exists to assign them to -- this is not a gap in the
outcome logic (which was tested and confirmed working via the pass-
matching and estimated-position paths), it is a direct, measured
consequence of Part C's finding.

## Coverage / limitations

- Ball position is available for only 990/1642 ball-track frames
  (60%) even before the confidence gate; `ESTIMATED_OFFSIDE_POSITION`
  / `ESTIMATED_ONSIDE_POSITION` are consequently shown on a minority of
  frames -- most frames correctly fall back to `LINE_RELATION_ONLY`.
- The defensive-line depth series (`offside_line_series`, v1's
  function, reused read-only) shows some large short-term swings
  (e.g. one run's "line depth" briefly implying the line sat ~86m from
  its own goal, i.e. deep in the opponent's third) -- almost certainly
  attributable to sparse/noisy raw tracking (previously measured: ~50%
  of raw tracks have a candidate >20m/s "impossible jump"), possibly
  compounded by an occasional mis-attributed second-last-defender pick
  when few defenders are visible. This is disclosed, not smoothed
  away or hidden -- see `graphs/01_defensive_line_depth_over_time.png`.
- 0 TRAP_ACTIVE frames / 0 trap-break candidates means Parts C-E have
  no positive examples to validate against on this specific clip; the
  detectors are exercised and produce sane, non-degenerate scores
  (Part C's 0.66/0.67 near-misses show the score genuinely responds to
  real defensive coordination, it simply doesn't sustain), but a clip
  containing an actual textbook offside trap would be needed to fully
  validate the TRAP_ACTIVE/break-candidate path end-to-end.
- All graphs are labeled "descriptive/exploratory only -- NOT
  statistical evidence"; no claim of statistical significance is made.

## Dashboard: `offside_dashboard_120s_v2.mp4`

Full 120s (3600 frames, 30fps, canvas 2304x1204), attacking_team=1 /
defending_team=0 (a disclosed, fixed choice for this render -- this
direction showed more TRAP_FORMING signal, 899 vs. 377 frames, making
the trap visualization more active on screen; the other direction's
analytics are fully covered in the static graphs above). Layout:
top-left original video (defensive line, lead-attacker highlight, RUN
labels on players inside a detected run, trap-state text); top-right
radar (pitch control, space-behind-line shading, defensive line,
attacker velocity arrows -- reused read-only from v1's `draw_right_top`);
bottom-left 3 live scrolling graphs -- **ATTACKER DEPTH vs
DEFENSIVE-LINE DEPTH** (dual-line, the brief's most-important graph),
trap synchrony score, relative closing rate onto the line; bottom-right
numeric metrics (trap synchrony + its 3 components, line relation,
estimated position state with ball-confidence caveat shown inline,
relative speed, nearest defender, space behind line, passing-lane
status, active-run info) + event/outcome text; bottom timeline strip
(trap-state color band + run-episode band + candidate-marker ticks,
current-time marker).

"Lead attacker" (used for the depth-vs-line graph and the metrics
panel) is defined per-frame as the currently-visible attacker with the
smallest depth-from-goal (i.e. the most advanced one) -- a
well-defined scalar quantity even though the specific player holding
that position can change frame to frame, avoiding identity-switch
artifacts in the scrolling graph.

## Visual QA (mandatory pass, full-clip frames)

See the QA log in this session; inspected frames at 0%, 25%, 50%, 75%,
~99.5% of the clip plus multiple mid-clip timestamps during development
testing (10s and 20s test segments) before the full render, confirming:

- Defensive line (red) follows the actual 2nd-last defender and tracks
  plausibly with on-screen defender positions in both the video panel
  and radar.
- `LEAD ATTACKER` highlight and `AHEAD OF DEFENSIVE LINE` /
  `BEHIND OF DEFENSIVE LINE` label were consistent with the attacker's
  visible position relative to the drawn line at every checked frame.
- `ESTIMATED_OFFSIDE_POSITION` only appears together with a confident
  ball position; frames with an unavailable/low-confidence ball
  correctly show `LINE_RELATION_ONLY` with an explicit
  "[ball not confident this frame]" tag in the metrics panel -- no
  legal-sounding offside claim is ever shown without ball context.
  Confirmed at multiple frames (e.g. t=3s: LINE_RELATION_ONLY with the
  tag shown; t=9s, t=15s: ESTIMATED_OFFSIDE_POSITION shown once a
  confident ball position was available).
- Trap-state text (`NO_TRAP` / `TRAP_FORMING`, gray/orange) matched the
  live trap-synchrony-score graph at every checked frame; `TRAP_ACTIVE`
  never appeared, consistent with the measured 0-TRAP_ACTIVE finding.
- The depth-vs-line dual-line graph correctly showed the orange
  (attacker) line sitting below the red (line) line exactly when the
  metrics panel said "AHEAD OF DEFENSIVE LINE" (lower depth-from-goal =
  closer to goal = ahead), consistent at every checked frame.
- "Passing Lane From Carrier" initially always showed a hardcoded
  "n/a" placeholder -- found and fixed during QA before the full
  render (see Bugs section below); reruns confirmed it now reports
  "open" / "blocked" / a specific reason when no confident carrier
  exists.
- No stale labels or overlapping legend text observed in any checked
  frame; timeline strip's current-time marker matched elapsed video
  time throughout.

No problems requiring a fix/re-render of the FULL clip were found
after the pre-render fixes below were applied (those were caught and
fixed during 10s/20s test-segment QA, before the full 120s render).

## Bugs found and fixed during V2 development

1. **Sign inversion in `compute_estimated_offside_position`** (my own
   draft, caught before shipping): initially wrote
   `ahead_of_line = attacker_depth > line_depth_cm` (and the analogous
   `>` for `ahead_of_ball`/`in_opponent_half`), which is the same class
   of error found in v1 (see above). Caught by testing against the
   same real frame-3 data used to find the v1 bug; corrected to `<`
   throughout, with the derivation recorded in the function's
   docstring.
2. **`detect_attacking_runs`'s `distance_to_line_*_cm` used the
   v1-style (inverted) sign convention** while `classify_trap_break_
   candidates` assumed the corrected convention -- found during code
   review before testing; fixed to use `compute_line_relation_cm`
   consistently.
3. **`RUN_MIN_FRAMES_FOR_CURVED_CHECK` raised from 20 to 45 frames**
   after measuring that the shorter cutoff classified ~49% of
   qualifying runs as "curved" from short-segment velocity-angle noise
   (see Part D).
4. **"Passing Lane From Carrier" hardcoded to "n/a"** in the first
   dashboard draft -- found during 10s-segment QA; fixed to compute the
   real lane status via `build_possession` + the existing
   `_passing_lane_open` helper (imported read-only from
   `pressing_structure.analytics.pressing`).
5. **The v1 `offside.py` `onside` sign inversion** (see top of this
   document) -- found, reported to the user, and left unfixed in v1 by
   explicit decision.
6. **"Relative Speed to Line" showed a physically absurd value**
   (+205.7 m/s, found during full-clip QA at t=30s): the metric was
   computed as a 1-frame finite difference of the smoothed line-depth
   series multiplied by fps. The "second-last defender" pick can
   switch between two players at different depths from one frame to
   the next (sparse/noisy raw tracking), and a 1-frame diff amplifies
   that into an instantaneous spike. Fixed to use a 15-frame (0.5s)
   baseline difference instead of a 1-frame one; re-verified at the
   same timestamp (now +4.5 m/s) and the full 120s dashboard was
   **re-rendered** with the fix before this summary was finalized.

## Runtime / storage

- Full analytics (line series + trap series + runs + candidates,
  BOTH directions): ~13s.
- Full 120s dashboard render: ~6 minutes (3600 frames).
- Static graph generation (`generate_offside_graphs.py`): ~20s,
  produced 10 PNGs.
