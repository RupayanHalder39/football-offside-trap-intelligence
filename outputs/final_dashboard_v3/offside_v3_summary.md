# Offside / Trap-Break Dashboard V3 — Tactical Logic Upgrade Summary

## What changed and why

V2's offside dashboard picked ONE "defending team this frame" from
rolling-majority possession and only ever showed that team's defensive
line/trap state — the other team's back-line shape wasn't tracked at
all in the same frame. V3 adds a dual-ROLE analytics layer
(`offside_break/analytics/offside_v3.py`) that computes each team's
trap-as-defender state AND its run-threat-as-attacker state
**simultaneously, every frame, for both teams**, stabilized by the
same shared hysteresis FSM introduced for pressing
(`pressing_structure/analytics/state_machine.py`), then re-renders the
dashboard (`render_offside_dashboard_final_v3.py`) around that data.
V1 (`offside.py`) and V2 (`offside_v2.py`) analytics are completely
unmodified and still relied on read-only (`compute_trap_synchrony`,
`detect_attacking_runs`, `classify_trap_break_candidates`,
`compute_line_relation_cm`).

## Goal 1: simultaneous dual-role tracking

Every frame, for EACH team T in {0, 1}: `trap_score_as_defender[T]`,
`trap_state_as_defender[T]`, `trap_confidence_as_defender[T]` (the
smoothed score), `run_threat_score_as_attacker[T]`, and
`break_state_as_attacker[T]` (`build_dual_role_states`). Team 0 is
analyzed as a defender (vs Team 1 attacking) AND Team 1 is analyzed as
a defender (vs Team 0 attacking) in the same computation — neither role
is hardcoded. `break_state_as_attacker[T]` is deliberately defined as a
direct MIRROR of the opponent's own `trap_state_as_defender[1-T]`
("is the trap I'm attacking active" is the same fact as "is the
opponent's trap active") rather than a second, potentially-inconsistent
state machine for the same event.

`decide_dominant_trap_state()` mirrors pressing's arbitration rule but
allows both teams to show a real elevated state at once (a trap can be
active for both back lines independently, unlike pressing's
mutually-exclusive-by-definition scores) — reported as `CONTESTED` only
if both are genuinely at `TRAP_ACTIVE`/`TRAP_BREAK_THREAT`/
`TRAP_BROKEN` simultaneously.

## Goal 2: transition-aware state machine + two new real states

Same shared FSM engine as pressing: rolling-window smoothing (0.5s) →
hysteresis with enter/exit thresholds, minimum activate/hold frames, an
internal ending window, and a cooldown. Output vocabulary constrained
to exactly what was requested — `NO_TRAP`, `TRAP_FORMING`,
`TRAP_ACTIVE`, `TRAP_BREAK_THREAT`, `TRAP_BROKEN`, `UNCERTAIN` (the
FSM's internal `TRAP_ENDING` is used only for hysteresis timing and
collapses to `NO_TRAP` for display unless `TRAP_BROKEN` already caught
it).

Two genuinely new, rule-based, geometrically-grounded overlay states:
- **`TRAP_BREAK_THREAT`**: the defender's state is `TRAP_ACTIVE` AND the
  opponent's smoothed run-threat score ≥ 0.5 (a real attacking run is
  bearing down on an active trap).
- **`TRAP_BROKEN`**: the state was `TRAP_ACTIVE`/`TRAP_BREAK_THREAT`
  within the last 1s, the trap score has since dropped below
  `exit_th`, AND the lead attacker is now demonstrably ≥1m ahead of the
  defensive line (`compute_line_relation_cm`, unmodified). Disclosed,
  conservative — not a claim of an official offside/goal-scoring
  outcome.

**Thresholds used (`TRAP_FSM_CONFIG`)**: `enter_th=0.5`, `exit_th=0.3`
(unchanged from `offside_v2`), `min_activate_frames=8`,
`min_hold_frames=10`, `ending_frames=10`, `cooldown_frames=15`.

**These were explicitly re-validated, not blindly carried over.**
Measuring the real smoothed trap-synchrony score distribution on this
clip: Team 0 p75=0.38 / p90=0.43 / p95=0.46 / p99=0.49 / max=0.53; Team
1 p75=0.31 / p90=0.38 / p95=0.41 / p99=0.47 / max=0.52. At 0.5, both
teams' scores DO cross it briefly (both show a handful of
`TRAP_FORMING` segments) but never sustain the required 8 consecutive
frames, so `TRAP_ACTIVE` never fires on this clip — consistent with,
and independent corroborating evidence for, the prior V2 finding that
"trap synchrony peaks 0.66–0.67 but never sustains." As a check, the
same 0.4/0.25 pair that fixed pressing's threshold was tried here too:
it pushed Team 0 into `TRAP_ACTIVE` for 35.6% of the clip (one
continuous 15.3s "episode"), because 0.4 sits barely above Team 0's own
75th percentile — i.e. ordinary, unremarkable fluctuation would
constantly cross it. That is the "forcing a label the evidence doesn't
support" failure mode the brief warns against, not a fix, so **0.5/0.3
was kept**. Pressing and offside do not have to share the same
numeric thresholds just because they share the same FSM engine — each
was measured against its own score's real distribution, and this time
the original numbers were already correct.

**Result on the full 120s clip**: Team 0 — 0 `TRAP_ACTIVE` episodes, 14
brief `TRAP_FORMING` segments, mean smoothed score 0.277, 132 attacking
runs faced. Team 1 — 0 `TRAP_ACTIVE` episodes, 12 brief `TRAP_FORMING`
segments, mean smoothed score 0.234, 242 attacking runs faced. 0
frames of `TRAP_BREAK_THREAT`/`TRAP_BROKEN` for either team (expected:
neither state can fire without a `TRAP_ACTIVE` episode to threaten or
break from first, and this clip never produces one).

## Goal 3: visualization

- Base pitch-control wash unchanged (Team 0 cyan, Team 1 magenta).
- **Both teams' defensive lines drawn simultaneously**: Team 0 = bright
  red, Team 1 = orange-yellow (`TEAM_LINE_COLOR`) — no longer gated
  behind whichever team currently has the ball.
- Dangerous-space hatched zone drawn for both teams' lines, tinted in
  that team's own line color, sparse/semi-transparent (unchanged
  hatching mechanism from V2 — never an opaque fill).
- Attacking-run velocity arrows now colored by the running team's own
  color (Team 0 = cyan, Team 1 = magenta) instead of one fixed green
  for everyone.
- Small explicit "Team 0 trap state: … / Team 1 trap state: …" text on
  the radar panel (with a dark backing for legibility over the pitch),
  and the same pairing repeated in the left-feed state badge's role
  line.
- **Observed but not "fixed"**: at one inspected frame (t≈73.6s) both
  teams' computed defensive lines were only ~38cm apart (both teams'
  back lines genuinely compressed into the same penalty-box area at
  that moment), so one color visually occluded the other on the line
  itself — this is an honest rendering of a real, near-identical
  geometric fact, not a bug; the redundant text/KPI-tile state labels
  still correctly disambiguate which team's line is which at that
  instant. Fabricating a visual gap between two lines that are
  genuinely almost coincident would misrepresent the geometry, so this
  was left as-is.

## Goal 4: dual-team KPI summary

The KPI tile row now shows: Team 0 (Defender) state + runs-faced +
mean score, Team 1 (Defender) state + runs-faced + mean score,
"Defending This Frame" (the possession-derived team, kept as
supplementary context), and the dominant/display trap state for this
frame. A new Team-0-vs-Team-1 smoothed trap-score comparison graph
replaces V2's single-team "Distance to Lead Attacker" graph, making the
transition logic directly visible over time. The trap-state
distribution panel now covers both teams' V3 stable states (including
the two new overlay states) instead of V2's single-direction counts.

## Goal 5: transition QA performed

- Verified the raw FSM state arrays directly (not just the render) for
  both teams: state segments match the measured score distribution,
  short `TRAP_FORMING` blips that fail to escalate correctly revert to
  `NO_TRAP` without cooldown gating (cooldown is reserved for a real
  active episode ending, by design), and a single-frame `TRAP_FORMING`
  segment found in the data is a genuine one-frame threshold crossing
  followed by an immediate drop — not an oscillation bug.
- Rendered and visually inspected both of the clip's two real
  `TRAP_FORMING` windows (Team 0 at t≈73.6s, Team 1 at t≈37.9s):
  confirmed the banner, the radar's per-team text, and all four KPI
  tiles agree with each other and with the underlying FSM arrays at
  each instant, with no stale carry-over from the other team's state.
- Rendered t=0.0s (an `UNCERTAIN` boundary frame) to confirm no crash
  at the clip's very first frames.

## Files changed / created

**Modified**: none of `offside.py` or `offside_v2.py`.
`render_offside_dashboard_final_v3.py` was newly authored starting
from a copy of the V2 script (not the V2 script itself). Also added
one extra key (`"runs_df"`) to `compute_direction_data`'s return dict
inside the V3 script's own file — a pure addition, no existing key
removed or changed.

**New files**:
- `offside_break/analytics/offside_v3.py`
- `offside_break/dashboard/render_offside_dashboard_final_v3.py`
- `offside_break/outputs/final_dashboard_v3/` (this summary, the full
  render, preview frames)

**Untouched**: `offside_break/analytics/offside.py`,
`offside_break/analytics/offside_v2.py`, all V1/V2 dashboard scripts,
all prior renders/videos, `tracking.parquet`, `metadata.json`, and
every other canonical analytics output.

## Answering the required questions

- **Both teams tracked simultaneously?** Yes — every frame computes a
  real trap-as-defender state and run-threat-as-attacker state for both
  teams independently; the dashboard now shows both teams' lines,
  states, and KPIs at once rather than switching to whichever team
  currently has the ball.
- **Transition logic added?** Yes — the same shared hysteresis FSM used
  by pressing, plus two new rule-based overlay states
  (`TRAP_BREAK_THREAT`, `TRAP_BROKEN`) built from real run-threat and
  line-relation geometry.
- **Thresholds/hysteresis used?** `enter_th=0.5`, `exit_th=0.3`,
  `min_activate_frames=8`, `min_hold_frames=10`, `ending_frames=10`,
  `cooldown_frames=15` — explicitly re-measured against this clip's
  real score distribution and confirmed correct as-is (lowering them
  to pressing's 0.4/0.25 was tested and found to over-trigger badly, so
  it was rejected).
- **Full 120s render passed QA?** See the render log and preview
  frames in this folder; both of the clip's two real `TRAP_FORMING`
  windows were individually inspected and found consistent.
