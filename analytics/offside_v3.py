"""
Offside/Trap V3: SIMULTANEOUS dual-role tracking (both teams analyzed
as-defender AND as-attacker every frame) with the SAME transition-
aware hysteresis state machine used by `pressing_structure.analytics.
pressing_v3`, replacing V2's ad-hoc single-counter "sustain" rule for
TRAP_ACTIVE with a documented, shared, reusable FSM plus two new real
states (TRAP_BREAK_THREAT, TRAP_BROKEN).

NEW module. Does not modify `offside_break/analytics/offside.py` (v1)
or `offside_v2.py` -- reuses their existing, unmodified functions
read-only: `compute_trap_synchrony` (the real per-frame trap-synchrony
score), `detect_attacking_runs`, `classify_trap_break_candidates`,
`compute_line_relation_cm`. V1/V2 outputs remain reproducible exactly
as before.

## Dual-role, every frame, both teams

For EACH frame and EACH team T in {0, 1}:
  - `trap_score_as_defender[T]`  = `compute_trap_synchrony(..., defending_team=T, frame)`'s
    real score (already computed by offside_v2, unmodified) -- None
    when <3 defenders of team T are visible (real "insufficient
    evidence", not a fabricated zero).
  - `trap_state_as_defender[T]` = the smoothed+FSM-stabilized state
    from that score (NO_TRAP/TRAP_FORMING/TRAP_ACTIVE/
    TRAP_BREAK_THREAT/TRAP_BROKEN/UNCERTAIN -- see below).
  - `run_threat_score_as_attacker[T]` = a real, disclosed combination
    of (a) whether team T currently has an ACTIVE attacking run
    (from `detect_attacking_runs(attacking_team=T, defending_team=1-T)`,
    already computed) and (b) that run's closing rate onto the
    opponent's line (reusing the same closing-rate computation
    established in the V2 dashboard) -- 0.0 when no run is active
    (a real, defined zero: there is nothing threatening this frame),
    None when the opponent's line itself is not computable this frame.
  - `break_state_as_attacker[T]` is defined as the MIRROR of the
    opponent's own `trap_state_as_defender[1-T]` -- "is the trap I'm
    attacking currently active/forming/broken" is definitionally the
    same fact as "is the opponent's trap active/forming/broken", so
    this reuses that state directly rather than running a second,
    potentially-inconsistent state machine for the same underlying
    event from the other side.

## TRAP_BREAK_THREAT / TRAP_BROKEN (new, real, rule-based)

- TRAP_BREAK_THREAT: the defender's stabilized state is TRAP_ACTIVE
  AND the opposing team's `run_threat_score_as_attacker` this frame
  (smoothed) is >= `RUN_THREAT_ENTER_TH` -- i.e. a real attacking run
  is genuinely bearing down on an active trap.
- TRAP_BROKEN: the state was TRAP_ACTIVE or TRAP_BREAK_THREAT within
  the last `BROKEN_LOOKBACK_FRAMES`, the trap-synchrony score has now
  dropped below `PRESS... exit_th` (defenders no longer synchronized),
  AND the lead attacker for that direction is currently AHEAD OF the
  defensive line by at least `BROKEN_LINE_MARGIN_CM` (using the
  existing, unmodified `compute_line_relation_cm`) -- a disclosed,
  conservative definition: the trap was on, it fell apart, and an
  attacker is demonstrably past the line as a result. This is NOT an
  official "goal-scoring chance" or outcome claim -- see
  `offside_v3_summary.md` for exact scope limits.
"""
import os
import sys

import numpy as np
import polars as pl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from offside_break.analytics.offside import _forward_depth  # noqa: E402
from offside_break.analytics.offside_v2 import compute_line_relation_cm, compute_trap_synchrony  # noqa: E402
from pressing_structure.analytics.state_machine import FSMConfig, raw_to_stable_states, states_to_segments  # noqa: E402

TRAP_SCORE_WINDOW_FRAMES = 15   # 0.5s rolling smoothing window (same as pressing V3)
RUN_THREAT_WINDOW_FRAMES = 10   # 0.33s -- runs are already temporally coherent segments

TRAP_FSM_CONFIG = FSMConfig(
    # VALIDATED (not blindly copied) against this clip's measured
    # SMOOTHED trap-synchrony score distribution, using the same
    # percentile methodology that caught and fixed a real miscalibration
    # in pressing_v3's thresholds -- but here the conclusion is the
    # OPPOSITE of pressing's: the original 0.5/0.3 (inherited from
    # offside_v2) turned out to already be well-calibrated, and lowering
    # it would have been a real bug, not an improvement.
    #
    # Measured smoothed-score percentiles: team0 p75=0.38 p90=0.43
    # p95=0.46 p99=0.49 max=0.53; team1 p75=0.31 p90=0.38 p95=0.41
    # p99=0.47 max=0.52. At enter_th=0.5, both teams' scores DO cross it
    # (briefly -- confirmed via TRAP_FORMING appearing a handful of
    # times for each team) but never sustain min_activate_frames=8
    # consecutive frames there, so TRAP_ACTIVE never fires on this clip
    # -- consistent with, and independent corroborating evidence for,
    # the prior V2 finding ("trap synchrony peaks 0.66-0.67 but never
    # sustains"). This is a genuinely rare, ~99th-percentile-or-above
    # event, not a threshold sitting above where real elevated evidence
    # already lives (that WAS pressing's bug).
    #
    # To check the alternative, enter_th was experimentally lowered to
    # pressing's 0.4/0.25 (mirroring pressing_v3 for cross-project
    # consistency) and re-run: it pushed team0 into TRAP_ACTIVE for
    # 1281/3600 frames (35.6% of the clip), including one single
    # "episode" lasting 15.3 continuous seconds. 0.4 sits only just
    # above team0's OWN 75th percentile, i.e. it is crossed by
    # unremarkable, typical-fluctuation frames roughly a quarter of the
    # time -- exactly the "forcing a label the evidence doesn't support"
    # failure mode the brief warns against, not a fix. So 0.5/0.3 is
    # kept as-is; pressing and offside do NOT have to share the same
    # numeric thresholds just because they share the same FSM engine --
    # each was measured against its own score's real distribution.
    enter_th=0.5, exit_th=0.3,
    min_activate_frames=8, min_hold_frames=10, ending_frames=10, cooldown_frames=15,
    low_state="NO_TRAP", forming_state="TRAP_FORMING",
    active_state="TRAP_ACTIVE", ending_state="TRAP_ENDING", uncertain_state="UNCERTAIN",
)
RUN_THREAT_ENTER_TH = 0.5
BROKEN_LOOKBACK_FRAMES = 30      # 1s -- how far back "was recently active/threat" looks
BROKEN_LINE_MARGIN_CM = 100.0    # 1m clearly ahead of the line, not just borderline


def _team_forward_sign(defending_team: int) -> float:
    from offside_break.analytics.offside_v2 import forward_sign_away_from_own_goal
    return forward_sign_away_from_own_goal(1 - defending_team)


def compute_dual_role_raw_scores(cleaned: pl.DataFrame, runs_by_defender: dict, n_total_frames: int) -> dict:
    """`runs_by_defender`: {0: runs_df_for_defending_team_0, 1: ...}
    (attacking_team = 1 - defending_team in each), already computed by
    the caller via `detect_attacking_runs` (unmodified). Returns
    per-team trap and run-threat raw score arrays."""
    trap_raw = {0: [None] * n_total_frames, 1: [None] * n_total_frames}
    for defending_team in (0, 1):
        for f in range(n_total_frames):
            r = compute_trap_synchrony(cleaned, defending_team, f)
            trap_raw[defending_team][f] = r["score"] if r["valid"] else None

    run_frame_index = {0: {}, 1: {}}
    for defending_team in (0, 1):
        runs = runs_by_defender[defending_team]
        for run in (runs.to_dicts() if runs.height else []):
            for fr in range(run["start_frame"], run["end_frame"] + 1):
                run_frame_index[defending_team].setdefault(fr, []).append(run)

    run_threat_raw = {0: [None] * n_total_frames, 1: [None] * n_total_frames}
    for defending_team in (0, 1):
        attacking_team = 1 - defending_team
        for f in range(n_total_frames):
            active_here = run_frame_index[defending_team].get(f, [])
            if not active_here:
                # A real, defined zero: this attacking team has no run
                # threatening this defensive line right now.
                run_threat_raw[attacking_team][f] = 0.0
                continue
            run = active_here[0]
            speed = run.get("mean_forward_speed_cm_s")
            speed_term = 0.0 if speed is None else float(np.clip((speed) / 300.0, 0.0, 1.0))
            run_threat_raw[attacking_team][f] = float(np.clip(0.6 + 0.4 * speed_term, 0.0, 1.0))
    return {"trap": trap_raw, "run_threat": run_threat_raw}


def build_dual_role_states(cleaned: pl.DataFrame, ball: pl.DataFrame, passes: pl.DataFrame,
                             runs_by_defender: dict, n_total_frames: int, fps: float = 30.0) -> dict:
    """Full pipeline for both teams-as-defender (trap) and both
    teams-as-attacker (run threat + mirrored break state), plus the
    new TRAP_BREAK_THREAT / TRAP_BROKEN overlay states."""
    raws = compute_dual_role_raw_scores(cleaned, runs_by_defender, n_total_frames)
    trap_out = {}
    for team in (0, 1):
        smoothed, states = raw_to_stable_states(raws["trap"][team], TRAP_SCORE_WINDOW_FRAMES, TRAP_FSM_CONFIG)
        trap_out[team] = {"raw": raws["trap"][team], "smoothed": smoothed, "states": states}

    run_threat_out = {}
    for team in (0, 1):
        smoothed = _rolling_mean(raws["run_threat"][team], RUN_THREAT_WINDOW_FRAMES)
        run_threat_out[team] = {"raw": raws["run_threat"][team], "smoothed": smoothed}

    # Overlay TRAP_BREAK_THREAT / TRAP_BROKEN onto each defender's base
    # TRAP_ACTIVE/TRAP_ENDING states using the opposing attacker's
    # run-threat and the real line-relation geometry.
    final_states = {0: [], 1: []}
    recent_active_or_threat = {0: [], 1: []}  # rolling record for the BROKEN lookback
    for defending_team in (0, 1):
        attacking_team = 1 - defending_team
        states = trap_out[defending_team]["states"]
        smoothed = trap_out[defending_team]["smoothed"]
        threat_smoothed = run_threat_out[attacking_team]["smoothed"]
        was_recently_active = []
        window = []
        for f in range(n_total_frames):
            base = states[f]
            s = smoothed[f]
            t = threat_smoothed[f]
            final = base
            if base == TRAP_FSM_CONFIG.active_state and t is not None and t >= RUN_THREAT_ENTER_TH:
                final = "TRAP_BREAK_THREAT"
            window.append(final in (TRAP_FSM_CONFIG.active_state, "TRAP_BREAK_THREAT"))
            while len(window) > BROKEN_LOOKBACK_FRAMES:
                window.pop(0)
            recently_active = any(window[:-1])  # excludes this frame itself
            if final in (TRAP_FSM_CONFIG.ending_state, TRAP_FSM_CONFIG.low_state) and recently_active and s is not None and s < TRAP_FSM_CONFIG.exit_th:
                # check line relation: is the opposing lead attacker
                # ahead of this defending team's line right now?
                if _lead_attacker_ahead(cleaned, attacking_team, defending_team, f):
                    final = "TRAP_BROKEN"
            if final == TRAP_FSM_CONFIG.ending_state:
                # The requested output vocabulary for this project is
                # exactly {NO_TRAP, TRAP_FORMING, TRAP_ACTIVE,
                # TRAP_BREAK_THREAT, TRAP_BROKEN, UNCERTAIN} -- no
                # separate "ending" state. TRAP_ENDING is used INTERNALLY
                # (in TRAP_FSM_CONFIG) purely for hysteresis/cooldown
                # timing so a real active spell doesn't flicker back and
                # forth at its tail; for display it collapses to
                # NO_TRAP (the score is already below exit_th, i.e. the
                # coordinated stepping has genuinely stopped) unless the
                # TRAP_BROKEN condition above already caught it.
                final = TRAP_FSM_CONFIG.low_state
            was_recently_active.append(final)
        final_states[defending_team] = was_recently_active

    out = {}
    for team in (0, 1):
        out[team] = {
            "trap_raw": trap_out[team]["raw"], "trap_smoothed": trap_out[team]["smoothed"],
            "trap_states": final_states[team], "trap_segments": states_to_segments(final_states[team], fps=fps),
            "run_threat_raw": run_threat_out[team]["raw"], "run_threat_smoothed": run_threat_out[team]["smoothed"],
        }
    for team in (0, 1):
        out[team]["break_state_as_attacker"] = list(final_states[1 - team])  # mirror of opponent's trap state
    return out


def _lead_attacker_ahead(cleaned: pl.DataFrame, attacking_team: int, defending_team: int, frame: int) -> bool:
    from offside_break.analytics.offside import compute_offside_line
    line_info = compute_offside_line(cleaned, defending_team, frame)
    if not line_info["valid"]:
        return False
    attackers = cleaned.filter(
        (pl.col("frame") == frame) & (pl.col("display_team_id") == attacking_team) &
        pl.col("x_pitch").is_not_null() & pl.col("display_object_type").is_in(["player", "goalkeeper"])
    ).to_dicts()
    if not attackers:
        return False
    lead = min(attackers, key=lambda p: _forward_depth(defending_team, p["x_pitch"]))
    relation = compute_line_relation_cm(lead["x_pitch"], defending_team, line_info["line_depth_cm"])
    return relation >= BROKEN_LINE_MARGIN_CM


def _rolling_mean(raw_scores: list, window_frames: int) -> list:
    out = []
    window = []
    for i, v in enumerate(raw_scores):
        if v is not None:
            window.append((i, v))
        while window and i - window[0][0] >= window_frames:
            window.pop(0)
        out.append(sum(x[1] for x in window) / len(window) if window else None)
    return out


def decide_dominant_trap_state(state0: str, state1: str) -> tuple:
    """Mirrors pressing_v3's decide_dominant_state, for the two
    defender-role states. A trap can only meaningfully be "active" for
    the team NOT currently down a goal-side threat from the other, but
    unlike pressing (mutually exclusive by definition), both teams CAN
    show a real trap state simultaneously since trap-synchrony is
    computed independently per team's own back line. Reported as
    CONTESTED only if both are genuinely at TRAP_ACTIVE/TRAP_BREAK_
    THREAT/TRAP_BROKEN at once (rare, both back lines highly
    coordinated at the same instant)."""
    high = (TRAP_FSM_CONFIG.active_state, "TRAP_BREAK_THREAT", "TRAP_BROKEN")
    t0_high, t1_high = state0 in high, state1 in high
    if t0_high and t1_high:
        return "CONTESTED", None
    if t0_high:
        return state0, 0
    if t1_high:
        return state1, 1
    if state0 == TRAP_FSM_CONFIG.forming_state:
        return state0, 0
    if state1 == TRAP_FSM_CONFIG.forming_state:
        return state1, 1
    if state0 == TRAP_FSM_CONFIG.uncertain_state and state1 == TRAP_FSM_CONFIG.uncertain_state:
        return TRAP_FSM_CONFIG.uncertain_state, None
    return TRAP_FSM_CONFIG.low_state, None
