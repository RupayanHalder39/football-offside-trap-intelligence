"""
Offside-trap-break V2 analytics. NEW module -- `offside_break/analytics/
offside.py` (v1) is UNCHANGED and still used by the v1 dashboard
exactly as before. This file imports v1's measured constants and pure
geometry helpers (DEFENDING_GOAL_X, `_forward_depth`,
`compute_offside_line`) rather than redefining them.

Three things v1 conflated that V2 separates, per the brief:
1. **Defensive-line relation** (AHEAD/BEHIND the second-last defender)
   -- pure geometry, no legality claim. v1 already computed this
   correctly as `onside`; V2 only fixes the ON-SCREEN LABEL (v1's
   display code called it "offside position", which overstated it).
2. **Estimated offside POSITION** -- a closer (still NOT legal-grade)
   approximation using the actual Law 11 ingredients this dataset can
   support: ahead of the second-last defender AND ahead of the ball AND
   in the opponent's half. Requires a confident ball position; when the
   ball isn't available, this module returns "LINE_RELATION_ONLY" and
   refuses to guess.
3. **Offside-trap CANDIDATE** -- a defensive line existing is not a
   trap; a trap requires the defenders to be doing something
   coordinated (stepping up together). See `trap_synchrony_score`.

See `offside_break/docs/METHODOLOGY.md`'s V2 section for the exact,
disclosed thresholds and everything this does NOT claim.
"""
import os
import sys

import numpy as np
import polars as pl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from offside_break.analytics.offside import (DEFENDING_GOAL_X, PITCH_LENGTH_CM, PITCH_WIDTH_CM,  # noqa: E402
                                               _forward_depth, compute_offside_line)

HALFWAY_DEPTH_CM = PITCH_LENGTH_CM / 2.0  # 6000cm from either goal

# Measured on this clip's ball track (990/1642 frames have ANY pitch
# position; of those, confidence ranges 0.25-0.86, mean 0.63). 0.4 drops
# only the least-confident ~9% of available ball detections while
# keeping the estimated-offside-position logic usable most of the time
# the ball is tracked at all -- a disclosed, round threshold, not fit to
# any outcome.
BALL_CONFIDENCE_MIN_FOR_OFFSIDE = 0.4


def gated_ball_x(ball_row: dict | None) -> float | None:
    """Returns the ball's x_pitch only if it exists AND its confidence
    clears BALL_CONFIDENCE_MIN_FOR_OFFSIDE; otherwise None, forcing
    callers to fall back to LINE_RELATION_ONLY rather than guessing."""
    if ball_row is None:
        return None
    x = ball_row.get("x_pitch")
    conf = ball_row.get("confidence")
    if x is None or conf is None or conf < BALL_CONFIDENCE_MIN_FOR_OFFSIDE:
        return None
    return x

# --- Trap synchrony thresholds (simple, interpretable, NOT fit to data) ---
TRAP_MIN_DEFENDERS = 3
FORWARD_STEP_SPEED_CM_S = 50.0     # a defender counts as "stepping forward" above this
DEPTH_DISPERSION_NORM_CM = 500.0   # 5m -- dispersion normalizer for the alignment score
VELOCITY_ALIGNMENT_NORM_CM_S = 200.0  # 2 m/s -- velocity-std normalizer
TRAP_FORMING_THRESHOLD = 0.4
TRAP_ACTIVE_THRESHOLD = 0.65
TRAP_ACTIVE_MIN_SUSTAIN_FRAMES = 15  # 0.5s -- must sustain >=TRAP_ACTIVE_THRESHOLD, not just spike one frame

# --- Run detection thresholds ---
RUN_MIN_FORWARD_SPEED_CM_S = 150.0   # sustained forward speed to count as "running", not walking
RUN_MIN_DURATION_FRAMES = 15         # 0.5s minimum
RUN_MAX_LINE_DISTANCE_CM = 3000.0    # 30m -- must be plausibly approaching the defensive zone
RUN_BRIDGE_GAP_FRAMES = 5            # very short gap only, per the brief's "bridge only very short gaps"
RUN_MIN_FRAMES_FOR_CURVED_CHECK = 45  # 1.5s -- checked empirically: at 20 frames (0.67s), noisy per-frame
# velocity angle on short segments pushed ~half of all qualifying runs into "curved" even for runs as
# short as the 0.5s minimum: not a defensible signal at that length. 45 frames keeps the complex classes
# for genuinely sustained runs; shorter ones fall back to the simple straight/diagonal/lateral split.
CURVED_ANGLE_STD_DEG = 25.0
CHECK_RUN_BACK_SPEED_CM_S = -50.0
CHECK_RUN_FORWARD_SPEED_CM_S = 150.0
IN_OUT_MIN_LATERAL_SHIFT_CM = 400.0   # 4m of lateral (y) travel to call it in-to-out / out-to-in

# --- Break-candidate thresholds ---
BREAK_LINE_THREATEN_CM = 200.0        # "threatens/crosses the line" if distance_to_line >= -this
BREAK_DEFENDER_RECOVERY_CM = 200.0    # nearest-defender gap closing by this much = "recovers"
BREAK_PASS_MATCH_TOLERANCE_FRAMES = 30  # 1s tolerance matching a pass to a run window


def forward_sign_away_from_own_goal(team_id: int) -> float:
    """+1 if moving with increasing x counts as this team's OWN attacking
    direction (their own goal is at x=0), else -1. Used for the
    DEFENDING team's own forward push (stepping up the pitch), which is
    the SAME direction as their own attacking direction."""
    from tactical_shared.coordinates import DEFAULT_PITCH
    return DEFAULT_PITCH.attacking_sign(team_id)


# ============================================================================
# 1. Defensive-line relation (pure geometry, no legality claim)
# ============================================================================
def compute_line_relation_cm(attacker_x: float, defending_team: int, line_depth_cm: float) -> float:
    """Positive = ahead of the defensive line (on the goal side of the
    second-last defender); negative = behind it. Defined as
    `line_depth_cm - attacker_depth` (NOT `attacker_depth -
    line_depth_cm`, which is v1's convention in
    `offside.py::compute_attacker_features` and is inverted there --
    see the module docstring's stop-and-report note)."""
    attacker_depth = _forward_depth(defending_team, attacker_x)
    return line_depth_cm - attacker_depth


def line_relation_label(distance_to_line_cm: float) -> str:
    """Replaces v1's overstated 'AHEAD OF LINE (offside position)' /
    'ONSIDE (behind line)' text with a pure line-relation label. Takes
    a distance already computed by `compute_line_relation_cm` (positive
    = ahead of the line)."""
    return ("LEVEL WITH DEFENSIVE LINE" if abs(distance_to_line_cm) < 1e-6 else
            "AHEAD OF DEFENSIVE LINE" if distance_to_line_cm > 0 else "BEHIND DEFENSIVE LINE")


# ============================================================================
# 2. Estimated offside POSITION (line + ball + halfway line)
# ============================================================================
def compute_estimated_offside_position(attacker_x: float, defending_team: int,
                                        line_depth_cm: float, ball_x: float | None) -> dict:
    """Returns {"state": "ESTIMATED_OFFSIDE_POSITION" |
    "ESTIMATED_ONSIDE_POSITION" | "LINE_RELATION_ONLY",
    "ahead_of_line": bool, "ahead_of_ball": bool|None, "in_opponent_half": bool,
    "reason": str}. Requires a confident ball x-position (`ball_x` not
    None) for anything beyond LINE_RELATION_ONLY -- never guesses the
    ball's side.

    NOTE on the sign convention: `_forward_depth(defending_team, x)` is
    the distance of a pitch position from the DEFENDING team's own
    goal. A SMALLER value means CLOSER to that goal, i.e. further
    advanced from the attacker's point of view. So "ahead of the line"
    (the offside side) is `attacker_depth < line_depth_cm`, not `>` --
    this was verified against real tracking rows (a defender line at
    depth 6212.9cm / x=5787 vs. an attacker at depth 4396cm / x=7604,
    i.e. clearly on the goal side of the line, must read as ahead of
    the line). See the V2 stop-and-report note for the matching bug
    found in v1's `offside.py::compute_attacker_features`, which is
    left untouched."""
    attacker_depth = _forward_depth(defending_team, attacker_x)
    ahead_of_line = attacker_depth < line_depth_cm
    in_opponent_half = attacker_depth < HALFWAY_DEPTH_CM
    if ball_x is None:
        return {"state": "LINE_RELATION_ONLY", "ahead_of_line": ahead_of_line,
                "ahead_of_ball": None, "in_opponent_half": in_opponent_half,
                "reason": "ball position unavailable this frame"}
    ball_depth = _forward_depth(defending_team, ball_x)
    ahead_of_ball = attacker_depth < ball_depth
    is_offside_position = ahead_of_line and ahead_of_ball and in_opponent_half
    return {"state": "ESTIMATED_OFFSIDE_POSITION" if is_offside_position else "ESTIMATED_ONSIDE_POSITION",
            "ahead_of_line": ahead_of_line, "ahead_of_ball": ahead_of_ball,
            "in_opponent_half": in_opponent_half,
            "reason": "all 3 spatial conditions met" if is_offside_position else "not all 3 spatial conditions met"}


# ============================================================================
# 3. Offside-trap candidate detector
# ============================================================================
def compute_trap_synchrony(cleaned, defending_team, frame):
    """Deprecated scalar API; uses a defensive unit and stable-member motion."""
    from offside_break.analytics.offside_v4 import defensive_unit,unit_motion,trap_synchrony,line_geometry,eligible_players
    from tactical_shared.coordinates import DEFAULT_PITCH
    history=[]
    for f in range(max(0,frame-15),frame+1):
        ps=cleaned.filter(pl.col('frame')==f).to_dicts()
        # REGRESSION FIX (found during V4 continuation, not present in the
        # original repair): the legacy `cleaned` view (unmodified
        # cleaned_tracking_view.build_cleaned_view) has no `motion_valid`
        # column at all, so `setdefault('motion_valid', False)` forced
        # EVERY row to invalid motion regardless of real vx_cm_s/vy_cm_s
        # data -- silently making this legacy-API wrapper (and therefore
        # the still-live offside_v3/render_offside_dashboard_final_v3
        # pipeline that calls it) always return INSUFFICIENT_UNIT_MOTION
        # for every frame. Fall back to the same "velocity present ->
        # trustworthy" convention the rest of the legacy codebase already
        # uses, while still respecting an explicit motion_valid flag from
        # a real V4 quality view when one is actually supplied.
        for p in ps: p.setdefault('motion_valid', p.get('vx_cm_s') is not None)
        history.append(dict(frame=f,unit=defensive_unit(ps,defending_team,f)))
    velocity,_=unit_motion(history,defending_team,frame)
    line=line_geometry(ps,defending_team,frame)
    attackers=eligible_players(ps,1-defending_team)
    context=None if not line['valid'] or not attackers else any(abs(line['line_depth_cm']-DEFAULT_PITCH.depth(defending_team,p['x_pitch'],frame))<=2000 for p in attackers)
    return trap_synchrony(history[-1]['unit'],defending_team,velocity,context,frame)


def trap_state_series(cleaned: pl.DataFrame, defending_team: int, frame_min: int, frame_max: int) -> pl.DataFrame:
    """Per-frame trap state with the sustain rule for TRAP_ACTIVE
    (score alone crossing 0.65 for ONE frame is TRAP_FORMING, not
    ACTIVE, until it has held for >=15 consecutive frames)."""
    rows = []
    for f in range(frame_min, frame_max + 1):
        r = compute_trap_synchrony(cleaned, defending_team, f)
        rows.append({"frame": f, **r})
    df = pl.DataFrame(rows)

    states = []
    sustain = 0
    for r in df.to_dicts():
        if not r["valid"]:
            states.append("UNCERTAIN")
            sustain = 0
            continue
        s = r["score"]
        if s >= TRAP_ACTIVE_THRESHOLD:
            sustain += 1
            states.append("TRAP_ACTIVE" if sustain >= TRAP_ACTIVE_MIN_SUSTAIN_FRAMES else "TRAP_FORMING")
        elif s >= TRAP_FORMING_THRESHOLD:
            sustain = 0
            states.append("TRAP_FORMING")
        else:
            sustain = 0
            states.append("NO_TRAP")
    return df.with_columns(pl.Series("trap_state", states))


# ============================================================================
# 4. Temporal run detection (per attacker, within one cleaned segment)
# ============================================================================
def classify_run_type_v2(angles: list, fwd_speeds: list, start_y: float, end_y: float) -> str:
    """Retrospective geometric shape only; ordered phases, unwrapped angles.
    A full preparation/check-run requires a separately observed backward phase.
    """
    valid = [v for v in angles if v is not None]
    if not valid: return "unknown"
    unwrapped = np.degrees(np.unwrap(np.radians(valid)))
    if len(valid) >= RUN_MIN_FRAMES_FOR_CURVED_CHECK:
        # Require three supported backward frames followed by three forward.
        back_end = next((i for i in range(2,len(fwd_speeds)) if all(v <= CHECK_RUN_BACK_SPEED_CM_S for v in fwd_speeds[i-2:i+1])), None)
        if back_end is not None and any(all(v >= CHECK_RUN_FORWARD_SPEED_CM_S for v in fwd_speeds[i:i+3]) for i in range(back_end+1,len(fwd_speeds)-2)):
            return "check-run"
        if np.std(unwrapped) >= CURVED_ANGLE_STD_DEG: return "curved"
        if abs(end_y-start_y) >= IN_OUT_MIN_LATERAL_SHIFT_CM:
            center=PITCH_WIDTH_CM/2
            return "outside-to-inside" if abs(start_y-center)>abs(end_y-center) else "inside-to-outside"
    angle=float(np.mean(np.abs(((np.array(valid)+180)%360)-180)))
    return "straight" if angle <=20 else "lateral" if angle>=70 else "diagonal"


def detect_attacking_runs(cleaned: pl.DataFrame, attacking_team: int, defending_team: int,
                            line_series: pl.DataFrame) -> pl.DataFrame:
    """Run episodes per (track_id, cleaned_segment_id) -- NEVER bridges
    across a rejected-jump segment boundary (trajectory continuity
    requirement). A run candidate requires sustained forward speed
    (>=150cm/s toward the defending goal) for >=0.5s while within 30m
    of the (current) defensive line depth. Short gaps (<=5 frames) in
    the qualifying-speed condition are bridged; a segment boundary is
    NEVER bridged."""
    line_by_frame = {r["frame"]: r["line_depth_cm_smoothed"] for r in line_series.to_dicts()}
    # Attacker's "forward" = toward the DEFENDING team's goal. With only
    # two goals on the pitch, that is the same direction as "away from
    # the attacking team's own goal", so this is the same helper used
    # for defenders' forward push, applied to the attacking team.
    sign = forward_sign_away_from_own_goal(attacking_team)

    rows = cleaned.filter(
        (pl.col("display_team_id") == attacking_team) & pl.col("x_pitch").is_not_null() &
        pl.col("display_object_type").is_in(["player", "goalkeeper"])
    ).sort(["track_id", "cleaned_segment_id", "frame"])

    runs = []
    run_id = 0
    for (tid, seg_id), grp in rows.group_by(["track_id", "cleaned_segment_id"], maintain_order=True):
        grp = grp.sort("frame")
        recs = grp.to_dicts()
        qualifying = []
        for r in recs:
            line_depth = line_by_frame.get(r["frame"])
            if line_depth is None or r.get("vx_cm_s") is None:
                qualifying.append(False)
                continue
            fwd_speed = r["vx_cm_s"] * sign
            attacker_depth = _forward_depth(defending_team, r["x_pitch"])
            near_line = abs(attacker_depth - line_depth) <= RUN_MAX_LINE_DISTANCE_CM
            qualifying.append(fwd_speed >= RUN_MIN_FORWARD_SPEED_CM_S and near_line)

        i = 0
        n = len(recs)
        while i < n:
            if not qualifying[i]:
                i += 1
                continue
            start = i
            j = i
            while j < n:
                if j > start and recs[j]["frame"] - recs[j - 1]["frame"] > RUN_BRIDGE_GAP_FRAMES:
                    break
                if qualifying[j]:
                    j += 1
                    continue
                k = j
                bridged = False
                while k < n and recs[k]["frame"] - recs[j - 1]["frame"] <= RUN_BRIDGE_GAP_FRAMES:
                    if qualifying[k]:
                        bridged = True
                        break
                    k += 1
                if bridged:
                    j = k + 1
                    continue
                break
            end = j - 1
            seg = recs[start:end + 1]
            duration_frames = seg[-1]["frame"] - seg[0]["frame"] + 1
            if sum(qualifying[start:end + 1]) >= RUN_MIN_DURATION_FRAMES:
                angles = []
                fwd_speeds = []
                for r in seg:
                    vx, vy = r.get("vx_cm_s"), r.get("vy_cm_s")
                    if vx is not None:
                        fwd = vx * sign
                        fwd_speeds.append(fwd)
                        angles.append(float(np.degrees(np.arctan2(vy, fwd))) if (fwd != 0 or vy != 0) else None)
                    else:
                        angles.append(None)
                run_type = classify_run_type_v2(angles, fwd_speeds, seg[0]["y_pitch"], seg[-1]["y_pitch"])
                line_depth_start = line_by_frame.get(seg[0]["frame"])
                line_depth_end = line_by_frame.get(seg[-1]["frame"])
                # Positive = ahead of the line (see compute_line_relation_cm);
                # NOT v1's inverted attacker_depth-minus-line_depth convention.
                dist_start = (compute_line_relation_cm(seg[0]["x_pitch"], defending_team, line_depth_start)
                              if line_depth_start is not None else None)
                dist_end = (compute_line_relation_cm(seg[-1]["x_pitch"], defending_team, line_depth_end)
                            if line_depth_end is not None else None)
                runs.append({
                    "run_id": run_id, "attacker_track_id": tid, "cleaned_segment_id": seg_id,
                    "start_frame": seg[0]["frame"], "end_frame": seg[-1]["frame"],
                    "start_time_sec": seg[0]["frame"] / 30.0, "end_time_sec": seg[-1]["frame"] / 30.0,
                    "duration_sec": sum(qualifying[start:end + 1]) / 30.0, "run_type": run_type,
                    "start_x": seg[0]["x_pitch"], "start_y": seg[0]["y_pitch"],
                    "end_x": seg[-1]["x_pitch"], "end_y": seg[-1]["y_pitch"],
                    "distance_to_line_start_cm": dist_start, "distance_to_line_end_cm": dist_end,
                    "mean_forward_speed_cm_s": float(np.mean(fwd_speeds)) if fwd_speeds else None,
                })
                run_id += 1
            i = j
    return pl.DataFrame(runs) if runs else pl.DataFrame()


# ============================================================================
# 5. Trap-break candidate detector
# ============================================================================
def classify_trap_break_candidates(runs, trap_series, cleaned, ball, passes, attacking_team, defending_team):
    """Deprecated tabular API, corrected to V4 contemporaneous release assessment.
    Use build_offside_v4 for complete runner-linked, timestamped lifecycles."""
    from offside_break.analytics.offside_v4 import line_geometry, release_assessment
    if not runs.height: return runs
    players = {}
    for p in cleaned.to_dicts(): players.setdefault(p['frame'], []).append(p)
    balls={r['frame']:r for r in ball.to_dicts()}
    lines={defending_team:{f:line_geometry(ps,defending_team,f) for f,ps in players.items()}}
    active={r['frame'] for r in trap_series.to_dicts() if r['trap_state']=='TRAP_ACTIVE'}
    used=set();out=[]
    for r in runs.sort('start_frame').to_dicts():
        candidate=any(f in active for f in range(r['start_frame'],r['end_frame']+1)) and (r.get('distance_to_line_end_cm') is not None and r['distance_to_line_end_cm']>=-200)
        outcome=None
        if candidate:
            outcome='UNCERTAIN'
            matched=[p for p in passes.sort('start_frame').to_dicts() if p['team_id']==attacking_team and p['target_track_id']==r['attacker_track_id'] and r['start_frame']<=p['start_frame']<=r['end_frame'] and p['pass_id'] not in used]
            if matched:
                p=matched[0];used.add(p['pass_id'])
                outcome,_=release_assessment({**r,'attacking_team':attacking_team},p,players,balls,lines)
        out.append({**r,'is_trap_break_candidate':candidate,'break_outcome':outcome})
    return pl.DataFrame(out, infer_schema_length=None)


def _nearest_defender_dist(cleaned: pl.DataFrame, attacker_track_id: int, frame: int, defending_team: int) -> float | None:
    a = cleaned.filter((pl.col("frame") == frame) & (pl.col("track_id") == attacker_track_id)).to_dicts()
    if not a or a[0]["x_pitch"] is None:
        return None
    defs = cleaned.filter((pl.col("frame") == frame) & (pl.col("display_team_id") == defending_team) &
                            pl.col("x_pitch").is_not_null()).to_dicts()
    if not defs:
        return None
    return float(min(np.hypot(d["x_pitch"] - a[0]["x_pitch"], d["y_pitch"] - a[0]["y_pitch"]) for d in defs))
