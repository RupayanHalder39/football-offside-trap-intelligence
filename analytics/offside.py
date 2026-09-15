"""
LEGACY API: geometry corrected in V4; use offside_v4 for quality-gated estimates.
Offside-trap-break analytics (Project 2) -- reusable, project-scoped to
`offside_break/` only. Builds on the SAME shared cleaned view as
Project 1 (`pressing_structure.analytics.cleaned_tracking_view` --
imported, not duplicated) and the EXISTING PhD `analytics.possession` /
`analytics.pitch_control` modules (unmodified).

See `offside_break/docs/METHODOLOGY.md` for exact definitions,
thresholds, and explicit scope limits (this NEVER claims an official
offside decision).

## Defending-end calibration (measured, not assumed)

Each team's own defending end was measured directly from its
goalkeeper's average pitch position across the whole clip (not
hardcoded): team 0's GK averages x=11168cm (defends the x=12000 end,
attacks toward x=0); team 1's GK averages x=1410cm (defends the x=0
end, attacks toward x=12000). See `offside_break/outputs/
offside_summary.md` for this measurement's own numbers.
"""
import os
import sys

import numpy as np
import polars as pl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analytics.pitch_control import compute_pitch_control_grid  # noqa: E402
from pressing_structure.analytics.pressing import LANE_BLOCK_DISTANCE_CM, _passing_lane_open  # noqa: E402

from tactical_shared.coordinates import DEFAULT_PITCH

PITCH_LENGTH_CM = DEFAULT_PITCH.length_cm
PITCH_WIDTH_CM = DEFAULT_PITCH.width_cm

# team_id -> own-goal x coordinate (measured from GK average position,
# NOT assumed) -- team's attacking direction is AWAY from this value.
DEFENDING_GOAL_X = {t: DEFAULT_PITCH.own_goal(t) for t in (0, 1)}  # legacy frame-0 alias

LINE_SMOOTH_WINDOW_FRAMES = 5      # rolling median, display stability only
CLOSING_SPEED_HIGH_CM_S = 200.0    # explicit "Break Potential" thresholds
SEPARATION_HIGH_CM = 300.0
CLOSING_SPEED_MED_CM_S = 80.0
SEPARATION_MED_CM = 150.0
STRAIGHT_ANGLE_DEG = 20.0
LATERAL_ANGLE_DEG = 70.0


def _forward_depth(team_id: int, x_pitch: float) -> float:
    """Distance from this team's OWN goal, i.e. how far forward
    (attacking-ward) a position is for that team. Always >=0 within the
    pitch."""
    return DEFAULT_PITCH.depth(team_id, x_pitch)


def _depth_to_x(team_id: int, depth: float) -> float:
    return DEFAULT_PITCH.x_from_depth(team_id, depth)


def compute_offside_line(cleaned: pl.DataFrame, defending_team: int, frame: int) -> dict:
    """Second-last-defender line (GK included in the depth ordering,
    per Law 11's own "second-last defender" wording -- if the GK is the
    deepest, the line is the deepest OUTFIELD defender, which is the
    common case; if the GK has stepped forward of another defender, the
    GK correctly is NOT automatically the reference). Returns {"valid",
    "reason", "line_x", "line_depth_cm", "defender_track_id",
    "n_defenders_visible"}."""
    rows = cleaned.filter(
        (pl.col("frame") == frame) & (pl.col("display_team_id") == defending_team) &
        pl.col("x_pitch").is_not_null() & pl.col("display_object_type").is_in(["player", "goalkeeper"])
    ).to_dicts()
    if len(rows) < 2:
        return {"valid": False, "reason": f"only {len(rows)} defenders visible (need >=2)",
                "line_x": None, "line_depth_cm": None, "defender_track_id": None, "n_defenders_visible": len(rows)}
    depths = sorted(((_forward_depth(defending_team, r["x_pitch"]), r) for r in rows),
                     key=lambda t: t[0])
    second_last_depth, second_last_row = depths[1]
    return {"valid": True, "reason": None, "line_x": second_last_row["x_pitch"],
            "line_depth_cm": second_last_depth, "defender_track_id": second_last_row["track_id"],
            "n_defenders_visible": len(rows)}


def offside_line_series(cleaned: pl.DataFrame, defending_team: int, frame_min: int, frame_max: int,
                         smooth_window: int = LINE_SMOOTH_WINDOW_FRAMES) -> pl.DataFrame:
    """Per-frame line depth, rolling-median-smoothed for display
    stability (never fabricates a line where <2 defenders are visible
    -- those frames stay null)."""
    rows = []
    for f in range(frame_min, frame_max + 1):
        r = compute_offside_line(cleaned, defending_team, f)
        rows.append({"frame": f, "valid": r["valid"], "line_depth_cm": r["line_depth_cm"],
                     "n_defenders_visible": r["n_defenders_visible"]})
    df = pl.DataFrame(rows)
    depths = df["line_depth_cm"].to_list()
    smoothed = []
    for i in range(len(depths)):
        window = [d for d in depths[max(0, i - smooth_window + 1):i + 1] if d is not None]
        smoothed.append(float(np.median(window)) if depths[i] is not None and window else None)
    return df.with_columns(pl.Series("line_depth_cm_smoothed", smoothed))


def classify_run_type(angle_deg: float | None) -> str:
    """Simple, disclosed, geometry-only classification. Only 3 classes
    are attempted (straight / diagonal / lateral) -- curved, check-run,
    third-man, inside-to-outside/outside-to-inside are EXPLICITLY NOT
    implemented in this first pass (would need multi-frame trajectory
    shape analysis this prototype does not yet build) and are never
    forced; see METHODOLOGY.md."""
    if angle_deg is None:
        return "unknown"
    a = abs(angle_deg)
    if a <= STRAIGHT_ANGLE_DEG:
        return "straight"
    if a >= LATERAL_ANGLE_DEG:
        return "lateral"
    return "diagonal"


def compute_break_potential(onside: bool, closing_speed_cm_s: float | None,
                             nearest_defender_distance_cm: float | None) -> str | None:
    """Explicit geometric thresholds only -- NEVER a fabricated
    probability. Returns None (omit) rather than guessing when required
    inputs are missing."""
    if closing_speed_cm_s is None or nearest_defender_distance_cm is None:
        return None
    if not onside:
        return "LOW"  # already ahead of the line -- would be an offside-position run, not a break
    if closing_speed_cm_s >= CLOSING_SPEED_HIGH_CM_S and nearest_defender_distance_cm >= SEPARATION_HIGH_CM:
        return "HIGH"
    if closing_speed_cm_s >= CLOSING_SPEED_MED_CM_S and nearest_defender_distance_cm >= SEPARATION_MED_CM:
        return "MEDIUM"
    return "LOW"


def compute_attacker_features(cleaned: pl.DataFrame, attacking_team: int, defending_team: int,
                               frame: int, ball_carrier_track_id: int | None,
                               line_info: dict) -> list[dict]:
    """One row per visible attacker at `frame`. Returns [] if the line
    itself is not valid this frame (never assessed against an
    unreliable line)."""
    if not line_info["valid"]:
        return []
    players = cleaned.filter(
        (pl.col("frame") == frame) & pl.col("x_pitch").is_not_null() &
        pl.col("display_object_type").is_in(["player", "goalkeeper"])
    ).to_dicts()
    attackers = [p for p in players if p["display_team_id"] == attacking_team]
    defenders = [p for p in players if p["display_team_id"] == defending_team]
    carrier = next((p for p in players if p["track_id"] == ball_carrier_track_id), None)

    out = []
    for a in attackers:
        # Both teams' "forward, toward the defending team's goal" is the
        # SAME shared x-axis direction -- express the attacker's depth
        # in that same frame as the line so the two are directly
        # comparable.
        a_depth_toward_def_goal = _forward_depth(defending_team, a["x_pitch"])
        distance_to_line_cm = line_info["line_depth_cm"] - a_depth_toward_def_goal
        onside = distance_to_line_cm <= 0  # behind (on their own side of) the line

        vx, vy = a.get("vx_cm_s"), a.get("vy_cm_s")
        angle_deg = None
        rel_velocity_to_line = None
        if vx is not None and vy is not None:
            # forward component = velocity pointing TOWARD the
            # defending team's goal -- a FIXED direction (that goal's
            # x-position never changes mid-clip), not dependent on the
            # attacker's own current position.
            direction = DEFAULT_PITCH.attacking_sign(attacking_team, frame)
            forward_v = vx * direction
            lateral_v = vy
            angle_deg = float(np.degrees(np.arctan2(lateral_v, forward_v))) if (forward_v != 0 or lateral_v != 0) else None
            rel_velocity_to_line = forward_v

        nearest_def_dist = None
        if defenders:
            nearest_def_dist = float(min(np.hypot(d["x_pitch"] - a["x_pitch"], d["y_pitch"] - a["y_pitch"])
                                          for d in defenders))

        lane_open = None
        if carrier is not None and carrier["track_id"] != a["track_id"]:
            lane_open = _passing_lane_open((carrier["x_pitch"], carrier["y_pitch"]),
                                            (a["x_pitch"], a["y_pitch"]), defenders)

        time_to_line_sec = None
        if onside and rel_velocity_to_line is not None and rel_velocity_to_line > 1e-6:
            time_to_line_sec = float((-distance_to_line_cm) / rel_velocity_to_line)

        break_potential = compute_break_potential(onside, rel_velocity_to_line, nearest_def_dist)
        run_type = classify_run_type(angle_deg)

        out.append({
            "frame": frame, "attacker_track_id": a["track_id"], "attacking_team_id": attacking_team,
            "x_pitch": a["x_pitch"], "y_pitch": a["y_pitch"],
            "distance_to_line_cm": distance_to_line_cm, "onside": onside,
            "relative_velocity_to_line_cm_s": rel_velocity_to_line, "run_angle_deg": angle_deg,
            "run_type": run_type, "nearest_defender_distance_cm": nearest_def_dist,
            "passing_lane_open_from_carrier": lane_open, "time_to_enter_behind_line_sec": time_to_line_sec,
            "break_potential": break_potential,
        })
    return out


OFFSIDE_FEATURE_SCHEMA = {
    "frame": pl.Int64, "attacker_track_id": pl.Int64, "attacking_team_id": pl.Int64,
    "x_pitch": pl.Float64, "y_pitch": pl.Float64,
    "distance_to_line_cm": pl.Float64, "onside": pl.Boolean,
    "relative_velocity_to_line_cm_s": pl.Float64, "run_angle_deg": pl.Float64,
    "run_type": pl.Utf8, "nearest_defender_distance_cm": pl.Float64,
    "passing_lane_open_from_carrier": pl.Boolean, "time_to_enter_behind_line_sec": pl.Float64,
    "break_potential": pl.Utf8,
}


def build_offside_features(cleaned: pl.DataFrame, poss: pl.DataFrame,
                            attacking_team: int, defending_team: int) -> pl.DataFrame:
    poss_by_frame = {r["frame"]: r for r in poss.to_dicts()}
    rows = []
    for frame in sorted(poss_by_frame.keys()):
        line_info = compute_offside_line(cleaned, defending_team, frame)
        carrier_id = poss_by_frame[frame]["possessing_track_id"] \
            if poss_by_frame[frame]["possession_state"] == "CONTROLLED" else None
        rows.extend(compute_attacker_features(cleaned, attacking_team, defending_team, frame, carrier_id, line_info))
    return pl.DataFrame(rows, schema=OFFSIDE_FEATURE_SCHEMA) if rows else pl.DataFrame(schema=OFFSIDE_FEATURE_SCHEMA)
