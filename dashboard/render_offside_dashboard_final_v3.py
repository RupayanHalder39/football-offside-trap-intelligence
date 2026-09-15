"""
Offside/trap-break dashboard -- SECOND-generation "final" layout, now
with broadcast-style polish (icon KPI row, team-comparison table/bars,
data-derived insights) AND dynamic team assignment (no team is
hardcoded as "the defending team" for the whole clip). Same V2
analytics, UNCHANGED: `offside_line_series`, `trap_state_series`,
`detect_attacking_runs`, `classify_trap_break_candidates`,
`compute_estimated_offside_position`, `compute_line_relation_cm`,
`line_relation_label` are imported read-only. V2 terminology preserved
exactly: AHEAD/BEHIND DEFENSIVE LINE, ESTIMATED_OFFSIDE_POSITION/
ESTIMATED_ONSIDE_POSITION/LINE_RELATION_ONLY, NO_TRAP/TRAP_FORMING/
TRAP_ACTIVE/UNCERTAIN.

Team-aware, not hardcoded: BOTH directions (team1-attacks-team0 and
team0-attacks-team1) are precomputed in full -- the LIVE panels then
pick, per frame, whichever direction matches the team currently OUT OF
possession (the same "presser team" convention pressing.py already
uses: defending team = 1 - ball_carrier_team_id). When possession is
UNCERTAIN this frame, the dashboard shows "Uncertain" and does not
guess which team is defending. Whole-clip AGGREGATES (KPI cards, the
team-comparison table, the insights panel) do NOT need this per-frame
switch -- "team X's defensive shape against team Y's attack" is a
well-defined fixed-direction computation on its own, so those reuse
the same two fixed-direction precomputes used throughout this
project's V2/graphs work.

Usage (preview -- ONE frame, no video written):
    external/sports/.venv/bin/python offside_break/dashboard/render_offside_dashboard_final_v2.py \\
        --preview_frame_sec 30.0 --out_path /tmp/offside_preview.png

Usage (full render, only after preview approval):
    external/sports/.venv/bin/python offside_break/dashboard/render_offside_dashboard_final_v2.py \\
        --out_path offside_break/outputs/final_dashboard_v2/offside_dashboard_120s_final_v2.mp4
"""
import argparse
import os
import pickle
import sys
from collections import deque

import cv2
import numpy as np
import polars as pl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from sports.annotators.soccer import draw_pitch  # noqa: E402
from sports.configs.soccer import SoccerPitchConfiguration  # noqa: E402

from analytics.pitch_control import compute_pitch_control_grid  # noqa: E402
from offside_break.analytics.offside import _forward_depth, compute_offside_line, offside_line_series  # noqa: E402
from offside_break.analytics.offside_v2 import (compute_estimated_offside_position, compute_line_relation_cm,  # noqa: E402
                                                  classify_trap_break_candidates, detect_attacking_runs,
                                                  forward_sign_away_from_own_goal, gated_ball_x,
                                                  line_relation_label, trap_state_series)
from offside_break.analytics.offside_v3 import (TRAP_FSM_CONFIG, build_dual_role_states,  # noqa: E402
                                                  decide_dominant_trap_state)
from pressing_structure.analytics.cleaned_tracking_view import build_cleaned_view, load_ball_view  # noqa: E402
from pressing_structure.analytics.pressing import _passing_lane_open, build_possession  # noqa: E402
from pressing_structure.dashboard.dashboard_style_v2 import (ACCENT_CYAN, ACCENT_GREEN, ACCENT_MAGENTA,  # noqa: E402
                                                               ACCENT_ORANGE, ACCENT_RED, BG, BORDER_SOFT, FONT,
                                                               TEXT_DIM, apply_color_wash_preserve_lines,
                                                               composite_state_badge, draw_big_dual_graph,
                                                               draw_big_graph, draw_hatched_zone, draw_header,
                                                               draw_icon_kpi_card, draw_insights_panel,
                                                               draw_radar_legend_box, draw_timeline_bucketed,
                                                               panel_frame, panel_title)
from pressing_structure.dashboard.live_graphs import compute_stable_y_range  # noqa: E402
from pressing_structure.dashboard.render_pressing_dashboard import _nearest_transformer  # noqa: E402

CONFIG = SoccerPitchConfiguration()
TEAM_COLOR = {0: ACCENT_CYAN, 1: ACCENT_MAGENTA}   # FIXED per-team colors (never flip with role)
LINE_COLOR = ACCENT_RED
LEAD_COLOR = (0, 215, 255)
RUN_COLOR = ACCENT_GREEN
# V3 Goal 3: defensive/trap-line overlays get a per-team color so BOTH
# teams' lines can be shown simultaneously and stay visually
# distinguishable -- Team 0 = bright red (unchanged from V2's single
# LINE_COLOR), Team 1 = orange-yellow (a clearly different strong hue,
# not reused elsewhere as a state color).
TEAM_LINE_COLOR = {0: ACCENT_RED, 1: (0, 200, 255)}
TRAP_STATE_COLOR = {"NO_TRAP": (150, 150, 150), "TRAP_FORMING": ACCENT_ORANGE,
                     "TRAP_ACTIVE": ACCENT_RED, "UNCERTAIN": (110, 110, 110),
                     # New V3 overlay states (offside_v3.build_dual_role_states) --
                     # kept visually distinct from the base 4 states above.
                     "TRAP_BREAK_THREAT": (0, 140, 255), "TRAP_BROKEN": (40, 40, 220),
                     "CONTESTED": (200, 60, 220)}
GRAPH_WINDOW_SEC = 8.0
LINE_VEL_BASELINE_FRAMES = 15  # 0.5s -- see offside_v2_summary.md's "relative speed" bug note
BUCKET_SEC = 2.0

# Geometry locked to the reference's spatial distribution: header ~5%,
# top row ~50%, lower analytics ~45% -- matches the pressing dashboard's
# budget (same canvas, same proportions), but the LOWER area uses a
# 4-column layout (KPI 2x2 | 2 stacked graphs | 2 stacked panels | 2
# stacked graphs) instead of pressing's stacked-row layout, per the
# offside reference's own structure.
HEADER_H = 60           # 5.0% of 1204
TOP_ROW_H = 602          # 50.0% of 1204
CONTENT_H = 470          # the 4-column area (excludes the timeline below it)
TIMELINE_H = 60
GAP = 6
CANVAS_W = 2304
LEFT_W = round(CANVAS_W * 0.52)
RIGHT_W = CANVAS_W - LEFT_W
KPI_COL_W = 460
GRAPH_COL_W = (CANVAS_W - KPI_COL_W - 4 * GAP) // 3
PANEL_H = (CONTENT_H - GAP) // 2  # each of the 2 stacked panels per column
assert HEADER_H + TOP_ROW_H + GAP + CONTENT_H + GAP + TIMELINE_H == 1204


def _pitch_xy_to_px(x, y, scale=0.1, padding=50):
    return int(x * scale) + padding, int(y * scale) + padding


def _draw_pitch_line_on_image(frame, transformer, line_x, color, thickness=3):
    if transformer is None:
        return frame
    try:
        inv_m = np.linalg.inv(transformer.m)
    except np.linalg.LinAlgError:
        return frame
    pts = np.array([(line_x, 0.0), (line_x, 7000.0)], dtype=np.float32).reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(pts, inv_m).reshape(-1, 2).astype(np.int32)
    cv2.line(frame, tuple(img_pts[0]), tuple(img_pts[1]), color, thickness, cv2.LINE_AA)
    return frame


def compute_direction_data(cleaned, ball, passes, n_total, attacking_team, defending_team):
    line_series = offside_line_series(cleaned, defending_team, 0, n_total - 1)
    trap_series = trap_state_series(cleaned, defending_team, 0, n_total - 1)
    runs = detect_attacking_runs(cleaned, attacking_team, defending_team, line_series)
    candidates = classify_trap_break_candidates(runs, trap_series, cleaned, ball, passes, attacking_team, defending_team)
    return {
        "line_by_frame": {r["frame"]: r["line_depth_cm_smoothed"] for r in line_series.to_dicts()},
        "trap_by_frame": {r["frame"]: r for r in trap_series.to_dicts()},
        "runs_list": runs.to_dicts() if runs.height else [],
        "runs_df": runs,  # NEW (v3): needed by offside_v3.build_dual_role_states's run-threat computation
        "candidates_list": candidates.filter(pl.col("is_trap_break_candidate")).to_dicts() if candidates.height else [],
    }


def compute_team_aggregates(dir0, dir1):
    """Whole-clip, fixed-direction aggregates per team-as-defender --
    real numbers, no possession-weighting needed (see module docstring
    for why fixed-direction is the right unit for this)."""
    agg = {}
    for team, d in ((0, dir0), (1, dir1)):
        scores = [r["score"] for r in d["trap_by_frame"].values() if r.get("valid")]
        states = [r["trap_state"] for r in d["trap_by_frame"].values()]
        n_active = sum(1 for s in states if s == "TRAP_ACTIVE")
        n_forming = sum(1 for s in states if s == "TRAP_FORMING")
        agg[team] = {
            "n_runs_against": len(d["runs_list"]),
            "n_candidates": len(d["candidates_list"]),
            "mean_trap_score": float(np.mean(scores)) if scores else None,
            "max_trap_score": float(np.max(scores)) if scores else None,
            "pct_frames_forming": (n_forming / len(states) * 100.0) if states else None,
            "pct_frames_active": (n_active / len(states) * 100.0) if states else None,
        }
    return agg


def draw_left_feed(frame, tracking_rows, attacking_team, defending_team, line_x0, line_x1, transformer,
                    lead_track_id, run_track_ids, banner_text, banner_color, role_text):
    vis = frame.copy()
    for r in tracking_rows:
        if r["object_type"] not in ("player", "goalkeeper"):
            continue
        color = TEAM_COLOR.get(r["team_id"], (180, 180, 180))
        thickness, label = 2, None
        if r["track_id"] in run_track_ids:
            thickness = 3
        if attacking_team is not None and r["team_id"] == attacking_team and r["track_id"] == lead_track_id:
            color, thickness, label = LEAD_COLOR, 3, "LEAD ATTACKER"
        x1, y1, x2, y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if label:
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.42, 2)
            cv2.rectangle(vis, (x1, y1 - th - 12), (x1 + tw + 8, y1 - 4), (0, 0, 0), -1)
            cv2.putText(vis, label, (x1 + 4, y1 - 8), FONT, 0.42, color, 2, cv2.LINE_AA)
    # V3 Goal 1/3: BOTH teams' defensive lines drawn simultaneously
    # (team 0 as defender's line AND team 1 as defender's line), each in
    # its own color -- not gated behind whichever team currently has
    # the ball, since a real defensive shape/line exists for both teams
    # independent of momentary possession.
    if line_x0 is not None:
        vis = _draw_pitch_line_on_image(vis, transformer, line_x0, TEAM_LINE_COLOR[0], 3)
    if line_x1 is not None:
        vis = _draw_pitch_line_on_image(vis, transformer, line_x1, TEAM_LINE_COLOR[1], 3)

    # compact, rounded, translucent state badge -- real per-frame state only
    vis = composite_state_badge(vis, 10, 10, banner_text, banner_color, role_text)

    # title bar above the feed, matching the radar panel's title-bar
    # treatment (the locked reference gives both top-row panels a
    # matching dark title strip)
    out = np.full((vis.shape[0] + 32, vis.shape[1], 3), BG, dtype=np.uint8)
    out[32:, :] = vis
    panel_title(out, "TACTICAL MATCH FEED", sub="defensive line  •  lead attacker  •  trap state", accent=LINE_COLOR)
    return out


def draw_radar(players_this_frame, ball_xy, line_x0, line_x1, state0, state1, w, h):
    """V3: both teams' defensive lines, dangerous-space zones, and
    attacking-run arrows drawn SIMULTANEOUSLY (Goal 1/3) -- no longer
    gated behind a single per-frame "the defending team" possession
    switch, since each team's own back-line shape is a real, always-
    computable fact independent of who currently has the ball."""
    team_a = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0} for p in players_this_frame if p["display_team_id"] == 0]
    team_b = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0} for p in players_this_frame if p["display_team_id"] == 1]
    pc = compute_pitch_control_grid(team_a, team_b) if (team_a and team_b) else {"valid": False}

    pitch = draw_pitch(config=CONFIG)
    pitch_dark = cv2.addWeighted(pitch, 0.7, np.zeros_like(pitch), 0.3, 0)
    if pc.get("valid"):
        ph, pw = pitch.shape[:2]
        up = cv2.resize(pc["grid"].astype(np.float32), (pw, ph), interpolation=cv2.INTER_CUBIC)
        up = np.clip(up, 0, 1)[..., None]
        color_field = (up * np.array(TEAM_COLOR[0]) + (1 - up) * np.array(TEAM_COLOR[1])).astype(np.uint8)
        pitch_dark = apply_color_wash_preserve_lines(pitch_dark, color_field, alpha=0.88)

    # Dangerous-space hatched zones for BOTH teams-as-defender, drawn
    # BEFORE the solid line strokes so the lines stay crisp on top.
    for defending_team, line_x in ((0, line_x0), (1, line_x1)):
        if line_x is None:
            continue
        goal_x = 12000.0 if defending_team == 0 else 0.0
        x0, x1 = sorted([line_x, goal_x])
        p0, p1 = _pitch_xy_to_px(x0, 0), _pitch_xy_to_px(x1, 7000)
        pitch_dark = draw_hatched_zone(pitch_dark, p0, p1, TEAM_LINE_COLOR[defending_team])
    for defending_team, line_x in ((0, line_x0), (1, line_x1)):
        if line_x is None:
            continue
        lp0, lp1 = _pitch_xy_to_px(line_x, 0), _pitch_xy_to_px(line_x, 7000)
        cv2.line(pitch_dark, lp0, lp1, TEAM_LINE_COLOR[defending_team], 3, cv2.LINE_AA)

    for p in players_this_frame:
        px, py = _pitch_xy_to_px(p["x_pitch"], p["y_pitch"])
        color = TEAM_COLOR.get(p["display_team_id"], (150, 150, 150))
        cv2.circle(pitch_dark, (px, py), 8, color, -1, cv2.LINE_AA)
        cv2.circle(pitch_dark, (px, py), 8, (10, 10, 10), 2, cv2.LINE_AA)
        # Attacking-run velocity arrows, colored by the RUNNING team
        # (Goal 3: Team 0 = cyan, Team 1 = magenta -- same as each
        # team's own base color, since these arrows belong to whichever
        # team is attacking at that instant, and both teams can be
        # attacking-in-transition/showing forward runs simultaneously).
        if p.get("vx_cm_s") is not None and p["display_team_id"] is not None:
            ex, ey = int(px + p["vx_cm_s"] * 0.03), int(py + p["vy_cm_s"] * 0.03)
            arrow_color = TEAM_COLOR[p["display_team_id"]]
            cv2.arrowedLine(pitch_dark, (px, py), (ex, ey), arrow_color, 2, cv2.LINE_AA, tipLength=0.3)
    if ball_xy is not None:
        bx, by = _pitch_xy_to_px(*ball_xy)
        cv2.circle(pitch_dark, (bx, by), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(pitch_dark, (bx, by), 6, (15, 15, 15), 1, cv2.LINE_AA)

    out = np.full((h, w, 3), BG, dtype=np.uint8)
    resized = cv2.resize(pitch_dark, (w, h - 34))
    out[34:h, 0:w] = resized
    panel_title(out, "OFFSIDE RADAR", sub="pitch control  •  both teams' lines  •  attacking runs", accent=LINE_COLOR)
    # Goal 3: small explicit "Team 0 trap/defensive state / Team 1
    # trap/defensive state" text, stacked below the title's divider line
    # (y=30) so it never collides with the title text itself. A small
    # dark backing keeps it legible over the busy pitch underneath.
    roi = out[34:70, 8:210]
    cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, dst=roi)
    cv2.putText(out, f"Team 0 trap state: {state0.replace('_', ' ').title()}", (14, 47), FONT, 0.34, TEAM_LINE_COLOR[0], 1, cv2.LINE_AA)
    cv2.putText(out, f"Team 1 trap state: {state1.replace('_', ' ').title()}", (14, 65), FONT, 0.34, TEAM_LINE_COLOR[1], 1, cv2.LINE_AA)
    draw_radar_legend_box(out, w - 172, 44, 160, 210,
                           [("Team 0 / Run", TEAM_COLOR[0]), ("Team 1 / Run", TEAM_COLOR[1]),
                            ("Ball", (255, 255, 255)), ("T0 Line", TEAM_LINE_COLOR[0]),
                            ("T1 Line", TEAM_LINE_COLOR[1])],
                           gradient=(TEAM_COLOR[0], TEAM_COLOR[1]),
                           gradient_labels=("Team 0", "Control", "Team 1"))
    return out


def draw_tiles_2x2(state0, state1, dominant_state, dominant_team, dual_trap_stats, defending_team, width, height):
    """V3 (Goal 4): dual-team KPI block -- Team0/1 current trap state
    (each packing that team's own runs-faced/mean-score into its sub-
    line so both teams' full metric set fits the reference's 2x2 grid),
    the possession-derived 'defending team this frame' fact, and the
    dominant/display trap-state call for this frame."""
    cell_w = (width - 3 * GAP) // 2
    cell_h = (height - 3 * GAP) // 2
    img = np.full((height, width, 3), BG, dtype=np.uint8)

    def _team_sub(team):
        st = dual_trap_stats[team]
        mean_txt = f"{st['mean_score']:.2f}" if st["mean_score"] is not None else "n/a"
        return f"{st['n_runs_faced']} runs faced, mean {mean_txt}"

    values = [
        ("Team 0 (Defender)", state0.replace("_", " ").title(), _team_sub(0), TRAP_STATE_COLOR.get(state0, ACCENT_ORANGE), "shield"),
        ("Team 1 (Defender)", state1.replace("_", " ").title(), _team_sub(1), TRAP_STATE_COLOR.get(state1, ACCENT_MAGENTA), "shield"),
        ("Defending This Frame", f"Team {defending_team}" if defending_team is not None else "Uncertain", "rolling possession majority (4s)", ACCENT_CYAN, "ruler"),
        ("Dominant Trap State", dominant_state.replace("_", " ").title(), f"Team {dominant_team}" if dominant_team is not None else "n/a", TRAP_STATE_COLOR.get(dominant_state, ACCENT_ORANGE), "bolt"),
    ]
    positions = [(GAP, GAP), (GAP * 2 + cell_w, GAP), (GAP, GAP * 2 + cell_h), (GAP * 2 + cell_w, GAP * 2 + cell_h)]
    for (label, val, sub, accent, glyph), (px, py) in zip(values, positions):
        tile = draw_icon_kpi_card(cell_w, cell_h, label, val, sub, accent=accent, glyph=glyph)
        img[py:py + cell_h, px:px + cell_w] = tile
    return img


def draw_trap_state_distribution(width, height, dual):
    """Real, whole-clip trap-state frame-fraction distribution (both
    teams-as-defender combined) -- rebuilt in V3 from the stable,
    transition-aware FSM states (`dual[team]["trap_states"]`) rather
    than V2's ad-hoc trap_state_series, and extended with the two new
    V3 overlay states (TRAP_BREAK_THREAT/TRAP_BROKEN)."""
    from collections import Counter
    counts = Counter()
    for team in (0, 1):
        counts.update(dual[team]["trap_states"])
    total = sum(counts.values()) or 1
    img = panel_frame(width, height)
    panel_title(img, "Trap State Distribution", sub="both teams, V3 stable states", accent=ACCENT_ORANGE)
    pad_l, pad_r, pad_t, pad_b = 16, 16, 42, 30
    x0, y0 = pad_l, pad_t
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    states = ["NO_TRAP", "TRAP_FORMING", "TRAP_ACTIVE", "TRAP_BREAK_THREAT", "TRAP_BROKEN", "UNCERTAIN"]
    bar_w = pw // len(states)
    max_pct = max((counts.get(s, 0) / total for s in states), default=1.0) or 1.0
    for i, s in enumerate(states):
        pct = counts.get(s, 0) / total
        bh_px = int((pct / max_pct) * ph)
        bx0 = x0 + i * bar_w + 8
        bx1 = x0 + (i + 1) * bar_w - 8
        cv2.rectangle(img, (bx0, y0 + ph - bh_px), (bx1, y0 + ph), TRAP_STATE_COLOR.get(s, (150, 150, 150)), -1)
        cv2.putText(img, f"{pct*100:.1f}%", (bx0, y0 + ph - bh_px - 6), FONT, 0.32, TEXT_DIM, 1, cv2.LINE_AA)
        label = s.replace("_", " ").title()
        cv2.putText(img, label, (bx0, y0 + ph + 18), FONT, 0.28, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.line(img, (x0, y0 + ph), (x0 + pw, y0 + ph), BORDER_SOFT, 1)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_video_path", type=str, default="ExternalDownlaodVideo/testVideo1_120s.mp4")
    parser.add_argument("--tracking_dir", type=str, default="outputs/tracking/testVideo1_120s")
    parser.add_argument("--analytics_dir", type=str, default="outputs/analytics/testVideo1_120s_v3")
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--end_sec", type=float, default=None)
    parser.add_argument("--preview_frame_sec", type=float, default=None,
                         help="If set, render exactly ONE frame at this timestamp to a PNG and exit.")
    args = parser.parse_args()

    tracking = pl.read_parquet(os.path.join(args.tracking_dir, "tracking.parquet"))
    cleaned = build_cleaned_view(tracking)
    ball = load_ball_view(args.analytics_dir)
    passes = pl.read_parquet(os.path.join(args.analytics_dir, "passes.parquet"))
    poss = build_possession(cleaned, ball)
    poss_by_frame = {r["frame"]: r for r in poss.to_dicts()}

    cap = cv2.VideoCapture(args.source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    span_sec = n_total / fps

    print("Computing BOTH directions (team-aware, not hardcoded)...", flush=True)
    dir_by_defender = {
        0: compute_direction_data(cleaned, ball, passes, n_total, attacking_team=1, defending_team=0),
        1: compute_direction_data(cleaned, ball, passes, n_total, attacking_team=0, defending_team=1),
    }
    print(f"  defender=0: {len(dir_by_defender[0]['runs_list'])} runs, {len(dir_by_defender[0]['candidates_list'])} candidates", flush=True)
    print(f"  defender=1: {len(dir_by_defender[1]['runs_list'])} runs, {len(dir_by_defender[1]['candidates_list'])} candidates", flush=True)
    agg = compute_team_aggregates(dir_by_defender[0], dir_by_defender[1])

    print("Computing V3 dual-role trap/break states (both teams simultaneously, transition-aware FSM)...", flush=True)
    runs_by_defender = {0: dir_by_defender[0]["runs_df"], 1: dir_by_defender[1]["runs_df"]}
    dual = build_dual_role_states(cleaned, ball, passes, runs_by_defender, n_total, fps=fps)
    dominant_trap_state_by_frame = [None] * n_total
    dominant_trap_team_by_frame = [None] * n_total
    for f in range(n_total):
        d, team = decide_dominant_trap_state(dual[0]["trap_states"][f], dual[1]["trap_states"][f])
        dominant_trap_state_by_frame[f] = d
        dominant_trap_team_by_frame[f] = team
    dual_trap_stats = {}
    for team in (0, 1):
        n_active_eps = sum(1 for s in dual[team]["trap_segments"] if s["state"] == TRAP_FSM_CONFIG.active_state)
        scores = [v for v in dual[team]["trap_smoothed"] if v is not None]
        dual_trap_stats[team] = {
            "n_active_episodes": n_active_eps,
            "mean_score": float(np.mean(scores)) if scores else None,
            "n_runs_faced": len(dir_by_defender[team]["runs_list"]),
        }
    n_break_threat = sum(1 for team in (0, 1) for s in dual[team]["trap_states"] if s == "TRAP_BREAK_THREAT")
    n_broken = sum(1 for team in (0, 1) for s in dual[team]["trap_states"] if s == "TRAP_BROKEN")
    print(f"  Team0 as defender: {dual_trap_stats[0]['n_active_episodes']} TRAP_ACTIVE episodes, "
          f"mean score {dual_trap_stats[0]['mean_score']:.3f}" if dual_trap_stats[0]['mean_score'] is not None else
          f"  Team0 as defender: {dual_trap_stats[0]['n_active_episodes']} TRAP_ACTIVE episodes", flush=True)
    print(f"  Team1 as defender: {dual_trap_stats[1]['n_active_episodes']} TRAP_ACTIVE episodes, "
          f"mean score {dual_trap_stats[1]['mean_score']:.3f}" if dual_trap_stats[1]['mean_score'] is not None else
          f"  Team1 as defender: {dual_trap_stats[1]['n_active_episodes']} TRAP_ACTIVE episodes", flush=True)
    hist_trap0_v3 = [(f / fps, dual[0]["trap_smoothed"][f]) for f in range(n_total)]
    hist_trap1_v3 = [(f / fps, dual[1]["trap_smoothed"][f]) for f in range(n_total)]
    y_trap_v3 = (0.0, 1.0)

    ball_by_frame = {r["frame"]: r for r in ball.to_dicts()}
    tracking_by_frame = {}
    for r in tracking.to_dicts():
        tracking_by_frame.setdefault(r["frame"], []).append(r)
    cleaned_by_frame = {}
    for r in cleaned.to_dicts():
        cleaned_by_frame.setdefault(r["frame"], []).append(r)
    run_frame_index = {0: {}, 1: {}}
    cand_frame_index = {0: {}, 1: {}}
    for team in (0, 1):
        for run in dir_by_defender[team]["runs_list"]:
            for fr in range(run["start_frame"], run["end_frame"] + 1):
                run_frame_index[team].setdefault(fr, []).append(run)
        for c in dir_by_defender[team]["candidates_list"]:
            for fr in range(c["start_frame"], c["end_frame"] + 1):
                cand_frame_index[team][fr] = c

    with open(os.path.join(args.analytics_dir, "homography_transformers.pkl"), "rb") as fh:
        transformers = pickle.load(fh)

    # Per-frame defending-team decision: 1 - ball_carrier_team_id,
    # resolved from a trailing ROLLING-MAJORITY window of confident-
    # carrier frames (not just the exact instantaneous frame). Ball
    # possession confidence is sparse on this dataset (previously
    # measured at ~7% of frames) -- requiring instantaneous certainty
    # would make the whole dashboard flicker to "Uncertain" on ~93% of
    # frames. A trailing window is still genuinely data-driven and
    # team-aware (never a hardcoded direction), just smoothed over a
    # few real seconds of recent evidence; it never looks into the
    # future. Only marked Uncertain when the window contains NO
    # confident carrier at all.
    ROLLING_WINDOW_SEC = 4.0
    carrier_team_by_frame = [None] * n_total
    for f in range(n_total):
        pr = poss_by_frame.get(f)
        if pr is not None and pr["possession_state"] == "CONTROLLED":
            row = next((p for p in cleaned_by_frame.get(f, []) if p["track_id"] == pr["possessing_track_id"]), None)
            if row is not None and row.get("display_team_id") is not None:
                carrier_team_by_frame[f] = row["display_team_id"]
    window_frames = int(ROLLING_WINDOW_SEC * fps)
    defending_team_by_frame = {}
    recent = deque()
    counts = {0: 0, 1: 0}
    for f in range(n_total):
        if carrier_team_by_frame[f] is not None:
            recent.append(f)
            counts[carrier_team_by_frame[f]] += 1
        while recent and f - recent[0] > window_frames:
            old = recent.popleft()
            counts[carrier_team_by_frame[old]] -= 1
        if counts[0] == 0 and counts[1] == 0:
            defending_team_by_frame[f] = None
        else:
            majority_carrier = 0 if counts[0] >= counts[1] else 1
            defending_team_by_frame[f] = 1 - majority_carrier

    print("Precomputing live-graph histories (team-aware per frame)...", flush=True)
    sign_cache = {0: forward_sign_away_from_own_goal(1), 1: forward_sign_away_from_own_goal(0)}
    hist_lead_depth, hist_line_depth, hist_trap_score, hist_closing_rate, hist_relation = [], [], [], [], []
    # (frame, defending_team, line_depth) -- kept PER-TEAM so a velocity
    # baseline never spans a frame where the defending team (hence which
    # goal "depth" is measured from) flipped; comparing depths measured
    # from two different goals would be physically meaningless.
    line_depth_history = []
    last_defending_team = None
    for f in range(n_total):
        t = f / fps
        defending_team = defending_team_by_frame[f]
        # A "depth" is only comparable across frames measured from the
        # SAME goal. When the inferred defending team flips, force a
        # visible break in every depth-based graph (a None sample at
        # this same timestamp closes the in-progress line segment)
        # rather than drawing a physically meaningless jump between two
        # different goals' reference frames.
        if defending_team is not None and last_defending_team is not None and defending_team != last_defending_team:
            for hist in (hist_line_depth, hist_lead_depth, hist_relation, hist_closing_rate):
                hist.append((t, None))
        last_defending_team = defending_team
        if defending_team is None:
            hist_line_depth.append((t, None)); hist_lead_depth.append((t, None))
            hist_relation.append((t, None)); hist_trap_score.append((t, None)); hist_closing_rate.append((t, None))
            line_depth_history.append((f, None, None))
            continue
        attacking_team = 1 - defending_team
        d = dir_by_defender[defending_team]
        line_depth = d["line_by_frame"].get(f)
        hist_line_depth.append((t, (line_depth / 100.0) if line_depth is not None else None))
        players = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None and p["display_team_id"] == attacking_team
                   and p["display_object_type"] in ("player", "goalkeeper")]
        lead = min(players, key=lambda p: _forward_depth(defending_team, p["x_pitch"])) if players else None
        lead_depth = _forward_depth(defending_team, lead["x_pitch"]) if lead is not None else None
        hist_lead_depth.append((t, (lead_depth / 100.0) if lead_depth is not None else None))
        relation = compute_line_relation_cm(lead["x_pitch"], defending_team, line_depth) if (lead is not None and line_depth is not None) else None
        hist_relation.append((t, (relation / 100.0) if relation is not None else None))
        tr = d["trap_by_frame"].get(f)
        hist_trap_score.append((t, tr["score"] if tr and tr["valid"] else None))
        line_depth_history.append((f, defending_team, line_depth))
        window = line_depth_history[-(LINE_VEL_BASELINE_FRAMES + 1):]
        recent = [(fr, dep) for fr, dt, dep in window if dt == defending_team and dep is not None]
        closing_rate = None
        if lead is not None and lead.get("vx_cm_s") is not None and line_depth is not None and len(recent) >= 2:
            f0, d0 = recent[0]
            dt_sec = (f - f0) / fps
            if dt_sec > 0:
                closing_rate = (line_depth - d0) / dt_sec + lead["vx_cm_s"] * sign_cache[defending_team]
        hist_closing_rate.append((t, (closing_rate / 100.0) if closing_rate is not None else None))
    y_depth = compute_stable_y_range([v for _, v in hist_lead_depth] + [v for _, v in hist_line_depth])
    y_trap = (0.0, 1.0)
    y_closing = compute_stable_y_range([v for _, v in hist_closing_rate])
    y_relation = compute_stable_y_range([v for _, v in hist_relation])
    # "Dangerous space behind the line" = the real line-depth-from-own-
    # goal measurement re-expressed as an area estimate (depth x the
    # pitch's actual width, 70m -- both real, measured quantities, not
    # a new fabricated metric) so this reference panel slot shows a
    # genuinely different-feeling number from the raw depth graph.
    PITCH_WIDTH_M = 70.0
    hist_space = [(t, (v * PITCH_WIDTH_M) if v is not None else None) for t, v in hist_line_depth]
    y_space = compute_stable_y_range([v for _, v in hist_space])

    n_buckets = int(np.ceil(span_sec / BUCKET_SEC))
    bucket_colors = []
    for b in range(n_buckets):
        f0, f1 = int(b * BUCKET_SEC * fps), int(min(n_total, (b + 1) * BUCKET_SEC * fps))
        states = []
        for f in range(f0, f1):
            dt = defending_team_by_frame.get(f)
            if dt is not None:
                tr = dir_by_defender[dt]["trap_by_frame"].get(f)
                if tr:
                    states.append(tr["trap_state"])
        majority = max(set(states), key=states.count) if states else "UNCERTAIN"
        bucket_colors.append(TRAP_STATE_COLOR.get(majority, (100, 100, 100)))
    all_runs = dir_by_defender[0]["runs_list"] + dir_by_defender[1]["runs_list"]
    all_cands = dir_by_defender[0]["candidates_list"] + dir_by_defender[1]["candidates_list"]
    run_dots = [(run["start_time_sec"], RUN_COLOR, 5) for run in all_runs]
    cand_dots = [(c["start_time_sec"], ACCENT_RED, 6) for c in all_cands]
    legend = [(k.replace("_", " ").title(), v) for k, v in TRAP_STATE_COLOR.items()] + [("Run Start", RUN_COLOR)]

    def render_frame(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            return None
        cur_time = f / fps
        defending_team = defending_team_by_frame[f]
        attacking_team = (1 - defending_team) if defending_team is not None else None

        # V3 Goal 1: BOTH teams' defensive lines are real, always-
        # computable facts (not gated behind whichever team currently
        # has the ball) -- compute both every frame.
        line_info0 = compute_offside_line(cleaned, 0, f)
        line_info1 = compute_offside_line(cleaned, 1, f)
        line_x0 = line_info0["line_x"] if line_info0.get("valid") else None
        line_x1 = line_info1["line_x"] if line_info1.get("valid") else None
        line_info = line_info0 if defending_team == 0 else line_info1 if defending_team == 1 else {"valid": False, "reason": "possession uncertain"}
        line_x = line_info["line_x"] if line_info.get("valid") else None

        # V3 Goal 1/2: the real, transition-aware, simultaneous per-team
        # states -- these replace V2's possession-gated single trap_info
        # as the SOURCE OF TRUTH for what's displayed as "the" state.
        state0, state1 = dual[0]["trap_states"][f], dual[1]["trap_states"][f]
        dominant_state, dominant_team = dominant_trap_state_by_frame[f], dominant_trap_team_by_frame[f]

        players_this_frame = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None]
        lead, lead_feat, run_track_ids, active_run, cand = None, None, set(), None, None
        if defending_team is not None:
            attackers = [p for p in players_this_frame if p["display_team_id"] == attacking_team
                         and p["display_object_type"] in ("player", "goalkeeper")]
            defenders = [p for p in players_this_frame if p["display_team_id"] == defending_team
                         and p["display_object_type"] in ("player", "goalkeeper")]
            lead = min(attackers, key=lambda p: _forward_depth(defending_team, p["x_pitch"])) if attackers else None
            if lead is not None and line_info.get("valid"):
                relation_cm = compute_line_relation_cm(lead["x_pitch"], defending_team, line_info["line_depth_cm"])
                ball_x = gated_ball_x(ball_by_frame.get(f))
                est = compute_estimated_offside_position(lead["x_pitch"], defending_team, line_info["line_depth_cm"], ball_x)
                nearest_def = min((np.hypot(dd["x_pitch"] - lead["x_pitch"], dd["y_pitch"] - lead["y_pitch"]) for dd in defenders), default=None)
                poss_row = poss_by_frame.get(f)
                lane_status = "n/a (no confident carrier)"
                if poss_row is not None and poss_row["possession_state"] == "CONTROLLED":
                    carrier_row = next((p for p in players_this_frame if p["track_id"] == poss_row["possessing_track_id"]), None)
                    if carrier_row is not None and carrier_row["track_id"] == lead["track_id"]:
                        lane_status = "n/a (lead attacker is the carrier)"
                    elif carrier_row is not None:
                        open_lane = _passing_lane_open((carrier_row["x_pitch"], carrier_row["y_pitch"]),
                                                         (lead["x_pitch"], lead["y_pitch"]), defenders)
                        lane_status = "open" if open_lane else "blocked"
                cr = hist_closing_rate[f][1]
                lead_feat = {"track_id": lead["track_id"], "relation_cm": relation_cm,
                             "relation_label": line_relation_label(relation_cm), "est_state": est["state"],
                             "ball_confident": ball_x is not None, "nearest_def_cm": nearest_def,
                             "lane_status": lane_status, "closing_rate_cm_s": (cr * 100.0 if cr is not None else None)}
            active_runs_here = run_frame_index[defending_team].get(f, [])
            run_track_ids = {r["attacker_track_id"] for r in active_runs_here}
            active_run = active_runs_here[0] if active_runs_here else None
            cand = cand_frame_index[defending_team].get(f)

        transformer = _nearest_transformer(transformers, f)
        # V3 Goal 3: the primary banner shows the DOMINANT simultaneous
        # call (decide_dominant_trap_state), and the small role line
        # explicitly states BOTH teams' own trap states -- never just
        # whichever team happens to hold the ball this frame.
        banner_text = dominant_state.replace("_", " ").title()
        banner_color = TRAP_STATE_COLOR.get(dominant_state, (100, 100, 100))
        role_text = (f"T0 trap: {state0.replace('_', ' ').title()}  |  T1 trap: {state1.replace('_', ' ').title()}"
                     + (f"  (Team {defending_team} defends now)" if defending_team is not None else ""))

        left = cv2.resize(draw_left_feed(frame, tracking_by_frame.get(f, []), attacking_team, defending_team,
                                          line_x0, line_x1, transformer, lead["track_id"] if lead else None, run_track_ids,
                                          banner_text, banner_color, role_text), (LEFT_W, TOP_ROW_H))
        ball_row = ball_by_frame.get(f)
        ball_xy = (ball_row["x_pitch"], ball_row["y_pitch"]) if ball_row and ball_row.get("is_observed") and ball_row.get("x_pitch") is not None else None
        radar = draw_radar(players_this_frame, ball_xy, line_x0, line_x1, state0, state1, RIGHT_W, TOP_ROW_H)
        top_row = np.hstack([left, radar])

        header_status = (f"Trap-break candidate: track {cand['attacker_track_id']} → {cand['break_outcome']}", ACCENT_RED) if cand \
            else (f"t = {cur_time:.1f}s", TEXT_DIM)
        header = draw_header(CANVAS_W, HEADER_H, "OFFSIDE / TRAP-BREAK ANALYTICS", header_status, accent=ACCENT_RED)

        # --- 4-column lower area (KPI 2x2 | line-depth graphs | trap
        # distribution + pattern summary | relation + dangerous-space
        # graphs), per the offside reference's own column structure ---
        kpi_block = draw_tiles_2x2(state0, state1, dominant_state, dominant_team, dual_trap_stats, defending_team, KPI_COL_W, CONTENT_H)

        cl_top = draw_big_graph(GRAPH_COL_W, PANEL_H, hist_line_depth, cur_time, GRAPH_WINDOW_SEC,
                                  "Defensive Line Depth", y_depth, "m", LINE_COLOR)
        # V3: replaces V2's single-team "Distance to Lead Attacker" with
        # the SIMULTANEOUS Team0-vs-Team1 trap-score comparison (Goal 2's
        # stable, FSM-smoothed score, not the raw per-frame value) --
        # this is the graph that makes the transition logic upgrade
        # directly visible/inspectable (Goal 5 QA).
        cl_bot = draw_big_dual_graph(GRAPH_COL_W, PANEL_H, hist_trap0_v3, hist_trap1_v3, cur_time, GRAPH_WINDOW_SEC,
                                       "Trap Score (smoothed): Team 0 vs Team 1", y_trap_v3, "Team 0", "Team 1", "",
                                       color_a=TEAM_LINE_COLOR[0], color_b=TEAM_LINE_COLOR[1])
        col_gap = np.full((GAP, GRAPH_COL_W, 3), BG, dtype=np.uint8)
        center_left_col = np.vstack([cl_top, col_gap, cl_bot])

        dist_panel = draw_trap_state_distribution(GRAPH_COL_W, PANEL_H, dual)
        pattern_lines = [f"{len(all_runs)} attacking runs detected (both directions).",
                          f"V3 FSM (enter={TRAP_FSM_CONFIG.enter_th}, exit={TRAP_FSM_CONFIG.exit_th}, "
                          f"min_activate={TRAP_FSM_CONFIG.min_activate_frames}f): Team0 {len(dual[0]['trap_segments'])} state "
                          f"segments ({dual_trap_stats[0]['n_active_episodes']} TRAP_ACTIVE), Team1 {len(dual[1]['trap_segments'])} "
                          f"segments ({dual_trap_stats[1]['n_active_episodes']} TRAP_ACTIVE).",
                          f"{n_break_threat} frame(s) TRAP_BREAK_THREAT, {n_broken} frame(s) TRAP_BROKEN (both teams combined).",
                          (f"Mean trap score: Team0 {dual_trap_stats[0]['mean_score']:.2f}, Team1 {dual_trap_stats[1]['mean_score']:.2f} "
                           f"-- thresholds validated against this clip's real score distribution (see offside_v3.py)."
                           if dual_trap_stats[0]["mean_score"] is not None and dual_trap_stats[1]["mean_score"] is not None
                           else "See offside_v3_summary_v3.md for full threshold validation notes.")]
        pattern_panel = draw_insights_panel(GRAPH_COL_W, PANEL_H, "Trap / Run Pattern Summary (V3)", pattern_lines, accent=ACCENT_ORANGE)
        center_col = np.vstack([dist_panel, col_gap, pattern_panel])

        rel_graph = draw_big_dual_graph(GRAPH_COL_W, PANEL_H, hist_lead_depth, hist_line_depth, cur_time, GRAPH_WINDOW_SEC,
                                          "Attacker / Line Relation", y_depth, "lead attacker", "defensive line", "m",
                                          color_a=LEAD_COLOR, color_b=LINE_COLOR)
        space_graph = draw_big_graph(GRAPH_COL_W, PANEL_H, hist_space, cur_time, GRAPH_WINDOW_SEC,
                                       "Dangerous Space Behind Line", y_space, "m2", ACCENT_ORANGE)
        right_col = np.vstack([rel_graph, col_gap, space_graph])

        gap_col = np.full((CONTENT_H, GAP, 3), BG, dtype=np.uint8)
        content_row = np.hstack([gap_col, kpi_block, gap_col, center_left_col, gap_col, center_col, gap_col, right_col, gap_col])
        if content_row.shape[1] != CANVAS_W:
            content_row = cv2.resize(content_row, (CANVAS_W, CONTENT_H))

        timeline = draw_timeline_bucketed(CANVAS_W, TIMELINE_H, span_sec, cur_time, bucket_colors, BUCKET_SEC,
                                           run_dots + cand_dots, legend, title="TRAP STATE (2s buckets) / RUN STARTS")

        row_gap = np.full((GAP, CANVAS_W, 3), BG, dtype=np.uint8)
        canvas = np.vstack([header, top_row, row_gap, content_row, row_gap, timeline])
        return canvas

    if args.preview_frame_sec is not None:
        f = int(args.preview_frame_sec * fps)
        canvas = render_frame(f)
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        cv2.imwrite(args.out_path, canvas)
        print(f"Wrote preview frame {args.out_path}: dims {canvas.shape[1]}x{canvas.shape[0]}")
        y = 0
        for name, h in [("header", HEADER_H), ("top_row(video+radar)", TOP_ROW_H), ("gap", GAP),
                         ("content_row(4 columns)", CONTENT_H), ("gap", GAP), ("timeline", TIMELINE_H)]:
            print(f"  {name:28s} y={y:4d} h={h:4d}  ({h/1204:.1%} of canvas height)")
            y += h
        print(f"  video x=0 w={LEFT_W} ({LEFT_W/CANVAS_W:.1%})  |  radar x={LEFT_W} w={RIGHT_W} ({RIGHT_W/CANVAS_W:.1%})")
        print(f"  content_row columns: kpi w={KPI_COL_W} | 3x graph col w={GRAPH_COL_W} each, panel_h={PANEL_H}")
        print(f"  canvas total height check: {y} (must be 1204)")
        cap.release()
        return

    f_start = int(args.start_sec * fps)
    f_end = int(args.end_sec * fps) if args.end_sec is not None else n_total - 1
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_path, fourcc, fps, (CANVAS_W, 1204))
    n_written = 0
    for f in range(f_start, f_end + 1):
        canvas = render_frame(f)
        if canvas is None:
            break
        writer.write(canvas)
        n_written += 1
        if n_written % 600 == 0:
            print(f"...{n_written} frames written ({f/fps:.1f}s)", flush=True)
    writer.release()
    cap.release()
    print(f"Wrote {args.out_path}: {n_written} frames, canvas {CANVAS_W}x1204")


if __name__ == "__main__":
    main()
