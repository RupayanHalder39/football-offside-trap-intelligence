"""
Offside-trap-break dashboard renderer (Project 2). NEW script, NEW
project folder -- does not touch the existing PhD `scripts/`/
`visualization/` packages; reuses existing, validated pieces
(`draw_pitch`, the homography-inverse-warp technique already used by
`visualization/voronoi.py` and Project 1's pressing dashboard,
`analytics.pitch_control`) rather than re-deriving them.

Usage:
    external/sports/.venv/bin/python offside_break/dashboard/render_offside_dashboard.py \\
        --start_sec 0 --end_sec 25 --attacking_team 1 \\
        --out_path offside_break/outputs/offside_dashboard.mp4
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np
import polars as pl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from sports.annotators.soccer import draw_pitch  # noqa: E402
from sports.configs.soccer import SoccerPitchConfiguration  # noqa: E402

from analytics.pitch_control import compute_pitch_control_grid  # noqa: E402
from offside_break.analytics.offside import DEFENDING_GOAL_X, build_offside_features, compute_offside_line  # noqa: E402
from pressing_structure.analytics.cleaned_tracking_view import build_cleaned_view, load_ball_view  # noqa: E402
from pressing_structure.analytics.pressing import build_possession  # noqa: E402
from pressing_structure.dashboard.render_pressing_dashboard import _nearest_transformer  # noqa: E402

CONFIG = SoccerPitchConfiguration()
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEAM_COLOR = {0: (147, 20, 255), 1: (255, 191, 0)}
ATTACKER_AHEAD_COLOR = (0, 140, 255)  # orange -- ahead of the line (offside-position)
LINE_COLOR = (0, 0, 255)


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


def draw_left_panel(frame, tracking_rows, attacking_team, defending_team, line_x, transformer, ahead_ids):
    vis = frame.copy()
    for r in tracking_rows:
        if r["object_type"] not in ("player", "goalkeeper"):
            continue
        color = TEAM_COLOR.get(r["team_id"], (200, 200, 200))
        thickness = 2
        label = None
        if r["team_id"] == attacking_team and r["track_id"] in ahead_ids:
            color, thickness, label = ATTACKER_AHEAD_COLOR, 3, "AHEAD"
        x1, y1, x2, y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(vis, label, (x1, max(12, y1 - 6)), FONT, 0.42, color, 2, cv2.LINE_AA)
    if line_x is not None:
        vis = _draw_pitch_line_on_image(vis, transformer, line_x, LINE_COLOR, 3)
    cv2.rectangle(vis, (0, 0), (300, 34), (0, 0, 0), -1)
    txt = f"DEFENSIVE LINE: team {defending_team}" if line_x is not None else "LINE: insufficient data"
    cv2.putText(vis, txt, (10, 24), FONT, 0.55, (255, 255, 255) if line_x is not None else (150, 150, 150), 2, cv2.LINE_AA)
    return vis


def draw_right_top(players_this_frame, ball_xy, attacking_team, defending_team, line_x, line_valid_reason):
    team_a = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0}
              for p in players_this_frame if p["display_team_id"] == 0]
    team_b = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0}
              for p in players_this_frame if p["display_team_id"] == 1]
    pc = compute_pitch_control_grid(team_a, team_b) if (team_a and team_b) else {"valid": False}

    pitch = draw_pitch(config=CONFIG)
    if pc.get("valid"):
        h, w = pitch.shape[:2]
        up = cv2.resize(pc["grid"].astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        up = np.clip(up, 0, 1)[..., None]
        color_field = (up * np.array(TEAM_COLOR[0]) + (1 - up) * np.array(TEAM_COLOR[1])).astype(np.uint8)
        pitch = cv2.addWeighted(color_field, 0.4, pitch, 0.6, 0)

    # shade the space behind the line (from the line to the defending
    # goal's byline) -- the attacking team's target territory
    if line_x is not None:
        goal_x = DEFENDING_GOAL_X[defending_team]
        x0, x1 = sorted([line_x, goal_x])
        p0 = _pitch_xy_to_px(x0, 0)
        p1 = _pitch_xy_to_px(x1, 7000)
        overlay = pitch.copy()
        cv2.rectangle(overlay, p0, p1, (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.15, pitch, 0.85, 0, dst=pitch)
        lp0 = _pitch_xy_to_px(line_x, 0)
        lp1 = _pitch_xy_to_px(line_x, 7000)
        cv2.line(pitch, lp0, lp1, LINE_COLOR, 3, cv2.LINE_AA)

    for p in players_this_frame:
        px, py = _pitch_xy_to_px(p["x_pitch"], p["y_pitch"])
        color = TEAM_COLOR.get(p["display_team_id"], (150, 150, 150))
        cv2.circle(pitch, (px, py), 6, color, -1, cv2.LINE_AA)
        if p.get("vx_cm_s") is not None and p["display_team_id"] == attacking_team:
            ex = int(px + p["vx_cm_s"] * 0.03)
            ey = int(py + p["vy_cm_s"] * 0.03)
            cv2.arrowedLine(pitch, (px, py), (ex, ey), (0, 100, 0), 2, cv2.LINE_AA, tipLength=0.3)

    if ball_xy is not None:
        bx, by = _pitch_xy_to_px(*ball_xy)
        cv2.circle(pitch, (bx, by), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(pitch, (bx, by), 6, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.putText(pitch, "OFFSIDE-LINE RADAR", (16, 30), FONT, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    sub = "Background: pitch-control | shaded: space behind line | arrows: attacker velocity" \
        if line_x is not None else f"Line unavailable: {line_valid_reason}"
    cv2.putText(pitch, sub, (16, 54), FONT, 0.36, (60, 60, 60), 1, cv2.LINE_AA)
    return pitch


def draw_right_bottom(focal, w=900, h=320):
    panel = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "OFFSIDE-BREAK METRICS", (16, 34), FONT, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    if focal is None:
        cv2.putText(panel, "No assessable attacker this frame (line unavailable or no attackers visible)",
                    (16, 70), FONT, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        return panel
    lines = [
        f"Attacker: track {focal['attacker_track_id']}",
        f"Distance to Line: {focal['distance_to_line_cm']/100:+.1f} m (negative = onside)",
        f"Relative Speed to Line: {(focal['relative_velocity_to_line_cm_s'] or 0)/100:.1f} m/s",
        f"Run Angle: {focal['run_angle_deg']:.0f} deg" if focal['run_angle_deg'] is not None else "Run Angle: n/a",
        f"Run Type: {focal['run_type']}",
        f"Estimated State: {'ONSIDE (behind line)' if focal['onside'] else 'AHEAD OF LINE (offside position)'}",
        f"Passing Lane from Carrier: {'open' if focal['passing_lane_open_from_carrier'] else 'blocked' if focal['passing_lane_open_from_carrier'] is not None else 'n/a (no confident carrier)'}",
        f"Time to Enter Behind-Line Region: {focal['time_to_enter_behind_line_sec']:.1f}s" if focal['time_to_enter_behind_line_sec'] is not None else "Time to Enter Behind-Line: n/a",
    ]
    if focal.get("break_potential"):
        lines.append(f"Break Potential: {focal['break_potential']}")
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (16, 64 + i * 28), FONT, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_video_path", type=str, default="ExternalDownlaodVideo/testVideo1_120s.mp4")
    parser.add_argument("--tracking_dir", type=str, default="outputs/tracking/testVideo1_120s")
    parser.add_argument("--analytics_dir", type=str, default="outputs/analytics/testVideo1_120s_v3")
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--end_sec", type=float, default=25.0)
    parser.add_argument("--attacking_team", type=int, required=True, choices=[0, 1])
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()
    attacking_team = args.attacking_team
    defending_team = 1 - attacking_team

    tracking = pl.read_parquet(os.path.join(args.tracking_dir, "tracking.parquet"))
    cleaned = build_cleaned_view(tracking)
    ball = load_ball_view(args.analytics_dir)
    poss = build_possession(cleaned, ball)
    features = build_offside_features(cleaned, poss, attacking_team, defending_team)

    with open(os.path.join(args.analytics_dir, "homography_transformers.pkl"), "rb") as fh:
        transformers = pickle.load(fh)

    feat_by_frame = {}
    for r in features.to_dicts():
        feat_by_frame.setdefault(r["frame"], []).append(r)
    tracking_by_frame = {}
    for r in tracking.to_dicts():
        tracking_by_frame.setdefault(r["frame"], []).append(r)
    cleaned_by_frame = {}
    for r in cleaned.to_dicts():
        cleaned_by_frame.setdefault(r["frame"], []).append(r)
    ball_by_frame = {r["frame"]: r for r in ball.to_dicts()}

    cap = cv2.VideoCapture(args.source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    f_start, f_end = int(args.start_sec * fps), int(args.end_sec * fps)

    right_w = 900
    left_w, left_h = 1144, 714
    sample_pitch = draw_pitch(config=CONFIG)
    right_top_h = int(round(right_w * sample_pitch.shape[0] / sample_pitch.shape[1]))
    right_bottom_h = 320
    canvas_w = left_w + right_w
    canvas_h = right_top_h + right_bottom_h
    left_y0 = (canvas_h - left_h) // 2

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_path, fourcc, fps, (canvas_w, canvas_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    n_written = 0
    for f in range(f_start, f_end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        line_info = compute_offside_line(cleaned, defending_team, f)
        line_x = line_info["line_x"] if line_info["valid"] else None
        feats = feat_by_frame.get(f, [])
        ahead_ids = {r["attacker_track_id"] for r in feats if not r["onside"]}
        transformer = _nearest_transformer(transformers, f)

        left = draw_left_panel(frame, tracking_by_frame.get(f, []), attacking_team, defending_team,
                                line_x, transformer, ahead_ids)
        left = cv2.resize(left, (left_w, left_h))

        players_this_frame = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None]
        ball_row = ball_by_frame.get(f)
        ball_xy = (ball_row["x_pitch"], ball_row["y_pitch"]) if ball_row and ball_row.get("is_observed") and ball_row.get("x_pitch") is not None else None
        right_top = draw_right_top(players_this_frame, ball_xy, attacking_team, defending_team, line_x, line_info["reason"])
        right_top = cv2.resize(right_top, (right_w, right_top_h))

        # focal attacker: highest break_potential (HIGH>MEDIUM>LOW>None), else closest to the line
        rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, None: 0}
        focal = None
        if feats:
            focal = sorted(feats, key=lambda r: (rank.get(r["break_potential"], 0), -abs(r["distance_to_line_cm"])),
                            reverse=True)[0]
        right_bottom = draw_right_bottom(focal, w=right_w, h=right_bottom_h)

        right_col = np.vstack([right_top, right_bottom])
        canvas = np.full((canvas_h, canvas_w, 3), 35, dtype=np.uint8)
        canvas[left_y0:left_y0 + left_h, 0:left_w] = left
        canvas[0:right_col.shape[0], left_w:left_w + right_w] = right_col
        cv2.line(canvas, (left_w, 0), (left_w, canvas_h), (80, 80, 80), 2)
        writer.write(canvas)
        n_written += 1

    writer.release()
    cap.release()
    print(f"Wrote {args.out_path}: {n_written} frames, canvas {canvas_w}x{canvas_h}, attacking_team={attacking_team}")


if __name__ == "__main__":
    main()
