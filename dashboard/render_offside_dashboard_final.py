"""
Offside/trap-break "final" dashboard -- same V2 analytics (unchanged
import, no logic touched), NEW dark-card visual language modeled on a
locked reference mockup (layout/hierarchy/color-treatment only -- no
values, labels, or event categories from the mockup are used; every
number shown here comes from `offside_break.analytics.offside_v2`'s
real outputs). V2 terminology is preserved exactly: AHEAD/BEHIND
DEFENSIVE LINE, ESTIMATED_OFFSIDE_POSITION / ESTIMATED_ONSIDE_POSITION
/ LINE_RELATION_ONLY, NO_TRAP / TRAP_FORMING / TRAP_ACTIVE / UNCERTAIN.

Canvas is fixed at 2304x1204. Top row (video + radar) occupies >60% of
the canvas height, matching the pressing final dashboard's layout
constants for visual consistency between the two.

Usage:
    external/sports/.venv/bin/python offside_break/dashboard/render_offside_dashboard_final.py \\
        --out_path offside_break/outputs/final_dashboard/offside_dashboard_120s_final.mp4
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
from offside_break.analytics.offside import _forward_depth, compute_offside_line, offside_line_series  # noqa: E402
from offside_break.analytics.offside_v2 import (compute_estimated_offside_position, compute_line_relation_cm,  # noqa: E402
                                                  classify_trap_break_candidates, detect_attacking_runs,
                                                  forward_sign_away_from_own_goal, gated_ball_x,
                                                  line_relation_label, trap_state_series)
from pressing_structure.analytics.cleaned_tracking_view import build_cleaned_view, load_ball_view  # noqa: E402
from pressing_structure.analytics.pressing import _passing_lane_open, build_possession  # noqa: E402
from pressing_structure.dashboard.dashboard_style import (ACCENT_CYAN, ACCENT_MAGENTA, BG_DARK, BORDER_COLOR,  # noqa: E402
                                                            FONT, PANEL_DARK, TEXT_DIM, TEXT_WHITE, draw_card,
                                                            draw_dark_dual_scrolling_graph, draw_dark_scrolling_graph,
                                                            draw_event_timeline_dark, draw_header_strip,
                                                            draw_metric_tile)
from pressing_structure.dashboard.live_graphs import compute_stable_y_range  # noqa: E402
from pressing_structure.dashboard.render_pressing_dashboard import _nearest_transformer  # noqa: E402

CONFIG = SoccerPitchConfiguration()
DEFEND_COLOR = (255, 200, 60)    # BGR cyan-ish, defending team
ATTACK_COLOR = (200, 60, 220)    # BGR magenta, attacking team
LINE_COLOR = (50, 50, 235)       # red defensive line
LEAD_COLOR = (0, 210, 255)       # amber/yellow lead-attacker highlight
RUN_COLOR = (110, 220, 90)
TRAP_STATE_COLOR = {"NO_TRAP": (150, 150, 150), "TRAP_FORMING": (0, 165, 255),
                     "TRAP_ACTIVE": (50, 50, 235), "UNCERTAIN": (110, 110, 110)}
GRAPH_WINDOW_SEC = 8.0
HEADER_H = 64
TIMELINE_H = 50
TILE_ROW_H = 92
GRAPH_ROW_H = 268
GAP = 10
LINE_VEL_BASELINE_FRAMES = 15  # 0.5s baseline, matches the V2 dashboard's fix for the same metric


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


def draw_left_feed(frame, tracking_rows, attacking_team, defending_team, line_x, transformer,
                    lead_track_id, run_track_ids, trap_state):
    vis = frame.copy()
    for r in tracking_rows:
        if r["object_type"] not in ("player", "goalkeeper"):
            continue
        color = DEFEND_COLOR if r["team_id"] == defending_team else ATTACK_COLOR if r["team_id"] == attacking_team else (180, 180, 180)
        thickness, label = 2, None
        # A "RUN" text label per runner collided with the LEAD ATTACKER
        # chip whenever two attackers stood close together on screen
        # (found during full-clip QA at t=30s) -- run membership is now
        # shown as a colored border ONLY (no text), which cannot collide;
        # the lead attacker keeps its chip since there is only ever one.
        if r["track_id"] in run_track_ids:
            color, thickness = RUN_COLOR, 3
        if r["team_id"] == attacking_team and r["track_id"] == lead_track_id:
            color, thickness, label = LEAD_COLOR, 3, "LEAD ATTACKER"
        x1, y1, x2, y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if label:
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.42, 2)
            cv2.rectangle(vis, (x1, y1 - th - 12), (x1 + tw + 8, y1 - 4), (0, 0, 0), -1)
            cv2.putText(vis, label, (x1 + 4, y1 - 8), FONT, 0.42, color, 2, cv2.LINE_AA)
    if line_x is not None:
        vis = _draw_pitch_line_on_image(vis, transformer, line_x, LINE_COLOR, 3)
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (330, 66), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, dst=vis)
    txt1 = f"DEFENSIVE LINE: team {defending_team}" if line_x is not None else "LINE: insufficient data"
    cv2.putText(vis, txt1, (12, 24), FONT, 0.48, TEXT_WHITE if line_x is not None else TEXT_DIM, 2, cv2.LINE_AA)
    cv2.putText(vis, trap_state, (12, 52), FONT, 0.58, TRAP_STATE_COLOR.get(trap_state, TEXT_DIM), 2, cv2.LINE_AA)
    return vis


def draw_radar(players_this_frame, ball_xy, attacking_team, defending_team, line_x, w, h):
    team_a = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0} for p in players_this_frame if p["display_team_id"] == 0]
    team_b = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"],
               "vx": p.get("vx_cm_s") or 0.0, "vy": p.get("vy_cm_s") or 0.0} for p in players_this_frame if p["display_team_id"] == 1]
    pc = compute_pitch_control_grid(team_a, team_b) if (team_a and team_b) else {"valid": False}
    color0 = DEFEND_COLOR if defending_team == 0 else ATTACK_COLOR
    color1 = DEFEND_COLOR if defending_team == 1 else ATTACK_COLOR

    pitch = draw_pitch(config=CONFIG)
    pitch = cv2.addWeighted(pitch, 0.7, np.zeros_like(pitch), 0.3, 0)
    if pc.get("valid"):
        ph, pw = pitch.shape[:2]
        up = cv2.resize(pc["grid"].astype(np.float32), (pw, ph), interpolation=cv2.INTER_CUBIC)
        up = np.clip(up, 0, 1)[..., None]
        color_field = (up * np.array(color0) + (1 - up) * np.array(color1)).astype(np.uint8)
        pitch = cv2.addWeighted(color_field, 0.6, pitch, 0.4, 0)
    if line_x is not None:
        goal_x = 12000.0 if defending_team == 0 else 0.0
        x0, x1 = sorted([line_x, goal_x])
        p0, p1 = _pitch_xy_to_px(x0, 0), _pitch_xy_to_px(x1, 7000)
        overlay = pitch.copy()
        cv2.rectangle(overlay, p0, p1, (0, 210, 255), -1)
        cv2.addWeighted(overlay, 0.28, pitch, 0.72, 0, dst=pitch)
        lp0, lp1 = _pitch_xy_to_px(line_x, 0), _pitch_xy_to_px(line_x, 7000)
        cv2.line(pitch, lp0, lp1, LINE_COLOR, 3, cv2.LINE_AA)
    for p in players_this_frame:
        px, py = _pitch_xy_to_px(p["x_pitch"], p["y_pitch"])
        color = DEFEND_COLOR if p["display_team_id"] == defending_team else ATTACK_COLOR if p["display_team_id"] == attacking_team else (150, 150, 150)
        cv2.circle(pitch, (px, py), 7, color, -1, cv2.LINE_AA)
        cv2.circle(pitch, (px, py), 7, (20, 20, 20), 1, cv2.LINE_AA)
        if p.get("vx_cm_s") is not None and p["display_team_id"] == attacking_team:
            ex, ey = int(px + p["vx_cm_s"] * 0.03), int(py + p["vy_cm_s"] * 0.03)
            cv2.arrowedLine(pitch, (px, py), (ex, ey), (90, 235, 90), 2, cv2.LINE_AA, tipLength=0.3)
    if ball_xy is not None:
        bx, by = _pitch_xy_to_px(*ball_xy)
        cv2.circle(pitch, (bx, by), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(pitch, (bx, by), 6, (20, 20, 20), 1, cv2.LINE_AA)
    overlay = pitch.copy()
    cv2.rectangle(overlay, (0, 0), (280, 38), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, pitch, 0.45, 0, dst=pitch)
    cv2.putText(pitch, "OFFSIDE RADAR", (14, 26), FONT, 0.6, TEXT_WHITE, 2, cv2.LINE_AA)
    return cv2.resize(pitch, (w, h))


def draw_tiles_row(trap_info, lead_feat, width, height):
    n = 4
    tile_w = (width - (n + 1) * GAP) // n
    img = np.full((height, width, 3), BG_DARK, dtype=np.uint8)
    trap_val = f"{trap_info['score']:.2f}" if trap_info.get("valid") else "n/a"
    trap_sub = trap_info.get("trap_state", "n/a")
    if lead_feat is not None:
        values = [
            ("TRAP STATE", trap_info.get("trap_state", "n/a"), trap_sub if trap_info.get("valid") else trap_info.get("reason", "")),
            ("TRAP SYNCHRONY SCORE", trap_val, "0-1 scale"),
            ("DISTANCE TO LINE", f"{lead_feat['relation_cm']/100:+.1f} m", lead_feat["relation_label"]),
            ("RELATIVE SPEED TO LINE", f"{lead_feat['closing_rate_cm_s']/100:+.1f} m/s" if lead_feat["closing_rate_cm_s"] is not None else "n/a", "closing rate onto line"),
        ]
    else:
        values = [("TRAP STATE", trap_info.get("trap_state", "n/a"), ""), ("TRAP SYNCHRONY SCORE", trap_val, "0-1 scale"),
                  ("DISTANCE TO LINE", "n/a", "no lead attacker"), ("RELATIVE SPEED TO LINE", "n/a", "")]
    x = GAP
    for label, val, sub in values:
        tile = draw_metric_tile(tile_w, height, label, val, sub, accent=ACCENT_CYAN)
        img[0:height, x:x + tile_w] = tile
        x += tile_w + GAP
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_video_path", type=str, default="ExternalDownlaodVideo/testVideo1_120s.mp4")
    parser.add_argument("--tracking_dir", type=str, default="outputs/tracking/testVideo1_120s")
    parser.add_argument("--analytics_dir", type=str, default="outputs/analytics/testVideo1_120s_v3")
    parser.add_argument("--attacking_team", type=int, default=1, choices=[0, 1])
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--end_sec", type=float, default=None)
    args = parser.parse_args()
    attacking_team = args.attacking_team
    defending_team = 1 - attacking_team

    tracking = pl.read_parquet(os.path.join(args.tracking_dir, "tracking.parquet"))
    cleaned = build_cleaned_view(tracking)
    ball = load_ball_view(args.analytics_dir)
    passes = pl.read_parquet(os.path.join(args.analytics_dir, "passes.parquet"))
    poss = build_possession(cleaned, ball)
    poss_by_frame = {r["frame"]: r for r in poss.to_dicts()}

    cap = cv2.VideoCapture(args.source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_start = int(args.start_sec * fps)
    f_end = int(args.end_sec * fps) if args.end_sec is not None else n_total - 1
    span_sec = n_total / fps

    print("Computing offside-line series...", flush=True)
    line_series = offside_line_series(cleaned, defending_team, 0, n_total - 1)
    line_by_frame = {r["frame"]: r["line_depth_cm_smoothed"] for r in line_series.to_dicts()}
    print("Computing trap-synchrony series...", flush=True)
    trap_series = trap_state_series(cleaned, defending_team, 0, n_total - 1)
    trap_series_dicts = trap_series.to_dicts()
    trap_by_frame = {r["frame"]: r for r in trap_series_dicts}
    print("Detecting attacking runs...", flush=True)
    runs = detect_attacking_runs(cleaned, attacking_team, defending_team, line_series)
    candidates = classify_trap_break_candidates(runs, trap_series, cleaned, ball, passes, attacking_team, defending_team)
    runs_list = runs.to_dicts() if runs.height else []
    candidates_list = candidates.filter(pl.col("is_trap_break_candidate")).to_dicts() if candidates.height else []
    print(f"  {len(runs_list)} runs, {len(candidates_list)} trap-break candidates", flush=True)

    ball_by_frame = {r["frame"]: r for r in ball.to_dicts()}
    tracking_by_frame = {}
    for r in tracking.to_dicts():
        tracking_by_frame.setdefault(r["frame"], []).append(r)
    cleaned_by_frame = {}
    for r in cleaned.to_dicts():
        cleaned_by_frame.setdefault(r["frame"], []).append(r)
    run_frame_index = {}
    for run in runs_list:
        for fr in range(run["start_frame"], run["end_frame"] + 1):
            run_frame_index.setdefault(fr, []).append(run)
    cand_frame_index = {}
    for c in candidates_list:
        for fr in range(c["start_frame"], c["end_frame"] + 1):
            cand_frame_index.setdefault(fr, c)

    sign = forward_sign_away_from_own_goal(attacking_team)

    with open(os.path.join(args.analytics_dir, "homography_transformers.pkl"), "rb") as fh:
        transformers = pickle.load(fh)

    print("Precomputing live-graph histories...", flush=True)
    hist_lead_depth, hist_line_depth, hist_trap_score, hist_closing_rate, hist_relation = [], [], [], [], []
    line_depth_history = []
    for f in range(n_total):
        t = f / fps
        line_depth = line_by_frame.get(f)
        hist_line_depth.append((t, (line_depth / 100.0) if line_depth is not None else None))
        players = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None and p["display_team_id"] == attacking_team
                   and p["display_object_type"] in ("player", "goalkeeper")]
        lead = min(players, key=lambda p: _forward_depth(defending_team, p["x_pitch"])) if players else None
        lead_depth = _forward_depth(defending_team, lead["x_pitch"]) if lead is not None else None
        hist_lead_depth.append((t, (lead_depth / 100.0) if lead_depth is not None else None))
        relation = compute_line_relation_cm(lead["x_pitch"], defending_team, line_depth) if (lead is not None and line_depth is not None) else None
        hist_relation.append((t, (relation / 100.0) if relation is not None else None))
        tr = trap_by_frame.get(f)
        hist_trap_score.append((t, tr["score"] if tr and tr["valid"] else None))

        if line_depth is not None:
            line_depth_history.append((f, line_depth))
        while line_depth_history and f - line_depth_history[0][0] > LINE_VEL_BASELINE_FRAMES:
            line_depth_history.pop(0)
        closing_rate = None
        if lead is not None and lead.get("vx_cm_s") is not None and line_depth is not None and len(line_depth_history) >= 2:
            f0, d0 = line_depth_history[0]
            dt = (f - f0) / fps
            if dt > 0:
                line_vel = (line_depth - d0) / dt
                closing_rate = line_vel + lead["vx_cm_s"] * sign
        hist_closing_rate.append((t, (closing_rate / 100.0) if closing_rate is not None else None))
    y_depth = compute_stable_y_range([v for _, v in hist_lead_depth] + [v for _, v in hist_line_depth])
    y_trap = (0.0, 1.0)
    y_closing = compute_stable_y_range([v for _, v in hist_closing_rate])
    y_relation = compute_stable_y_range([v for _, v in hist_relation])

    canvas_w = 2304
    top_row_h = 730
    sample_pitch = draw_pitch(config=CONFIG)
    right_w = int(round(top_row_h * sample_pitch.shape[1] / sample_pitch.shape[0]))
    left_w = canvas_w - right_w
    canvas_h = HEADER_H + top_row_h + TILE_ROW_H + GRAPH_ROW_H + TIMELINE_H
    n_graphs = 4
    graph_w = (canvas_w - (n_graphs + 1) * GAP) // n_graphs

    # The trap-state timeline has one segment PER FRAME (3600 of them) --
    # redrawing all of them every output frame would mean ~13M rectangle
    # draws for the full clip. Render the static background (segments +
    # run/candidate markers + legend) ONCE, then per-frame only draw the
    # cheap moving current-time marker on a copy.
    segments = [(r["frame"] / fps, (r["frame"] + 1) / fps, TRAP_STATE_COLOR.get(r["trap_state"], (150, 150, 150))) for r in trap_series_dicts]
    run_markers = [(run["start_time_sec"], RUN_COLOR) for run in runs_list]
    cand_markers = [(c["start_time_sec"], (50, 50, 235)) for c in candidates_list]
    legend = [(k, v) for k, v in TRAP_STATE_COLOR.items()] + [("RUN", RUN_COLOR)]
    timeline_bg = draw_event_timeline_dark(canvas_w, TIMELINE_H, span_sec, -1e9, segments,
                                             run_markers + cand_markers, legend, title="TRAP STATE / RUN / CANDIDATE TIMELINE")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_path, fourcc, fps, (canvas_w, canvas_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    n_written = 0
    for f in range(f_start, f_end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        cur_time = f / fps
        line_info = compute_offside_line(cleaned, defending_team, f)
        line_x = line_info["line_x"] if line_info["valid"] else None
        trap_info = trap_by_frame.get(f, {"valid": False, "reason": "n/a", "score": None,
                                           "fraction_stepping_forward": None, "depth_dispersion_score": None,
                                           "velocity_alignment_score": None})
        trap_state = trap_info.get("trap_state", "UNCERTAIN")

        players_this_frame = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None]
        attackers = [p for p in players_this_frame if p["display_team_id"] == attacking_team
                     and p["display_object_type"] in ("player", "goalkeeper")]
        defenders = [p for p in players_this_frame if p["display_team_id"] == defending_team
                     and p["display_object_type"] in ("player", "goalkeeper")]
        lead = min(attackers, key=lambda p: _forward_depth(defending_team, p["x_pitch"])) if attackers else None

        lead_feat = None
        if lead is not None and line_info["valid"]:
            relation_cm = compute_line_relation_cm(lead["x_pitch"], defending_team, line_info["line_depth_cm"])
            ball_x = gated_ball_x(ball_by_frame.get(f))
            est = compute_estimated_offside_position(lead["x_pitch"], defending_team, line_info["line_depth_cm"], ball_x)
            nearest_def = min((np.hypot(d["x_pitch"] - lead["x_pitch"], d["y_pitch"] - lead["y_pitch"]) for d in defenders), default=None)
            poss_row = poss_by_frame.get(f)
            lane_status = "n/a (no confident carrier)"
            if poss_row is not None and poss_row["possession_state"] == "CONTROLLED":
                carrier_row = next((p for p in players_this_frame if p["track_id"] == poss_row["possessing_track_id"]), None)
                if carrier_row is not None and carrier_row["track_id"] == lead["track_id"]:
                    lane_status = "n/a (lead attacker IS the carrier)"
                elif carrier_row is not None:
                    open_lane = _passing_lane_open((carrier_row["x_pitch"], carrier_row["y_pitch"]),
                                                     (lead["x_pitch"], lead["y_pitch"]), defenders)
                    lane_status = "open" if open_lane else "blocked"
            lead_feat = {"track_id": lead["track_id"], "relation_cm": relation_cm,
                         "relation_label": line_relation_label(relation_cm), "est_state": est["state"],
                         "ball_confident": ball_x is not None, "nearest_def_cm": nearest_def,
                         "lane_status": lane_status, "closing_rate_cm_s": None}
            cr = hist_closing_rate[f][1]
            lead_feat["closing_rate_cm_s"] = cr * 100.0 if cr is not None else None

        active_runs_here = run_frame_index.get(f, [])
        run_track_ids = {r["attacker_track_id"] for r in active_runs_here}
        cand = cand_frame_index.get(f)
        candidate_msg = None
        if cand is not None:
            candidate_msg = f"TRAP-BREAK CANDIDATE: track {cand['attacker_track_id']} ({cand['run_type']}) -> {cand['break_outcome']}"

        transformer = _nearest_transformer(transformers, f)
        left = cv2.resize(draw_left_feed(frame, tracking_by_frame.get(f, []), attacking_team, defending_team,
                                          line_x, transformer, lead["track_id"] if lead else None, run_track_ids, trap_state),
                           (left_w, top_row_h))
        ball_row = ball_by_frame.get(f)
        ball_xy = (ball_row["x_pitch"], ball_row["y_pitch"]) if ball_row and ball_row.get("is_observed") and ball_row.get("x_pitch") is not None else None
        radar = draw_radar(players_this_frame, ball_xy, attacking_team, defending_team, line_x, right_w, top_row_h)
        top_row = np.hstack([left, radar])

        header_right = [(candidate_msg or "(no trap-break candidate active)", (50, 50, 235) if candidate_msg else TEXT_DIM),
                         (f"t={cur_time:5.1f}s", TEXT_DIM)]
        header = draw_header_strip(canvas_w, HEADER_H, "OFFSIDE / TRAP-BREAK ANALYTICS", "rule-based | V2 logic", header_right, accent=LINE_COLOR)

        tiles = draw_tiles_row(trap_info, lead_feat, canvas_w, TILE_ROW_H)

        g1 = draw_dark_dual_scrolling_graph(hist_lead_depth, hist_line_depth, cur_time, GRAPH_WINDOW_SEC, graph_w, GRAPH_ROW_H,
                                              "Attacker Depth vs Line Depth", f"last {GRAPH_WINDOW_SEC:.0f}s", y_depth,
                                              "lead attacker", "defensive line", "m", color_a=LEAD_COLOR, color_b=LINE_COLOR)
        g2 = draw_dark_scrolling_graph(hist_trap_score, cur_time, GRAPH_WINDOW_SEC, graph_w, GRAPH_ROW_H,
                                         "Trap Synchrony Score", f"last {GRAPH_WINDOW_SEC:.0f}s", y_trap, "0-1", (0, 165, 255))
        g3 = draw_dark_scrolling_graph(hist_closing_rate, cur_time, GRAPH_WINDOW_SEC, graph_w, GRAPH_ROW_H,
                                         "Relative Closing Rate onto Line", f"last {GRAPH_WINDOW_SEC:.0f}s", y_closing, "m/s", ACCENT_MAGENTA)
        g4 = draw_dark_scrolling_graph(hist_relation, cur_time, GRAPH_WINDOW_SEC, graph_w, GRAPH_ROW_H,
                                         "Distance to Lead Attacker (line relation)", f"last {GRAPH_WINDOW_SEC:.0f}s", y_relation, "m", RUN_COLOR)
        gap_col = np.full((GRAPH_ROW_H, GAP, 3), BG_DARK, dtype=np.uint8)
        graphs_row = np.hstack([gap_col, g1, gap_col, g2, gap_col, g3, gap_col, g4, gap_col])
        if graphs_row.shape[1] != canvas_w:
            graphs_row = cv2.resize(graphs_row, (canvas_w, GRAPH_ROW_H))

        timeline = timeline_bg.copy()
        cx = int(cur_time / span_sec * canvas_w)
        cv2.line(timeline, (cx, 0), (cx, TIMELINE_H), TEXT_WHITE, 2, cv2.LINE_AA)

        canvas = np.vstack([header, top_row, tiles, graphs_row, timeline])
        writer.write(canvas)
        n_written += 1
        if n_written % 600 == 0:
            print(f"...{n_written} frames written ({cur_time:.1f}s)", flush=True)

    writer.release()
    cap.release()
    print(f"Wrote {args.out_path}: {n_written} frames, canvas {canvas_w}x{canvas_h}, "
          f"attacking_team={attacking_team}, defending_team={defending_team}")


if __name__ == "__main__":
    main()
