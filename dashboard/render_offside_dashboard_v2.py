"""
Offside-trap-break V2 dashboard renderer -- full 120s, terminology-safe
line-relation labels, estimated-offside-position state, trap-synchrony
detector, temporal run episodes, trap-break candidates, live scrolling
graphs, richer event timeline. NEW script; v1's
`render_offside_dashboard.py` and its output are UNCHANGED and remain
reproducible exactly as before.

Attacking/defending direction: FIXED for the whole clip (matching v1's
own single-direction-per-render convention), chosen as attacking_team=1
/ defending_team=0 because that direction showed substantially more
trap-forming signal on this clip (899 vs. 377 TRAP_FORMING frames out
of 3600 -- see offside_v2_summary.md) -- a disclosed choice, not
cherry-picked for a nicer-looking result; the OTHER direction's
analytics are still computed and reported in the full-clip static
graphs (Part 4).

Usage:
    external/sports/.venv/bin/python offside_break/dashboard/render_offside_dashboard_v2.py \\
        --out_path offside_break/outputs/v2_120s/offside_dashboard_120s_v2.mp4
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

from offside_break.analytics.offside import (_forward_depth, compute_offside_line,  # noqa: E402
                                               offside_line_series)
from offside_break.analytics.offside_v2 import (compute_estimated_offside_position,  # noqa: E402
                                                  compute_line_relation_cm, classify_trap_break_candidates,
                                                  detect_attacking_runs, forward_sign_away_from_own_goal,
                                                  gated_ball_x, line_relation_label, trap_state_series)
from offside_break.dashboard.render_offside_dashboard import (ATTACKER_AHEAD_COLOR, LINE_COLOR, TEAM_COLOR,  # noqa: E402
                                                                 _draw_pitch_line_on_image, draw_right_top)
from pressing_structure.analytics.cleaned_tracking_view import build_cleaned_view, load_ball_view  # noqa: E402
from pressing_structure.analytics.pressing import _passing_lane_open, build_possession  # noqa: E402
from pressing_structure.dashboard.live_graphs import compute_stable_y_range, draw_dual_scrolling_graph, draw_scrolling_graph  # noqa: E402
from pressing_structure.dashboard.render_pressing_dashboard import _nearest_transformer  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
CONFIG = SoccerPitchConfiguration()
GRAPH_WINDOW_SEC = 8.0
TRAP_STATE_COLOR = {"NO_TRAP": (140, 140, 140), "TRAP_FORMING": (0, 165, 255),
                     "TRAP_ACTIVE": (0, 0, 255), "UNCERTAIN": (100, 100, 100)}
RUN_ARROW_COLOR = (0, 200, 0)


def draw_left_panel(frame, tracking_rows, attacking_team, defending_team, line_x, transformer,
                     lead_track_id, run_track_ids, trap_state):
    vis = frame.copy()
    for r in tracking_rows:
        if r["object_type"] not in ("player", "goalkeeper"):
            continue
        color = TEAM_COLOR.get(r["team_id"], (200, 200, 200))
        thickness, label = 2, None
        if r["team_id"] == attacking_team and r["track_id"] == lead_track_id:
            color, thickness, label = ATTACKER_AHEAD_COLOR, 3, "LEAD ATTACKER"
        if r["track_id"] in run_track_ids:
            thickness = 3
        x1, y1, x2, y2 = int(r["bbox_x1"]), int(r["bbox_y1"]), int(r["bbox_x2"]), int(r["bbox_y2"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(vis, label, (x1, max(12, y1 - 6)), FONT, 0.42, color, 2, cv2.LINE_AA)
        elif r["track_id"] in run_track_ids:
            cv2.putText(vis, "RUN", (x1, min(vis.shape[0] - 4, y2 + 16)), FONT, 0.4, RUN_ARROW_COLOR, 2, cv2.LINE_AA)
    if line_x is not None:
        vis = _draw_pitch_line_on_image(vis, transformer, line_x, LINE_COLOR, 3)
    cv2.rectangle(vis, (0, 0), (300, 60), (0, 0, 0), -1)
    txt1 = f"DEFENSIVE LINE: team {defending_team}" if line_x is not None else "LINE: insufficient data"
    cv2.putText(vis, txt1, (10, 22), FONT, 0.5, (255, 255, 255) if line_x is not None else (150, 150, 150), 2, cv2.LINE_AA)
    color = TRAP_STATE_COLOR.get(trap_state, (200, 200, 200))
    cv2.putText(vis, trap_state, (10, 48), FONT, 0.6, color, 2, cv2.LINE_AA)
    return vis


def draw_metrics_panel(trap_info, lead_feat, active_run, candidate_msg, w, h):
    panel = np.full((h, w, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "OFFSIDE-BREAK METRICS", (16, 30), FONT, 0.68, (20, 20, 20), 2, cv2.LINE_AA)
    y = 62
    lines = []
    if trap_info["valid"]:
        lines.append(f"Trap Synchrony Score: {trap_info['score']:.2f}  "
                      f"(stepping {trap_info['fraction_stepping_forward']:.0%} / "
                      f"depth-align {trap_info['depth_dispersion_score']:.2f} / "
                      f"vel-align {trap_info['velocity_alignment_score']:.2f})")
    else:
        lines.append(f"Trap Synchrony: n/a ({trap_info['reason']})")
    if lead_feat is not None:
        lines.append(f"Lead Attacker: track {lead_feat['track_id']}  "
                      f"Line Relation: {lead_feat['relation_cm']/100:+.1f} m ({lead_feat['relation_label']})")
        lines.append(f"Estimated Position: {lead_feat['est_state']}"
                      + ("" if lead_feat["ball_confident"] else "  [ball not confident this frame]"))
        lines.append(f"Relative Speed to Line: {lead_feat['closing_rate_cm_s']/100:+.1f} m/s" if lead_feat["closing_rate_cm_s"] is not None else "Relative Speed to Line: n/a")
        lines.append(f"Nearest Defender: {lead_feat['nearest_def_cm']/100:.1f} m" if lead_feat["nearest_def_cm"] is not None else "Nearest Defender: n/a")
        lines.append(f"Space Behind Line: {lead_feat['space_behind_line_cm']/100:.1f} m" if lead_feat["space_behind_line_cm"] is not None else "Space Behind Line: n/a")
        lines.append(f"Passing Lane From Carrier: {lead_feat['lane_status']}")
    else:
        lines.append("Lead Attacker: n/a (no confident line / attacker this frame)")
    if active_run is not None:
        lines.append(f"Active Run: track {active_run['attacker_track_id']}  type={active_run['run_type']}  "
                      f"{active_run['duration_sec']:.1f}s so far")
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (16, y + i * 26), FONT, 0.44, (30, 30, 30), 1, cv2.LINE_AA)
    y2 = y + len(lines) * 26 + 14
    cv2.line(panel, (16, y2), (w - 16, y2), (210, 210, 210), 1)
    y2 += 26
    cv2.putText(panel, "EVENT / OUTCOME", (16, y2), FONT, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    y2 += 26
    cv2.putText(panel, candidate_msg or "(no trap-break candidate active)", (16, y2), FONT, 0.46,
                (0, 0, 200) if candidate_msg else (120, 120, 120), 1 if not candidate_msg else 2, cv2.LINE_AA)
    return panel


def draw_timeline(trap_series_dicts, runs_list, candidates_list, cur_time, fps, w, h):
    bar = np.full((h, w, 3), 230, dtype=np.uint8)
    n_total = len(trap_series_dicts)
    span = max(n_total / fps, cur_time, 1)
    # trap-state color strip (thin band)
    for r in trap_series_dicts:
        x0 = int(r["frame"] / fps / span * w)
        x1 = max(int((r["frame"] + 1) / fps / span * w), x0 + 1)
        cv2.rectangle(bar, (x0, 6), (x1, 16), TRAP_STATE_COLOR.get(r["trap_state"], (180, 180, 180)), -1)
    # run episodes (green band)
    for run in runs_list:
        x0 = int(run["start_time_sec"] / span * w)
        x1 = max(int(run["end_time_sec"] / span * w), x0 + 2)
        cv2.rectangle(bar, (x0, 20), (x1, 28), RUN_ARROW_COLOR, -1)
    # break-candidate markers (red tick + outcome text would overlap at this scale -- tick only)
    for c in candidates_list:
        x0 = int(c["start_time_sec"] / span * w)
        cv2.line(bar, (x0, 4), (x0, h - 16), (0, 0, 220), 2)
    cx = int(cur_time / span * w)
    cv2.line(bar, (cx, 0), (cx, h), (0, 0, 0), 2)
    cv2.putText(bar, "TRAP STATE / RUN EPISODES / CANDIDATE MARKERS", (8, h - 4), FONT, 0.32, (40, 40, 40), 1, cv2.LINE_AA)
    return bar


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
    run_frame_index = {}  # frame -> [run dicts active this frame]
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

    # --- precompute full-clip histories for the live graphs ---
    print("Precomputing live-graph histories...", flush=True)
    hist_lead_depth, hist_line_depth, hist_trap_score, hist_closing_rate = [], [], [], []
    # Line velocity uses a 15-frame (0.5s) BASELINE difference, not a
    # 1-frame diff: the "second-last defender" pick can switch between
    # two players at noticeably different depths from one frame to the
    # next (sparse/noisy raw tracking, previously measured), and a
    # 1-frame diff multiplied by fps amplifies that into a physically
    # absurd instantaneous "velocity" (found during QA: a single-frame
    # jump produced a displayed +205.7 m/s). A 0.5s baseline is long
    # enough to average over that noise while still being short enough
    # to track genuine within-clip changes in line push/pace.
    LINE_VEL_BASELINE_FRAMES = 15
    line_depth_history = []  # (frame, depth) pairs, trimmed to the baseline window
    for f in range(n_total):
        t = f / fps
        line_depth = line_by_frame.get(f)
        hist_line_depth.append((t, (line_depth / 100.0) if line_depth is not None else None))
        players = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None and p["display_team_id"] == attacking_team
                   and p["display_object_type"] in ("player", "goalkeeper")]
        lead = min(players, key=lambda p: _forward_depth(defending_team, p["x_pitch"])) if players else None
        lead_depth = _forward_depth(defending_team, lead["x_pitch"]) if lead is not None else None
        hist_lead_depth.append((t, (lead_depth / 100.0) if lead_depth is not None else None))
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
                line_vel = (line_depth - d0) / dt  # cm/s, over up to 0.5s baseline
                closing_rate = line_vel + lead["vx_cm_s"] * sign
        hist_closing_rate.append((t, (closing_rate / 100.0) if closing_rate is not None else None))
    y_depth = compute_stable_y_range([v for _, v in hist_lead_depth] + [v for _, v in hist_line_depth])
    y_trap = (0.0, 1.0)
    y_closing = compute_stable_y_range([v for _, v in hist_closing_rate])

    left_w, left_h = 1144, 714
    sample_pitch = draw_pitch(config=CONFIG)
    right_w = int(round(left_h * sample_pitch.shape[1] / sample_pitch.shape[0]))
    graph_h = 150
    bottom_h = graph_h * 3
    timeline_h = 40
    canvas_w = left_w + right_w
    canvas_h = left_h + bottom_h + timeline_h

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
            lead_feat = {
                "track_id": lead["track_id"], "relation_cm": relation_cm,
                "relation_label": line_relation_label(relation_cm),
                "est_state": est["state"], "ball_confident": ball_x is not None,
                "closing_rate_cm_s": None, "nearest_def_cm": nearest_def,
                "space_behind_line_cm": line_info["line_depth_cm"], "lane_status": lane_status,
            }
            cr = hist_closing_rate[f][1]
            lead_feat["closing_rate_cm_s"] = cr * 100.0 if cr is not None else None

        active_runs_here = run_frame_index.get(f, [])
        active_run = active_runs_here[0] if active_runs_here else None
        run_track_ids = {r["attacker_track_id"] for r in active_runs_here}
        cand = cand_frame_index.get(f)
        candidate_msg = None
        if cand is not None:
            candidate_msg = f"TRAP-BREAK CANDIDATE: track {cand['attacker_track_id']} ({cand['run_type']}) -> {cand['break_outcome']}"

        transformer = _nearest_transformer(transformers, f)
        left = draw_left_panel(frame, tracking_by_frame.get(f, []), attacking_team, defending_team, line_x,
                                transformer, lead["track_id"] if lead else None, run_track_ids, trap_state)
        left = cv2.resize(left, (left_w, left_h))

        ball_row = ball_by_frame.get(f)
        ball_xy = (ball_row["x_pitch"], ball_row["y_pitch"]) if ball_row and ball_row.get("is_observed") and ball_row.get("x_pitch") is not None else None
        right_top = draw_right_top(players_this_frame, ball_xy, attacking_team, defending_team, line_x, line_info["reason"])
        right_top = cv2.resize(right_top, (right_w, left_h))

        g1 = draw_dual_scrolling_graph(hist_lead_depth, hist_line_depth, cur_time, GRAPH_WINDOW_SEC, left_w, graph_h,
                                        "ATTACKER DEPTH vs DEFENSIVE-LINE DEPTH", f"last {GRAPH_WINDOW_SEC:.0f}s",
                                        y_depth, "lead attacker", "defensive line", "m",
                                        color_a=ATTACKER_AHEAD_COLOR, color_b=LINE_COLOR)
        g2 = draw_scrolling_graph(hist_trap_score, cur_time, GRAPH_WINDOW_SEC, left_w, graph_h,
                                   "Trap Synchrony Score", f"last {GRAPH_WINDOW_SEC:.0f}s", y_trap, "0-1", (0, 130, 200))
        g3 = draw_scrolling_graph(hist_closing_rate, cur_time, GRAPH_WINDOW_SEC, left_w, graph_h,
                                   "Relative Closing Rate onto Line", f"last {GRAPH_WINDOW_SEC:.0f}s", y_closing, "m/s", (150, 0, 200))
        bottom_left = np.vstack([g1, g2, g3])
        bottom_right = draw_metrics_panel(trap_info, lead_feat, active_run, candidate_msg, right_w, bottom_h)

        timeline = draw_timeline(trap_series_dicts, runs_list, candidates_list, cur_time, fps, canvas_w, timeline_h)

        top = np.hstack([left, right_top])
        bottom = np.hstack([bottom_left, bottom_right])
        canvas = np.vstack([top, bottom, timeline])
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
