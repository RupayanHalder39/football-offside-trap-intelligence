"""
Offside/Trap dashboard V4 -- PREVIEW ONLY (Phase 10 of the 2026-09-07
audit continuation). Same locked V3 visual geometry and team-
distinguishable color scheme (Team 0 = cyan/red line, Team 1 =
magenta/amber line) as the approved V3 dashboard, but EVERY value comes
from the corrected V4 snapshot (`tactical_shared.snapshot.build_snapshot`,
merging `offside_v4.build_offside_v4` -- separate line/trap/run models,
causal per-run threat, both teams computed independently every frame)
-- never a separate V2 call for the timeline/candidates (B8's fix). The
radar's pitch-control wash uses the registration-corrected transform
(C6's fix).

Two modes: pass one or more `--frame`/`--name` pairs for individual
preview PNGs (the default), or `--out_path <file>.mp4` to render the
full clip (or `--start_frame`/`--end_frame`) to video, sequentially
(single seek, then cap.read() in order) for efficiency.
"""
import argparse
import os
import pickle
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import cv2
import numpy as np
import polars as pl

from pressing_structure.analytics.cleaned_tracking_view import build_cleaned_view, load_ball_view
from pressing_structure.dashboard.dashboard_style_v2 import (
    ACCENT_CYAN, ACCENT_GREEN, ACCENT_MAGENTA, ACCENT_ORANGE, ACCENT_RED, BG, FONT, TEXT_DIM,
    composite_state_badge, draw_big_dual_graph, draw_big_graph, draw_hatched_zone, draw_header, draw_icon_kpi_card,
    draw_insights_panel, draw_radar_legend_box, draw_team_table, draw_timeline_bucketed, panel_title,
)
from pressing_structure.dashboard.live_graphs import compute_stable_y_range
from pressing_structure.dashboard.render_pressing_dashboard import _nearest_transformer
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from tactical_shared.coordinates import DEFAULT_PITCH
from tactical_shared.radar_geometry import apply_registered_control_wash, pitch_xy_to_px

CONFIG = SoccerPitchConfiguration()
TEAM_COLOR = {0: ACCENT_CYAN, 1: ACCENT_MAGENTA}
TEAM_LINE_COLOR = {0: ACCENT_RED, 1: (0, 200, 255)}  # red vs amber/orange-yellow
TRAP_STATE_COLOR = {"NO_TRAP": (150, 150, 150), "TRAP_FORMING": ACCENT_ORANGE, "TRAP_ACTIVE": ACCENT_RED,
                     "TRAP_ENDING": (60, 200, 220), "UNCERTAIN": (90, 90, 90),
                     "TRAP_BREAK_THREAT": (0, 140, 255), "TRAP_BROKEN": (40, 40, 220), "CONTESTED": (200, 60, 220)}

HEADER_H, TOP_ROW_H, TILE_ROW_H, GRAPH_ROW_H, ROW3_H, TIMELINE_H, GAP = 60, 602, 88, 216, 154, 60, 6
CANVAS_W = 2304
LEFT_W = round(CANVAS_W * 0.52)
RIGHT_W = CANVAS_W - LEFT_W
assert HEADER_H + TOP_ROW_H + TILE_ROW_H + GRAPH_ROW_H + ROW3_H + TIMELINE_H + 4 * GAP == 1204


def draw_left_feed(frame, tracking_rows, snap_f, transformer):
    vis = frame.copy()
    for r in tracking_rows:
        if r["object_type"] not in ("player", "goalkeeper"):
            continue
        color = TEAM_COLOR.get(r["team_id"], (180, 180, 180))
        cv2.rectangle(vis, (int(r["bbox_x1"]), int(r["bbox_y1"])), (int(r["bbox_x2"]), int(r["bbox_y2"])), color, 2)
    dom = snap_f["offside"]["dominant_state"]
    team = snap_f["offside"]["dominant_team"]
    banner = dom.replace("_", " ").title()
    color = TRAP_STATE_COLOR.get(dom, (100, 100, 100))
    role = f"T0 trap: {snap_f['offside'][0]['state'].replace('_',' ').title()}   T1 trap: {snap_f['offside'][1]['state'].replace('_',' ').title()}"
    if team is not None:
        role += f"   (dominant: Team {team})"
    vis = composite_state_badge(vis, 10, 10, banner, color, role, w=460)
    out = np.full((vis.shape[0] + 32, vis.shape[1], 3), BG, dtype=np.uint8)
    out[32:, :] = vis
    panel_title(out, "TACTICAL MATCH FEED", sub="V4 evidence-aware trap/line state -- both teams", accent=ACCENT_RED)
    return out


def draw_radar(players_this_frame, ball_xy, snap_f, w, h):
    team_a = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"]} for p in players_this_frame if p["display_team_id"] == 0]
    team_b = [{"track_id": p["track_id"], "x_pitch": p["x_pitch"], "y_pitch": p["y_pitch"]} for p in players_this_frame if p["display_team_id"] == 1]
    pitch = draw_pitch(config=CONFIG)
    pitch_dark = cv2.addWeighted(pitch, 0.7, np.zeros_like(pitch), 0.3, 0)
    if team_a and team_b:
        from analytics.pitch_control import compute_pitch_control_grid
        pc = compute_pitch_control_grid([{**p, "vx": 0., "vy": 0.} for p in team_a], [{**p, "vx": 0., "vy": 0.} for p in team_b])
        if pc.get("valid"):
            pitch_dark = apply_registered_control_wash(pitch_dark, pc["grid"], TEAM_COLOR[0], TEAM_COLOR[1],
                                                         DEFAULT_PITCH.length_cm, DEFAULT_PITCH.width_cm, alpha=0.88)

    for team in (0, 1):
        line = snap_f["offside"][team]["line"]
        if not line["valid"]:
            continue
        goal_x = DEFAULT_PITCH.own_goal(team)
        x0, x1 = sorted([line["line_x"], goal_x])
        p0, p1 = pitch_xy_to_px(x0, 0), pitch_xy_to_px(x1, DEFAULT_PITCH.width_cm)
        pitch_dark = draw_hatched_zone(pitch_dark, p0, p1, TEAM_LINE_COLOR[team])
    for team in (0, 1):
        line = snap_f["offside"][team]["line"]
        if not line["valid"]:
            continue
        lp0, lp1 = pitch_xy_to_px(line["line_x"], 0), pitch_xy_to_px(line["line_x"], DEFAULT_PITCH.width_cm)
        cv2.line(pitch_dark, lp0, lp1, TEAM_LINE_COLOR[team], 3, cv2.LINE_AA)

    for p in players_this_frame:
        px, py = pitch_xy_to_px(p["x_pitch"], p["y_pitch"])
        color = TEAM_COLOR.get(p["display_team_id"], (150, 150, 150))
        cv2.circle(pitch_dark, (px, py), 8, color, -1, cv2.LINE_AA)
        cv2.circle(pitch_dark, (px, py), 8, (10, 10, 10), 2, cv2.LINE_AA)
        if p.get("vx_cm_s") is not None and p["display_team_id"] is not None:
            ex, ey = int(px + p["vx_cm_s"] * 0.03), int(py + p["vy_cm_s"] * 0.03)
            cv2.arrowedLine(pitch_dark, (px, py), (ex, ey), TEAM_COLOR[p["display_team_id"]], 2, cv2.LINE_AA, tipLength=0.3)
    if ball_xy is not None:
        bx, by = pitch_xy_to_px(*ball_xy)
        cv2.circle(pitch_dark, (bx, by), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(pitch_dark, (bx, by), 6, (15, 15, 15), 1, cv2.LINE_AA)

    out = np.full((h, w, 3), BG, dtype=np.uint8)
    resized = cv2.resize(pitch_dark, (w, h - 34))
    out[34:h, 0:w] = resized
    panel_title(out, "OFFSIDE / TRAP RADAR (V4)", sub="registration-corrected  •  both teams' estimated lines", accent=ACCENT_RED)
    roi = out[34:78, 8:280]
    cv2.addWeighted(np.zeros_like(roi), 0.55, roi, 0.45, 0, dst=roi)
    s0 = snap_f["offside"][0]["state"].replace("_", " ").title()
    s1 = snap_f["offside"][1]["state"].replace("_", " ").title()
    cv2.putText(out, f"T0 trap state: {s0}", (14, 50), FONT, 0.36, TEAM_LINE_COLOR[0], 1, cv2.LINE_AA)
    cv2.putText(out, f"T1 trap state: {s1}", (14, 70), FONT, 0.36, TEAM_LINE_COLOR[1], 1, cv2.LINE_AA)
    draw_radar_legend_box(out, w - 178, 44, 166, 210,
                           [("Team 0 / Run", TEAM_COLOR[0]), ("Team 1 / Run", TEAM_COLOR[1]), ("Ball", (255, 255, 255)),
                            ("T0 Line", TEAM_LINE_COLOR[0]), ("T1 Line", TEAM_LINE_COLOR[1])],
                           gradient=(TEAM_COLOR[0], TEAM_COLOR[1]), gradient_labels=("Team 0", "Control", "Team 1"))
    return out


def draw_tiles(snap_f, width, height):
    cell_w = (width - 5 * GAP) // 4
    img = np.full((height, width, 3), BG, dtype=np.uint8)
    o0, o1 = snap_f["offside"][0], snap_f["offside"][1]

    def _sub(o):
        # CURRENT (this-instant, raw) trap score shown separately from the
        # SUSTAINED (smoothed) score the state's own hysteresis reacts to
        # -- 2026-09-07 semantic-QA fix for the "Active but score reads
        # 0.00" apparent contradiction (a held ACTIVE/ENDING state can have
        # a low current-frame reading while the smoothed value that drives
        # the lifecycle is still above the exit threshold).
        if o["raw_score"] is None:
            return "no evidence"
        smoothed = o.get("score")
        smoothed_txt = f"{smoothed:.2f}" if smoothed is not None else "n/a"
        return f"now {o['raw_score']:.2f} / sustained {smoothed_txt} · conf {o['confidence']:.2f}"

    values = [
        ("Team 0 Trap State", o0["state"].replace("_", " ").title(), _sub(o0),
         TRAP_STATE_COLOR.get(o0["state"], ACCENT_CYAN), "shield"),
        ("Team 1 Trap State", o1["state"].replace("_", " ").title(), _sub(o1),
         TRAP_STATE_COLOR.get(o1["state"], ACCENT_MAGENTA), "shield"),
        ("Line Depth (T0 / T1)",
         f"{(o0['line']['line_depth_cm'] or 0)/100:.1f}m / {(o1['line']['line_depth_cm'] or 0)/100:.1f}m",
         f"conf {o0['line'].get('confidence',0):.2f} / {o1['line'].get('confidence',0):.2f}", ACCENT_ORANGE, "ruler"),
        ("Run Threat (T0 atk / T1 atk)",
         f"{(o0.get('run_threat_score') or 0):.2f} / {(o1.get('run_threat_score') or 0):.2f}",
         "causal, per-frame", ACCENT_GREEN, "bolt"),
    ]
    for i, (label, val, sub, accent, glyph) in enumerate(values):
        px = GAP + i * (cell_w + GAP)
        tile = draw_icon_kpi_card(cell_w, height, label, val, sub, accent=accent, glyph=glyph)
        img[0:height, px:px + cell_w] = tile
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_video_path", default="ExternalDownlaodVideo/testVideo1_120s.mp4")
    ap.add_argument("--tracking_dir", default="outputs/tracking/testVideo1_120s")
    ap.add_argument("--analytics_dir", default="outputs/analytics/testVideo1_120s_v3")
    ap.add_argument("--snapshot_pkl", default="audit_reports/2026-09-06_tactical_review_v4/v4_snapshot.pkl")
    ap.add_argument("--out_dir", default="offside_break/outputs/correctness_v4/dashboard_previews")
    ap.add_argument("--frame", type=int, action="append", default=None)
    ap.add_argument("--name", action="append", default=None)
    ap.add_argument("--out_path", default=None, help="if set, render the FULL clip to this .mp4 path instead of individual preview PNGs")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=None, help="inclusive; defaults to the last snapshot frame")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.snapshot_pkl, "rb") as fh:
        snapshot = pickle.load(fh)

    tracking = pl.read_parquet(os.path.join(args.tracking_dir, "tracking.parquet"))
    cleaned = build_cleaned_view(tracking)
    tracking_by_frame, cleaned_by_frame = {}, {}
    for r in tracking.to_dicts():
        tracking_by_frame.setdefault(r["frame"], []).append(r)
    for r in cleaned.to_dicts():
        cleaned_by_frame.setdefault(r["frame"], []).append(r)
    ball_by_frame = {r["frame"]: r for r in load_ball_view(args.analytics_dir).to_dicts()}
    with open(os.path.join(args.analytics_dir, "homography_transformers.pkl"), "rb") as fh:
        transformers = pickle.load(fh)
    cap = cv2.VideoCapture(args.source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Precomputed ONCE (video-render perf fix, not a semantic change --
    # depends only on the fixed `snapshot`, never on the current frame).
    n_buckets = 60
    bucket_sec = (len(snapshot) / fps) / n_buckets
    bucket_colors = []
    for b in range(n_buckets):
        bf0, bf1 = int(b * bucket_sec * fps), int(min(len(snapshot), (b + 1) * bucket_sec * fps))
        states = [snapshot[i]["offside"]["dominant_state"] for i in range(bf0, max(bf0 + 1, bf1))]
        majority = max(set(states), key=states.count) if states else "UNCERTAIN"
        bucket_colors.append(TRAP_STATE_COLOR.get(majority, (100, 100, 100)))
    legend = [(k.replace("_", " ").title(), v) for k, v in TRAP_STATE_COLOR.items()]

    def render_frame(f, seek=True):
        snap_f = snapshot[f]
        if seek:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            return None
        transformer = _nearest_transformer(transformers, f)
        players_this_frame = [p for p in cleaned_by_frame.get(f, []) if p["x_pitch"] is not None]
        ball_row = ball_by_frame.get(f)
        ball_xy = (ball_row["x_pitch"], ball_row["y_pitch"]) if ball_row and ball_row.get("is_observed") and ball_row.get("x_pitch") is not None else None

        left = cv2.resize(draw_left_feed(frame, tracking_by_frame.get(f, []), snap_f, transformer), (LEFT_W, TOP_ROW_H))
        radar = draw_radar(players_this_frame, ball_xy, snap_f, RIGHT_W, TOP_ROW_H)
        top_row = np.hstack([left, radar])

        header = draw_header(CANVAS_W, HEADER_H, "OFFSIDE / TRAP-BREAK ANALYTICS -- V4 CORRECTNESS PREVIEW",
                              (f"t = {snap_f['time_sec']:.1f}s", TEXT_DIM), accent=ACCENT_RED)
        tiles = draw_tiles(snap_f, CANVAS_W, TILE_ROW_H)

        window_sec = 14.0
        f0, f1 = max(0, f - int(window_sec * fps)), f
        hist_s0 = [(snapshot[i]["time_sec"], snapshot[i]["offside"][0]["raw_score"]) for i in range(f0, f1 + 1)]
        hist_s1 = [(snapshot[i]["time_sec"], snapshot[i]["offside"][1]["raw_score"]) for i in range(f0, f1 + 1)]
        hist_d0 = [(snapshot[i]["time_sec"], (snapshot[i]["offside"][0]["line"]["line_depth_cm"] or None) if snapshot[i]["offside"][0]["line"]["valid"] else None) for i in range(f0, f1 + 1)]
        hist_d1 = [(snapshot[i]["time_sec"], (snapshot[i]["offside"][1]["line"]["line_depth_cm"] or None) if snapshot[i]["offside"][1]["line"]["valid"] else None) for i in range(f0, f1 + 1)]
        graph_w = (CANVAS_W - 4 * GAP) // 3
        g1 = draw_big_dual_graph(graph_w, GRAPH_ROW_H, hist_s0, hist_s1, snap_f["time_sec"], window_sec,
                                  "Trap Score: Team 0 vs Team 1", (0.0, 1.0), "Team 0", "Team 1", "",
                                  color_a=TEAM_LINE_COLOR[0], color_b=TEAM_LINE_COLOR[1])
        y_depth = compute_stable_y_range([v / 100.0 if v is not None else None for _, v in hist_d0 + hist_d1])
        hist_d0_m = [(t, v / 100.0 if v is not None else None) for t, v in hist_d0]
        hist_d1_m = [(t, v / 100.0 if v is not None else None) for t, v in hist_d1]
        g2 = draw_big_dual_graph(graph_w, GRAPH_ROW_H, hist_d0_m, hist_d1_m, snap_f["time_sec"], window_sec,
                                  "Estimated Defensive Line Depth", y_depth, "Team 0", "Team 1", "m",
                                  color_a=TEAM_LINE_COLOR[0], color_b=TEAM_LINE_COLOR[1])
        hist_rt0 = [(snapshot[i]["time_sec"], snapshot[i]["offside"][0].get("run_threat_score")) for i in range(f0, f1 + 1)]
        hist_rt1 = [(snapshot[i]["time_sec"], snapshot[i]["offside"][1].get("run_threat_score")) for i in range(f0, f1 + 1)]
        g3 = draw_big_dual_graph(graph_w, GRAPH_ROW_H, hist_rt0, hist_rt1, snap_f["time_sec"], window_sec,
                                  "Run Threat (as attacker): Team 0 vs Team 1", (0.0, 1.0), "Team 0", "Team 1", "",
                                  color_a=ACCENT_CYAN, color_b=ACCENT_MAGENTA)
        gap_col = np.full((GRAPH_ROW_H, GAP, 3), BG, dtype=np.uint8)
        graph_row = np.hstack([g1, gap_col, g2, gap_col, g3])
        if graph_row.shape[1] != CANVAS_W:
            graph_row = cv2.resize(graph_row, (CANVAS_W, GRAPH_ROW_H))

        insights = [
            "V4 separates line / trap-intent / run-threat as independent per-team models (they are NOT the same "
            "concept): a valid estimated line does not require an active trap, and vice versa.",
            "Trap requires a real, causally-estimated defensive-unit forward VELOCITY (not just depth alignment) -- "
            "a stationary or retreating line scores 0 by construction, verified by fixture tests.",
            "Confirming TRAP_ACTIVE requires a GENERIC, non-clip-tuned bar (>=2 observations, >=0.25s real support, "
            ">=1/3 coverage of the forming window) after the FSM's exact-unit-identity context reset (previously the "
            "dominant, unjustified cause of early termination) was fixed. 0/13 real episodes this clip currently "
            "clear that bar -- reported honestly, not lowered to manufacture an active trap.",
        ]
        insights_panel = draw_insights_panel(graph_w * 2 + GAP, ROW3_H, "V4 Notes (data-derived)", insights, accent=ACCENT_ORANGE)
        table = draw_team_table(graph_w - GAP, ROW3_H, "Team Comparison (V4)",
                                 ["State", "Score", "Line (m)"],
                                 [("Team 0", [snap_f["offside"][0]["state"].replace("_", " ").title(),
                                              f"{(snap_f['offside'][0]['raw_score'] or 0):.2f}",
                                              f"{(snap_f['offside'][0]['line']['line_depth_cm'] or 0)/100:.1f}"]),
                                  ("Team 1", [snap_f["offside"][1]["state"].replace("_", " ").title(),
                                              f"{(snap_f['offside'][1]['raw_score'] or 0):.2f}",
                                              f"{(snap_f['offside'][1]['line']['line_depth_cm'] or 0)/100:.1f}"])],
                                 [TEAM_COLOR[0], TEAM_COLOR[1]])
        row3 = np.hstack([table, np.full((ROW3_H, GAP, 3), BG, dtype=np.uint8), insights_panel])
        if row3.shape[1] != CANVAS_W:
            row3 = cv2.resize(row3, (CANVAS_W, ROW3_H))

        timeline = draw_timeline_bucketed(CANVAS_W, TIMELINE_H, len(snapshot) / fps, snap_f["time_sec"],
                                           bucket_colors, bucket_sec, [], legend, title="DOMINANT TRAP STATE (V4, both teams)")

        row_gap = np.full((GAP, CANVAS_W, 3), BG, dtype=np.uint8)
        canvas = np.vstack([header, top_row, row_gap, tiles, row_gap, graph_row, row_gap, row3, row_gap, timeline])
        return canvas

    if args.out_path:
        f_start = args.start_frame
        f_end = args.end_frame if args.end_frame is not None else len(snapshot) - 1
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_path, fourcc, fps, (CANVAS_W, 1204))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)  # seek ONCE, then read sequentially
        n_written = 0
        t0 = time.time()
        for f in range(f_start, f_end + 1):
            canvas = render_frame(f, seek=False)
            if canvas is None:
                print(f"stopped at frame {f}: source video ended early")
                break
            writer.write(canvas)
            n_written += 1
            if n_written % 300 == 0:
                elapsed = time.time() - t0
                print(f"...{n_written} frames written ({f/fps:.1f}s of {(f_end-f_start+1)/fps:.1f}s) "
                      f"in {elapsed:.1f}s ({n_written/max(elapsed,1e-6):.1f} fps)", flush=True)
        writer.release()
        cap.release()
        print(f"Wrote {args.out_path}: {n_written} frames, canvas {CANVAS_W}x1204, {time.time()-t0:.1f}s total")
        return

    for f, name in zip(args.frame or [], args.name or []):
        canvas = render_frame(f)
        if canvas is None:
            print(f"frame {f}: could not read video frame")
            continue
        out_path = os.path.join(args.out_dir, f"{name}.png")
        cv2.imwrite(out_path, canvas)
        print(f"wrote {out_path} ({canvas.shape[1]}x{canvas.shape[0]})")
    cap.release()


if __name__ == "__main__":
    main()
