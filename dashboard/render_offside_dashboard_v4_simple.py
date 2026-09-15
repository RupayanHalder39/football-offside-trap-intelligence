"""
Offside/Trap dashboard V4 -- FINAL simplified broadcast layout (2026-09-07
implementation pass). Same V4 data source as `render_offside_dashboard_v4.py`
(`tactical_shared.snapshot.build_snapshot` / `offside_v4.build_offside_v4`,
separate line/trap/run models, evidence-aware FSM, both teams computed
independently every frame) and the same 2304x1204 canvas / video-output
compatibility. This is the VISUAL VALIDATION layer for Paper 2
(Offside-Trap Recognition & Trap-Break Optimization) -- it answers "where
is the line, is it moving, are defenders stepping together, is an
attacker approaching it?", not the paper's scientific contribution itself.

LAYOUT: exactly two rows.
  ROW 1: tactical video feed (left) + offside/trap radar (right), both
         teams' estimated lines shown simultaneously, one compact state
         badge on the video only.
  ROW 2: exactly four graphs -- Defensive Unit Centre Depth, Trap
         Synchrony Score, Relevant Attacker Distance to Estimated
         Offside Reference (signed, zero reference line, never stitched
         across different attackers), Defensive-Line Forward Velocity
         (signed, zero reference line). Every graph carries a large
         title, an explicit X-axis label ("Match Time (seconds)"), an
         explicit Y-axis label with units, a one-line interpretation
         subtitle, and readable absolute-time tick labels.

TWO TIME MODES:
  - "rolling" (default; what the final --out_path video uses): a
    trailing 20s window, tick labels at ABSOLUTE match-second values,
    current-time cursor at the right edge.
  - "paper": the full 0-<duration>s clip, ticks every 30s -- for static
    paper-figure/report use.
  Both modes only ever plot data up to the current frame (causal) and
  preserve real gaps -- no interpolation, no forward-fill, no event or
  attacker stitching, ever.

Kept as a SEPARATE file from `render_offside_dashboard_v4.py` (not an
in-place edit): that script produced the already-approved canonical
120s render and must stay reproducible exactly as before. No analytics
logic changes -- every value here is read as-is from the frozen V4
snapshot (plus one disclosed RENDER-SIDE selection rule for "which
runner is the relevant attacker this frame"). "Team 0"/"Team 1" internal
indices are UNCHANGED -- only the user-facing TEXT/COLOR is Team A
(green) / Team B (gold-yellow).
"""
import argparse
import math
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
    ACCENT_RED, BG, BORDER_SOFT, FONT, GRID, TEXT_DIM, TEXT_WHITE, composite_state_badge, draw_header, panel_frame,
)
from pressing_structure.dashboard.render_pressing_dashboard import _nearest_transformer
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration
from tactical_shared.coordinates import DEFAULT_PITCH
from tactical_shared.radar_geometry import pitch_xy_to_px, playing_rect_px
from tactical_shared.tracking import build_quality_view, current_roles
from offside_break.analytics.offside_v4 import build_offside_v4
from offside_break.analytics.pass_release_v9 import (
    PasserCandidateTracker, detect_release, build_pass_run_events_v9, associate_pass_to_runner,
    detect_runner_crossing, effective_reference_at_release, timing_margin_sec, PASSER_PREROLL_SEC,
)
# 2026-09-09 -- COACH-MODE second-row graphs (per
# audit_reports/2026-09-09_offside_graph_feasibility/). New, additive,
# presentation-only helpers -- no existing offside analytics is touched.
from offside_break.analytics.coach_trap_signal import cumulative_funnel_counts, nice_bar_axis, FUNNEL_STAGES
# 2026-09-08 -- FINAL VISUALIZATION UPGRADE: shared perspective-radar
# camera/projection helper (rendering only). OFFSIDE analytics/graph
# semantics are unchanged this round -- only how the radar is DRAWN.
from tactical_shared.perspective_radar import (
    RadarCameraConfig, DEFAULT_RADAR_CAMERA, compute_radar_homography, project_point_int,
    draw_pitch_markings_perspective, marker_scale_factor, warp_raster_perspective,
)
# 2026-09-08 V9.1 Part 9 -- PRESSING IS LOCKED this round ("do NOT modify
# pressing analytics/visualization/thresholds"). This import is READ-ONLY
# reuse of the already-existing, unchanged `behind_line_threat_field`/
# `pitch_grid` functions (added to pressure_field.py back in V8 for
# exactly this purpose) -- no pressing file is edited by this round's work.
from pressing_structure.analytics.pressure_field import behind_line_threat_field, pitch_grid

CONFIG = SoccerPitchConfiguration()

# ---------------------------------------------------------------------------
# Simple football palette -- Team A = green, Team B = gold/yellow, ball =
# white, IDENTICAL across video, radar (players AND lines), and graphs.
# ---------------------------------------------------------------------------
TEAM_A_COLOR = (50, 180, 50)     # BGR -- green
TEAM_B_COLOR = (10, 200, 235)    # BGR -- gold / yellow
BALL_COLOR = (255, 255, 255)
TEAM_COLOR = {0: TEAM_A_COLOR, 1: TEAM_B_COLOR}
TEAM_LINE_COLOR = TEAM_COLOR     # same hue for a team's line as for its players
TEAM_LETTER = {0: "A", 1: "B"}   # UI-label-only remap

RUN_APPROACH_COLOR = (0, 140, 255)   # orange -- RUN_APPROACHING_LINE
RUN_BREAK_COLOR = (30, 30, 230)      # red    -- BREAK_THREAT
SELECTED_RING_COLOR = (255, 255, 255)   # white -- the ONE Graph-3-selected relevant attacker
CROSSING_MARK_COLOR = (255, 255, 255)   # white -- geometric line-crossing reference (never "offside")

# 2026-09-07 V5 pass -- three structure lines per team (A1): front/first
# (light, dashed), last/deepest (bright, solid), reference/offside line
# (the pre-existing defs[1] line -- unchanged value, distinct dotted style).
# 2026-09-07 V6 terminology/styling (1C/1G): three DISTINCT football
# objects, never conflated -- LINE 1 back-line front (dashed, light),
# LINE 2 deepest outfield defender (thin solid, medium), LINE 3
# estimated offside reference (thick solid, BRIGHTEST -- the pre-
# existing, unchanged defs[1] value). LINE 1/2 come from the NEW
# `back_line` cluster; LINE 3 stays exactly the old `line` field.
# 2026-09-08 V8 -- visual hierarchy fix (O1/O7 of the offside-refinement
# pass): the DEFENSIVE UNIT CENTRE is now the visually DOMINANT line
# (thick, brightest -- it is the PRIMARY trap-analysis reference); the
# DEEPEST OUTFIELD line stays thin/secondary; the ESTIMATED OFFSIDE
# REFERENCE is drawn DASHED so it reads as a clearly distinct, separate
# concept rather than competing for boldness with the unit centre (the
# prior pass had this backwards -- reference was the boldest line).
CENTRE_LINE_COLOR = {0: (20, 230, 20), 1: (0, 190, 255)}         # BRIGHT green / BRIGHT gold, thick solid -- PRIMARY
DEEP_LINE_COLOR = {0: (50, 180, 50), 1: (10, 200, 235)}          # medium green / gold, thin solid -- SECONDARY
REFERENCE_LINE_COLOR = {0: (140, 225, 140), 1: (150, 225, 245)}  # light green / light gold, DASHED -- clearly distinct
UNIT_RING_COLOR = (255, 255, 255)        # plain unit-member outline (no stepping evidence)
STEPPING_RING_COLOR = {0: (20, 230, 20), 1: (0, 190, 255)}    # coordinated-step highlight (reuses reference-line bright hue)
CANDIDATE_RING_COLOR = (150, 150, 150)   # A6: lighter ring -- candidate (not yet confirmed) member

# 2026-09-08 V9 -- passer/release/runner-timing objects (item "CRITICAL
# NEXT ALGORITHM: PASSER + PASS RELEASE"). Distinct from every existing
# hue above so these never read as a team color or an existing line.
PASSER_RING_COLOR = (255, 230, 0)     # cyan -- the ONE evidence-scored passer candidate
RELEASE_MARKER_COLOR = (255, 255, 255)  # white "x" -- release instant, brief on-screen persistence only
PASS_VECTOR_COLOR = (255, 230, 0)     # same cyan as the passer ring -- one continuous concept
RELEASE_MARKER_PERSIST_SEC = 0.3
DEEP_TICK_LEN_PX = 16                 # (V9) normal-mode DEEPEST OUTFIELD marker -- a short tick, not a full-pitch line

TRAP_STATE_COLOR = {"NO_TRAP": (150, 150, 150), "TRAP_FORMING": (0, 140, 255), "TRAP_ACTIVE": (30, 30, 230),
                     "TRAP_ENDING": (60, 200, 220), "UNCERTAIN": (90, 90, 90),
                     "TRAP_BREAK_THREAT": (0, 140, 255), "TRAP_BROKEN": (40, 40, 220), "CONTESTED": (30, 100, 180)}

# ---------------------------------------------------------------------------
# 2026-09-09 FINAL COACH-DASHBOARD GRAPH POLISH -- second row reduced to 3
# graphs (Behind-Line Space removed from the COACH LAYOUT only -- its
# analytics/precompute stay in `main()` unused), Graph 1 restyled to
# STRUCTURAL (not team) colors, Graph 2 (Margin) restyled with a clearer
# zero line + single-accent-or-team-split logic, Graph 3 (Funnel) restyled
# with distinct per-stage bar colors, and the graph PANEL background is
# lightened. Every constant below is NEW and offside-local -- none of the
# shared, pressing-owned `dashboard_style_v2.py` constants (BG, GRID,
# BORDER_SOFT, TEXT_DIM, ...) are modified; this file only ever calls the
# SHARED `panel_frame()` with different bg/border ARGUMENTS (that function
# already accepts them), which changes nothing under pressing_structure/.
GRAPH_PANEL_BG = (110, 95, 78)        # BGR -- lighter desaturated navy/slate-blue graph-panel fill (vs pressing's near-black CARD_BG)
GRAPH_PANEL_BORDER = (170, 150, 125)  # BGR -- brighter panel/plot-area border for the lighter panel
GRAPH_GRID = (78, 66, 52)             # BGR -- subtly darker than the panel fill: visible gridlines, still subtle
GRAPH_TICK_TEXT = (225, 218, 205)     # BGR -- higher-contrast tick/axis/legend/subtitle text than shared TEXT_DIM
ZERO_LINE_COLOR = (0, 225, 255)       # BGR -- bright amber zero/reference line (clearly distinct from the white NOW cursor)

# Graph 1 (Defensive Line Dynamics): STRUCTURAL colors -- one per REAL
# object (Unit Centre / Estimated Reference / Deepest Defender), the SAME
# color regardless of which team the series belongs to, so the three
# structural references are what the color distinguishes (not team
# identity -- team identity is instead a tiny "A"/"B" badge near the NOW
# cursor, see `draw_line_dynamics_graph`).
STRUCTURE_COLOR = {
    "unit_centre": (255, 220, 40),   # bright cyan / electric blue, solid
    "reference": (255, 255, 255),    # bright white, dashed
    "deepest": (225, 60, 235),       # magenta, dotted
}

# Graph 2 (Most-Threatening-Attacker Margin): one strong single-accent
# color used when only ONE attacking team has real data in the current
# rolling window (the common case) -- team-split coloring (TEAM_COLOR) is
# still used, unchanged, whenever BOTH teams genuinely have real samples
# in-window at once (see `draw_lead_margin_graph`).
MARGIN_ACCENT_COLOR = (0, 165, 255)   # BGR -- bright orange, single-accent margin line (clearly distinct from Team A green / Team B gold, unlike a lime green would be)

# Graph 3 (Trap-Break Event Funnel): distinct per-stage bar colors --
# never team colors (checked explicitly in tests below).
FUNNEL_STAGE_COLOR = {
    "Pass Releases": (235, 175, 40),         # blue/cyan
    "Runner Associations": (220, 90, 150),   # violet
    "Observed Crossings": (0, 140, 255),     # orange
    "Successful Breaks": (70, 215, 90),      # green
    "Failed Breaks": (90, 60, 235),          # red/pink
}

# ROW1 (video+radar) kept at the exact same height as the approved V4
# correctness-preview dashboard; Row 2 is one taller graph strip.
HEADER_H, TOP_ROW_H, GRAPH_ROW_H, GAP = 60, 602, 536, 6
CANVAS_W = 2304
LEFT_W = round(CANVAS_W * 0.52)
RIGHT_W = CANVAS_W - LEFT_W
assert HEADER_H + TOP_ROW_H + GRAPH_ROW_H + GAP == 1204


def _project_pitch_points(transformer, pts_cm):
    """Pitch (cm) -> broadcast image (px) projection, via the SAME
    inverse-homography technique already used by
    `render_pressing_dashboard._draw_pitch_circle_on_image` (not
    reinvented here). Returns None if the transformer is missing or
    singular for this frame -- callers must then skip drawing rather
    than fabricate a broadcast overlay ("if a relationship cannot be
    projected reliably into broadcast video, show it only in radar")."""
    if transformer is None:
        return None
    try:
        inv_m = np.linalg.inv(transformer.m)
    except np.linalg.LinAlgError:
        return None
    pts = np.array(pts_cm, dtype=np.float32).reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(pts, inv_m).reshape(-1, 2)
    return img_pts


def _dashed_line(img, p0, p1, color, thickness=2, dash_len=14, gap_len=10):
    """Straight dashed segment from p0 to p1 -- cv2 has no native dashed
    line primitive. Used ONLY for the front/first-defender line, per A5's
    'light ... dashed' styling; last-defender and reference lines stay solid."""
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    total = float(np.linalg.norm(p1 - p0))
    if total < 1e-6:
        return
    direction = (p1 - p0) / total
    d = 0.0
    while d < total:
        seg_end = min(d + dash_len, total)
        a = tuple(np.round(p0 + direction * d).astype(int))
        b = tuple(np.round(p0 + direction * seg_end).astype(int))
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
        d += dash_len + gap_len


def _clipped_line(img, p0, p1, color, thickness, dashed=False):
    """cv2.clipLine (standard Liang-Barsky-family clipping, not
    reinvented) restricts the projected pitch-space line to the
    VISIBLE FRAME bounds (1H: 'clip correctly to the visible pitch
    polygon; do NOT draw lines outside the visible field') -- skips
    drawing entirely if the projected segment never intersects the
    frame at all."""
    h, w = img.shape[:2]
    ok, c0, c1 = cv2.clipLine((0, 0, w, h), p0, p1)
    if not ok:
        return
    if dashed:
        _dashed_line(img, c0, c1, color, thickness)
    else:
        cv2.line(img, c0, c1, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 2026-09-08 V9.1 Part 9 -- BEHIND-LINE THREAT wash, subtle and SECONDARY to
# every tactical line/player. Same low-alpha colormap-wash TECHNIQUE the
# (locked, unmodified) pressing dashboard already uses -- not a copy of its
# code, a fresh, small implementation living entirely in this file so
# nothing pressing-side is touched. low=near-transparent, medium=yellow/
# orange, high=red; capped alpha so it can never saturate the screen.
THREAT_ALPHA_MAX = 0.55
THREAT_COLORMAP = cv2.COLORMAP_HOT
THREAT_GAMMA = 0.5  # sqrt-boost so genuinely real but modest (~0.2-0.4) field values stay visibly distinct from zero


def _threat_field_to_color_alpha(field):
    """field in [0,1) -> (bgr_color_map_uint8, alpha_uint8) both shaped
    like `field`. Values below a small floor are fully transparent
    (never a faint wash over the ENTIRE pitch). A mild gamma keeps this
    STILL secondary to every tactical line/player (capped alpha, hot-map
    low end reads dark), while remaining visible enough to review --
    `behind_line_threat_field`'s own values are typically well under 1.0
    even in a genuinely dangerous pocket of space (see V8/V9 audit)."""
    raw = np.clip(field, 0.0, 1.0)
    norm = raw ** THREAT_GAMMA
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), THREAT_COLORMAP)
    alpha = norm * THREAT_ALPHA_MAX
    alpha[raw < 0.03] = 0.0
    return color, alpha


def apply_behind_line_threat_wash_radar(pitch_img, field):
    if field is None or float(field.max()) <= 0.0:
        return pitch_img
    x0, y0, x1, y1 = playing_rect_px(DEFAULT_PITCH.length_cm, DEFAULT_PITCH.width_cm)
    color, alpha = _threat_field_to_color_alpha(field)
    color_up = cv2.resize(color, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    alpha_up = cv2.resize(alpha, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)[..., None]
    out = pitch_img.copy()
    region = out[y0:y1, x0:x1].astype(np.float32)
    region = region * (1 - alpha_up) + color_up.astype(np.float32) * alpha_up
    out[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
    return out


def apply_behind_line_threat_wash_video(frame, field, transformer, grid_step_cm=200.0):
    """Projects the IDENTICAL pitch-space field into the broadcast frame
    through the frame's own valid homography (same `_project_pitch_points`
    inverse-homography technique every other video overlay in this file
    uses) -- never a second, divergent model. Returns `frame` unchanged
    if the transformer is missing/singular (never fabricate a broadcast
    overlay from an invalid registration)."""
    if field is None or transformer is None or float(field.max()) <= 0.0:
        return frame
    try:
        inv_m = np.linalg.inv(transformer.m)
    except np.linalg.LinAlgError:
        return frame
    color, alpha = _threat_field_to_color_alpha(field)
    grid_to_pitch = np.array([[grid_step_cm, 0, 0], [0, grid_step_cm, 0], [0, 0, 1]], dtype=np.float64)
    m = inv_m @ grid_to_pitch
    h, w = frame.shape[:2]
    warped_color = cv2.warpPerspective(color, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_alpha = cv2.warpPerspective(alpha, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)[..., None]
    out = frame.astype(np.float32) * (1 - warped_alpha) + warped_color.astype(np.float32) * warped_alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_behind_line_threat_wash_radar_perspective(pitch_img, field, H, grid_step_cm=200.0):
    """(Perspective radar) SAME `_threat_field_to_color_alpha` color/
    alpha this project already computes for the flat radar and the
    broadcast video, warped through the SAME shared
    `compute_radar_homography()` matrix every other perspective-radar
    object uses -- never a second, divergent color/alpha model."""
    if field is None or float(field.max()) <= 0.0:
        return pitch_img
    color, alpha = _threat_field_to_color_alpha(field)
    h, w = pitch_img.shape[:2]
    warped_color, warped_alpha = warp_raster_perspective(color, alpha, H, grid_step_cm, w, h)
    out = pitch_img.astype(np.float32) * (1 - warped_alpha[..., None]) + warped_color * warped_alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_structure_lines_on_image(vis, transformer, snap_f, players_this_frame=None, debug=False):
    """(1H, V7 A4): the SAME three pitch-space lines the radar draws --
    (1) DEFENSIVE-UNIT CENTRE (solid, primary -- robust median x of the
    NEW temporal-membership confirmed unit, 2026-09-08 V7; the PRIMARY
    reference for trap movement), (2) DEEPEST OUTFIELD DEFENDER (thin
    solid; rear member of that SAME confirmed unit), (3) ESTIMATED
    OFFSIDE REFERENCE (thick solid, brightest; the pre-existing,
    UNCHANGED `line` defs[1] value, for attacker legality only) --
    projected via the frame's own inverse homography, then clipped to
    the visible frame. Never conflated; skipped entirely (per team, per
    line) when the transformer is invalid or that field is None.

    2026-09-08 V9: DEEPEST OUTFIELD is now a full-pitch line only in
    `debug=True` (research) mode -- normal-mode broadcast video instead
    gets a short marker/tick at the responsible player's own position
    (see `players_this_frame`), per this round's explicit instruction
    not to clutter the clean video with a 4th full-pitch line."""
    by_track = {p["track_id"]: p for p in (players_this_frame or [])}
    for team in (0, 1):
        line = snap_f["offside"][team]["line"]
        back_line = snap_f["offside"][team].get("back_line") or {}
        if back_line.get("centre_x") is not None:
            pts = _project_pitch_points(transformer, [(back_line["centre_x"], 0), (back_line["centre_x"], DEFAULT_PITCH.width_cm)])
            if pts is not None:
                _clipped_line(vis, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[1]).astype(int)),
                              CENTRE_LINE_COLOR[team], 3)
        if back_line.get("deep_x") is not None:
            if debug:
                pts = _project_pitch_points(transformer, [(back_line["deep_x"], 0), (back_line["deep_x"], DEFAULT_PITCH.width_cm)])
                if pts is not None:
                    _clipped_line(vis, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[1]).astype(int)),
                                  DEEP_LINE_COLOR[team], 1)
            else:
                deep_p = by_track.get(back_line.get("deep_track_id"))
                if deep_p is not None:
                    pts = _project_pitch_points(transformer, [(deep_p["x_pitch"], deep_p["y_pitch"] - 400),
                                                               (deep_p["x_pitch"], deep_p["y_pitch"] + 400)])
                    if pts is not None:
                        _clipped_line(vis, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[1]).astype(int)),
                                      DEEP_LINE_COLOR[team], 2)
        if line["valid"]:
            pts = _project_pitch_points(transformer, [(line["line_x"], 0), (line["line_x"], DEFAULT_PITCH.width_cm)])
            if pts is not None:
                _clipped_line(vis, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[1]).astype(int)),
                              REFERENCE_LINE_COLOR[team], 2, dashed=True)


def draw_left_feed(frame, tracking_rows, snap_f, transformer, selected_runner_track_id,
                    players_this_frame=None, debug=False, passer=None, release_active=False,
                    pass_vector_end_xy=None, threat_field=None, debug_text_lines=None):
    vis = frame.copy()
    # (Part 9) behind-line threat wash drawn FIRST -- strictly secondary
    # to every tactical line/player drawn on top of it.
    vis = apply_behind_line_threat_wash_video(vis, threat_field, transformer)
    for r in tracking_rows:
        if r["object_type"] not in ("player", "goalkeeper"):
            continue
        color = TEAM_COLOR.get(r["team_id"], (180, 180, 180))
        cv2.rectangle(vis, (int(r["bbox_x1"]), int(r["bbox_y1"])), (int(r["bbox_x2"]), int(r["bbox_y2"])), color, 2)
        if debug and r.get("track_id") is not None:  # (Part 11) debug-only track IDs
            cv2.putText(vis, str(r["track_id"]), (int(r["bbox_x1"]), int(r["bbox_y1"]) - 4), FONT, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    _draw_structure_lines_on_image(vis, transformer, snap_f, players_this_frame, debug)

    # (4) relevant attacker highlight -- same track_id Graph 3 uses,
    # exact bbox match, no homography needed for the player marker itself.
    if selected_runner_track_id is not None:
        row = next((r for r in tracking_rows if r["track_id"] == selected_runner_track_id), None)
        if row is not None:
            cx = int((row["bbox_x1"] + row["bbox_x2"]) / 2)
            cy = int(row["bbox_y2"])
            rw = max(24, int((row["bbox_x2"] - row["bbox_x1"]) * 0.9))
            cv2.ellipse(vis, (cx, cy), (rw, max(8, rw // 3)), 0, 0, 360, SELECTED_RING_COLOR, 2, cv2.LINE_AA)

    # (V9) passer candidate -- distinct cyan ring, SAME track_id used by
    # the radar's passer marker (never re-derived per-surface).
    passer_row = None
    if passer is not None and passer.get("passer_track_id") is not None:
        passer_row = next((r for r in tracking_rows if r["track_id"] == passer["passer_track_id"]), None)
        if passer_row is not None:
            cx = int((passer_row["bbox_x1"] + passer_row["bbox_x2"]) / 2)
            cy = int(passer_row["bbox_y2"])
            rw = max(22, int((passer_row["bbox_x2"] - passer_row["bbox_x1"]) * 0.85))
            cv2.ellipse(vis, (cx, cy), (rw, max(7, rw // 3)), 0, 0, 360, PASSER_RING_COLOR, 2, cv2.LINE_AA)
            if release_active:
                cv2.drawMarker(vis, (cx, cy - rw), RELEASE_MARKER_COLOR, cv2.MARKER_TILTED_CROSS, 16, 2)

    # (V9) pass vector -- projected via the SAME inverse-homography
    # helper every other pitch-space video overlay uses; only drawn once
    # a passer AND a release candidate exist (never speculative before that).
    if passer_row is not None and pass_vector_end_xy is not None and transformer is not None:
        p_row = next((p for p in (players_this_frame or []) if p.get("track_id") == passer["passer_track_id"]), None)
        if p_row is not None and p_row.get("x_pitch") is not None:
            pts = _project_pitch_points(transformer, [(p_row["x_pitch"], p_row["y_pitch"]), pass_vector_end_xy])
            if pts is not None:
                a = tuple(np.round(pts[0]).astype(int))
                b = tuple(np.round(pts[1]).astype(int))
                h, w = vis.shape[:2]
                ok, c0, c1 = cv2.clipLine((0, 0, w, h), a, b)
                if ok:
                    cv2.arrowedLine(vis, c0, c1, PASS_VECTOR_COLOR, 2, cv2.LINE_AA, tipLength=0.08)

    dom = snap_f["offside"]["dominant_state"]
    team = snap_f["offside"]["dominant_team"]
    banner = dom.replace("_", " ").title()
    color = TRAP_STATE_COLOR.get(dom, (100, 100, 100))
    role = (f"Team A: {snap_f['offside'][0]['state'].replace('_',' ').title()}   "
            f"Team B: {snap_f['offside'][1]['state'].replace('_',' ').title()}")
    if team is not None:
        role += f"   (dominant: Team {TEAM_LETTER[team]})"
    vis = composite_state_badge(vis, 10, 10, banner, color, role, w=460)
    out = np.full((vis.shape[0] + 32, vis.shape[1], 3), BG, dtype=np.uint8)
    out[32:, :] = vis
    cv2.rectangle(out, (12, 8), (15, 24), ACCENT_RED, -1)
    cv2.putText(out, "TACTICAL MATCH FEED", (22, 22), FONT, 0.46, TEXT_WHITE, 2, cv2.LINE_AA)
    # (Part 11) debug-only numerical readout -- release evidence, effective-
    # reference source, timing margin, lifecycle label. Never in normal mode.
    if debug and debug_text_lines:
        ty = out.shape[0] - 14 - 16 * (len(debug_text_lines) - 1)
        for line in debug_text_lines:
            cv2.putText(out, line, (14, ty), FONT, 0.36, (0, 220, 255), 1, cv2.LINE_AA)
            ty += 16
    return out


# ---------------------------------------------------------------------------
# 2026-09-10 FINAL VISUAL UPGRADE -- OFFSIDE / TRAP RADAR ONLY: compact 2.5D
# player glyphs replacing the old flat filled-circle markers. MARKER
# RENDERING ONLY -- pitch geometry, the perspective camera, every
# offside/reference line, goal positions, player PITCH coordinates, team
# assignments, and radar orientation are all completely untouched (this
# code only decides how a player is DRAWN at the already-projected (px,
# py) screen point every other radar element -- rings, connector lines,
# distance labels, the passer ring -- already anchors to; that anchor
# itself is never modified here). Cheap OpenCV primitives only (Option A
# from the task: procedural draw, no external assets, no meshes/textures,
# nothing loaded from disk per frame) -- kept fast enough for a 3600-frame
# render (see the performance benchmark in this round's audit report).
GLYPH_SCALE_MIN, GLYPH_SCALE_MAX = 0.60, 1.45  # 2026-09-10 POLISH: re-audited
# for the new, taller human glyph (was 0.55-1.6 for the shorter "pawn"
# glyph -- narrowed per this round's visual QA so a near-camera player can
# never balloon and a far one never vanishes). The upstream `scale_at()`
# (marker_scale_factor / the flat-mode constant 1.0) is already bounded,
# but the glyph itself never trusts an unbounded input either way.


def _player_glyph_colors(team_color):
    """Two shading variants of the SAME dominant team color -- darker for
    the legs/lower body, brighter for the torso/head highlight -- so the
    glyph reads as subtly 3D while the team hue stays instantly
    recognizable (never a different hue, only a lightness change)."""
    dark = tuple(int(max(0, c * 0.55)) for c in team_color)
    bright = tuple(int(min(255, c * 1.18 + 20)) for c in team_color)
    return dark, bright


def _draw_player_glyph(img, px, py, team_color, scale=1.0, outline=(12, 12, 12)):
    """Compact 2.5D football-player glyph: ground shadow -> two short
    legs -> torso -> small head, all built from cheap ellipse/line/circle
    primitives (no meshes, no textures, no per-frame file I/O).

    Anchored at (px, py) -- the EXACT SAME ground/base screen point every
    existing ring, connector line, arrow, and distance label in
    `draw_radar` already uses; this function only changes how the player
    is DRAWN there, never where. The body is built rising upward
    (decreasing y) from that anchor, so it visually sits ABOVE the
    existing highlight rings that `draw_radar`'s own (unmodified) later
    code still draws centered on this same anchor -- those rings read as
    a halo at the player's feet/base, with the new body above them.

    `scale`: the ONLY perspective-aware knob (from the existing
    `scale_at(x, y)` -- 1.0 in flat radar mode, a bounded [0.72, 1.28]
    factor in perspective mode) -- clamped again here defensively so the
    glyph can never balloon or vanish regardless of caller input."""
    s = max(GLYPH_SCALE_MIN, min(GLYPH_SCALE_MAX, float(scale)))
    dark, bright = _player_glyph_colors(team_color)

    shadow_rx, shadow_ry = max(3, round(5 * s)), max(1, round(2 * s))
    cv2.ellipse(img, (px, py + 1), (shadow_rx, shadow_ry), 0, 0, 360, (8, 8, 8), -1, cv2.LINE_AA)

    leg_h = max(2, round(4 * s))
    torso_rx = max(2, round(3 * s))
    torso_ry = max(2, round(4 * s))
    head_r = max(2, round(2.4 * s))
    leg_dx = max(1, round(1.5 * s))
    leg_th = max(1, round(1.6 * s))

    hip_y = py
    shoulder_y = hip_y - leg_h
    torso_cy = shoulder_y - torso_ry
    head_cy = torso_cy - torso_ry - head_r + 1

    # two short legs, dark-shaded (lower-body shade)
    cv2.line(img, (px - leg_dx, hip_y), (px - leg_dx, shoulder_y), dark, leg_th, cv2.LINE_AA)
    cv2.line(img, (px + leg_dx, hip_y), (px + leg_dx, shoulder_y), dark, leg_th, cv2.LINE_AA)

    # torso, dominant team color, with a brighter shoulder-line highlight
    # hint for a subtle 3D read (never a different hue)
    cv2.ellipse(img, (px, torso_cy), (torso_rx, torso_ry), 0, 0, 360, team_color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (px, torso_cy - max(1, torso_ry // 2)), (max(1, torso_rx - 1), max(1, torso_ry // 2)),
                0, 0, 360, bright, -1, cv2.LINE_AA)
    cv2.ellipse(img, (px, torso_cy), (torso_rx, torso_ry), 0, 0, 360, outline, 1, cv2.LINE_AA)

    # head, brighter highlight shade, thin outline for readability
    cv2.circle(img, (px, head_cy), head_r, bright, -1, cv2.LINE_AA)
    cv2.circle(img, (px, head_cy), head_r, outline, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 2026-09-10 FINAL VISUAL POLISH -- OFFSIDE / TRAP RADAR ONLY: a more
# clearly HUMAN/footballer-like glyph replacing the "pawn"-style
# `_draw_player_glyph` above (kept, unmodified, purely for the A/B/C
# comparison this round's audit report documents -- no longer called by
# `draw_radar`). Same constraints as before: cheap OpenCV primitives only,
# same (px, py) anchor as every other radar element, marker rendering
# only. Split into a SHADOW pass (drawn early, same position in the
# sequence as the old player loop) and a BODY pass (drawn LATE, after
# every existing highlight ring/connector/label) per this round's explicit
# "ground shadow -> highlight ring -> body" layering request.
HUMAN_GLYPH_BASE_H = 22.0         # nominal total height (px) at scale=1.0 -- a
# first pass at exactly the requested "+15-20% vs the old ~16.8px pawn
# glyph" (19px) rounded every body part to 1px at this size, which
# CRUSHED the head/arms into indistinguishable outline-only blobs on
# real dense-cluster frames (caught during this round's own zoomed A/B/C
# QA -- see the audit report) -- bumped to 22px (~+31%) specifically so
# every part keeps at least 2px of real fill, per this round's own
# stated priority ("human-like silhouette quality is more important
# than rotation" -- and, by the same logic, more important than hitting
# the height figure exactly). Still clamped small enough for dense
# clusters (see the performance/QA sections of this round's report).
HUMAN_HEAD_FRAC = 0.15           # head diameter as a fraction of HUMAN_GLYPH_BASE_H
HUMAN_NECK_FRAC = 0.05
HUMAN_TORSO_FRAC = 0.35
HUMAN_HIP_FRAC = 0.125
HUMAN_LEG_FRAC = 0.375
HEADING_MIN_SPEED_CM_S = 80.0    # below this, a player reads as "stationary" ->
# neutral forward-facing pose (never a fabricated direction for a standing player)
HEADING_LOOKAHEAD_SEC = 0.3      # small forward time-step used ONLY to sample a
# second SCREEN point from the player's own current/past causal velocity
# (never a future tracked position) -- see `draw_radar`'s body-pass comment.


def _heading_cue(dx, dy):
    """A cheap, SAFE motion cue for the human glyph: `(mirror_sign,
    lean_dx, lean_dy)`. `(dx, dy)` is a SCREEN-space direction (already
    projected through whichever camera -- flat or perspective -- is
    active, so this never needs its own homography/Jacobian math).

    Deliberately NOT a full-body rotation: an earlier version of this
    glyph rotated the whole upright skeleton to face the real ground-
    plane heading angle, and for a player running mostly LEFT/RIGHT
    across the pitch (the common case, not toward/away from camera) that
    tipped the human silhouette onto its side -- it read as "lying down",
    not "running" (this is a top-down/2.5D radar view, not a side-on
    view, so a full-body rotation from a ground-plane angle does not
    translate the way it would in a side-view game sprite). Caught during
    this round's own A/B/C visual QA and fixed by switching to this
    silhouette-preserving alternative instead, per the task's own
    "human-like silhouette quality is more important than rotation"
    priority: the body is ALWAYS drawn upright (head above feet); only
    (a) which leg/arm leads mirrors left/right to match real horizontal
    travel direction, and (b) the upper body (torso/neck/head only --
    never the planted feet) leans a few px toward the real heading.

    `(0, 0)` (the caller's own "too slow to trust" gate) yields the
    neutral default: mirror_sign=1.0 (default stance), no lean."""
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 1.0, 0.0, 0.0
    mirror_sign = 1.0 if dx >= 0 else -1.0
    return mirror_sign, dx / norm, dy / norm


def _draw_player_shadow(img, px, py, scale=1.0):
    """Ground shadow only -- drawn FIRST (see `draw_radar`), so it reads
    as the base the highlight ring and body both sit on top of."""
    s = max(GLYPH_SCALE_MIN, min(GLYPH_SCALE_MAX, float(scale)))
    rx, ry = max(3, round(5.5 * s)), max(1, round(2 * s))
    cv2.ellipse(img, (px, py + 1), (rx, ry), 0, 0, 360, (8, 8, 8), -1, cv2.LINE_AA)


def _draw_player_body_human(img, px, py, team_color, scale=1.0, mirror_sign=1.0, lean=(0.0, 0.0),
                             outline=(10, 10, 10)):
    """Compact HUMAN-proportioned 2.5D footballer glyph: two separate
    legs (subtly staggered, tiny feet) -> shorts -> a tapered torso
    (shoulders wider than the waist) -> two short arms -> a short neck ->
    a small, clearly-separated head. All cheap `cv2` line/polygon/circle
    primitives -- no meshes, no textures, nothing loaded from disk.

    Anchored at (px, py) exactly like every other radar element (never
    moved) -- the body is ALWAYS drawn upright (head above feet, never
    rotated to face a ground-plane heading angle; see `_heading_cue`'s
    own docstring for why). `mirror_sign` (+1/-1) flips which leg/arm
    leads to match real horizontal travel direction; `lean` (a small unit
    vector, `(0,0)` = no lean) nudges the UPPER body only (torso/neck/
    head -- never the planted feet) a couple of px toward the real
    heading, for a mild forward-lean cue without ever rotating the
    skeleton. `scale` is clamped exactly like the pawn glyph."""
    s = max(GLYPH_SCALE_MIN, min(GLYPH_SCALE_MAX, float(scale)))
    H = HUMAN_GLYPH_BASE_H * s
    # every floor below is a legibility guarantee, not just an
    # anti-zero-size guard: a head/arm/leg that rounds to a 1px sliver is
    # visually indistinguishable from its own outline (see the
    # HUMAN_GLYPH_BASE_H comment above) -- each part keeps at least 2px
    # of real fill even at the smallest allowed `scale`.
    head_r = max(2, round(H * HUMAN_HEAD_FRAC / 2))
    neck_h = max(1, round(H * HUMAN_NECK_FRAC))
    torso_h = max(3, round(H * HUMAN_TORSO_FRAC))
    hip_h = max(2, round(H * HUMAN_HIP_FRAC))
    leg_h = max(3, round(H * HUMAN_LEG_FRAC))

    shoulder_half = max(2, round(H * 0.18))
    waist_half = max(2, round(H * 0.11))
    shorts_half = max(2, round(H * 0.13))
    leg_half = max(2, round(H * 0.09))   # wide enough that two legs never merge into one stem
    arm_len = max(2, round(H * 0.20))
    leg_stagger = max(1, round(leg_h * 0.18)) * mirror_sign   # subtle -- never an exaggerated stride
    arm_stagger = max(1, round(arm_len * 0.5)) * mirror_sign

    lean_px = min(1.6, 1.6 * s)
    lean_x, lean_y = lean[0] * lean_px, lean[1] * lean_px

    def pt(right_off, fwd_off, upper=False):
        # fixed upright basis: +right_off moves screen-right, +fwd_off
        # moves screen-up (toward the head) -- the skeleton itself is
        # NEVER rotated (see docstring); `upper=True` adds the small,
        # heading-driven lean to torso/neck/head/arm points only.
        x = px + right_off
        y = py - fwd_off
        if upper:
            x += lean_x
            y += lean_y
        return (int(round(x)), int(round(y)))

    f_hip = leg_h
    f_waist = leg_h + hip_h
    f_shoulder = leg_h + hip_h + torso_h
    f_neck_top = f_shoulder + neck_h
    f_head = f_neck_top + head_r

    dark, _bright = _player_glyph_colors(team_color)
    shorts_color = tuple(int(max(0, c * 0.7)) for c in team_color)
    head_color = (222, 222, 222)  # small neutral/light highlight -- schematic, never a real skin tone

    leg_th = max(1, round(1.5 * s))
    foot_r = max(1, leg_th // 2 + 1)
    # two SEPARATE legs, planted at the real anchor (no lean) -- one foot
    # staggered slightly forward, the other slightly back (mirrored by
    # real horizontal travel direction) for a mild running/ready stance,
    # never an exaggerated stride.
    cv2.line(img, pt(-leg_half, f_hip), pt(-leg_half, leg_stagger), dark, leg_th, cv2.LINE_AA)
    cv2.line(img, pt(leg_half, f_hip), pt(leg_half, -leg_stagger), dark, leg_th, cv2.LINE_AA)
    cv2.circle(img, pt(-leg_half, leg_stagger), foot_r, dark, -1, cv2.LINE_AA)
    cv2.circle(img, pt(leg_half, -leg_stagger), foot_r, dark, -1, cv2.LINE_AA)

    # shorts / hip region -- a darker shade of the SAME team color
    shorts_poly = np.array([pt(-shorts_half, f_hip), pt(shorts_half, f_hip),
                             pt(waist_half, f_waist), pt(-waist_half, f_waist)], dtype=np.int32)
    cv2.fillPoly(img, [shorts_poly], shorts_color, cv2.LINE_AA)

    # torso -- tapered trapezoid (shoulders wider than the waist), the
    # dominant team color (the "shirt"), never a simple vertical oval --
    # gets the small upper-body lean.
    torso_poly = np.array([pt(-waist_half, f_waist, upper=True), pt(waist_half, f_waist, upper=True),
                            pt(shoulder_half, f_shoulder, upper=True), pt(-shoulder_half, f_shoulder, upper=True)],
                           dtype=np.int32)
    cv2.fillPoly(img, [torso_poly], team_color, cv2.LINE_AA)
    cv2.polylines(img, [torso_poly], True, outline, 1, cv2.LINE_AA)

    # two short arms, subtly asymmetric (mirrored with the legs), dark
    # limb shade, upper-body lean applied
    arm_th = max(1, round(1.3 * s))
    cv2.line(img, pt(-shoulder_half, f_shoulder, upper=True),
              pt(-shoulder_half - arm_len, f_shoulder - arm_stagger, upper=True), dark, arm_th, cv2.LINE_AA)
    cv2.line(img, pt(shoulder_half, f_shoulder, upper=True),
              pt(shoulder_half + arm_len, f_shoulder + arm_stagger, upper=True), dark, arm_th, cv2.LINE_AA)

    # short neck -- keeps the head visibly separated from the shoulders
    cv2.line(img, pt(0, f_shoulder, upper=True), pt(0, f_neck_top, upper=True), dark, max(1, round(1.4 * s)), cv2.LINE_AA)

    # head -- small, clearly separated, neutral highlight shade
    cv2.circle(img, pt(0, f_head, upper=True), head_r, head_color, -1, cv2.LINE_AA)
    cv2.circle(img, pt(0, f_head, upper=True), head_r, outline, 1, cv2.LINE_AA)


def draw_radar(players_this_frame, ball_xy, snap_f, w, h, lead_runners, crossed_teams,
                debug=False, passer=None, release_active=False, pass_vector_end_xy=None, threat_field=None,
                radar_view="flat", H=None):
    """Radar kept deliberately minimal: Team A/B players, ball, both
    teams' estimated lines, and the SINGLE currently-relevant attacker
    per team (the exact same selection Graph 3 plots) -- no
    pitch-control wash, no shaded zone fill, no corner text blocks, no
    legend box.

    `lead_runners`: {team: runner_sample_or_None} from
    `_select_lead_runner` -- the one attacker highlighted, connected to
    the line, and labeled, consistent with Graph 3 (never "all active
    runs", to avoid the radar showing a different attacker than the
    graph is tracking).
    `crossed_teams`: set of teams whose selected runner's relation_cm
    sign flipped vs. the previous frame of the SAME run_id -- a purely
    geometric line-crossing reference, drawn as a small marker, never
    labeled or implied as an official offside decision."""
    # 2026-09-08 FINAL VISUALIZATION UPGRADE -- ONE shared projection for
    # every radar object below (pitch markings, lines, players, threat
    # wash, passer/runner/pass-vector). `project(x,y)` is the only thing
    # that changes between flat and perspective mode.
    perspective = radar_view == "perspective" and H is not None
    if perspective:
        w0, h0 = 1300, 800  # SAME intrinsic working canvas size draw_pitch() itself produces
        base = draw_pitch_markings_perspective(H, CONFIG, w0, h0)
        project = lambda x, y: project_point_int(H, x, y)
        scale_at = lambda x, y: marker_scale_factor(H, x, y, DEFAULT_PITCH)
    else:
        base = draw_pitch(config=CONFIG)
        w0, h0 = base.shape[1], base.shape[0]
        project = pitch_xy_to_px
        scale_at = lambda x, y: 1.0
    pitch_dark = cv2.addWeighted(base, 0.7, np.zeros_like(base), 0.3, 0)
    # (Part 9) behind-line threat wash -- drawn FIRST, strictly secondary
    # to every tactical line/player/marker drawn on top of it.
    if perspective:
        pitch_dark = apply_behind_line_threat_wash_radar_perspective(pitch_dark, threat_field, H)
    else:
        pitch_dark = apply_behind_line_threat_wash_radar(pitch_dark, threat_field)

    # (A4/A6) THREE DISTINCT football objects, ONLY in canonical pitch
    # coordinates -- never conflated:
    #   (1) DEFENSIVE-UNIT CENTRE = robust median x of the NEW temporal-
    #       membership CONFIRMED unit (2026-09-08 V7) -- solid, light --
    #       the PRIMARY reference for trap movement/synchrony.
    #   (2) DEEPEST OUTFIELD DEFENDER = deepest member of that SAME
    #       confirmed unit -- thin solid, medium.
    #   (3) ESTIMATED OFFSIDE REFERENCE = second-last OPPONENT among ALL
    #       eligible opponents incl. GK (the pre-existing, UNCHANGED
    #       `line` defs[1] value) -- thick solid, brightest. NOT the
    #       unit-centre and never called "the offside line" outright.
    # Tiny endpoint labels only; the specific responsible player gets a
    # subtle ring (never a giant legend).
    by_track = {p["track_id"]: p for p in players_this_frame}
    for team in (0, 1):
        line = snap_f["offside"][team]["line"]
        back_line = snap_f["offside"][team].get("back_line") or {}
        label_x_off = -52 if team == 0 else 6
        if back_line.get("centre_x") is not None:
            cp0, cp1 = project(back_line["centre_x"], 0), project(back_line["centre_x"], DEFAULT_PITCH.width_cm)
            cv2.line(pitch_dark, cp0, cp1, CENTRE_LINE_COLOR[team], 3, cv2.LINE_AA)
            cv2.putText(pitch_dark, "UNIT CENTRE", (cp0[0] + label_x_off - 20, cp0[1] + 12), FONT, 0.26, CENTRE_LINE_COLOR[team], 1, cv2.LINE_AA)
        if back_line.get("deep_x") is not None:
            if debug:
                lp0, lp1 = project(back_line["deep_x"], 0), project(back_line["deep_x"], DEFAULT_PITCH.width_cm)
                cv2.line(pitch_dark, lp0, lp1, DEEP_LINE_COLOR[team], 1, cv2.LINE_AA)
                cv2.putText(pitch_dark, "DEEPEST OUTFIELD", (lp0[0] + label_x_off - 30, lp0[1] + 26), FONT, 0.24, DEEP_LINE_COLOR[team], 1, cv2.LINE_AA)
            else:
                # (V9) normal-mode: a short TICK at the responsible
                # player's own position, not a full-pitch line (avoids a
                # 4th competing line on the clean broadcast radar).
                deep_p = by_track.get(back_line.get("deep_track_id"))
                deep_y = deep_p["y_pitch"] if deep_p is not None else DEFAULT_PITCH.width_cm / 2
                dpx, dpy = project(back_line["deep_x"], deep_y)
                cv2.line(pitch_dark, (dpx, dpy - DEEP_TICK_LEN_PX), (dpx, dpy + DEEP_TICK_LEN_PX), DEEP_LINE_COLOR[team], 2, cv2.LINE_AA)
                cv2.putText(pitch_dark, "DEEPEST", (dpx + 8, dpy + 4), FONT, 0.22, DEEP_LINE_COLOR[team], 1, cv2.LINE_AA)
        if line["valid"]:
            rp0, rp1 = project(line["line_x"], 0), project(line["line_x"], DEFAULT_PITCH.width_cm)
            _dashed_line(pitch_dark, rp0, rp1, REFERENCE_LINE_COLOR[team], 2)
            cv2.putText(pitch_dark, "OFFSIDE REF", (rp0[0] + label_x_off - 12, rp0[1] + 40), FONT, 0.26, REFERENCE_LINE_COLOR[team], 1, cv2.LINE_AA)

    # 2026-09-10 FINAL VISUAL POLISH: human-proportioned 2.5D player
    # glyphs -- same projected (px, py) anchor and same `scale_at`
    # perspective scaling every other radar element already used, nothing
    # about WHERE a player is drawn changes, only HOW. Far-to-near draw
    # order (by screen-y, ascending) so nearer glyphs correctly overlap
    # farther ones instead of a visually-impossible reverse overlap.
    #
    # Per this round's explicit "ground shadow -> highlight ring -> body"
    # layering request, only the SHADOW is drawn here (same position in
    # the sequence the old flat-circle/pawn-glyph player loop used); the
    # actual BODY is drawn in a second pass further below, AFTER every
    # existing highlight ring/connector/label, so a taller human glyph
    # never buries those rings under an opaque body drawn too early.
    projected = [(project(p["x_pitch"], p["y_pitch"]), p) for p in players_this_frame]
    projected_sorted = sorted(projected, key=lambda item: item[0][1])
    for (px, py), p in projected_sorted:
        scale = scale_at(p["x_pitch"], p["y_pitch"])
        _draw_player_shadow(pitch_dark, px, py, scale=scale)

    # (A6) player-level membership visualization from the NEW temporal
    # state (2026-09-08 V7) -- confirmed=strong ring (brighter still if
    # individually part of a real coordinated step), candidate=lighter
    # ring, temporarily missing (carried by temporal state but not
    # visible this frame -- drawn only if somehow still resolvable, e.g.
    # a stale position, which in practice means this rarely draws;
    # documented for completeness)=dashed ring, non-member=no extra ring.
    for team in (0, 1):
        back_line = snap_f["offside"][team].get("back_line") or {}
        confirmed_ids = set(back_line.get("member_ids") or [])
        candidate_ids = set(back_line.get("candidate_ids") or [])
        stepping = bool(snap_f["offside"][team].get("raw_score"))
        stepping_ids = set(snap_f["offside"][team].get("metrics", {}).get("participants", [])) if stepping else set()
        for tid in confirmed_ids:
            p = by_track.get(tid)
            if p is None:
                continue
            px, py = project(p["x_pitch"], p["y_pitch"])
            if tid in stepping_ids:
                cv2.circle(pitch_dark, (px, py), 12, STEPPING_RING_COLOR[team], 2, cv2.LINE_AA)
            else:
                cv2.circle(pitch_dark, (px, py), 11, UNIT_RING_COLOR, 1, cv2.LINE_AA)
            if p.get("vx_cm_s") is not None and p.get("vy_cm_s") is not None:
                ex, ey = int(px + p["vx_cm_s"] * 0.03), int(py + p["vy_cm_s"] * 0.03)
                arrow_color = STEPPING_RING_COLOR[team] if tid in stepping_ids else UNIT_RING_COLOR
                cv2.arrowedLine(pitch_dark, (px, py), (ex, ey), arrow_color, 1, cv2.LINE_AA, tipLength=0.35)
        for tid in candidate_ids:
            p = by_track.get(tid)
            if p is None:
                continue
            px, py = project(p["x_pitch"], p["y_pitch"])
            cv2.circle(pitch_dark, (px, py), 10, CANDIDATE_RING_COLOR, 1, cv2.LINE_AA)
        for tid in (back_line.get("missing_ids") or []):
            p = by_track.get(tid)  # usually None (genuinely absent this frame) -- drawn only if resolvable
            if p is None:
                continue
            px, py = project(p["x_pitch"], p["y_pitch"])
            _dashed_line(pitch_dark, (px - 11, py), (px + 11, py), UNIT_RING_COLOR, 1, dash_len=4, gap_len=3)

    # Mark the specific player responsible for the deep/reference
    # boundaries with a subtle ring in that line's own color (1G).
    for team in (0, 1):
        line = snap_f["offside"][team]["line"]
        back_line = snap_f["offside"][team].get("back_line") or {}
        for tid, ring_color in ((back_line.get("deep_track_id"), DEEP_LINE_COLOR[team]),
                                 (line.get("defender_track_id") if line["valid"] else None, REFERENCE_LINE_COLOR[team])):
            p = by_track.get(tid)
            if p is not None:
                px, py = project(p["x_pitch"], p["y_pitch"])
                cv2.circle(pitch_dark, (px, py), 14, ring_color, 1, cv2.LINE_AA)
    for attacking_team, r in lead_runners.items():
        if r is None:
            continue
        p = by_track.get(r["track_id"])
        if p is None:
            continue
        px, py = project(p["x_pitch"], p["y_pitch"])
        highlight = RUN_BREAK_COLOR if r["state"] == "BREAK_THREAT" else RUN_APPROACH_COLOR

        # (5) attacker-to-line connector -- nearest point on the
        # OPPOSING team's estimated line (line is a fixed-x vertical
        # line in pitch space, so the nearest point is (line_x, attacker_y)).
        defending_team = 1 - attacking_team
        line = snap_f["offside"][defending_team]["line"]
        if line["valid"]:
            lx, ly = project(line["line_x"], p["y_pitch"])
            cv2.line(pitch_dark, (px, py), (lx, ly), highlight, 1, cv2.LINE_AA)
            # (8) line-crossing point -- purely geometric reference,
            # only drawn on the frame(s) where a real sign-flip was
            # detected for this exact run_id (see render_frame).
            if attacking_team in crossed_teams:
                cv2.circle(pitch_dark, (lx, py), 7, CROSSING_MARK_COLOR, 2, cv2.LINE_AA)

        # (7) run direction arrow -- real motion vector only.
        if p.get("vx_cm_s") is not None and p.get("vy_cm_s") is not None:
            ex, ey = int(px + p["vx_cm_s"] * 0.03), int(py + p["vy_cm_s"] * 0.03)
            cv2.arrowedLine(pitch_dark, (px, py), (ex, ey), highlight, 2, cv2.LINE_AA, tipLength=0.3)

        # (4) relevant attacker highlight -- the one selected runner only.
        cv2.circle(pitch_dark, (px, py), 12, SELECTED_RING_COLOR, 2, cv2.LINE_AA)
        cv2.circle(pitch_dark, (px, py), 9, highlight, 2, cv2.LINE_AA)

        # (6) distance-to-line label -- exact same relation_cm/100 value Graph 3 plots.
        label = f"{r['relation_cm'] / 100.0:+.1f} m"
        (tw_, th_), _ = cv2.getTextSize(label, FONT, 0.32, 1)
        lx2, ly2 = px + 14, py - 14
        cv2.rectangle(pitch_dark, (lx2 - 3, ly2 - th_ - 3), (lx2 + tw_ + 3, ly2 + 3), (20, 14, 10), -1)
        cv2.putText(pitch_dark, label, (lx2, ly2), FONT, 0.32, TEXT_WHITE, 1, cv2.LINE_AA)

    # (V9) passer candidate -- SAME track_id the video panel highlights
    # (never re-derived per-surface), distinct cyan ring so it never
    # reads as a team color or the existing runner highlight.
    passer_p = None
    if passer is not None and passer.get("passer_track_id") is not None:
        passer_p = by_track.get(passer["passer_track_id"])
        if passer_p is not None:
            ppx, ppy = project(passer_p["x_pitch"], passer_p["y_pitch"])
            cv2.circle(pitch_dark, (ppx, ppy), 13, PASSER_RING_COLOR, 2, cv2.LINE_AA)
            cv2.putText(pitch_dark, "PASSER", (ppx + 10, ppy - 10), FONT, 0.24, PASSER_RING_COLOR, 1, cv2.LINE_AA)
            if release_active:
                cv2.drawMarker(pitch_dark, (ppx, ppy - 20), RELEASE_MARKER_COLOR, cv2.MARKER_TILTED_CROSS, 14, 2)
                cv2.putText(pitch_dark, "RELEASE", (ppx + 10, ppy - 22), FONT, 0.22, RELEASE_MARKER_COLOR, 1, cv2.LINE_AA)

    # (V9) pass vector -- pitch-space geometry only, drawn once a passer
    # AND a release candidate exist (never speculative beforehand).
    if passer_p is not None and pass_vector_end_xy is not None:
        epx, epy = project(*pass_vector_end_xy)
        cv2.arrowedLine(pitch_dark, (ppx, ppy), (epx, epy), PASS_VECTOR_COLOR, 2, cv2.LINE_AA, tipLength=0.08)

    # 2026-09-10 FINAL VISUAL POLISH: player BODIES, drawn LAST (after
    # every ring/connector/label above) so the human silhouette sits
    # visually on top -- per this round's explicit layering request --
    # while every ring above stays intact underneath (largely still
    # visible around the body's narrower silhouette; only the small
    # chords a leg/arm happens to cross are covered, same trade-off this
    # round's own instructions accepted). Far-to-near order reused from
    # the shadow pass above (no second sort, no second `project()` call
    # per player for position).
    for (px, py), p in projected_sorted:
        scale = scale_at(p["x_pitch"], p["y_pitch"])
        color = TEAM_COLOR.get(p["display_team_id"], (150, 150, 150))
        # Optional visual-only heading: a SECOND screen point sampled a
        # small lookahead ahead along the player's own CURRENT/PAST
        # causal velocity (`vx_cm_s`/`vy_cm_s` are backward-difference,
        # never future frames -- see `cleaned_tracking_view.py`), then
        # the SAME `project()` this whole panel already uses turns that
        # into a screen-space direction -- correct in both flat and
        # perspective radar modes without needing any homography
        # Jacobian math. Below the speed gate (or missing/invalid
        # velocity), the glyph gets the neutral forward-facing default.
        vx, vy = p.get("vx_cm_s"), p.get("vy_cm_s")
        if vx is not None and vy is not None and math.hypot(vx, vy) >= HEADING_MIN_SPEED_CM_S:
            ahead_px, ahead_py = project(p["x_pitch"] + vx * HEADING_LOOKAHEAD_SEC,
                                          p["y_pitch"] + vy * HEADING_LOOKAHEAD_SEC)
            mirror_sign, lean_dx, lean_dy = _heading_cue(ahead_px - px, ahead_py - py)
        else:
            mirror_sign, lean_dx, lean_dy = _heading_cue(0.0, 0.0)
        _draw_player_body_human(pitch_dark, px, py, color, scale=scale,
                                 mirror_sign=mirror_sign, lean=(lean_dx, lean_dy))

    if ball_xy is not None:
        bx, by = project(*ball_xy)
        br = int(round(6 * scale_at(*ball_xy)))
        # tiny ground shadow only -- the ball stays its own small, plain
        # white marker (never turned into a player-style glyph), just a
        # touch of the same 2.5D ground-contact cue the player glyphs use.
        cv2.ellipse(pitch_dark, (bx, by + max(1, br // 2)), (max(2, br), max(1, br // 3)), 0, 0, 360, (8, 8, 8), -1, cv2.LINE_AA)
        cv2.circle(pitch_dark, (bx, by), br, BALL_COLOR, -1, cv2.LINE_AA)
        cv2.circle(pitch_dark, (bx, by), br, (15, 15, 15), 1, cv2.LINE_AA)

    out = np.full((h, w, 3), BG, dtype=np.uint8)
    resized = cv2.resize(pitch_dark, (w, h - 34))
    out[34:h, 0:w] = resized
    cv2.rectangle(out, (12, 8), (15, 24), ACCENT_RED, -1)
    cv2.putText(out, "OFFSIDE / TRAP RADAR", (22, 22), FONT, 0.46, TEXT_WHITE, 2, cv2.LINE_AA)
    return out


def _select_lead_runner(runners):
    """Fixed, disclosed selection rule for 'the relevant attacker this
    frame': among the attacking team's currently-tracked (already
    confirmed, >= RUN_MIN_OBSERVATIONS) runs, pick the one with the
    HIGHEST threat score; ties/all-None threat fall back to the
    smallest |relation_cm|. Returns None if no runner is active --
    an honest gap, never a fabricated value."""
    if not runners:
        return None
    with_threat = [r for r in runners if r.get("threat") is not None]
    if with_threat:
        return max(with_threat, key=lambda r: r["threat"])
    return min(runners, key=lambda r: abs(r["relation_cm"]))


def _lead_series_with_breaks(snapshot, team):
    """Full-clip (time, value) series for Graph O3, breaking the line
    with an explicit None whenever the selected runner's `run_id`
    changes between consecutive samples -- so two DIFFERENT attackers'
    episodes are never visually joined by one continuous stroke.
    Sign convention (documented here and on-screen): relation_cm =
    line_depth_cm - attacker_depth, so POSITIVE = attacker still on the
    safe/pre-line side, 0 = level with the line, NEGATIVE = attacker has
    moved beyond the estimated line."""
    out = []
    last_run_id = None
    for s in snapshot:
        t = s["time_sec"]
        r = _select_lead_runner(s["offside"][team].get("runners") or [])
        if r is None:
            out.append((t, None))
            last_run_id = None
            continue
        if last_run_id is not None and r["run_id"] != last_run_id:
            out.append((t, None))
        out.append((t, r["relation_cm"] / 100.0))
        last_run_id = r["run_id"]
    return out


def _full_series(snapshot, fn):
    return [(s["time_sec"], fn(s)) for s in snapshot]


def _signed_y_range(values, pad_frac=0.1, floor_zero=False):
    """Robust 1st-99th-percentile range, WITHOUT
    `live_graphs.compute_stable_y_range`'s unconditional 0-floor --
    kept local so genuinely SIGNED quantities (line velocity, attacker
    distance to line) are never silently clamped off-axis."""
    vals = [v for v in values if v is not None]
    if not vals:
        return (0.0, 1.0) if floor_zero else (-1.0, 1.0)
    lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
    if hi <= lo:
        hi = lo + 1.0
    pad = (hi - lo) * pad_frac
    lo_out = lo - pad
    if floor_zero:
        lo_out = max(0.0, lo_out)
    return (lo_out, hi + pad)


def _to_px(t, v, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph):
    x = x0 + int((t - t_lo) / max(1e-6, (t_hi - t_lo)) * pw)
    y = y0 + ph - int((v - y_lo) / max(1e-6, (y_hi - y_lo)) * ph)
    return x, y


def _chrome(w, h, title, subtitle, y_label, y_range, tick_fmt, t_lo, t_hi, tick_spacing, accent=ACCENT_RED):
    """Beginner-readable FULL chrome: large title + interpretation
    subtitle, explicit Y-axis label (with units, human wording), an
    explicit 'Match Time (seconds)' X-axis label, readable absolute-
    second tick labels, and a zero reference line whenever the axis
    range actually spans zero."""
    # 2026-09-09 polish: lighter slate/blue-grey graph-panel fill -- calls
    # the SAME shared, unmodified `panel_frame()` with different bg/border
    # ARGUMENTS (it already accepts them); the outer dashboard shell
    # (header, top row) keeps calling the shared function with ITS OWN
    # defaults elsewhere in this file, completely untouched.
    img = panel_frame(w, h, bg=GRAPH_PANEL_BG, border=GRAPH_PANEL_BORDER)
    cv2.rectangle(img, (12, 8), (15, 24), accent, -1)
    cv2.putText(img, title, (22, 22), FONT, 0.50, TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (14, 40), FONT, 0.32, GRAPH_TICK_TEXT, 1, cv2.LINE_AA)

    pad_l, pad_r, pad_t, pad_b = 54, 16, 60, 48
    x0, y0 = pad_l, pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    y_lo, y_hi = y_range
    fmt = tick_fmt

    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), GRAPH_PANEL_BORDER, 1)
    for frac in (0.25, 0.5, 0.75):
        gy = y0 + int(ph * (1 - frac))
        cv2.line(img, (x0, gy), (x0 + pw, gy), GRAPH_GRID, 1, cv2.LINE_AA)
        cv2.putText(img, fmt.format(y_lo + frac * (y_hi - y_lo)), (4, gy + 4), FONT, 0.30, GRAPH_TICK_TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, fmt.format(y_hi), (4, y0 + 8), FONT, 0.30, GRAPH_TICK_TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, fmt.format(y_lo), (4, y0 + ph + 4), FONT, 0.30, GRAPH_TICK_TEXT, 1, cv2.LINE_AA)

    if y_lo < 0 < y_hi:
        # brighter, thicker, amber zero/reference line -- "make it very
        # clear" (explicit instruction for the Margin graph; every other
        # offside graph that genuinely spans zero benefits identically).
        zy = y0 + int(ph * (1 - (0 - y_lo) / (y_hi - y_lo)))
        x = x0
        while x < x0 + pw:
            cv2.line(img, (x, zy), (min(x + 11, x0 + pw), zy), ZERO_LINE_COLOR, 3, cv2.LINE_AA)
            x += 17
        cv2.putText(img, "0", (4, zy + 4), FONT, 0.36, TEXT_WHITE, 2, cv2.LINE_AA)

    t = math.ceil(t_lo / tick_spacing) * tick_spacing
    while t <= t_hi + 1e-6:
        x = x0 + int((t - t_lo) / max(1e-6, (t_hi - t_lo)) * pw)
        cv2.line(img, (x, y0), (x, y0 + ph), GRAPH_GRID, 1, cv2.LINE_AA)
        label = f"{int(round(t))}s"
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.30, 1)
        cv2.putText(img, label, (max(0, x - tw // 2), y0 + ph + 17), FONT, 0.30, GRAPH_TICK_TEXT, 1, cv2.LINE_AA)
        t += tick_spacing

    cv2.putText(img, y_label, (pad_l, pad_t - 6), FONT, 0.33, GRAPH_TICK_TEXT, 1, cv2.LINE_AA)
    (xw, _), _ = cv2.getTextSize("Match Time (seconds)", FONT, 0.32, 1)
    cv2.putText(img, "Match Time (seconds)", (x0 + pw - xw, h - 8), FONT, 0.32, GRAPH_TICK_TEXT, 1, cv2.LINE_AA)
    return img, x0, y0, pw, ph, y_lo, y_hi


def _draw_legend(img, x0, pw, label_a, label_b, color_a, color_b):
    lx = x0 + pw - 6
    for lbl, color in ((label_b, color_b), (label_a, color_a)):
        tw = cv2.getTextSize(lbl, FONT, 0.33, 1)[0][0]
        lx -= tw
        cv2.putText(img, lbl, (lx, 20), FONT, 0.33, TEXT_WHITE, 1, cv2.LINE_AA)
        lx -= 16
        cv2.rectangle(img, (lx, 12), (lx + 10, 20), color, -1)
        lx -= 10


def _draw_structure_legend(img, x0, pw):
    """2026-09-09 polish -- Graph 1's own legend: three STRUCTURAL swatches
    (color + line style), replacing the old Team A/Team B legend now that
    color encodes structural identity rather than team (see
    `STRUCTURE_COLOR`/`draw_line_dynamics_graph`)."""
    items = [("Unit Centre", STRUCTURE_COLOR["unit_centre"], "solid"),
             ("Reference", STRUCTURE_COLOR["reference"], "dashed"),
             ("Deepest", STRUCTURE_COLOR["deepest"], "dotted")]
    lx = x0 + pw - 6
    for label, color, style in reversed(items):
        tw = cv2.getTextSize(label, FONT, 0.28, 1)[0][0]
        lx -= tw
        cv2.putText(img, label, (lx, 20), FONT, 0.28, TEXT_WHITE, 1, cv2.LINE_AA)
        lx -= 22
        if style == "solid":
            cv2.line(img, (lx, 16), (lx + 16, 16), color, 3, cv2.LINE_AA)
        elif style == "dashed":
            _dashed_line(img, (lx, 16), (lx + 16, 16), color, thickness=2, dash_len=5, gap_len=3)
        else:
            _dashed_line(img, (lx, 16), (lx + 16, 16), color, thickness=2, dash_len=2, gap_len=3)
        lx -= 14


def _plot_line(img, hist, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color, max_gap_sec):
    history = [(t, v) for t, v in hist if t_lo - 1e-9 <= t <= cur_time + 1e-9]
    segment, last_t = [], None
    for t, v in history:
        gap = (last_t is not None) and (t - last_t > max_gap_sec)
        if v is None or gap:
            if len(segment) >= 2:
                cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, color, 3, cv2.LINE_AA)
            elif len(segment) == 1:
                cv2.circle(img, segment[0], 3, color, -1, cv2.LINE_AA)
            segment = []
        if v is not None:
            segment.append(_to_px(t, np.clip(v, y_lo, y_hi), t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph))
        last_t = t
    if len(segment) >= 2:
        cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, color, 3, cv2.LINE_AA)
    elif len(segment) == 1:
        cv2.circle(img, segment[0], 3, color, -1, cv2.LINE_AA)


def draw_dual_graph(w, h, full_a, full_b, cur_time, t_lo, t_hi, tick_spacing, title, subtitle, y_label, y_range,
                     tick_fmt, label_a, label_b, color_a, color_b, accent=ACCENT_RED, max_gap_sec=0.5):
    img, x0, y0, pw, ph, y_lo, y_hi = _chrome(w, h, title, subtitle, y_label, y_range, tick_fmt, t_lo, t_hi, tick_spacing, accent)
    _plot_line(img, full_a, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color_a, max_gap_sec)
    _plot_line(img, full_b, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color_b, max_gap_sec)
    cx, _ = _to_px(cur_time, y_lo, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
    cv2.line(img, (cx, y0), (cx, y0 + ph), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (cx, y0), (cx, y0 + ph), TEXT_WHITE, 2, cv2.LINE_AA)
    _draw_legend(img, x0, pw, label_a, label_b, color_a, color_b)
    return img


# ---------------------------------------------------------------------------
# 2026-09-09 FINAL COACH-DASHBOARD GRAPH POLISH -- Graph 2 (Most-
# Threatening-Attacker Margin, Candidate C). Kept as a SEPARATE function
# from `draw_dual_graph` (which stays exactly as-is, still used for the
# now-coach-layout-removed Behind-Line Space research output and any
# other future dual-team graph) so this graph's own single-accent/
# team-split decision and clearer zero line never affect `draw_dual_graph`
# callers. No signal/value/threshold logic changes -- `full_a`/`full_b`
# are the SAME unmodified `_lead_series_with_breaks()` output as before.
def draw_lead_margin_graph(w, h, full_a, full_b, cur_time, t_lo, t_hi, tick_spacing, y_range,
                            accent=ACCENT_RED, max_gap_sec=0.5):
    """Graph 2 -- Most-Threatening-Attacker Margin. Tactical ESTIMATED-
    REFERENCE margin only -- explicitly never framed as an official
    offside decision. Sign convention unchanged: 0m = level with the
    estimated reference, negative = attacker has moved beyond it,
    positive = attacker safely behind it (see `_lead_series_with_breaks`).

    Coloring is a RENDER-ONLY decision: if the current rolling window
    genuinely contains real samples for BOTH attacking teams at once, the
    existing Team A/Team B split coloring is preserved unchanged (real
    data is never hidden); otherwise -- the common case, since `team` here
    is the ATTACKING side and open play usually has one attacking team's
    runners active at a time -- one strong single-accent color is used
    plus a small 'Attacking Team: Team X' context label instead of a
    two-team legend nobody needs."""
    def _has_data(hist):
        return any(v is not None for t, v in hist if t_lo - 1e-9 <= t <= cur_time + 1e-9)

    a_has, b_has = _has_data(full_a), _has_data(full_b)
    img, x0, y0, pw, ph, y_lo, y_hi = _chrome(
        w, h, "MOST-THREATENING-ATTACKER MARGIN",
        "Tactical estimated-reference margin -- NOT an official offside decision",
        "Margin to Estimated Reference (metres)", y_range, "{:.0f}", t_lo, t_hi, tick_spacing, accent)

    if a_has and b_has:
        _plot_line(img, full_a, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, TEAM_COLOR[0], max_gap_sec)
        _plot_line(img, full_b, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, TEAM_COLOR[1], max_gap_sec)
        _draw_legend(img, x0, pw, "Team A", "Team B", TEAM_COLOR[0], TEAM_COLOR[1])
    elif a_has or b_has:
        hist = full_a if a_has else full_b
        _plot_line(img, hist, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, MARGIN_ACCENT_COLOR, max_gap_sec)
        team_label = f"Attacking Team: Team {'A' if a_has else 'B'}"
        (tw_, _), _ = cv2.getTextSize(team_label, FONT, 0.30, 1)
        cv2.putText(img, team_label, (x0 + pw - tw_ - 4, 20), FONT, 0.30, MARGIN_ACCENT_COLOR, 1, cv2.LINE_AA)
    # else: neither team has a real sample in this window -- an honest
    # empty panel, never a fabricated line or label.

    if y_lo < 0 < y_hi:
        # graph-specific clarification of what the (already-emphasized,
        # see `_chrome`) zero line MEANS here.
        zy = y0 + int(ph * (1 - (0 - y_lo) / (y_hi - y_lo)))
        cv2.putText(img, "0m = estimated reference level", (x0 + 6, zy - 6), FONT, 0.28, ZERO_LINE_COLOR, 1, cv2.LINE_AA)

    cx, _ = _to_px(cur_time, y_lo, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
    cv2.line(img, (cx, y0), (cx, y0 + ph), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (cx, y0), (cx, y0 + ph), TEXT_WHITE, 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# 2026-09-08 V9.1 Part 12 -- GRAPH 4 VARIANT B: Release-to-Cross Timing
# Margin. Deliberately event-limited and NEVER stitched: each event with
# a real, computable margin draws its OWN short horizontal segment
# (release_time -> cross_time, at height = margin) plus a release-time
# marker dot -- no polyline connects two different events, and a frame
# before an event's own release time never shows that event at all
# (causal). Missing margins simply are not drawn.
def draw_timing_margin_graph(w, h, events, passes_by_id, cur_time, t_lo, t_hi, tick_spacing, accent=ACCENT_RED):
    values = [e["timing_margin_sec"] for e in events if e["timing_margin_sec"] is not None]
    y_lo, y_hi = _signed_y_range(values) if values else (-1.0, 1.0)
    # ZERO is the natural reference for this graph -- always keep it on
    # axis (a strong zero line), even when every real event so far
    # happens to fall on one side of it.
    y_range = (min(y_lo, -0.1), max(y_hi, 0.1))
    img, x0, y0, pw, ph, y_lo, y_hi = _chrome(
        w, h, "Release-to-Cross Timing Margin",
        "Positive = release before crossing; Negative = crossing before release",
        "Timing Margin (seconds)", y_range, "{:.2f}", t_lo, t_hi, tick_spacing, accent)
    for e in events:
        if e["timing_margin_sec"] is None or e["release_time_sec"] is None or e["observed_crossing_time_sec"] is None:
            continue
        if e["release_time_sec"] > cur_time + 1e-9:  # causal -- never show a future event early
            continue
        if e["observed_crossing_time_sec"] < t_lo - 1e-9 and e["release_time_sec"] < t_lo - 1e-9:
            continue  # entirely scrolled out of the rolling window
        color = TEAM_COLOR.get(e["passer_team"], (200, 200, 200))
        y_val = float(np.clip(e["timing_margin_sec"], y_lo, y_hi))
        t_rel = float(np.clip(e["release_time_sec"], t_lo, t_hi))
        t_cross = float(np.clip(min(e["observed_crossing_time_sec"], cur_time), t_lo, t_hi))
        p_rel = _to_px(t_rel, y_val, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
        p_cross = _to_px(t_cross, y_val, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
        cv2.line(img, p_rel, p_cross, color, 3, cv2.LINE_AA)
        cv2.circle(img, p_rel, 5, TEXT_WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, p_rel, 5, color, 2, cv2.LINE_AA)
    cx, _ = _to_px(cur_time, y_lo, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
    cv2.line(img, (cx, y0), (cx, y0 + ph), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (cx, y0), (cx, y0 + ph), TEXT_WHITE, 2, cv2.LINE_AA)
    _draw_legend(img, x0, pw, "Team A", "Team B", TEAM_COLOR[0], TEAM_COLOR[1])
    return img


# ---------------------------------------------------------------------------
# 2026-09-09 -- COACH-MODE second-row graphs (per
# audit_reports/2026-09-09_offside_graph_feasibility/). Graphs 1-3 below
# share a rolling 24s window (x_min=max(0,cur_time-24), x_max=cur_time,
# 4s major ticks) -- the SAME formula/tick convention the now-locked
# pressing dashboard uses, kept consistent per the audit's own
# recommendation. Graph 4 (funnel) is cumulative, not rolling.
COACH_GRAPH_WINDOW_SEC = 24.0
COACH_GRAPH_TICK_SEC = 4.0


def _plot_line_styled(img, hist, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi, color, max_gap_sec,
                       style="solid", thickness=2):
    """Same gap-aware, causal run-detection as `_plot_line` (a real gap
    -- a `None` sample or a timestamp jump beyond `max_gap_sec` -- always
    breaks the line, never bridged/interpolated); adds a `style` choice
    ("solid"/"dashed"/"dotted", via the project's own existing
    `_dashed_line` primitive) so multiple real series can share one
    team color without being visually conflated."""
    history = [(t, v) for t, v in hist if t_lo - 1e-9 <= t <= cur_time + 1e-9]

    def flush(segment):
        if len(segment) >= 2:
            if style == "solid":
                cv2.polylines(img, [np.array(segment, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
            else:
                dash_len, gap_len = (10, 7) if style == "dashed" else (2, 6)
                for i in range(len(segment) - 1):
                    _dashed_line(img, segment[i], segment[i + 1], color, thickness, dash_len=dash_len, gap_len=gap_len)
        elif len(segment) == 1:
            cv2.circle(img, segment[0], 3, color, -1, cv2.LINE_AA)

    segment, last_t = [], None
    for t, v in history:
        gap = (last_t is not None) and (t - last_t > max_gap_sec)
        if v is None or gap:
            flush(segment)
            segment = []
        if v is not None:
            segment.append(_to_px(t, np.clip(v, y_lo, y_hi), t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph))
        last_t = t
    flush(segment)


def draw_line_dynamics_graph(w, h, series_by_team, cur_time, t_lo, t_hi, tick_spacing,
                              release_times, crossing_times, y_range, accent=ACCENT_RED, max_gap_sec=0.5):
    """Graph 1 -- Defensive Line Dynamics. `series_by_team`:
    {team: {"unit_centre": hist, "reference": hist, "deepest": hist}}
    for team in (0,1), each `hist` a real `[(t,v_or_None), ...]` series
    (already in metres). THREE separate series per team.

    2026-09-09 polish: colored by STRUCTURAL identity (Unit Centre /
    Estimated Reference / Deepest Defender -- `STRUCTURE_COLOR`), not by
    team -- the purpose of this graph is to distinguish the three
    structural references at a glance, not to compare teams (both teams'
    own three lines still draw, same structural color, distinguished by
    line STYLE: solid/dashed/dotted, per this project's existing
    three-object distinction). Team identity is instead a tiny "A"/"B"
    badge near each team's own most-recent Unit Centre sample -- never a
    third coloring dimension competing with the structural one.
    `release_times`/`crossing_times`: real event timestamps (already
    causally valid) -- drawn as subtle vertical markers, filtered here
    AGAIN against `[t_lo, cur_time]` so a future event can never leak
    in even if the caller's list were ever built wrong."""
    STYLE = {"unit_centre": "solid", "reference": "dashed", "deepest": "dotted"}
    img, x0, y0, pw, ph, y_lo, y_hi = _chrome(
        w, h, "DEFENSIVE LINE DYNAMICS",
        "Unit Centre (cyan solid) / Reference (white dashed) / Deepest (magenta dotted)",
        "Depth (metres)", y_range, "{:.0f}", t_lo, t_hi, tick_spacing, accent)

    any_marker_drawn = False
    for rt in release_times:
        if t_lo - 1e-9 <= rt <= cur_time + 1e-9:
            xr, _ = _to_px(rt, y_lo, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
            _dashed_line(img, (xr, y0), (xr, y0 + ph), (170, 170, 170), thickness=1, dash_len=6, gap_len=5)
            any_marker_drawn = True
    for ct in crossing_times:
        if t_lo - 1e-9 <= ct <= cur_time + 1e-9:
            xc, _ = _to_px(ct, y_lo, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
            cv2.line(img, (xc, y0), (xc, y0 + ph), (0, 215, 255), 1, cv2.LINE_AA)
            any_marker_drawn = True

    for team in (0, 1):
        for kind, hist in series_by_team[team].items():
            color = STRUCTURE_COLOR[kind]
            _plot_line_styled(img, hist, cur_time, x0, y0, pw, ph, t_lo, t_hi, y_lo, y_hi,
                               color, max_gap_sec, style=STYLE[kind])

    # small team-identity badge (not a coloring dimension) -- one tiny
    # "A"/"B" tag at each team's own latest in-window Unit Centre sample.
    for team in (0, 1):
        hist = series_by_team[team]["unit_centre"]
        latest = None
        for t, v in hist:
            if v is not None and t_lo - 1e-9 <= t <= cur_time + 1e-9:
                latest = (t, v)
        if latest is not None:
            bx, by = _to_px(latest[0], np.clip(latest[1], y_lo, y_hi), t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
            _shadow_text_offside(img, TEAM_LETTER[team], (bx + 6, by - 6), scale=0.34, color=TEAM_COLOR[team])

    cx, _ = _to_px(cur_time, y_lo, t_lo, t_hi, y_lo, y_hi, x0, y0, pw, ph)
    cv2.line(img, (cx, y0), (cx, y0 + ph), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (cx, y0), (cx, y0 + ph), TEXT_WHITE, 2, cv2.LINE_AA)
    _draw_structure_legend(img, x0, pw)
    if any_marker_drawn:
        # only shown when a marker is ACTUALLY visible this frame -- never
        # keyed on whether the caller's full-clip lists are merely non-empty
        # (which would show the note even before any real marker exists yet).
        cv2.putText(img, "-- release   -- crossing", (x0 + 4, y0 + 14), FONT, 0.26, GRAPH_TICK_TEXT, 1, cv2.LINE_AA)
    return img


def draw_trap_break_funnel_graph(w, h, counts, accent=ACCENT_RED):
    """Graph 4 -- Trap-Break Event Funnel. `counts`: the real, causal
    `cumulative_funnel_counts(...)` output for the CURRENT frame's
    `cur_time` (computed by the caller -- this function only draws).
    A single, monotonically-expanding "nice" Y-axis ceiling (never
    hard-coded, never shrinks -- `nice_bar_axis` is monotone and the
    counts themselves are non-decreasing in time, see that function's
    own docstring)."""
    img = panel_frame(w, h, bg=GRAPH_PANEL_BG, border=GRAPH_PANEL_BORDER)
    cv2.rectangle(img, (12, 8), (15, 24), accent, -1)
    cv2.putText(img, "TRAP-BREAK EVENT FUNNEL", (22, 22), FONT, 0.50, TEXT_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "Cumulative pipeline: where do break attempts reach, and where do they fail?",
                (14, 40), FONT, 0.30, GRAPH_TICK_TEXT, 1, cv2.LINE_AA)

    pad_l, pad_r, pad_t, pad_b = 44, 16, 56, 46
    x0, y0 = pad_l, pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    max_count = max([0] + list(counts.values()))
    y_hi = nice_bar_axis(max_count)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), GRAPH_PANEL_BORDER, 1)
    step = max(1, math.ceil(y_hi / 5))
    yv = 0
    while yv <= y_hi:
        gy = y0 + int(ph * (1 - yv / y_hi))
        cv2.line(img, (x0, gy), (x0 + pw, gy), GRAPH_GRID, 1, cv2.LINE_AA)
        cv2.putText(img, str(yv), (6, gy + 4), FONT, 0.26, GRAPH_TICK_TEXT, 1, cv2.LINE_AA)
        yv += step

    n = len(FUNNEL_STAGES)
    bar_gap = 16
    bar_w = (pw - bar_gap * (n - 1)) / n
    for i, stage in enumerate(FUNNEL_STAGES):
        c = counts[stage]
        bar_color = FUNNEL_STAGE_COLOR[stage]
        bx0 = int(x0 + i * (bar_w + bar_gap))
        bx1 = int(bx0 + bar_w)
        by1 = y0 + ph
        bar_h = int(ph * (c / y_hi)) if y_hi > 0 else 0
        by0 = by1 - bar_h
        if c > 0:
            cv2.rectangle(img, (bx0, by0), (bx1, by1), bar_color, -1)
        else:
            cv2.rectangle(img, (bx0, by1 - 2), (bx1, by1), bar_color, -1)  # honest visible sliver at a real 0
        _shadow_text_offside(img, str(c), (bx0, max(y0 + 10, by0 - 6)))
        label = stage.replace(" ", "\n")
        for li, line in enumerate(label.split("\n")):
            (tw_, _), _ = cv2.getTextSize(line, FONT, 0.26, 1)
            cv2.putText(img, line, (int(bx0 + bar_w / 2 - tw_ / 2), y0 + ph + 16 + li * 12), FONT, 0.26, GRAPH_TICK_TEXT, 1, cv2.LINE_AA)
    return img


def _shadow_text_offside(img, text, org, scale=0.30, color=TEXT_WHITE):
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), FONT, scale, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_video_path", default="ExternalDownlaodVideo/testVideo1_120s.mp4")
    ap.add_argument("--tracking_dir", default="outputs/tracking/testVideo1_120s")
    ap.add_argument("--analytics_dir", default="outputs/analytics/testVideo1_120s_v3")
    ap.add_argument("--snapshot_pkl", default="audit_reports/2026-09-06_tactical_review_v4/v4_snapshot.pkl")
    ap.add_argument("--out_dir", default="offside_break/outputs/correctness_v4/dashboard_previews_simple")
    ap.add_argument("--frame", type=int, action="append", default=None)
    ap.add_argument("--name", action="append", default=None)
    ap.add_argument("--time_mode", choices=["rolling", "paper"], default="rolling",
                     help="rolling = trailing 20s window with absolute-time ticks (what --out_path renders); "
                          "paper = full 0-duration static timeline, for paper figures/reports")
    ap.add_argument("--out_path", default=None, help="if set, render the FULL clip to this .mp4 path (always rolling-20s mode)")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=None, help="inclusive; defaults to the last snapshot frame")
    ap.add_argument("--debug", action="store_true",
                     help="(V9) research/debug mode: full-pitch DEEPEST OUTFIELD line (normal mode draws a short "
                          "marker/tick instead). Does not affect the clean final video unless explicitly passed.")
    ap.add_argument("--graph4", choices=["velocity", "timing"], default="velocity",
                     help="(Part 12) Graph 4 variant: 'velocity' (current, default -- Defensive-Line Forward "
                          "Velocity) or 'timing' (Release-to-Cross Timing Margin, event-limited). The current "
                          "velocity graph is NOT permanently replaced -- pass --graph4 timing to preview variant B.")
    ap.add_argument("--radar-view", choices=["flat", "perspective"], default="flat",
                     help="Radar rendering mode: 'flat' (unchanged existing top-down radar) or 'perspective' "
                          "(NEW tilted tactical-board camera, same underlying pitch geometry/analytics).")
    ap.add_argument("--camera-elevation-deg", type=float, default=None, help="override RadarCameraConfig.elevation_deg")
    ap.add_argument("--camera-fov-deg", type=float, default=None, help="override RadarCameraConfig.fov_deg")
    ap.add_argument("--camera-distance-scale", type=float, default=None, help="override RadarCameraConfig.distance_scale")
    args = ap.parse_args()

    camera_kwargs = {}
    if args.camera_elevation_deg is not None:
        camera_kwargs["elevation_deg"] = args.camera_elevation_deg
    if args.camera_fov_deg is not None:
        camera_kwargs["fov_deg"] = args.camera_fov_deg
    if args.camera_distance_scale is not None:
        camera_kwargs["distance_scale"] = args.camera_distance_scale
    radar_camera = RadarCameraConfig(**camera_kwargs) if camera_kwargs else DEFAULT_RADAR_CAMERA
    RADAR_H = compute_radar_homography(DEFAULT_PITCH, radar_camera, 1300, 800) if args.radar_view == "perspective" else None

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

    # 2026-09-08 V9.1 -- FINAL INTEGRATION: the real per-frame/per-run
    # pipeline output (`build_offside_v4()`'s own `runs`/`frames`, the
    # SAME inputs that built `--snapshot_pkl`) feeds
    # `build_pass_run_events_v9()` ONCE here, producing one real,
    # already-resolved event record per detected pass (passer, release,
    # effective reference, association, crossing, timing margin,
    # lifecycle, trap-link) -- the render loop below only VISUALIZES
    # these, it never re-derives them per frame. V6/V7/V8 defensive-unit
    # membership stays FROZEN (unchanged `build_offside_v4` call).
    passes_path = os.path.join(args.analytics_dir, "passes.parquet")
    passes = pl.read_parquet(passes_path).to_dicts() if os.path.exists(passes_path) else []
    passes_by_id = {p["pass_id"]: p for p in passes}
    if passes:
        quality = build_quality_view(tracking, fps=fps, pitch=DEFAULT_PITCH)
        players_by_frame_q = {}
        for r in quality.to_dicts():
            players_by_frame_q.setdefault(r["frame"], []).append(r)
        roles = current_roles(players_by_frame_q, ball_by_frame, len(snapshot), fps=fps, pitch=DEFAULT_PITCH)
        off_raw = build_offside_v4(players_by_frame_q, roles, ball_by_frame, passes, fps=fps, pitch=DEFAULT_PITCH)
        all_events_v9 = build_pass_run_events_v9(passes, off_raw["runs"], off_raw["frames"], ball_by_frame,
                                                  players_by_frame_q, fps=fps, pitch=DEFAULT_PITCH)
    else:
        all_events_v9 = []
    PREROLL_FRAMES = max(0, round(PASSER_PREROLL_SEC * fps))
    PERSIST_FRAMES = max(0, round(RELEASE_MARKER_PERSIST_SEC * fps))

    def _active_event_for_frame(f):
        for e in all_events_v9:
            p = passes_by_id.get(e["pass_event_id"])
            if p is None:
                continue
            lo, hi = p["start_frame"] - PREROLL_FRAMES, p["end_frame"] + PERSIST_FRAMES
            if lo <= f <= hi:
                return e, p
        return None, None

    gx, gy = pitch_grid(DEFAULT_PITCH)
    duration_sec = len(snapshot) / fps

    # ---- FULL-CLIP series, precomputed ONCE (not per frame) ----
    # 2026-09-08 V9 correctness fix: Graph 1 is the DEFENSIVE UNIT CENTRE
    # (the PRIMARY trap-analysis reference per this round's explicit
    # three-object separation), not the ESTIMATED OFFSIDE REFERENCE --
    # it previously plotted `line["line_depth_cm"]` (the reference),
    # which is Graph 3's own object, not Graph 1's. Fixed to read
    # `back_line["centre_depth_cm"]` instead; no other graph's data
    # source changes.
    full_d0 = _full_series(snapshot, lambda s: (s["offside"][0]["back_line"]["centre_depth_cm"] / 100.0) if s["offside"][0]["back_line"]["valid"] else None)
    full_d1 = _full_series(snapshot, lambda s: (s["offside"][1]["back_line"]["centre_depth_cm"] / 100.0) if s["offside"][1]["back_line"]["valid"] else None)
    y_depth = _signed_y_range([v for _, v in full_d0 + full_d1], floor_zero=True)

    full_s0 = _full_series(snapshot, lambda s: s["offside"][0]["raw_score"])
    full_s1 = _full_series(snapshot, lambda s: s["offside"][1]["raw_score"])

    full_lead0_m = _lead_series_with_breaks(snapshot, 0)
    full_lead1_m = _lead_series_with_breaks(snapshot, 1)
    y_lead = _signed_y_range([v for _, v in full_lead0_m + full_lead1_m])

    full_lv0 = _full_series(snapshot, lambda s: s["offside"][0].get("line_velocity_cm_s"))
    full_lv1 = _full_series(snapshot, lambda s: s["offside"][1].get("line_velocity_cm_s"))
    full_lv0_ms = [(t, v / 100.0 if v is not None else None) for t, v in full_lv0]
    full_lv1_ms = [(t, v / 100.0 if v is not None else None) for t, v in full_lv1]
    y_lv = _signed_y_range([v for _, v in full_lv0_ms + full_lv1_ms])

    # ---------------------------------------------------------------------
    # 2026-09-09 -- COACH-MODE second-row graphs (Candidates A/G/C/H, per
    # audit_reports/2026-09-09_offside_graph_feasibility/). All read
    # EXISTING fields already baked into the frozen snapshot (`line`/
    # `back_line`, the SAME structures `build_offside_v4` itself
    # produces -- verified identical schema) or the existing
    # `all_events_v9`/`behind_line_threat_field` -- no new analytics.
    #
    # Graph 1 (Defensive Line Dynamics) additionally needs the ESTIMATED
    # OFFSIDE REFERENCE and DEEPEST OUTFIELD DEFENDER series -- kept
    # SEPARATE from `full_d0`/`full_d1` (Unit Centre) above, never
    # conflated, per this project's own three-object distinction.
    full_ref0 = _full_series(snapshot, lambda s: (s["offside"][0]["line"]["line_depth_cm"] / 100.0) if s["offside"][0]["line"]["valid"] else None)
    full_ref1 = _full_series(snapshot, lambda s: (s["offside"][1]["line"]["line_depth_cm"] / 100.0) if s["offside"][1]["line"]["valid"] else None)
    full_deep0 = _full_series(snapshot, lambda s: (s["offside"][0]["back_line"]["deep_depth_cm"] / 100.0) if s["offside"][0]["back_line"]["valid"] else None)
    full_deep1 = _full_series(snapshot, lambda s: (s["offside"][1]["back_line"]["deep_depth_cm"] / 100.0) if s["offside"][1]["back_line"]["valid"] else None)
    y_line_dynamics = _signed_y_range([v for _, v in full_d0 + full_d1 + full_ref0 + full_ref1 + full_deep0 + full_deep1], floor_zero=True)

    # Graph 2 (Behind-Line Space, Candidate G): `behind_line_threat_field`
    # reduced to its mean value over the behind-reference region --
    # identical definition to the feasibility audit's own Plot 6, and
    # the SAME function the existing top-row radar wash already calls
    # (read-only reuse, never modified). Computed once per team per
    # frame from the real, already-loaded `cleaned_by_frame`/`gx`/`gy`.
    full_behind0_raw = [None] * len(snapshot)
    full_behind1_raw = [None] * len(snapshot)
    for f in range(len(snapshot)):
        players_f = [p for p in cleaned_by_frame.get(f, []) if p.get("x_pitch") is not None]
        for defending in (0, 1):
            attacking = 1 - defending
            bl = snapshot[f]["offside"][defending]["back_line"]
            if not bl["valid"] or bl["centre_depth_cm"] is None:
                continue
            att_players = [p for p in players_f if p["display_team_id"] == attacking and p["display_object_type"] in ("player", "goalkeeper")]
            def_players = [p for p in players_f if p["display_team_id"] == defending and p["display_object_type"] in ("player", "goalkeeper")]
            if not att_players or not def_players:
                continue
            field = behind_line_threat_field(att_players, def_players, gx, gy, defending, bl["centre_depth_cm"], DEFAULT_PITCH)
            (full_behind0_raw if defending == 0 else full_behind1_raw)[f] = float(field.mean())
    full_behind0 = [(snapshot[f]["time_sec"], full_behind0_raw[f]) for f in range(len(snapshot))]
    full_behind1 = [(snapshot[f]["time_sec"], full_behind1_raw[f]) for f in range(len(snapshot))]
    y_behind = _signed_y_range([v for _, v in full_behind0 + full_behind1], floor_zero=True)

    # Graph 1 event markers: real release/crossing times, straight from
    # the already-computed, unmodified `all_events_v9` -- never
    # re-derived, never a future event shown before it actually happens
    # (the draw function itself also re-checks against `cur_time`).
    release_marker_times = sorted(e["release_time_sec"] for e in all_events_v9 if e.get("release_valid") and e.get("release_time_sec") is not None)
    crossing_marker_times = sorted(e["observed_crossing_time_sec"] for e in all_events_v9 if e.get("cross_valid") and e.get("observed_crossing_time_sec") is not None)

    def render_frame(f, seek=True, time_mode="rolling"):
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

        # Consolidated relevant-attacker identity, per attacking team --
        # the EXACT same _select_lead_runner rule Graph 3 uses, so radar
        # and graph always point at the same player.
        lead_runners = {t: _select_lead_runner(snap_f["offside"][t].get("runners") or []) for t in (0, 1)}
        # (8) line-crossing detection: compare this frame's selected
        # runner to the PREVIOUS frame's selected runner for the SAME
        # run_id -- a real sign-flip of the same attacker's relation_cm,
        # never inferred across different attackers.
        crossed_teams = set()
        if f > 0:
            prev_f = snapshot[f - 1]
            for t in (0, 1):
                cur_r = lead_runners[t]
                if cur_r is None:
                    continue
                prev_r = _select_lead_runner(prev_f["offside"][t].get("runners") or [])
                if prev_r is not None and prev_r["run_id"] == cur_r["run_id"]:
                    if (prev_r["relation_cm"] <= 0) != (cur_r["relation_cm"] <= 0):
                        crossed_teams.add(t)
        selected_track_id = next((r["track_id"] for r in lead_runners.values() if r is not None), None)

        # (V9.1) the ONE real, already-integrated event covering this
        # frame (if any) -- passer/release/association/crossing/timing/
        # lifecycle all read from `all_events_v9`, never re-derived here.
        active_event, active_pass = _active_event_for_frame(f)
        passer, release_active, pass_vector_end_xy, debug_lines = None, False, None, []
        defending_team_for_threat = None
        if active_event is not None:
            passer = dict(passer_track_id=active_event["passer_track_id"], team=active_event["passer_team"],
                          confidence=active_event["passer_confidence"])
            defending_team_for_threat = 1 - active_event["passer_team"]
            if active_event["release_valid"]:
                release_active = active_event["release_frame"] <= f <= active_event["release_frame"] + PERSIST_FRAMES
                if f >= active_event["release_frame"]:
                    runner_row = next((p for p in players_this_frame if p.get("track_id") == active_event["runner_track_id"]), None)
                    pass_vector_end_xy = ((runner_row["x_pitch"], runner_row["y_pitch"]) if runner_row is not None
                                          else (active_pass["end_x"], active_pass["end_y"]))
            if args.debug:
                debug_lines = [
                    f"pass#{active_event['pass_event_id']} release={'VALID' if active_event['release_valid'] else 'INVALID'}"
                    f" evidence={active_event['release_evidence']}",
                    f"eff_ref_source={active_event['effective_reference_source']}  assoc_valid={active_event['association_valid']}",
                    f"timing_margin={active_event['timing_margin_sec']}  ({active_event['timing_margin_label']})",
                    f"lifecycle={active_event['lifecycle_outcome']}  ({active_event['lifecycle_reason']})",
                ]
        else:
            lr = next(((t, r) for t, r in lead_runners.items() if r is not None), None)
            if lr is not None:
                defending_team_for_threat = 1 - lr[0]

        # (Part 9) behind-line threat -- only when the DEFENDING team's
        # unit centre is valid AND a relevant attacking presence exists.
        threat_field = None
        if defending_team_for_threat is not None:
            bl = snap_f["offside"][defending_team_for_threat]["back_line"]
            if bl.get("valid") and bl.get("centre_depth_cm") is not None:
                attacking_team_for_threat = 1 - defending_team_for_threat
                attacking_players = [p for p in players_this_frame if p["display_team_id"] == attacking_team_for_threat
                                      and p["display_object_type"] in ("player", "goalkeeper")]
                defending_players = [p for p in players_this_frame if p["display_team_id"] == defending_team_for_threat
                                      and p["display_object_type"] in ("player", "goalkeeper")]
                if attacking_players and defending_players:
                    threat_field = behind_line_threat_field(attacking_players, defending_players, gx, gy,
                                                             defending_team_for_threat, bl["centre_depth_cm"], DEFAULT_PITCH)

        left = cv2.resize(draw_left_feed(frame, tracking_by_frame.get(f, []), snap_f, transformer, selected_track_id,
                                          players_this_frame, args.debug, passer, release_active, pass_vector_end_xy,
                                          threat_field, debug_lines), (LEFT_W, TOP_ROW_H))
        radar = draw_radar(players_this_frame, ball_xy, snap_f, RIGHT_W, TOP_ROW_H, lead_runners, crossed_teams,
                            args.debug, passer, release_active, pass_vector_end_xy, threat_field,
                            radar_view=args.radar_view, H=RADAR_H)
        top_row = np.hstack([left, radar])

        header = draw_header(CANVAS_W, HEADER_H, "OFFSIDE / TRAP-BREAK ANALYTICS -- V4",
                              (f"t = {snap_f['time_sec']:.1f}s", TEXT_DIM), accent=ACCENT_RED)

        cur_time = snap_f["time_sec"]
        if time_mode == "rolling":
            t_lo, t_hi, tick_spacing = max(0.0, cur_time - 20.0), cur_time, 4
        else:
            t_lo, t_hi, tick_spacing = 0.0, duration_sec, 30

        # 2026-09-09 -- COACH-MODE second-row graphs. Graphs 1-3 use their
        # OWN rolling 24s window (`COACH_GRAPH_WINDOW_SEC`), DELIBERATELY
        # separate from the `--time_mode`-driven `t_lo`/`t_hi` above (kept,
        # unused by these graphs, only for any future research-style
        # variant) -- same convention the now-locked pressing dashboard
        # already uses. Graph 4 is cumulative, not rolling, so it takes
        # no window at all.
        t_lo24 = max(0.0, cur_time - COACH_GRAPH_WINDOW_SEC)
        t_hi24 = cur_time
        # 2026-09-09 FINAL POLISH (+ 2026-09-09 CORRECTION) -- exactly THREE
        # coach graphs: DEFENSIVE LINE DYNAMICS / BEHIND-LINE SPACE /
        # TRAP-BREAK EVENT FUNNEL. The correction round swapped the middle
        # panel back to Behind-Line Space (Candidate G -- stronger real
        # coverage/continuity per the audit) and OUT Most-Threatening-
        # Attacker Margin; that graph's own precompute
        # (`full_lead0_m`/`full_lead1_m`/`y_lead` above) and its drawing
        # function (`draw_lead_margin_graph`) are both kept, just unused
        # here, for research purposes -- same pattern the PREVIOUS round
        # used to keep Behind-Line Space's own precompute around while
        # temporarily out of the layout. Panel widths stay ~36%/34%/30%
        # across 2 gaps -- Graph 1 gets the most room (3 series + event
        # markers), Graph 2 enough width to read the threat-field trend,
        # Graph 3 can be narrowest (categorical funnel).
        total_gap = 2 * GAP
        avail = CANVAS_W - total_gap
        w1 = round(avail * 0.36)
        w2 = round(avail * 0.34)
        w3 = avail - w1 - w2
        series_by_team = {
            0: {"unit_centre": full_d0, "reference": full_ref0, "deepest": full_deep0},
            1: {"unit_centre": full_d1, "reference": full_ref1, "deepest": full_deep1},
        }
        g1 = draw_line_dynamics_graph(w1, GRAPH_ROW_H, series_by_team, cur_time, t_lo24, t_hi24, COACH_GRAPH_TICK_SEC,
                                       release_marker_times, crossing_marker_times, y_line_dynamics)
        g2 = draw_dual_graph(w2, GRAPH_ROW_H, full_behind0, full_behind1, cur_time, t_lo24, t_hi24, COACH_GRAPH_TICK_SEC,
                              "BEHIND-LINE SPACE",
                              "Dangerous space available behind the defensive reference",
                              "Behind-Line Threat (real value)", y_behind, "{:.2f}", "Team A", "Team B",
                              TEAM_COLOR[0], TEAM_COLOR[1])
        funnel_counts = cumulative_funnel_counts(all_events_v9, cur_time)
        g3 = draw_trap_break_funnel_graph(w3, GRAPH_ROW_H, funnel_counts)
        gap_col1 = np.full((GRAPH_ROW_H, GAP, 3), BG, dtype=np.uint8)
        gap_col2 = np.full((GRAPH_ROW_H, GAP, 3), BG, dtype=np.uint8)
        graph_row = np.hstack([g1, gap_col1, g2, gap_col2, g3])
        if graph_row.shape[1] != CANVAS_W:
            graph_row = cv2.resize(graph_row, (CANVAS_W, GRAPH_ROW_H))

        row_gap = np.full((GAP, CANVAS_W, 3), BG, dtype=np.uint8)
        canvas = np.vstack([header, top_row, row_gap, graph_row])
        return canvas

    if args.out_path:
        f_start = args.start_frame
        f_end = args.end_frame if args.end_frame is not None else len(snapshot) - 1
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_path, fourcc, fps, (CANVAS_W, 1204))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
        n_written = 0
        t0 = time.time()
        for f in range(f_start, f_end + 1):
            canvas = render_frame(f, seek=False, time_mode="rolling")
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
        canvas = render_frame(f, time_mode=args.time_mode)
        if canvas is None:
            print(f"frame {f}: could not read video frame")
            continue
        out_path = os.path.join(args.out_dir, f"{name}.png")
        cv2.imwrite(out_path, canvas)
        print(f"wrote {out_path} ({canvas.shape[1]}x{canvas.shape[0]})")
    cap.release()


if __name__ == "__main__":
    main()
