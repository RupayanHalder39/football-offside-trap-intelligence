"""
2026-09-08 V9 -- PASSER IDENTITY + PASS-RELEASE TIMING for the offside/
trap-break paper. This is the module the V8 report flagged as the
project's central missing timing event: "WHEN DID THE PASS LEAVE THE
PASSER?"

DELIBERATE REUSE, NOT REINVENTION: this repo already has an audited,
validated pass detector (`analytics/pass_detection.py`'s
`detect_passes_possession()`/`detect_passes_hybrid()`, exported to
`outputs/analytics/testVideo1_120s_v3/passes.parquet`) built on top of
`analytics/possession.py`'s possession-transition evidence
(`source_track_id`, `team_id`, `confidence`, `source_possession_confidence`,
`trajectory_support`, ...). A manual audit already found that
trajectory-geometry-only detection missed 11/15 real passes on this
clip -- there is no reason to re-derive "who passed" from raw ball
proximity here. What THAT pipeline does NOT compute is a frame-precise
RELEASE INSTANT inside its own [start_frame, end_frame] window, which
is exactly this paper's research question (trap-break timing). This
module adds ONLY that, plus the reference/runner/timing-margin/lifecycle
machinery built on top of it.

Nothing here claims an official offside decision -- see
`ESTIMATED_...` naming throughout, matching `offside_v4.py`'s existing
convention.

All temporal parameters are declared in SECONDS and converted to frame
counts via `round(sec * fps)` -- fps is never hard-coded to 30.
"""
import math

import numpy as np

from tactical_shared.coordinates import DEFAULT_PITCH
from offside_break.analytics.offside_v4 import ball_x, effective_offside_reference_depth

# Reused, not reinvented: the SAME possession "confident control" radius
# `analytics/possession.py` already calibrated for this clip's own
# camera/detection/pitch-mapping precision (see that module's docstring).
PASSER_PROXIMITY_SCALE_CM = 200.0

# ---------------------------------------------------------------------
# A. PASSER CANDIDATE DETECTION
# ---------------------------------------------------------------------
PASSER_PREROLL_SEC = 0.3          # a short build-up window before the pass detector's own start_frame
PASSER_CONFIRM_THRESHOLD = 0.5


class PasserCandidateTracker:
    """Temporal passer identity. NEVER "nearest player to the ball every
    frame" -- a passer is only ever assigned inside a REAL, already-
    detected pass event's own window (`source_track_id`, `confidence`
    from the possession-evidenced detector), with one further REAL
    check layered on top: actual ball-to-passer proximity at the
    queried frame, from this frame's own tracking data, so a temporally
    stale/misattributed upstream label cannot silently pass through as
    "confirmed" without present-frame support."""

    def __init__(self, passes, fps=30., preroll_sec=PASSER_PREROLL_SEC, pitch=DEFAULT_PITCH):
        self.fps = fps
        self.preroll_frames = max(0, round(preroll_sec * fps))
        self.pitch = pitch
        self.events = sorted(passes, key=lambda p: p['start_frame'])

    def _active_event(self, frame):
        for p in self.events:
            lo = p['start_frame'] - self.preroll_frames
            if lo <= frame <= p['end_frame']:
                return p
            if p['start_frame'] - self.preroll_frames > frame:
                break
        return None

    def step(self, frame, players, ball_row):
        active = self._active_event(frame)
        if active is None:
            return dict(passer_track_id=None, team=None, confidence=0.0,
                        state='NO_ACTIVE_PASS_EVENT', pass_id=None, scene_id=None)
        passer_row = next((p for p in players if p.get('track_id') == active['source_track_id']
                            and p.get('x_pitch') is not None), None)
        ball_xy = None
        if ball_row and ball_row.get('is_observed') and ball_row.get('x_pitch') is not None:
            ball_xy = (ball_row['x_pitch'], ball_row['y_pitch'])
        if passer_row is not None and ball_xy is not None:
            d = math.hypot(passer_row['x_pitch'] - ball_xy[0], passer_row['y_pitch'] - ball_xy[1])
            proximity_term = math.exp(-d / PASSER_PROXIMITY_SCALE_CM)
        else:
            # ball or passer position unavailable THIS frame -- neutral,
            # never fabricated as confirmed from an absent reading.
            proximity_term = 0.5
        confidence = float(0.6 * (active.get('confidence') or 0.0) + 0.4 * proximity_term)
        state = 'CONFIRMED_PASSER' if confidence >= PASSER_CONFIRM_THRESHOLD else 'CANDIDATE_PASSER'
        return dict(passer_track_id=active['source_track_id'], team=active['team_id'],
                    confidence=round(confidence, 4), state=state,
                    pass_id=active['pass_id'], scene_id=active.get('scene_id', 0), event=active)


# ---------------------------------------------------------------------
# B/C. PASS-RELEASE FRAME + CONFIDENCE
# ---------------------------------------------------------------------
# Four cues, none decisive alone (an evidence-aware combination, per the
# explicit instruction not to rely on a single threshold):
#   near_term:        ball within the SAME possession-control radius of
#                      the passer at frame f (cue 1).
#   speed_term:       ball speed rises from f to a short look-ahead --
#                      RELEASE_SPEED_JUMP_CM_S is a disclosed modeling
#                      choice (a brisk pass speed increase), not fit to
#                      this clip's own outcomes (cue 2).
#   separation_term:  ball-to-passer distance GROWS over that same
#                      look-ahead (cue 3).
#   direction_term:   the ball's own instantaneous heading agrees with
#                      the bearing toward the pass event's already-known
#                      end point (cue 4 -- "direction becomes consistent
#                      with a pass"; cue 5, "a receiver exists in a
#                      plausible downstream direction", is already
#                      implied by the upstream detector's own
#                      target_track_id/end_x/end_y, reused here rather
#                      than re-derived).
RELEASE_LOOKAHEAD_SEC = 0.2
RELEASE_SPEED_JUMP_CM_S = 300.0
RELEASE_SEPARATION_CM = 150.0
RELEASE_DIRECTION_TOLERANCE_DEG = 60.0
RELEASE_MIN_EVIDENCE = 0.5
W_NEAR, W_SPEED, W_SEPARATION, W_DIRECTION = 0.25, 0.30, 0.25, 0.20


def _bearing_deg(x0, y0, x1, y1):
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def _angle_diff_deg(a, b):
    return abs((a - b + 180) % 360 - 180)


def detect_release(event, ball_by_frame, players_by_frame, fps=30.):
    """Evidence-scored release frame inside `event`'s own
    [start_frame, end_frame] window (from the already-detected pass).
    Returns valid=False with an explicit reason -- never a fabricated
    release -- whenever ball tracking in the window is too poor to
    support a decision."""
    lookahead = max(1, round(RELEASE_LOOKAHEAD_SEC * fps))
    best = None
    for f in range(event['start_frame'], event['end_frame'] + 1):
        ball_f = ball_by_frame.get(f)
        ball_after = ball_by_frame.get(f + lookahead)
        players = players_by_frame.get(f, [])
        passer_f = next((p for p in players if p.get('track_id') == event['source_track_id']
                          and p.get('x_pitch') is not None), None)
        if not (ball_f and ball_f.get('is_observed') and ball_f.get('x_pitch') is not None
                and ball_after and ball_after.get('is_observed') and ball_after.get('x_pitch') is not None
                and passer_f is not None):
            continue
        d_at = math.hypot(passer_f['x_pitch'] - ball_f['x_pitch'], passer_f['y_pitch'] - ball_f['y_pitch'])
        d_after = math.hypot(passer_f['x_pitch'] - ball_after['x_pitch'], passer_f['y_pitch'] - ball_after['y_pitch'])
        near_term = 1.0 if d_at <= PASSER_PROXIMITY_SCALE_CM else math.exp(-(d_at - PASSER_PROXIMITY_SCALE_CM) / PASSER_PROXIMITY_SCALE_CM)
        speed_before = ball_f.get('speed_cm_s') or 0.0
        speed_after = ball_after.get('speed_cm_s') or 0.0
        speed_term = float(np.clip((speed_after - speed_before) / RELEASE_SPEED_JUMP_CM_S, 0.0, 1.0))
        separation_term = float(np.clip((d_after - d_at) / RELEASE_SEPARATION_CM, 0.0, 1.0))
        bearing_ball = _bearing_deg(ball_f['x_pitch'], ball_f['y_pitch'], ball_after['x_pitch'], ball_after['y_pitch'])
        bearing_target = _bearing_deg(ball_f['x_pitch'], ball_f['y_pitch'], event['end_x'], event['end_y'])
        direction_term = float(np.clip(1.0 - _angle_diff_deg(bearing_ball, bearing_target) / RELEASE_DIRECTION_TOLERANCE_DEG, 0.0, 1.0))
        evidence = W_NEAR * near_term + W_SPEED * speed_term + W_SEPARATION * separation_term + W_DIRECTION * direction_term
        # Tie-break toward the LATER frame: a wide look-ahead window can
        # make several consecutive pre-release frames score identically
        # (the ball hasn't moved yet at any of them) -- the frame
        # closest to the actual separation is the more useful "release
        # instant" than the earliest indistinguishable one.
        if best is None or evidence >= best['evidence']:
            best = dict(frame=f, time_sec=f / fps, evidence=round(float(evidence), 4),
                        ball_x=ball_f['x_pitch'], ball_y=ball_f['y_pitch'],
                        near_term=round(near_term, 4), speed_term=round(speed_term, 4),
                        separation_term=round(separation_term, 4), direction_term=round(direction_term, 4))
    if best is None:
        return dict(valid=False, frame=None, time_sec=None, evidence=0.0,
                    reason='BALL_OR_PASSER_UNOBSERVED_IN_WINDOW')
    if best['evidence'] < RELEASE_MIN_EVIDENCE:
        return {**best, 'valid': False, 'reason': 'INSUFFICIENT_RELEASE_EVIDENCE'}
    return {**best, 'valid': True, 'reason': None}


# ---------------------------------------------------------------------
# D. EFFECTIVE OFFSIDE REFERENCE AT RELEASE
# ---------------------------------------------------------------------
def effective_reference_at_release(release, defending_team, lines_by_team, ball_by_frame, pitch=DEFAULT_PITCH):
    """Reuses the EXACT same `effective_offside_reference_depth` (V8)
    comparison -- second-last opponent vs the ball, whichever is closer
    to goal -- evaluated AT the release frame specifically. Never
    'official offside'; ESTIMATED only."""
    if not release.get('valid'):
        return dict(valid=False, reference_depth_cm=None, source=None,
                    reason=release.get('reason', 'RELEASE_INVALID'))
    f = release['frame']
    line = lines_by_team.get(defending_team, {}).get(f)
    if not line or not line.get('valid'):
        return dict(valid=False, reference_depth_cm=None, source=None,
                    reason='DEFENSIVE_LINE_UNAVAILABLE_AT_RELEASE')
    ball_val = ball_x(ball_by_frame.get(f))
    depth, source = effective_offside_reference_depth(line, ball_val, defending_team, f, pitch)
    if depth is None:
        return dict(valid=False, reference_depth_cm=None, source=None, reason='LINE_INVALID_AT_RELEASE')
    return dict(valid=True, reference_depth_cm=depth, source=source, frame=f, reason=None)


# ---------------------------------------------------------------------
# E. RUNNER POSITION AT RELEASE
# ---------------------------------------------------------------------
def runner_state_at_release(runner_row, release_frame, effective_reference_depth_cm,
                             attacking_team, defending_team, pitch=DEFAULT_PITCH):
    """`distance_to_reference_cm` is measured in the DEFENDING team's
    own depth frame (matching the reference's own frame): positive ==
    the runner sits BEHIND the reference (an ESTIMATED legal position),
    negative == the runner is AHEAD of it (an ESTIMATED offside
    position) -- never an official call."""
    if runner_row is None or runner_row.get('x_pitch') is None:
        return dict(valid=False, reason='RUNNER_UNOBSERVED_AT_RELEASE')
    runner_depth = pitch.depth(defending_team, runner_row['x_pitch'], release_frame)
    distance_to_reference_cm = effective_reference_depth_cm - runner_depth
    forward_speed_cm_s = None
    if runner_row.get('motion_valid') and runner_row.get('vx_cm_s') is not None:
        forward_speed_cm_s = runner_row['vx_cm_s'] * pitch.attacking_sign(attacking_team, release_frame)
    run_direction_deg = None
    if runner_row.get('motion_valid') and runner_row.get('vx_cm_s') is not None and runner_row.get('vy_cm_s') is not None:
        run_direction_deg = math.degrees(math.atan2(runner_row['vy_cm_s'], runner_row['vx_cm_s']))
    estimated_time_to_crossing_sec = None
    if forward_speed_cm_s is not None and forward_speed_cm_s > 1e-6 and distance_to_reference_cm > 0:
        estimated_time_to_crossing_sec = distance_to_reference_cm / forward_speed_cm_s
    return dict(valid=True, runner_depth_cm=runner_depth, distance_to_reference_cm=distance_to_reference_cm,
                forward_speed_cm_s=forward_speed_cm_s, run_direction_deg=run_direction_deg,
                estimated_time_to_crossing_sec=estimated_time_to_crossing_sec,
                estimated_legal_at_release=distance_to_reference_cm > 0)


# ---------------------------------------------------------------------
# L. PASS TIMING METRICS
# ---------------------------------------------------------------------
def timing_margin_sec(t_release_sec, t_cross_sec):
    """TimingMargin = t_cross - t_release. Positive: pass released
    before the runner's estimated crossing. ~Zero: nearly simultaneous.
    Negative: the runner's estimated crossing preceded release. This is
    ESTIMATED TRACKING GEOMETRY, never an official onside/offside call."""
    if t_release_sec is None or t_cross_sec is None:
        return None
    return t_cross_sec - t_release_sec


def actual_crossing_time_sec(run_samples, fps=30.):
    """The FIRST observed frame in this run's own recorded samples
    where `relation_cm` goes from <=0 to >0 (an ACTUAL observed
    crossing, matching `classify_break_lifecycle`'s existing
    `crossings` definition) -- None if no such transition was observed
    (an honest gap, never interpolated)."""
    for a, b in zip(run_samples, run_samples[1:]):
        if a['relation_cm'] <= 0 < b['relation_cm']:
            return b['frame'] / fps
    return None


# ---------------------------------------------------------------------
# F. RUN LIFECYCLE (v9) -- standalone, NOT yet wired into
# `build_offside_v4()`'s main per-frame loop this round (see the V9
# report's "algorithmic changes made" item for why: it needs the full
# passes list + ball_by_frame + per-team lines threaded together at the
# per-run level, which the existing pipeline does not currently do, and
# retrofitting that into the already-tested V4-V8 loop mid-round risked
# destabilizing 141 passing tests for a feature this module can fully
# exercise standalone first). Deliberately requires the FULL evidence
# chain before SUCCESSFUL_BREAK; any missing link -> UNCERTAIN.
LIFECYCLE_V9_STATES = ('RUN_DEVELOPING', 'LINE_THREAT', 'RELEASE_WINDOW', 'LINE_CROSSED',
                       'POTENTIAL_BREAK', 'SUCCESSFUL_BREAK', 'FAILED_BREAK', 'UNCERTAIN')


def classify_run_lifecycle_v9(run, release, runner_at_release, run_samples, fps=30.,
                               reception_evidence=None):
    """`run`: an existing `RunTracker` run dict (has start_frame/
    end_frame/attacker_track_id/...). `release`: `detect_release()`
    output. `runner_at_release`: `runner_state_at_release()` output at
    that same release frame. `reception_evidence`: True/False/None
    (None == not evaluated -- kept UNCERTAIN, never assumed)."""
    lifecycle = [dict(state='RUN_DEVELOPING', event_frame=run['start_frame'])]
    threat = next((s for s in run_samples if s.get('threat') is not None and s['threat'] >= .5
                   and -500 <= s['relation_cm'] <= 100), None)
    if threat:
        lifecycle.append(dict(state='LINE_THREAT', event_frame=threat['frame']))
    if not release.get('valid'):
        lifecycle.append(dict(state='UNCERTAIN', event_frame=run.get('end_frame', run['start_frame'])))
        return dict(lifecycle=lifecycle, outcome='UNCERTAIN', reason='RELEASE_NOT_DETECTED')
    lifecycle.append(dict(state='RELEASE_WINDOW', event_frame=release['frame']))
    t_cross = actual_crossing_time_sec(run_samples, fps)
    if t_cross is not None:
        lifecycle.append(dict(state='LINE_CROSSED', event_frame=round(t_cross * fps)))
        lifecycle.append(dict(state='POTENTIAL_BREAK', event_frame=round(t_cross * fps)))
    if not runner_at_release.get('valid'):
        lifecycle.append(dict(state='UNCERTAIN', event_frame=release['frame']))
        return dict(lifecycle=lifecycle, outcome='UNCERTAIN', reason='RUNNER_UNOBSERVED_AT_RELEASE')
    legal_at_release = runner_at_release.get('estimated_legal_at_release', False)
    crossed_after_release = t_cross is not None and t_cross >= release['time_sec']
    margin = timing_margin_sec(release['time_sec'], t_cross)
    if not legal_at_release:
        lifecycle.append(dict(state='FAILED_BREAK', event_frame=release['frame']))
        return dict(lifecycle=lifecycle, outcome='FAILED_BREAK',
                    reason='ESTIMATED_OFFSIDE_AT_RELEASE_NOT_OFFICIAL_OFFENCE', timing_margin_sec=margin)
    if t_cross is None:
        lifecycle.append(dict(state='UNCERTAIN', event_frame=release['frame']))
        return dict(lifecycle=lifecycle, outcome='UNCERTAIN', reason='NO_OBSERVED_LINE_CROSSING', timing_margin_sec=margin)
    if not crossed_after_release:
        lifecycle.append(dict(state='UNCERTAIN', event_frame=release['frame']))
        return dict(lifecycle=lifecycle, outcome='UNCERTAIN', reason='CROSSING_PRECEDED_RELEASE', timing_margin_sec=margin)
    if reception_evidence is not True:
        lifecycle.append(dict(state='UNCERTAIN', event_frame=round(t_cross * fps)))
        return dict(lifecycle=lifecycle, outcome='UNCERTAIN', reason='RECEPTION_NOT_EVALUATED', timing_margin_sec=margin)
    lifecycle.append(dict(state='SUCCESSFUL_BREAK', event_frame=round(t_cross * fps)))
    return dict(lifecycle=lifecycle, outcome='SUCCESSFUL_BREAK',
                reason='ESTIMATED_LEGAL_RELEASE_PLUS_OBSERVED_CROSSING_PLUS_RECEPTION', timing_margin_sec=margin)


# ---------------------------------------------------------------------
# K. BEHIND-LINE THREAT VALUE (a single interpretable time differential,
# built on the SAME time-to-reach physics as
# `pressing_structure.analytics.pressure_field` -- duplicated here as a
# small pure function, rather than importing/modifying that module, to
# respect this round's pressing freeze: the offside paper reuses the
# CONCEPT, not the pressing module itself.)
BEHIND_LINE_BASE_SPEED_CM_S = 400.0     # same disclosed modeling choice as pressure_field.BASE_SPEED_CM_S
BEHIND_LINE_MIN_SPEED_CM_S = 100.0
BEHIND_LINE_MAX_SPEED_CM_S = 824.0      # this project's own measured 95th-pct sprint speed (reused, not reinvented)


def _time_to_reach_sec(x, y, target_x, target_y, vx, vy, motion_valid):
    d = math.hypot(target_x - x, target_y - y)
    if motion_valid and vx is not None and vy is not None and d > 1e-6:
        closing = (vx * (target_x - x) + vy * (target_y - y)) / d
    else:
        closing = 0.0
    eff_speed = float(np.clip(BEHIND_LINE_BASE_SPEED_CM_S + closing, BEHIND_LINE_MIN_SPEED_CM_S, BEHIND_LINE_MAX_SPEED_CM_S))
    return d / eff_speed


def runner_behind_line_advantage(runner_row, target_xy, nearest_defender_row):
    """Simple, interpretable quantity (K): the defending team's nearest
    recovery time-to-reach the target point MINUS the attacking
    runner's own arrival time-to-reach it. Positive == the runner gets
    there first (an advantage); negative == the defender recovers
    first. No composite/invented scoring -- a single time differential."""
    if runner_row is None or runner_row.get('x_pitch') is None or nearest_defender_row is None or nearest_defender_row.get('x_pitch') is None:
        return dict(valid=False, reason='RUNNER_OR_DEFENDER_UNOBSERVED')
    tx, ty = target_xy
    attacking_arrival_sec = _time_to_reach_sec(runner_row['x_pitch'], runner_row['y_pitch'], tx, ty,
                                                runner_row.get('vx_cm_s'), runner_row.get('vy_cm_s'),
                                                runner_row.get('motion_valid', False))
    defending_recovery_sec = _time_to_reach_sec(nearest_defender_row['x_pitch'], nearest_defender_row['y_pitch'], tx, ty,
                                                 nearest_defender_row.get('vx_cm_s'), nearest_defender_row.get('vy_cm_s'),
                                                 nearest_defender_row.get('motion_valid', False))
    return dict(valid=True, attacking_arrival_sec=round(attacking_arrival_sec, 3),
                defending_recovery_sec=round(defending_recovery_sec, 3),
                advantage_sec=round(defending_recovery_sec - attacking_arrival_sec, 3))


# ---------------------------------------------------------------------
# M. FUTURE COUNTERFACTUAL TIMING -- structure only, NO fabricated
# values. `delta_run_sec`/`delta_pass_sec` name the grid the eventual
# V(delta_run, delta_pass) optimization will sweep; nothing here
# evaluates that function yet.
COUNTERFACTUAL_RUN_OFFSETS_SEC = (-0.4, -0.2, 0.0, 0.2, 0.4)
COUNTERFACTUAL_PASS_OFFSETS_SEC = (-0.4, -0.2, 0.0, 0.2, 0.4)


def counterfactual_timing_grid():
    """Returns the (delta_run_sec, delta_pass_sec) grid cells for the
    FUTURE V(delta_run, delta_pass) optimization -- a plain cartesian
    product, no evaluation. Deliberately not computed against real data
    this round."""
    return [(dr, dp) for dr in COUNTERFACTUAL_RUN_OFFSETS_SEC for dp in COUNTERFACTUAL_PASS_OFFSETS_SEC]


# ===========================================================================
# 2026-09-08 V9.1 -- FINAL OFFSIDE INTEGRATION ROUND. Connects every piece
# above into ONE coherent tactical event pipeline, operating on the REAL
# output of `build_offside_v4()` (its own `frames`/`runs`, not a
# reimplementation) -- never touches V6/V7/V8 defensive-unit membership,
# which stays FROZEN this round exactly as instructed. This is a
# deliberate design choice, disclosed in the report: rather than inlining
# this machinery into `build_offside_v4()`'s own per-frame loop (risking
# the 177 tests already passing against that pipeline), it is a SEPARATE
# post-processing pass that consumes that pipeline's real, already-
# computed `frames` (trap/line/back_line/state, per frame) and `runs`
# (RunTracker output, per attacking run) -- i.e. "the real per-frame/
# per-run pipeline", just not the SAME Python loop.
# ===========================================================================

# ---------------------------------------------------------------------
# PART 3 -- PASS -> RUNNER ASSOCIATION. Simple, explainable, two-term
# score (never a black-box model): how close the runner sits to the
# pass's OWN recorded end point (distance_term), and how well the
# runner's OWN instantaneous velocity heading agrees with the pass
# vector's bearing (angle_term) -- both already-real quantities, no
# new tracking invented. A runner must also be the SAME attacking team
# and have an active run sample within TEMPORAL_OVERLAP_SEC of release
# ("runner active around release" + "temporal overlap" evidence).
TEMPORAL_OVERLAP_SEC = 0.5
ASSOCIATION_DISTANCE_SCALE_CM = 2000.0
ASSOCIATION_ANGLE_TOLERANCE_DEG = 90.0
ASSOCIATION_VALID_THRESHOLD = 0.5
W_ASSOC_DISTANCE, W_ASSOC_ANGLE = 0.5, 0.5


def _run_sample_near(run, frame, window_frames):
    return next((s for s in run['samples'] if abs(s['frame'] - frame) <= window_frames), None)


def associate_pass_to_runner(event, release, runs, players_by_frame, fps=30., pitch=DEFAULT_PITCH):
    """Returns the single best-scoring SAME-TEAM active runner for this
    pass event, or an explicitly unassociated result -- never guesses
    an implausible runner, and never associates a different team's run."""
    if not release.get('valid'):
        return dict(runner_track_id=None, run_id=None, association_score=0.0, association_valid=False,
                    association_reason='RELEASE_INVALID', endpoint_distance_cm=None)
    f = release['frame']
    window = max(1, round(TEMPORAL_OVERLAP_SEC * fps))
    pass_bearing = _bearing_deg(release['ball_x'], release['ball_y'], event['end_x'], event['end_y'])
    best = None
    for run in runs:
        if run['attacking_team'] != event['team_id']:
            continue  # (1) attacking-team match -- a hard filter, never scored/relaxed
        sample = _run_sample_near(run, f, window)
        if sample is None:
            continue  # not active around release -- not a plausible candidate
        dist = math.hypot(sample['x_pitch'] - event['end_x'], sample['y_pitch'] - event['end_y'])
        distance_term = math.exp(-dist / ASSOCIATION_DISTANCE_SCALE_CM)
        runner_row = next((p for p in players_by_frame.get(sample['frame'], [])
                            if p.get('track_id') == run['attacker_track_id'] and p.get('x_pitch') is not None), None)
        if runner_row is not None and runner_row.get('motion_valid') and runner_row.get('vx_cm_s') is not None:
            runner_bearing = math.degrees(math.atan2(runner_row['vy_cm_s'], runner_row['vx_cm_s']))
            angle_term = float(np.clip(1.0 - _angle_diff_deg(runner_bearing, pass_bearing) / ASSOCIATION_ANGLE_TOLERANCE_DEG, 0.0, 1.0))
        else:
            angle_term = 0.5  # motion untrustworthy this frame -- neutral, never fabricated
        score = W_ASSOC_DISTANCE * distance_term + W_ASSOC_ANGLE * angle_term
        if best is None or score > best[1]:
            best = (run, score, dist)
    if best is None:
        return dict(runner_track_id=None, run_id=None, association_score=0.0, association_valid=False,
                    association_reason='NO_SAME_TEAM_RUNNER_ACTIVE_NEAR_RELEASE', endpoint_distance_cm=None)
    run, score, dist = best
    valid = score >= ASSOCIATION_VALID_THRESHOLD
    if not valid:
        # explicitly "leave the pass unassociated" -- the best candidate's
        # own score/distance are kept for diagnostics, but NOT surfaced
        # as a runner_track_id/run_id (no implausible association).
        return dict(runner_track_id=None, run_id=None, association_score=round(score, 4), association_valid=False,
                    association_reason='BEST_CANDIDATE_BELOW_THRESHOLD', endpoint_distance_cm=round(dist, 1),
                    rejected_candidate_track_id=run['attacker_track_id'])
    return dict(runner_track_id=run['attacker_track_id'], run_id=run['run_id'], association_score=round(score, 4),
                association_valid=True, association_reason=None, endpoint_distance_cm=round(dist, 1))


# ---------------------------------------------------------------------
# PART 5 -- RUNNER CROSSING, restricted to the ONE associated run's own
# `samples` list (never another run/track_id -- no stitching possible
# by construction, since `run['samples']` already belongs to a single
# RunTracker episode).
def detect_runner_crossing(run, fps=30.):
    samples = run['samples']
    for a, b in zip(samples, samples[1:]):
        if a['relation_cm'] <= 0 < b['relation_cm']:
            return dict(runner_track_id=run['attacker_track_id'], run_id=run['run_id'], cross_frame=b['frame'],
                        cross_time_sec=b['frame'] / fps, relation_before=a['relation_cm'], relation_after=b['relation_cm'],
                        cross_valid=True, cross_uncertainty_reason=None)
    return dict(runner_track_id=run['attacker_track_id'], run_id=run['run_id'], cross_frame=None, cross_time_sec=None,
                relation_before=None, relation_after=None, cross_valid=False,
                cross_uncertainty_reason='NO_SIGN_CHANGE_OBSERVED')


# ---------------------------------------------------------------------
# PART 8 -- TRAP x RUN INTERACTION: a real snapshot of the DEFENDING
# team's trap/unit state at an exact frame, read directly from
# `build_offside_v4()`'s own `frames` output -- never re-derived.
def _trap_snapshot(frames, defending_team, frame):
    if frame is None or frame < 0 or frame >= len(frames):
        return dict(unit_centre_depth_cm=None, unit_centre_velocity_cm_s=None,
                    trap_synchrony_score=None, trap_state=None, confirmed_unit_ids=[])
    rec = frames[frame]['teams'][defending_team]
    bl = rec.get('back_line') or {}
    return dict(unit_centre_depth_cm=bl.get('centre_depth_cm'), unit_centre_velocity_cm_s=rec.get('line_velocity_cm_s'),
                trap_synchrony_score=(rec.get('metrics') or {}).get('score'), trap_state=rec.get('state'),
                confirmed_unit_ids=list(bl.get('member_ids') or []))


# ---------------------------------------------------------------------
# Reception evidence (feeds lifecycle SUCCESSFUL_BREAK gate #5) -- a
# real geometric check: does the ball come back within the SAME
# possession-control radius of the associated runner within a short
# window after the observed crossing? True/False, or None when
# crossing itself is invalid/unevaluated (never assumed).
RECEPTION_SEARCH_SEC = 1.5


def _reception_evidence(run, crossing, ball_by_frame, fps=30., radius_cm=PASSER_PROXIMITY_SCALE_CM):
    if not crossing.get('cross_valid'):
        return None
    f = crossing['cross_frame']
    horizon = max(1, round(RECEPTION_SEARCH_SEC * fps))
    samples_by_frame = {s['frame']: s for s in run['samples']}
    for df in range(0, horizon + 1):
        ball_row = ball_by_frame.get(f + df)
        runner_sample = samples_by_frame.get(f + df)
        if ball_row is None or runner_sample is None or not ball_row.get('is_observed') or ball_row.get('x_pitch') is None:
            continue
        d = math.hypot(ball_row['x_pitch'] - runner_sample['x_pitch'], ball_row['y_pitch'] - runner_sample['y_pitch'])
        if d <= radius_cm:
            return True
    return False


# ---------------------------------------------------------------------
# PART 2/4/6/7 -- THE integration entry point: one EVENT record per
# REAL detected pass, wiring passer -> release -> effective reference
# -> association -> runner-at-release -> crossing -> timing margin ->
# trap-link -> lifecycle. Every field is None + an explicit reason when
# its own upstream evidence is unavailable -- nothing here is ever
# fabricated to fill a gap.
def build_pass_run_events_v9(passes, runs, frames, ball_by_frame, players_by_frame, fps=30., pitch=DEFAULT_PITCH):
    events_out = []
    for event in sorted(passes, key=lambda p: p['start_frame']):
        attacking_team = event['team_id']
        defending_team = 1 - attacking_team
        release = detect_release(event, ball_by_frame, players_by_frame, fps=fps)
        assoc = associate_pass_to_runner(event, release, runs, players_by_frame, fps=fps, pitch=pitch)
        run = next((r for r in runs if r.get('run_id') == assoc.get('run_id')), None)

        eff_ref = dict(valid=False, reference_depth_cm=None, source=None, reason='RELEASE_INVALID')
        second_last_opponent_depth = None
        ball_depth_at_release = None
        if release.get('valid'):
            line_f = frames[release['frame']]['teams'][defending_team]['line']
            eff_ref = effective_reference_at_release(release, defending_team, {defending_team: {release['frame']: line_f}},
                                                       ball_by_frame, pitch)
            if line_f.get('valid'):
                second_last_opponent_depth = line_f['line_depth_cm']
            ball_val = ball_x(ball_by_frame.get(release['frame']))
            if ball_val is not None:
                ball_depth_at_release = pitch.depth(defending_team, ball_val, release['frame'])

        runner_row = None
        runner_state = dict(valid=False, reason='NO_ASSOCIATED_RUNNER')
        crossing = dict(runner_track_id=None, run_id=None, cross_frame=None, cross_time_sec=None,
                        relation_before=None, relation_after=None, cross_valid=False,
                        cross_uncertainty_reason='NO_ASSOCIATED_RUNNER')
        margin_sec = None
        reception = None
        lifecycle = dict(lifecycle=[], outcome='UNCERTAIN', reason='RELEASE_NOT_DETECTED', timing_margin_sec=None)

        if run is not None and release.get('valid'):
            # `run` is only ever non-None here when association_valid was
            # True (see associate_pass_to_runner -- an invalid association
            # returns run_id=None), so this branch always has a real,
            # confidently-associated runner.
            runner_row = next((p for p in players_by_frame.get(release['frame'], [])
                                if p.get('track_id') == run['attacker_track_id'] and p.get('x_pitch') is not None), None)
            if eff_ref.get('valid'):
                runner_state = runner_state_at_release(runner_row, release['frame'], eff_ref['reference_depth_cm'],
                                                        attacking_team, defending_team, pitch)
            else:
                runner_state = dict(valid=False, reason='EFFECTIVE_REFERENCE_UNAVAILABLE')
            crossing = detect_runner_crossing(run, fps=fps)
            if crossing['cross_valid']:
                margin_sec = timing_margin_sec(release['time_sec'], crossing['cross_time_sec'])
                reception = _reception_evidence(run, crossing, ball_by_frame, fps=fps)
            lifecycle = classify_run_lifecycle_v9(run, release, runner_state, run['samples'], fps=fps,
                                                   reception_evidence=reception)
        elif release.get('valid'):
            # release WAS detected but no runner could be confidently
            # associated -- an honest, disclosed gap, never a guessed runner.
            lifecycle = dict(lifecycle=[dict(state='RELEASE_WINDOW', event_frame=release['frame'])],
                              outcome='UNCERTAIN', reason='NO_RUNNER_ASSOCIATED', timing_margin_sec=None)

        trap_at_release = _trap_snapshot(frames, defending_team, release.get('frame'))
        trap_at_cross = _trap_snapshot(frames, defending_team, crossing.get('cross_frame'))

        events_out.append(dict(
            pass_event_id=event['pass_id'], scene_id=event.get('scene_id', 0),
            start_time_sec=round(event['start_frame'] / fps, 3),
            passer_track_id=event['source_track_id'], passer_team=attacking_team,
            passer_confidence=event.get('confidence'),
            release_frame=release.get('frame'), release_time_sec=release.get('time_sec'),
            release_valid=release.get('valid'), release_evidence=release.get('evidence'),
            release_confidence=release.get('evidence'), release_uncertainty_reason=release.get('reason'),
            ball_x_at_release=release.get('ball_x'), ball_y_at_release=release.get('ball_y'),
            second_last_opponent_depth_at_release=second_last_opponent_depth,
            ball_depth_at_release=ball_depth_at_release,
            effective_reference_depth_at_release=eff_ref.get('reference_depth_cm'),
            effective_reference_source=eff_ref.get('source'),
            association_score=assoc.get('association_score'), association_valid=assoc.get('association_valid'),
            association_reason=assoc.get('association_reason'),
            runner_track_id=assoc.get('runner_track_id'), run_id=assoc.get('run_id'),
            runner_x_at_release=(runner_row or {}).get('x_pitch'), runner_y_at_release=(runner_row or {}).get('y_pitch'),
            runner_depth_at_release=runner_state.get('runner_depth_cm'),
            runner_relation_to_reference_cm=runner_state.get('distance_to_reference_cm'),
            runner_forward_speed_cm_s=runner_state.get('forward_speed_cm_s'),
            runner_relative_speed_to_reference_cm_s=runner_state.get('forward_speed_cm_s'),
            estimated_crossing_time_sec=runner_state.get('estimated_time_to_crossing_sec'),
            observed_crossing_frame=crossing.get('cross_frame'), observed_crossing_time_sec=crossing.get('cross_time_sec'),
            cross_valid=crossing.get('cross_valid'), cross_uncertainty_reason=crossing.get('cross_uncertainty_reason'),
            relation_before_cross=crossing.get('relation_before'), relation_after_cross=crossing.get('relation_after'),
            timing_margin_sec=margin_sec,
            timing_margin_label=_timing_margin_label(margin_sec),
            reception_evidence=reception,
            lifecycle_outcome=lifecycle.get('outcome'), lifecycle_reason=lifecycle.get('reason'),
            lifecycle_states=[s['state'] for s in lifecycle.get('lifecycle', [])],
            trap_state_at_release=trap_at_release['trap_state'], trap_score_at_release=trap_at_release['trap_synchrony_score'],
            unit_centre_depth_at_release=trap_at_release['unit_centre_depth_cm'],
            unit_centre_velocity_at_release=trap_at_release['unit_centre_velocity_cm_s'],
            confirmed_unit_ids_at_release=trap_at_release['confirmed_unit_ids'],
            trap_state_at_cross=trap_at_cross['trap_state'], trap_score_at_cross=trap_at_cross['trap_synchrony_score'],
            unit_centre_depth_at_cross=trap_at_cross['unit_centre_depth_cm'],
        ))
    return events_out


def _timing_margin_label(margin_sec, near_zero_tolerance_sec=0.1):
    """PART 6 wording -- never an official-sounding onside/offside verdict."""
    if margin_sec is None:
        return 'UNCERTAIN'
    if abs(margin_sec) <= near_zero_tolerance_sec:
        return 'NEAR_SIMULTANEOUS'
    return 'ESTIMATED_RELEASE_BEFORE_CROSS' if margin_sec > 0 else 'ESTIMATED_CROSS_BEFORE_RELEASE'
