"""V4 visible-opponent geometry, defensive-unit motion and runner-linked events.
Geometric offside position is an estimate, never an official offence decision.
All per-frame scores use past/current data. Pass/run outcomes are retrospective
records with separate event, confirmation and display timestamps.
"""
from collections import deque
import numpy as np
from tactical_shared.coordinates import DEFAULT_PITCH
from tactical_shared.temporal import EvidenceFSM,EvidenceConfig,dominant
from pressing_structure.analytics.pressing_v4 import evidence_quality

# min_observations/min_supported_sec/min_coverage_ratio: REVERTED
# (2026-09-07 semantic-QA pass) from an earlier recalibration to this
# clip's own measured ceiling (5 observations/0.167s) -- the same Case-B
# concern as PRESS_CONFIG (see pressing_v4.py's comment): a bar set to
# match one clip's best episode will always confirm something on that
# clip by construction. Left at `EvidenceConfig`'s generic, structurally-
# justified defaults (2 observations / 0.25s support / 1/3 coverage,
# IDENTICAL to PRESS_CONFIG since both share the same FSM engine and the
# same generic temporal-support philosophy -- only `enter`/`exit`, the
# trap SCORE's own measured scale, are project-specific here).
TRAP_CONFIG=EvidenceConfig(enter=.5,exit=.3,
                          low='NO_TRAP',forming='TRAP_FORMING',active='TRAP_ACTIVE',ending='TRAP_ENDING')
RUN_MIN_OBSERVATIONS=15
RUN_MAX_GAP=3

def eligible_players(players,team=None):
    return [p for p in players if p.get('x_pitch') is not None and p.get('y_pitch') is not None
            and p['display_object_type'] in ('player','goalkeeper') and (team is None or p['display_team_id']==team)]

def line_geometry(players,team,frame=0,pitch=DEFAULT_PITCH):
    """Ordering rule (2026-09-07 V5 audit -- unchanged, only now also
    DISCLOSED via new fields below): `defs` is every one of this team's
    visible, in-bounds players (goalkeeper included, matching IFAB Law
    11's 'second-to-last opponent' -- the keeper counts as a defender),
    sorted by `pitch.depth(team,x,frame)` ASCENDING -- i.e. by distance
    from THIS team's own goal in ITS OWN attacking direction, NEVER a
    raw/image-relative x-coordinate. Ties broken by track_id for a
    deterministic, reproducible order.

    `depth` ascending means index 0 = the player closest to their own
    goal (the LAST/deepest defender), and the last index = the player
    furthest from their own goal (the FRONT-most/most-advanced member
    of that team). `defs[1]` (second-nearest-to-own-goal) is the
    existing, UNCHANGED reference/offside line (`line_x`/`line_depth_cm`
    below -- byte-identical to the pre-audit formula, verified in the
    2026-09-07 V5 geometry-audit report). `last_*`/`front_*` are NEW,
    purely ADDITIVE fields exposing `defs[0]`/`defs[-1]` -- no existing
    field's value or the FSM/trap-synchrony inputs derived from them
    changed as a result of adding these."""
    defs=[p for p in eligible_players(players,team) if pitch.inside(p['x_pitch'],p['y_pitch'])]
    defs.sort(key=lambda p:(pitch.depth(team,p['x_pitch'],frame),p['track_id']))
    unknown=sum(p['display_team_id'] is None for p in eligible_players(players))
    base=dict(valid=False,line_x=None,line_depth_cm=None,defender_track_id=None,n_visible=len(defs),
              confidence=min(len(defs)/11,1.)*(1 if not unknown else .5),unknown_identities=unknown,
              line_kind='RAW_VISIBLE_OPPONENT_ESTIMATE',geometric_area_m2=None,
              last_x=None,last_depth_cm=None,last_track_id=None,
              front_x=None,front_depth_cm=None,front_track_id=None)
    if len(defs)<2:return base
    p=defs[1];depth=pitch.depth(team,p['x_pitch'],frame)
    last=defs[0];front=defs[-1]
    return {**base,'valid':True,'line_x':p['x_pitch'],'line_depth_cm':depth,'defender_track_id':p['track_id'],
            'geometric_area_m2':depth*pitch.width_cm/10000,
            'last_x':last['x_pitch'],'last_depth_cm':pitch.depth(team,last['x_pitch'],frame),'last_track_id':last['track_id'],
            'front_x':front['x_pitch'],'front_depth_cm':pitch.depth(team,front['x_pitch'],frame),'front_track_id':front['track_id']}

def defensive_unit(players,team,frame=0,pitch=DEFAULT_PITCH):
    """PRE-2026-09-07-V6 method. Kept unmodified (still directly unit-
    tested by test_offside_press_v4.py) but NO LONGER called by
    `build_offside_v4()` -- see `identify_back_line()`/`BackLineTracker`
    below, which replace it as the pipeline's actual unit-membership
    source (2026-09-07 V6 defensive-line-identification audit: a fixed
    1200cm window from the single deepest player is not football-aware --
    it has no concept of a GAP to the next distinct group, so one
    stray deep outfield player, or a back-4 that happens to be more than
    12m end-to-end, silently pulls in or excludes real members)."""
    rows=[p for p in eligible_players(players,team) if p['display_object_type']=='player' and pitch.inside(p['x_pitch'],p['y_pitch'])]
    rows.sort(key=lambda p:pitch.depth(team,p['x_pitch'],frame))
    if not rows:return []
    anchor=pitch.depth(team,rows[0]['x_pitch'],frame)
    return [p for p in rows if pitch.depth(team,p['x_pitch'],frame)-anchor<=1200][:5]

# ---------------------------------------------------------------------------
# 2026-09-07 V6 -- defensive back-line identification, rebuilt.
#
# AUDIT (this clip, `outputs/tracking/testVideo1_120s`, subsampled every
# 3rd frame, both teams, outfield players only): consecutive depth-gaps
# among a team's own visible outfield players do NOT separate into a
# clean bimodal "within-line" vs "to-next-unit" distribution --
#   gaps within the presumed back-4 (1st-2nd/2nd-3rd/3rd-4th): p50=211cm, p75=381cm, p90=631cm
#   gap from the presumed 4th to 5th deepest player:            p50=349cm, p75=544cm, p90=788cm
# these two distributions overlap substantially. This is an honest,
# disclosed limitation of position-only clustering without role labels
# on a real broadcast clip (a back line is not always a clean back-4;
# full-backs push on, midfielders drop in) -- there is no threshold that
# perfectly separates "same line" from "different unit" in this data.
# BACK_LINE_GAP_THRESHOLD_CM is therefore a data-INFORMED (75th
# percentile of the within-back-4 gap distribution, rounded), not a
# sharply-separating, choice: a greedy consecutive-gap walk from the
# deepest outfield player outward, stopping at the first gap this large,
# capped at MAX_BACK_LINE_SIZE. This directly satisfies the brief's own
# requirements better than a fixed absolute window from one anchor
# (`defensive_unit()`, above): it represents 3/4/5-man lines naturally,
# and a genuinely isolated advanced player (a winger, a midfielder
# stepping past the back line) breaks the chain instead of being pulled
# in or silently capped away.
BACK_LINE_GAP_THRESHOLD_CM=400.0
MIN_BACK_LINE_SIZE=3
MAX_BACK_LINE_SIZE=5

def identify_back_line(players,team,frame=0,pitch=DEFAULT_PITCH,
                        gap_threshold_cm=BACK_LINE_GAP_THRESHOLD_CM,
                        min_size=MIN_BACK_LINE_SIZE,max_size=MAX_BACK_LINE_SIZE):
    """RAW per-frame candidate back-line cluster (no temporal continuity
    -- see `BackLineTracker` for that). Goalkeeper EXCLUDED (1D: treat
    the keeper separately from outfield back-line estimation) via the
    same `display_object_type=='player'` filter `defensive_unit()`
    already used. Sorted by `pitch.depth` (own-goal-relative, team-
    direction-aware -- never raw x); a midfielder sitting several metres
    ahead of the real back line breaks the chain (the gap to them
    exceeds `gap_threshold_cm`) and is correctly excluded, never
    becoming the 'front' of the line. Returns [] (an honest gap, not a
    fabricated 1- or 2-man 'line') when fewer than `min_size` outfield
    players are visible at all."""
    rows=[p for p in eligible_players(players,team) if p['display_object_type']=='player' and pitch.inside(p['x_pitch'],p['y_pitch'])]
    rows.sort(key=lambda p:(pitch.depth(team,p['x_pitch'],frame),p['track_id']))
    if len(rows)<min_size:return []
    cluster=[rows[0]]
    for p in rows[1:]:
        gap=pitch.depth(team,p['x_pitch'],frame)-pitch.depth(team,cluster[-1]['x_pitch'],frame)
        if gap<=gap_threshold_cm and len(cluster)<max_size:cluster.append(p)
        else:break
    return cluster if len(cluster)>=min_size else []

class BackLineTracker:
    """Evidence-aware MEMBERSHIP continuity for the back-line cluster
    (1E) -- does NOT smooth the final x-coordinate; it smooths WHICH
    player IDs count as the unit, then always reads their CURRENT
    position fresh. Two seconds-denominated parameters (never a raw
    frame count):

    `min_stable_sec`: a newly-observed candidate membership (as an
    exact id-set) must recur for this long before it is ACCEPTED,
    absorbing single-frame measurement jitter (a near-tie 5th/6th
    player swap) without flip-flopping the visualized unit every frame.

    `max_missing_sec`: a previously-accepted member who is briefly not
    visible (occlusion, a missed detection) stays counted for up to
    this long (missing-track tolerance) -- their position is still read
    fresh from `players` the instant they reappear, never frozen.

    A GENUINE change (an entry/exit that persists) is still accepted
    once its own stability window elapses; this only suppresses
    one-frame flicker, it never masks a real, sustained defensive
    reshape."""
    def __init__(self,team,fps=30.,min_stable_sec=0.1,max_missing_sec=0.1,pitch=DEFAULT_PITCH):
        self.team=team;self.fps=fps;self.pitch=pitch
        self.min_stable_frames=max(1,round(min_stable_sec*fps))
        self.max_missing_frames=max(1,round(max_missing_sec*fps))
        self.accepted_ids=frozenset();self.last_seen={}
        self.candidate_ids=None;self.candidate_streak=0

    def step(self,players,frame):
        cluster=identify_back_line(players,self.team,frame,self.pitch)
        cand_ids=frozenset(p['track_id'] for p in cluster)
        by_id={p['track_id']:p for p in players if p.get('track_id') is not None and p.get('x_pitch') is not None and p.get('y_pitch') is not None}

        if cand_ids==self.candidate_ids:self.candidate_streak+=1
        else:self.candidate_ids=cand_ids;self.candidate_streak=1

        for tid in self.accepted_ids:
            if tid in by_id:self.last_seen[tid]=frame
        working_ids={tid for tid in self.accepted_ids if frame-self.last_seen.get(tid,frame)<=self.max_missing_frames}

        if self.candidate_streak>=self.min_stable_frames or len(working_ids)<MIN_BACK_LINE_SIZE:
            working_ids=cand_ids
            for tid in working_ids:self.last_seen[tid]=frame

        self.accepted_ids=working_ids
        rows=[by_id[tid] for tid in working_ids if tid in by_id]
        rows.sort(key=lambda p:self.pitch.depth(self.team,p['x_pitch'],frame))
        return rows

# ---------------------------------------------------------------------------
# 2026-09-07 V7 -- TEMPORAL defensive-unit MEMBERSHIP MODEL.
#
# WHY: the V6 `identify_back_line()` gap-walk is still a single hard
# per-frame threshold (400cm) on one signal (consecutive depth gap).
# The V6 audit itself found within-line and to-next-unit gap
# distributions overlap substantially -- no single depth threshold can
# be robust alone. This section adds a continuous, multi-evidence
# MEMBERSHIP SCORE plus an explicit temporal state machine, instead of
# a bigger/smaller hard threshold.
#
# P_backline(i,t) -- interpretable weighted sum of THREE evidence terms
# (each already in [0,1], so the weights are a genuine convex
# combination, not rescaled ad hoc):
#
#   depth_term(i,t)    = exp(-|depth_i - center_depth| / DEPTH_SCALE_CM)
#       -- how close player i's own-goal-relative depth is to the
#       CURRENT candidate group's center depth (median depth of the
#       V6 gap-cluster, reused as a seed -- see below). DEPTH_SCALE_CM=300
#       sits between the V6 audit's own measured p50=211cm and p75=381cm
#       within-line gap statistics -- reused, not re-derived.
#
#   temporal_term(i,t) = 1.0 if i was a CONFIRMED member last frame,
#                         else 0.5 (neutral -- a new/unseen player is
#                         neither penalized nor assumed a member)
#       -- this IS "temporal continuity" (A1 item 2) entering the score
#       itself, not just a post-hoc smoothing pass.
#
#   motion_term(i,t)   = exp(-|fwd_velocity_i - median(seed velocities)|
#                             / VELOCITY_SCALE_CM_S) if i's motion is
#                        trustworthy, else 0.5 (neutral -- never
#                        fabricated from an untrustworthy reading).
#       VELOCITY_SCALE_CM_S=150 is HALF of `trap_synchrony`'s own
#       existing, already-used velocity-alignment scale (200cm/s, see
#       `aligned=clip(1-std(velocities)/200,...)` below) -- reused, not
#       invented.
#
#   P_backline(i,t) = 0.5*depth_term + 0.3*temporal_term + 0.2*motion_term
#   (weights disclosed: depth is the strongest, most direct signal;
#   temporal continuity next; motion coherence last since velocity is
#   the noisiest input available here.)
#
# NOT separately implemented this pass (disclosed, not silently
# skipped): A1 item 3 "lateral structure" -- audited informally and not
# found to meaningfully separate real back-line members from non-
# members position-only in this dataset; flagged as future work rather
# than added as a weak/arbitrary term. Item 6 "group size prior" is
# enforced structurally (MIN/MAX_BACK_LINE_SIZE caps below), not as a
# fourth scoring term.
DEPTH_SCALE_CM=300.0
VELOCITY_SCALE_CM_S=150.0
W_DEPTH=0.5
W_TEMPORAL=0.3
W_MOTION=0.2
MEMBERSHIP_THRESHOLD=0.5

def score_backline_membership(players,team,frame,pitch,confirmed_ids_prev,
                                depth_scale_cm=DEPTH_SCALE_CM,velocity_scale_cms=VELOCITY_SCALE_CM_S):
    """Returns {track_id: (score, player_row)} for every visible,
    in-bounds outfield player of `team` this frame. `confirmed_ids_prev`:
    the SET of track_ids CONFIRMED as members in the immediately
    preceding frame (from `DefensiveUnitTracker`), feeding temporal_term."""
    rows=[p for p in eligible_players(players,team) if p['display_object_type']=='player' and pitch.inside(p['x_pitch'],p['y_pitch'])]
    if not rows:return {}
    seed=identify_back_line(players,team,frame,pitch)
    if seed:
        center_depth=float(np.median([pitch.depth(team,p['x_pitch'],frame) for p in seed]))
        seed_v=[p['vx_cm_s']*pitch.attacking_sign(team,frame) for p in seed if p.get('motion_valid') and p.get('vx_cm_s') is not None]
        center_v=float(np.median(seed_v)) if seed_v else None
    else:
        # Honest fallback when the V6 seed itself can't form (e.g. <3
        # outfield players close together): use the deepest-K (K<=4)
        # outfield players' median depth as a rough seed rather than
        # returning nothing -- membership scores are still genuinely
        # computed and gated by MEMBERSHIP_THRESHOLD below, never assumed.
        by_depth=sorted(rows,key=lambda p:pitch.depth(team,p['x_pitch'],frame))
        k=min(4,len(by_depth))
        center_depth=float(np.median([pitch.depth(team,p['x_pitch'],frame) for p in by_depth[:k]]))
        center_v=None
    scores={}
    for p in rows:
        depth=pitch.depth(team,p['x_pitch'],frame)
        depth_term=float(np.exp(-abs(depth-center_depth)/depth_scale_cm))
        temporal_term=1.0 if p['track_id'] in confirmed_ids_prev else 0.5
        if center_v is not None and p.get('motion_valid') and p.get('vx_cm_s') is not None:
            v=p['vx_cm_s']*pitch.attacking_sign(team,frame)
            motion_term=float(np.exp(-abs(v-center_v)/velocity_scale_cms))
        else:
            motion_term=0.5
        score=W_DEPTH*depth_term+W_TEMPORAL*temporal_term+W_MOTION*motion_term
        scores[p['track_id']]=(score,p)
    return scores

class DefensiveUnitTracker:
    """A2: explicit temporal state machine over `score_backline_membership`,
    not a plain persistence filter. Per track_id status:
        none -> candidate -> confirmed -> (missing) -> removed
    All parameters in SECONDS (never a hard-coded frame count):
      `min_support_sec`: a candidate must score >=MEMBERSHIP_THRESHOLD
        for this long, continuously, before being CONFIRMED.
      `max_missing_sec`: a CONFIRMED member absent from `players`
        entirely (no position this frame) is still counted CONFIRMED
        for up to this long (missing-track tolerance) before REMOVED.
      `removal_sec`: a CONFIRMED member whose score drops below
        MEMBERSHIP_THRESHOLD (but who is still VISIBLE) must stay below
        it continuously for this long before being demoted back to
        candidate -- a real, sustained score drop, not one noisy frame.
    Group-size prior (A1 item 6) enforced structurally: at most
    MAX_BACK_LINE_SIZE confirmed members are kept (lowest-scoring
    dropped back to candidate when over)."""
    def __init__(self,team,fps=30.,min_support_sec=0.15,max_missing_sec=0.15,removal_sec=0.2,pitch=DEFAULT_PITCH):
        self.team=team;self.fps=fps;self.pitch=pitch
        self.min_support_frames=max(1,round(min_support_sec*fps))
        self.max_missing_frames=max(1,round(max_missing_sec*fps))
        self.removal_frames=max(1,round(removal_sec*fps))
        self.state={}   # track_id -> dict(status, support_frames, missing_frames, below_frames)
        self.confirmed_ids=set()

    def step(self,players,frame):
        scores=score_backline_membership(players,self.team,frame,self.pitch,self.confirmed_ids)
        present_ids=set(scores.keys())
        # Members confirmed but with NO position at all this frame (not
        # merely a low score -- genuinely absent from `players`).
        for tid in list(self.confirmed_ids):
            if tid not in present_ids:
                st=self.state.setdefault(tid,dict(status='confirmed',support_frames=0,missing_frames=0,below_frames=0))
                st['missing_frames']+=1
                if st['missing_frames']>self.max_missing_frames:
                    st['status']='removed';self.confirmed_ids.discard(tid)
        for tid,(score,_) in scores.items():
            st=self.state.setdefault(tid,dict(status='none',support_frames=0,missing_frames=0,below_frames=0))
            above=score>=MEMBERSHIP_THRESHOLD
            if tid in self.confirmed_ids:
                st['missing_frames']=0
                if above:
                    st['below_frames']=0
                else:
                    st['below_frames']+=1
                    if st['below_frames']>=self.removal_frames:
                        st['status']='candidate';st['support_frames']=0;self.confirmed_ids.discard(tid)
            else:
                if above:
                    st['support_frames']+=1;st['status']='candidate'
                    if st['support_frames']>=self.min_support_frames:
                        st['status']='confirmed';self.confirmed_ids.add(tid);st['missing_frames']=0
                else:
                    st['support_frames']=0
                    st['status']='candidate' if st['status']=='candidate' else 'none'
        # Group-size prior: keep at most MAX_BACK_LINE_SIZE confirmed,
        # dropping the lowest-scoring first (never fabricates extra members).
        confirmed=[(tid,scores[tid][0]) for tid in self.confirmed_ids if tid in scores]
        confirmed.sort(key=lambda x:-x[1])
        for tid,_ in confirmed[MAX_BACK_LINE_SIZE:]:
            self.confirmed_ids.discard(tid)
            self.state[tid]['status']='candidate';self.state[tid]['support_frames']=0
        confirmed_rows=[scores[tid][1] for tid in self.confirmed_ids if tid in scores]
        candidate_rows=[scores[tid][1] for tid,st in self.state.items() if st['status']=='candidate' and tid in scores]
        missing_ids=[tid for tid in self.confirmed_ids if tid not in present_ids]
        confirmed_rows.sort(key=lambda p:self.pitch.depth(self.team,p['x_pitch'],frame))
        return dict(confirmed=confirmed_rows,candidate=candidate_rows,missing_ids=missing_ids,scores=scores)

def unit_motion(history,team,frame,fps=30.,pitch=DEFAULT_PITCH):
    """Median motion of >=3 same-segment unit identities continuously present
    for a full .5s baseline. Never differentiate the second-nearest order statistic.
    """
    span=round(.5*fps);prior=[h for h in history if frame-span<=h['frame']<=frame]
    if len(prior)!=span+1 or prior[0]['frame']!=frame-span:return None,[]
    if any(pitch.period(h['frame'])!=pitch.period(frame) for h in prior):return None,[]
    keys=[{(p['track_id'],p['cleaned_segment_id']) for p in h['unit'] if p.get('motion_valid',False) and p['display_team_id']==team} for h in prior]
    common=set.intersection(*keys) if keys else set()
    if len(common)<3:return None,[]
    a={(p['track_id'],p['cleaned_segment_id']):p for p in prior[0]['unit']}
    b={(p['track_id'],p['cleaned_segment_id']):p for p in prior[-1]['unit']}
    velocities=[(b[k]['x_pitch']-a[k]['x_pitch'])/(span/fps)*pitch.attacking_sign(team,frame) for k in common]
    if max(abs(v) for v in velocities)>1200 or np.std(velocities)>200:return None,[]
    return float(np.median(velocities)),sorted(common)

def trap_synchrony(unit,team,line_velocity,attack_context,frame=0,pitch=DEFAULT_PITCH):
    valid=[p for p in unit if p.get('motion_valid',p.get('vx_cm_s') is not None) and p.get('vx_cm_s') is not None]
    coverage=len(valid)/max(1,len(unit))
    base=dict(valid=False,score=None,confidence=coverage,coverage=coverage,participants=[],n_defenders=len(unit),reason='INSUFFICIENT_UNIT_MOTION')
    if len(valid)<3 or coverage<.75 or line_velocity is None:return base
    velocities=[p['vx_cm_s']*pitch.attacking_sign(team,frame) for p in valid]
    participants=[p['track_id'] for p,v in zip(valid,velocities) if v>=50]
    fraction=len(participants)/len(valid)
    depths=[pitch.depth(team,p['x_pitch'],frame) for p in unit]
    compact=float(np.clip(1-np.std(depths)/500,0,1))
    aligned=float(np.clip(1-np.std(velocities)/200,0,1))
    if attack_context is None:return {**base,'reason':'ATTACK_CONTEXT_UNAVAILABLE'}
    # Multiplicative movement gate makes stationary/retreating alignment zero.
    stepped=line_velocity>=50 and fraction>=.75 and len(participants)>=3
    score=float(fraction*(.5+.25*compact+.25*aligned)) if stepped and attack_context else 0.
    return {**base,'valid':True,'score':score,'participants':participants,'reason':None,
            'fraction_stepping_forward':fraction,'depth_dispersion_score':compact,'velocity_alignment_score':aligned}

# ---------------------------------------------------------------------------
# 2026-09-07 V7 -- trap-synchrony REDESIGN (A5). AUDIT of the function
# above: `valid`/`coverage`/`fraction_stepping_forward` (fraction of
# unit individually stepping forward >=50cm/s)/`depth_dispersion_score`
# ("compact", 1-std(depths)/500)/`velocity_alignment_score` ("aligned",
# 1-std(velocities)/200) and the `stepped` GATE
# (line_velocity>=50 AND fraction>=.75 AND >=3 participants) are ALL
# REUSED UNCHANGED below via `_unit_kinematics()` -- none of that
# evidence or its thresholds were touched. What changes is the SCORE
# MAGNITUDE formula once the gate is satisfied: the old
# `fraction*(.5+.25*compact+.25*aligned)` gives compact/aligned only
# 25% weight EACH and an unconditional 50% baseline just for the
# fraction-stepping gate already being true (i.e. compactness/alignment
# can barely move the score). The new formula gives compact/aligned
# real, roughly equal weight and adds the brief's explicitly-requested
# "duration of coordinated stepping" as its own disclosed term instead
# of folding everything into one flat baseline:
#
#   TrapSynchrony_t = fraction_stepping * (0.4*depth_coherence
#                                           + 0.4*velocity_alignment
#                                           + 0.2*duration_bonus)   if stepped AND attack_context, else 0.
#   duration_bonus = clip(stepping_streak_sec / DURATION_SCALE_SEC, 0, 1)
#
# DURATION_SCALE_SEC=0.5 reuses this project's own already-established
# `EvidenceConfig.smooth_sec` convention (the trailing window every
# other V4 evidence-smoothing decision in this codebase already uses),
# not a newly-invented number. `stepping_streak_sec` is maintained by
# the CALLER (`build_offside_v4`, a stateful per-team frame counter of
# CONSECUTIVE prior frames where `stepped` held) and passed in here --
# this function itself stays a pure, stateless per-frame computation,
# consistent with every other V4 scoring function.
DURATION_SCALE_SEC=0.5

def _unit_kinematics(unit,team,frame,pitch):
    """The REUSED evidence terms from `trap_synchrony()`, factored out
    so both the old (still-tested) and new formulas share one
    computation -- no duplicated/divergent logic."""
    valid=[p for p in unit if p.get('motion_valid',p.get('vx_cm_s') is not None) and p.get('vx_cm_s') is not None]
    coverage=len(valid)/max(1,len(unit)) if unit else 0.
    if len(valid)<3 or coverage<.75:return None,coverage
    velocities=[p['vx_cm_s']*pitch.attacking_sign(team,frame) for p in valid]
    participants=[p['track_id'] for p,v in zip(valid,velocities) if v>=50]
    fraction=len(participants)/len(valid)
    depths=[pitch.depth(team,p['x_pitch'],frame) for p in unit]
    compact=float(np.clip(1-np.std(depths)/500,0,1))
    aligned=float(np.clip(1-np.std(velocities)/200,0,1))
    return dict(coverage=coverage,fraction=fraction,participants=participants,compact=compact,aligned=aligned),coverage

def trap_synchrony_v7(unit,team,line_velocity,attack_context,stepping_streak_sec=0.,frame=0,pitch=DEFAULT_PITCH,
                        duration_scale_sec=DURATION_SCALE_SEC):
    kin,coverage=_unit_kinematics(unit,team,frame,pitch)
    base=dict(valid=False,score=None,confidence=coverage,coverage=coverage,participants=[],n_defenders=len(unit),
              reason='INSUFFICIENT_UNIT_MOTION',stepped=False,duration_bonus=0.)
    if kin is None or line_velocity is None:return base
    if attack_context is None:
        return {**base,'valid':True,'reason':'ATTACK_CONTEXT_UNAVAILABLE',
                'fraction_stepping_forward':kin['fraction'],'depth_dispersion_score':kin['compact'],
                'velocity_alignment_score':kin['aligned']}
    stepped=line_velocity>=50 and kin['fraction']>=.75 and len(kin['participants'])>=3
    duration_bonus=float(np.clip(stepping_streak_sec/duration_scale_sec,0,1)) if stepped else 0.
    score=float(kin['fraction']*(0.4*kin['compact']+0.4*kin['aligned']+0.2*duration_bonus)) if stepped and attack_context else 0.
    return {**base,'valid':True,'score':score,'participants':kin['participants'],'reason':None,'stepped':stepped,
            'duration_bonus':duration_bonus,'fraction_stepping_forward':kin['fraction'],
            'depth_dispersion_score':kin['compact'],'velocity_alignment_score':kin['aligned']}

def estimated_position(attacker_x,ball_x,line,team,frame=0,pitch=DEFAULT_PITCH,margin_cm=0.):
    if not line.get('valid') or attacker_x is None:return 'UNCERTAIN'
    if ball_x is None:return 'LINE_RELATION_ONLY'
    ad=pitch.depth(team,attacker_x,frame);bd=pitch.depth(team,ball_x,frame);ld=line['line_depth_cm']
    if ad>=pitch.length_cm/2 or ad>=ld or ad>=bd:return 'ESTIMATED_ONSIDE_POSITION'
    if min(ld-ad,bd-ad,pitch.length_cm/2-ad)<margin_cm:return 'UNCERTAIN'
    return 'ESTIMATED_OFFSIDE_POSITION'

def ball_x(row):
    return row.get('x_pitch') if row and row.get('is_observed') and (row.get('confidence') or 0)>=.4 else None

# ---------------------------------------------------------------------------
# 2026-09-08 V8 -- O3/O8: EFFECTIVE OFFSIDE REFERENCE preparation. Real
# offside law's own rule is "the second-to-last opponent OR the ball,
# whichever is closer to the goal line" -- `estimated_position()` above
# ALREADY uses this comparison inline (`ad>=bd`) at the exact moment of
# a pass release. This function exposes the SAME comparison as a
# general PER-FRAME, disclosed field (not gated on a release event) so
# it can be inspected/prepared for the future pass-release module --
# it does NOT drive any classification decision by itself yet.
def effective_offside_reference_depth(line,ball_x_val,team,frame,pitch=DEFAULT_PITCH):
    """Returns (effective_depth_cm, source) where source is 'opponent'
    or 'ball'. None,None if the reference line itself is invalid."""
    if not line.get('valid'):return None,None
    line_depth=line['line_depth_cm']
    if ball_x_val is None:return line_depth,'opponent'
    ball_depth=pitch.depth(team,ball_x_val,frame)
    if ball_depth<line_depth:return ball_depth,'ball'
    return line_depth,'opponent'

# ---------------------------------------------------------------------------
# 2026-09-08 V8 -- O4: LATERAL STRUCTURE candidate feature. NOT wired
# into `score_backline_membership`/`DefensiveUnitTracker` (the actual
# pipeline) -- exposed ONLY as a separate "V8 candidate score" for the
# manual-validation comparison (O5), per the explicit instruction to
# audit whether lateral information improves manual agreement BEFORE
# adopting it. A true back line has coherent width/spacing, not just
# similar depth: `lateral_term(i)` rewards a player sitting close (in
# the pitch's WIDTH/y axis) to at least one other candidate member,
# penalizing an isolated lateral outlier -- same exp-decay form as the
# other V7 terms for consistent interpretability.
LATERAL_SCALE_CM=2000.0   # ~a quarter of the pitch width -- a generous plausible neighbour spacing, disclosed not fit
W_DEPTH_V8=0.4
W_TEMPORAL_V8=0.25
W_MOTION_V8=0.15
W_LATERAL_V8=0.2

def score_backline_membership_v8_candidate(players,team,frame,pitch,confirmed_ids_prev):
    """UNVALIDATED candidate scorer (O4/O5) -- same depth/temporal/motion
    terms as `score_backline_membership` (V7), REWEIGHTED to make room
    for a new `lateral_term`, computed against the OTHER V7-scored
    candidates' lateral (y) positions. For comparison/manual-validation
    only; the actual `DefensiveUnitTracker` pipeline is unchanged."""
    base_scores=score_backline_membership(players,team,frame,pitch,confirmed_ids_prev)
    if not base_scores:return {}
    all_y=[p['y_pitch'] for _,p in base_scores.values()]
    out={}
    for tid,(base_score,p) in base_scores.items():
        others_y=[y for y in all_y if y!=p['y_pitch']] or all_y
        nearest_gap=min(abs(p['y_pitch']-y) for y in others_y) if others_y else 0.
        lateral_term=float(np.exp(-nearest_gap/LATERAL_SCALE_CM))
        # Recover the individual V7 terms is not needed -- reweight using
        # the SAME base_score components proportionally is not exact, so
        # recompute directly from the base building blocks for honesty:
        depth_component=base_score  # base_score already = 0.5*depth+0.3*temporal+0.2*motion
        # Renormalize: scale the V7 combination down to its new smaller
        # weight budget (0.4+0.25+0.15=0.8) and add the lateral term at 0.2.
        v8_score=depth_component*(W_DEPTH_V8+W_TEMPORAL_V8+W_MOTION_V8)+W_LATERAL_V8*lateral_term
        out[tid]=(v8_score,p)
    return out

class RunTracker:
    def __init__(self,team,fps=30.,pitch=DEFAULT_PITCH):
        self.team,self.fps,self.pitch=team,fps,pitch;self.pending={};self.runs=[];self.next_id=0
    def close(self,key,frame,reason):
        r=self.pending.pop(key)
        if r['observations']>=RUN_MIN_OBSERVATIONS:
            r.update(end_frame=r['last_frame'],end_time_sec=r['last_frame']/self.fps,
                     duration_sec=r['observations']/self.fps,span_sec=(r['last_frame']-r['start_frame']+1)/self.fps,
                     coverage=r['observations']/(r['last_frame']-r['start_frame']+1),termination_reason=reason,termination_frame=frame)
            self.runs.append(r)
    def step(self,frame,players,line,line_velocity,role):
        team=self.team;defender=1-team;visible=[]
        valid_keys={(p['track_id'],p['cleaned_segment_id'],self.pitch.period(frame)) for p in eligible_players(players,team)}
        # A visible contradictory team or segment is a discontinuity, not a bridge.
        tracks={p['track_id']:(p['display_team_id'],p['cleaned_segment_id']) for p in players}
        for key in list(self.pending):
            r=self.pending[key]
            changed=key[2]!=self.pitch.period(frame) or (key[0] in tracks and tracks[key[0]]!=(team,key[1]))
            if changed or frame-r['last_frame']>RUN_MAX_GAP:
                self.close(key,frame,'IDENTITY_OR_PERIOD_CHANGE' if changed else 'SUPPORT_GAP')
        if not line['valid']:return []
        for p in eligible_players(players,team):
            if p['display_object_type']!='player' or not p.get('motion_valid',False):continue
            forward=p['vx_cm_s']*self.pitch.attacking_sign(team,frame)
            relation=line['line_depth_cm']-self.pitch.depth(defender,p['x_pitch'],frame)
            key=(p['track_id'],p['cleaned_segment_id'],self.pitch.period(frame))
            if forward<150 or abs(relation)>3000:continue
            if key not in self.pending:
                self.pending[key]=dict(run_id=f'R{team}-{p["track_id"]}-{frame}',attacking_team=team,defending_team=defender,
                    attacker_track_id=p['track_id'],cleaned_segment_id=p['cleaned_segment_id'],period=key[2],start_frame=frame,
                    start_time_sec=frame/self.fps,confirmation_frame=None,observations=0,last_frame=frame,samples=[])
            r=self.pending[key];r['observations']+=1;r['last_frame']=frame
            closing=None if line_velocity is None else line_velocity+forward
            # Current/past data only. No whole-run mean, and no missing-line zero.
            threat=None if closing is None else float(np.clip(.45*(1-min(abs(relation)/3000,1))+.35*np.clip(closing/500,0,1)+.2*np.clip(forward/500,0,1),0,1))
            sample=dict(frame=frame,x_pitch=p['x_pitch'],y_pitch=p['y_pitch'],relation_cm=relation,
                forward_speed_cm_s=forward,relative_closing_cm_s=closing,threat=threat,
                confidence=min(p.get('team_confidence',1.),line['confidence']),carrier_context=role['carrier_team'])
            r['samples'].append(sample)
            if r['observations']>=RUN_MIN_OBSERVATIONS:
                if r['confirmation_frame'] is None:r['confirmation_frame']=frame
                visible.append({**sample,'run_id':r['run_id'],'track_id':p['track_id'],'team':team,'segment':key[1],
                                'start_frame':r['start_frame'],'confirmation_frame':r['confirmation_frame'],
                                'state':'BREAK_THREAT' if threat is not None and threat>=.5 and -500<=relation<=100 else 'RUN_APPROACHING_LINE'})
        return visible
    def finish(self,n):
        for key in list(self.pending):self.close(key,n,'CLIP_END')
        return self.runs

def release_assessment(run,p,players_by_frame,ball_by_frame,lines_by_team,pitch=DEFAULT_PITCH):
    """Compare runner, ball, and line at the exact release timestamp.
    A pass record is not sufficient: same identity/segment at release and receipt,
    observed ball at both ends, quality, and goal-side receipt are required.
    """
    team=run['attacking_team'];defender=1-team;f=p['start_frame'];end=p['end_frame']
    find=lambda fr:next((a for a in players_by_frame.get(fr,[]) if a['track_id']==run['attacker_track_id'] and a['display_team_id']==team and a['cleaned_segment_id']==run['cleaned_segment_id'] and a.get('x_pitch') is not None),None)
    a=find(f);z=find(end);line=lines_by_team[defender].get(f);end_line=lines_by_team[defender].get(end)
    valid,reason=evidence_quality(p,'pass',players_by_frame)
    if not valid:return 'UNCERTAIN',reason
    if not a or not line or not line['valid'] or ball_x(ball_by_frame.get(f)) is None:return 'UNCERTAIN','MISSING_SAME_TIME_RELEASE_GEOMETRY'
    if line.get('confidence',0)<.7:return 'UNCERTAIN','INSUFFICIENT_VISIBLE_LINE_COVERAGE'
    est=estimated_position(a['x_pitch'],ball_x(ball_by_frame[f]),line,defender,f,pitch,50.)
    if est=='ESTIMATED_OFFSIDE_POSITION':return 'OFFSIDE_OR_EARLY','ESTIMATED_POSITION_AT_RELEASE_NOT_OFFICIAL_OFFENCE'
    if est!='ESTIMATED_ONSIDE_POSITION':return 'UNCERTAIN','RELEASE_MARGIN_OR_BALL_UNKNOWN'
    if not z or not end_line or not end_line['valid'] or ball_x(ball_by_frame.get(end)) is None:return 'UNCERTAIN','RECEIPT_UNOBSERVED'
    at_release=line['line_depth_cm']-pitch.depth(defender,a['x_pitch'],f)
    at_receipt=end_line['line_depth_cm']-pitch.depth(defender,z['x_pitch'],end)
    between=[find(fr) for fr in range(f,end+1)]
    present=[i for i,q in enumerate(between) if q]
    continuity=len(present)/len(between)>=.8 and all(b-a<=RUN_MAX_GAP for a,b in zip(present,present[1:]))
    b=ball_by_frame[end];received=np.hypot(b['x_pitch']-z['x_pitch'],b['y_pitch']-z['y_pitch'])<=200
    penetration=pitch.progress(team,a['x_pitch'],z['x_pitch'],f)>=100
    if continuity and received and penetration and at_release<=0 and at_receipt>=100 and end_line.get('confidence',0)>=.7:
        return 'SUCCESSFUL_BREAK','ESTIMATED_ONSIDE_RELEASE_AND_OBSERVED_GOAL_SIDE_RECEIPT'
    return 'UNCERTAIN','NO_CONFIRMED_CONTINUOUS_PENETRATION_RECEIPT'

def classify_break_lifecycle(runs,passes,players_by_frame,ball_by_frame,lines,trap_frames,fps=30.,pitch=DEFAULT_PITCH):
    events=[];used=set()
    for run in sorted(runs,key=lambda r:(r['start_frame'],r['run_id'])):
        samples=run['samples'];team=run['attacking_team'];defender=1-team
        crossings=[b for a,b in zip(samples,samples[1:]) if b['frame']-a['frame']<=RUN_MAX_GAP and a['relation_cm']<=0 and b['relation_cm']>0]
        matched=sorted([p for p in passes if p['team_id']==team and p['target_track_id']==run['attacker_track_id'] and run['start_frame']<=p['start_frame']<=run['end_frame'] and (p.get('scene_id',0),p['pass_id']) not in used],key=lambda p:p['start_frame'])
        relevant=bool(crossings or matched or any(abs(s['relation_cm'])<=200 for s in samples))
        if not relevant:continue
        lifecycle=[dict(state='RUN_APPROACHING_LINE',event_frame=run['start_frame'],confirmation_frame=run['confirmation_frame'])]
        threat=next((s for s in samples if s['frame']>=run['confirmation_frame'] and s['threat'] is not None and s['threat']>=.5 and -500<=s['relation_cm']<=100),None)
        if threat:lifecycle.append(dict(state='BREAK_THREAT',event_frame=threat['frame'],confirmation_frame=threat['frame']))
        if crossings:
            c=crossings[0];lifecycle.append(dict(state='LINE_CROSSED',event_frame=c['frame'],confirmation_frame=max(run['confirmation_frame'],c['frame'])))
            lifecycle.append(dict(state='POTENTIAL_BREAK',event_frame=c['frame'],confirmation_frame=max(run['confirmation_frame'],c['frame'])))
        label='UNCERTAIN';reason='NO_PASS_EVIDENCE';evidence_id=None;event_frame=run['end_frame'];confirmation=run['termination_frame']
        if matched:
            p=matched[0];used.add((p.get('scene_id',0),p['pass_id']));evidence_id=f'pass:{p.get("scene_id",0)}:{p["pass_id"]}'
            lifecycle.append(dict(state='RELEASE_WINDOW',event_frame=p['start_frame'],confirmation_frame=p['end_frame']))
            label,reason=release_assessment(run,p,players_by_frame,ball_by_frame,lines,pitch)
            event_frame=p['end_frame'];confirmation=max(p['end_frame'],run['confirmation_frame'])
        else:
            observed=sum(ball_x(ball_by_frame.get(f)) is not None for f in range(run['start_frame'],run['end_frame']+1))/(run['end_frame']-run['start_frame']+1)
            # Recovery is geometrical and runner-linked, never inferred from a score drop.
            recovered=bool(crossings and samples[-1]['relation_cm']<=-100 and samples[-1]['relative_closing_cm_s'] is not None and samples[-1]['frame']>crossings[0]['frame'])
            if recovered:label,reason='DEFENDER_RECOVERY','RUNNER_RETURNED_BEHIND_LINE_AFTER_CROSSING'
            elif observed>=.8:label,reason='PASS_NOT_PLAYED','NO_MATCHED_PASS_WITH_OBSERVED_BALL_COVERAGE'
        trap_ids={trap_frames[f]['teams'][defender]['episode_id'] for f in range(run['start_frame'],min(event_frame,len(trap_frames)-1)+1) if trap_frames[f]['teams'][defender]['state']=='TRAP_ACTIVE'}
        lifecycle.append(dict(state=label,event_frame=event_frame,confirmation_frame=confirmation))
        for item in lifecycle:
            item['display_frame']=max(item['event_frame'],item['confirmation_frame'])
            item['event_time']=item['event_frame']/fps;item['confirmation_time']=item['confirmation_frame']/fps;item['display_time']=item['display_frame']/fps
        events.append(dict(event_id='B-'+run['run_id'],run_id=run['run_id'],attacking_team=team,defending_team=defender,
            attacker_track_id=run['attacker_track_id'],outcome=label,reason=reason,outcome_evidence_id=evidence_id,
            event_frame=event_frame,confirmation_frame=confirmation,display_frame=max(event_frame,confirmation),
            is_trap_break=bool(trap_ids and label=='SUCCESSFUL_BREAK'),trap_episode_ids=sorted(x for x in trap_ids if x),lifecycle=sorted(lifecycle,key=lambda e:(e['display_frame'],e['event_frame']))))
    return events

def build_offside_v4(players_by_frame,roles,ball_by_frame,passes,fps=30.,pitch=DEFAULT_PITCH):
    machines={t:EvidenceFSM(t,TRAP_CONFIG,fps) for t in (0,1)};hist={t:deque(maxlen=round(fps)+1) for t in (0,1)}
    runners={t:RunTracker(t,fps,pitch) for t in (0,1)};frames=[];lines={t:{} for t in (0,1)}
    # 2026-09-07 V7: DefensiveUnitTracker (temporal MEMBERSHIP SCORE +
    # explicit candidate/confirmed/missing state machine) REPLACES
    # BackLineTracker as the pipeline's actual unit-membership source
    # feeding `unit_motion()`/`trap_synchrony_v7()` -- see the V7 audit
    # report for the measured effect vs. V6's fixed-gap clustering.
    unit_trackers={t:DefensiveUnitTracker(t,fps,pitch=pitch) for t in (0,1)}
    stepping_streak_frames={0:0,1:0}
    for f,role in enumerate(roles):
        players=players_by_frame.get(f,[]);pair=[]
        for team in (0,1):
            line=line_geometry(players,team,f,pitch)
            # O3/O8 (2026-09-08 V8): PREPARED effective-reference field
            # (second-last opponent OR the ball, whichever is closer to
            # goal) -- additive, does not change `line['valid']`/
            # `line_x`/`line_depth_cm` and does NOT drive any
            # classification here; `estimated_position()` already does
            # its own release-moment version of this comparison
            # independently for actual pass-release assessment.
            eff_depth,eff_source=effective_offside_reference_depth(line,ball_x(ball_by_frame.get(f)),team,f,pitch)
            line=dict(line,effective_reference_depth_cm=eff_depth,effective_reference_source=eff_source)
            lines[team][f]=line
            step_result=unit_trackers[team].step(players,f)
            unit=step_result['confirmed'];hist[team].append(dict(frame=f,unit=unit))
            velocity,common=unit_motion(hist[team],team,f,fps,pitch)
            attackers=eligible_players(players,1-team)
            context=None if not line['valid'] or not attackers else any(abs(line['line_depth_cm']-pitch.depth(team,p['x_pitch'],f))<=2000 for p in attackers)
            streak_sec=stepping_streak_frames[team]/fps
            metric=trap_synchrony_v7(unit,team,velocity,context,streak_sec,f,pitch)
            stepping_streak_frames[team]=stepping_streak_frames[team]+1 if metric.get('stepped') else 0
            # A3/A4: THREE separate defensive references, never conflated.
            # (1) unit centre -- robust MEDIAN x/depth of the CONFIRMED
            #     temporal-membership unit; the PRIMARY reference for
            #     trap movement/synchrony analysis (not a single player).
            # (2) deepest outfield defender -- rear member of that SAME
            #     confirmed unit.
            # (3) `line` (below, unchanged) -- the 2nd-nearest-to-goal
            #     OPPONENT incl. GK, for attacker legality/relation only.
            if unit:
                unit_sorted=sorted(unit,key=lambda p:pitch.depth(team,p['x_pitch'],f))
                deep_p,front_p=unit_sorted[0],unit_sorted[-1]
                centre_x=float(np.median([p['x_pitch'] for p in unit_sorted]))
                back_line=dict(valid=True,member_ids=[p['track_id'] for p in unit_sorted],
                    candidate_ids=[p['track_id'] for p in step_result['candidate']],
                    missing_ids=list(step_result['missing_ids']),
                    front_x=front_p['x_pitch'],front_depth_cm=pitch.depth(team,front_p['x_pitch'],f),front_track_id=front_p['track_id'],
                    deep_x=deep_p['x_pitch'],deep_depth_cm=pitch.depth(team,deep_p['x_pitch'],f),deep_track_id=deep_p['track_id'],
                    centre_x=centre_x,centre_depth_cm=pitch.depth(team,centre_x,f))
            else:
                back_line=dict(valid=False,member_ids=[],candidate_ids=[p['track_id'] for p in step_result['candidate']],
                    missing_ids=list(step_result['missing_ids']),
                    front_x=None,front_depth_cm=None,front_track_id=None,
                    deep_x=None,deep_depth_cm=None,deep_track_id=None,centre_x=None,centre_depth_cm=None)
            # FSM context is PERIOD-only (2026-09-07 continuation fix): an
            # earlier version also included the exact 5-member unit-identity
            # SET here, on the theory that unit membership changing should
            # end the episode. Measured effect: it was the DOMINANT
            # termination reason for every trap episode in the full replay
            # (visibility blips -- a defender briefly leaving/re-entering
            # camera view, or a marginal 5th/6th-deepest swap -- reset the
            # whole set almost every frame, capping every real episode at
            # 2-6 observations regardless of true underlying continuity).
            # `unit_motion()` ALREADY enforces its own, more appropriate
            # continuity requirement internally (>=3 IDENTICAL, continuously
            # motion-valid members across the full trailing 0.5s window,
            # returning None otherwise) -- that is the real substantive
            # membership-stability gate; re-enforcing exact-set equality a
            # second time at the FSM layer was redundant and, in practice,
            # far stricter than the real football concept of "the same back
            # line" (which tolerates a personnel blip without ceasing to be
            # a coordinated unit).
            identity_context=pitch.period(f)
            state=machines[team].step(f,f/fps,metric['score'],metric['valid'],True if metric['valid'] else None,identity_context,metric['confidence'])
            pair.append({**state,'line':line,'line_velocity_cm_s':velocity,'unit_ids':[p['track_id'] for p in unit],
                         'unit_motion_ids':common,'metrics':metric,'back_line':back_line,'runners':[]})
        for attacking in (0,1):
            opposing=pair[1-attacking]
            running=runners[attacking].step(f,players,opposing['line'],opposing['line_velocity_cm_s'],role)
            pair[attacking]['runners']=running
            observed_attackers=eligible_players(players,attacking)
            coverage=sum(p.get('motion_valid',False) for p in observed_attackers)/max(1,len(observed_attackers))
            available=opposing['line']['valid'] and coverage>=.75 and opposing['line_velocity_cm_s'] is not None
            scores=[r['threat'] for r in running if r['threat'] is not None]
            pair[attacking]['run_threat_score']=max(scores) if scores else 0. if available and not running else None
            pair[attacking]['break_state_as_attacker']='BREAK_THREAT' if any(r['state']=='BREAK_THREAT' for r in running) else 'RUN_APPROACHING_LINE' if running else 'NO_RUN' if available else 'UNCERTAIN'
        state,team=dominant(pair,'NO_TRAP')
        frames.append(dict(frame=f,time_sec=f/fps,teams=pair,dominant_state=state,dominant_team=team,role=role))
    episodes=[e for m in machines.values() for e in m.finish(len(roles),len(roles)/fps)]
    runs=[r for tracker in runners.values() for r in tracker.finish(len(roles))]
    events=classify_break_lifecycle(runs,passes,players_by_frame,ball_by_frame,lines,frames,fps,pitch)
    # Stable lifecycle display: confirmed outcome has priority for its own runner;
    # never replace a newer active trap with an old retrospective outcome banner.
    for event in events:
        for step in event['lifecycle']:
            f=step['display_frame']
            if f>=len(frames):continue
            team=event['attacking_team'];record=frames[f]['teams'][team]
            record.setdefault('lifecycle_updates',[]).append(dict(event_id=event['event_id'],run_id=event['run_id'],**step))
        if event['is_trap_break']:
            f=event['display_frame']
            if f<len(frames):frames[f]['teams'][event['defending_team']].setdefault('confirmed_trap_break_events',[]).append(event['event_id'])
    return dict(frames=frames,runs=runs,events=events,episodes=episodes,config=TRAP_CONFIG.__dict__)
