"""
Coach-mode DISPLAY helpers for the offside/trap dashboard's second-row
graphs (2026-09-09 implementation, following the
`audit_reports/2026-09-09_offside_graph_feasibility/` audit). Pure,
presentation-only functions built on top of the UNCHANGED, existing
`build_offside_v4()`/`build_pass_run_events_v9()` outputs -- no
existing offside analytics formula, threshold, or FSM logic is
duplicated or modified here.

Fully independent of `pressing_structure/` -- PRESSING IS LOCKED this
round; nothing in this file imports from or references it.
"""
import math

# ---------------------------------------------------------------------------
# Trap-Break Event Funnel: dynamic "nice" Y-axis ceiling. Identical
# design/table to the one validated for pressing's outcome bars, but a
# fresh, independent implementation here (pressing's own module is not
# imported, per this round's lock).
NICE_AXIS_SCALE = (1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000)


def nice_bar_axis(max_count):
    """Smallest table ceiling STRICTLY greater than `max_count` (falls
    back to the next power-of-ten beyond the table for very large
    counts). Recomputing this fresh from the CURRENT max_count every
    frame is already monotonic (never shrinks) by construction, because
    `cumulative_funnel_counts` below is itself non-decreasing in
    `cur_time` -- no separate stateful tracker is needed."""
    if max_count <= 0:
        return 1
    for s in NICE_AXIS_SCALE:
        if s > max_count:
            return s
    return int(10 ** math.ceil(math.log10(max_count)))


FUNNEL_STAGES = ("Pass Releases", "Runner Associations", "Observed Crossings", "Successful Breaks", "Failed Breaks")


def cumulative_funnel_counts(events, cur_time):
    """Causal cumulative counts for the Trap-Break Event Funnel, up to
    and including `cur_time`. Each event contributes to a stage only
    once that stage's own real evidence is KNOWN as of `cur_time`:

      - Pass Releases / Runner Associations: known at `release_time_sec`
        (release detection and the at-release association check both
        resolve atomically at the same real moment, per
        `runner_state_at_release`).
      - Observed Crossings: known at `observed_crossing_time_sec`
        (only counted when a crossing was actually observed --
        `cross_valid`).
      - Successful/Failed Breaks: `build_pass_run_events_v9` flattens
        the lifecycle to a single `lifecycle_outcome` scalar with no
        per-state timestamp retained on the final event record, so this
        function uses a documented, causal, honest proxy for "when the
        final outcome became known": `observed_crossing_time_sec` when
        a crossing was actually observed (the outcome cannot be fully
        resolved before that evidence exists), otherwise
        `release_time_sec`. This matches the real mechanism for every
        FAILED_BREAK in this project's reference clip, which is decided
        immediately at release (the runner is already in an estimated
        illegal position) with no crossing ever observed at all --
        verified directly against that real event, not assumed.

    Never fabricates a count, never decrements once incremented (counts
    are strictly cumulative -- a later `cur_time` can only add to an
    earlier call's result, never remove from it)."""
    counts = {k: 0 for k in FUNNEL_STAGES}
    for e in events:
        release_t = e.get("release_time_sec")
        if e.get("release_valid") and release_t is not None and release_t <= cur_time + 1e-9:
            counts["Pass Releases"] += 1
            if e.get("association_valid"):
                counts["Runner Associations"] += 1

        cross_t = e.get("observed_crossing_time_sec")
        cross_known = bool(e.get("cross_valid")) and cross_t is not None
        if cross_known and cross_t <= cur_time + 1e-9:
            counts["Observed Crossings"] += 1

        if cross_known:
            outcome_known_time = cross_t
        elif e.get("release_valid") and release_t is not None:
            outcome_known_time = release_t
        else:
            outcome_known_time = None

        if outcome_known_time is not None and outcome_known_time <= cur_time + 1e-9:
            outcome = e.get("lifecycle_outcome")
            if outcome == "SUCCESSFUL_BREAK":
                counts["Successful Breaks"] += 1
            elif outcome == "FAILED_BREAK":
                counts["Failed Breaks"] += 1
    return counts
