# Longer-Video Strategy (V2, Part 5) -- Offside-Break

**Status: analysis and cost estimate only. No new tracking export was
run.** See `pressing_structure/docs/LONGER_VIDEO_STRATEGY_V2.md` for
the full cost derivation (shared source video, shared tracking
pipeline, shared per-frame cost measurements) -- reproduced here in
summary since these are separate projects and each should be able to
stand on its own.

## Candidate segment and cost

Candidate: seconds 0-300 (5 minutes) of `testVideo1.mp4` (334.87s
total) = existing tracked 0-120s + a new, unprocessed 120-300s (180s /
5,400-frame) tail. Extrapolated from this project's own prior
measurements (`PLAN.md`): **~4.0 hours** for the base tracking pipeline
on the new tail (`export_tracking.py` ~3.02h, `detect_ball_
specialized.py` ~0.73h, `replay_homography.py` ~0.29h), plus ~15-20
minutes for offside V2 analytics + dashboard render on the resulting
300s clip. **Grand total ~4.3-4.5 hours, not run without approval.**

## Extrapolated event counts (from this session's measured 120s rate)

| Metric | 120s (measured) | 300s (extrapolated x2.5) |
|---|---|---|
| Attacking runs (both directions) | 374 | ~935 |
| Trap-state TRAP_FORMING frames | 899 + 377 (per direction) | scales proportionally with clip length |
| Trap-break candidates | 0 | **not reliably extrapolable from a zero count** -- see caveat |

**Caveat**: the trap-synchrony score reached 0.66-0.67 in brief,
un-sustained spikes on this 120s clip (just under the 0.65-for-15-
frames TRAP_ACTIVE threshold). Whether a longer sample would contain a
genuine sustained trap cannot be predicted from zero observed
instances; no number is claimed here.

## Event-centered clips: rendered from the EXISTING 120s clip instead

The 5-minute expansion was not run, so event-centered clips were
produced from data already computed and QA'd on the 120s clip:

- **0 trap-break candidates exist**, so no "break" clips could be
  produced (there is nothing to center them on -- inventing one would
  violate the project's no-fabrication rule).
- Instead, `offside_break/outputs/event_clips/` contains clips
  centered on the **5 longest-duration attacking runs** (the same ones
  illustrated in `graphs/09_attacker_vs_line_depth_selected_runs.png`
  for the primary direction, plus 2 more from the other direction) --
  explicitly labeled as "notable runs by duration," NOT as trap-break
  events, since none were detected. See that folder's `README.md` for
  the exact list and the disclosed selection rule (top-N by
  `duration_sec`, not by any outcome judgment).
