# Offside-Trap Break — Prototype Summary

Input: `outputs/tracking/testVideo1_120s/tracking.parquet` (READ-ONLY,
unchanged) through this project's shared `cleaned_tracking_view.py`
(same module Project 1 uses), plus the EXISTING PhD `analytics.
possession` / `analytics.pitch_control` modules (unmodified).

## Defending-end calibration (measured)

Team 0's goalkeeper averages x=11168cm -> team 0 defends the x=12000
end. Team 1's goalkeeper averages x=1410cm -> team 1 defends the x=0
end. Measured directly from the tracking data, not assumed.

## What was run

Full 120s clip, both attacking directions (team 1 attacking vs. team 0
defending, and vice versa), analytics only (12.5s runtime total).
Offside-line + attacker features computed for every frame with >=2
visible defenders (no ball-possession gating required for the line
itself; passing-lane-from-carrier is the only sub-feature that needs a
confident carrier and is `n/a` otherwise).

## Headline numbers (team 1 attacking vs. team 0 defending direction)

- Frames assessed: 3597 / 3598.
- Onside vs. ahead-of-line attacker-frames: 22190 onside (60.1%), 14701
  ahead of the line (39.9%) -- a high ahead-of-line rate, expected since
  this counts EVERY attacking-team outfield player every frame, most of
  whom are simply advanced up the pitch in normal play, not making an
  offside-beating run.
- Run type: lateral 23993, diagonal 7660, straight 4574, unknown 664 --
  most player-frames are lateral/positional movement, not a directed
  run, which is realistic (most of a match is not a sprint).
- Break potential: LOW 30649, MEDIUM 2664, HIGH 2914 (out of ~40987
  assessed attacker-frames).
- "HIGH break potential" episodes (>=1 attacker sustaining HIGH for
  >=1 frame, gap-bridged at <=0.5s): **409** across the full 120s, both
  directions combined. Longest: 14.6s (track 2, team 1 attacking,
  frames 784-1222).

## Prototype dashboard

Rendered on the SAME 0-25s window as Project 1 (for direct
cross-project comparability), team 1 attacking vs. team 0 defending.

`offside_break/outputs/offside_dashboard.mp4` -- 751 frames, 2044x874,
~25s.

## Visual QA

Inspected beginning, middle, end, and 3 additional intermediate frames.
All checked consistent: defensive line position tracks visually with
the actual defenders' depth, "AHEAD"-labeled attackers are visually on
the correct side of the drawn line, run-type/angle/break-potential
values in the metrics panel matched what was computed independently by
code (spot-checked). One visual double-take (a lime-green-looking box)
was checked by zooming into the actual pixels and confirmed to be the
goalkeeper's own bright jersey color in the source footage, not a
rendering bug.

## Known limitations

- **"HIGH break potential" is a geometric threshold crossing, NOT a
  validated outcome** -- there is no ground truth for "this run actually
  beat the trap and received a dangerous pass" in this dataset. 409
  episodes measures how often the geometry crosses explicit thresholds,
  not how often a break actually succeeded.
- Run-type classification only attempts 3 classes (straight/diagonal/
  lateral); curved, check-run, third-man, and inside-to-outside/
  outside-to-inside runs are explicitly NOT implemented (would need
  multi-frame trajectory-shape analysis this prototype does not yet
  build) -- never forced into a wrong class, but also never attempted.
- The offside line is a "second-last defender depth" reading, smoothed
  for display only -- it does NOT model the exact frame of a pass
  release, so no output here should be read as a genuine offside/onside
  legal decision. Explicitly labeled "estimated state," never "decision."
- Space/pitch-control behind the line reuses the same simplified
  time-to-intercept model as Project 1 -- same disclosed limitations.
- Same underlying ball/tracking sparsity as the main PhD pipeline and
  Project 1 (raw track fragmentation into ~10 segments/track on
  average) limits velocity-based feature confidence over longer spans.

## Is supervised modeling currently justified?

**No.** There is no labeled "successful break" outcome in this dataset
at all (unlike pressing, which had real turnovers/passes to anchor
SUCCESS/FAILED) -- only a geometric threshold. Building a supervised
model on top of a threshold that defines its own label would not learn
anything beyond the threshold itself. This remains a rule-based
analytics + visualization prototype.
