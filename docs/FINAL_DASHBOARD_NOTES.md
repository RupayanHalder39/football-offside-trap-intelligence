# Final Dashboard Visual Redesign (Offside / Trap-Break)

Pure visual/layout restyle -- **no analytics logic changed**. Same V2
functions (`offside_line_series`, `trap_state_series`,
`detect_attacking_runs`, `classify_trap_break_candidates`,
`compute_estimated_offside_position`, `compute_line_relation_cm`,
`line_relation_label`) reused unmodified from `offside_break/
analytics/offside.py` and `offside_v2.py`. See `offside_v2_summary.md`
for the actual trap/run/candidate findings, which are unchanged by
this restyle. **V2 terminology is preserved exactly**: AHEAD OF
DEFENSIVE LINE / BEHIND DEFENSIVE LINE, ESTIMATED_OFFSIDE_POSITION /
ESTIMATED_ONSIDE_POSITION / LINE_RELATION_ONLY,
NO_TRAP / TRAP_FORMING / TRAP_ACTIVE / UNCERTAIN -- no "OFFSIDE" legal
decision or fake "trap success"/"successful break" label is ever
shown.

## What changed

- New dark-card visual language (shared `dashboard_style.py` module,
  same one used by the pressing final dashboard) modeled on a locked
  reference mockup -- layout/hierarchy/color-treatment only, no values
  or labels copied from it.
- Canvas 2304x1204 (unchanged). Layout mirrors the pressing final
  dashboard's row heights for visual consistency between the two
  projects: dark header (title + trap-break-candidate status + clock)
  -> top row (video feed + radar, 730px / 60.6% of canvas height) -> 4
  metric tiles (trap state, trap synchrony score, distance to line,
  relative speed to line) -> 4 graph panels (attacker depth vs.
  defensive-line depth [dual line, the most important graph per the
  brief], trap synchrony score, relative closing rate onto the line,
  distance to lead attacker over time) -> full-width timeline (trap
  state color band + run-episode markers + trap-break-candidate
  markers, none present on this clip).
- The offside radar now also washes in the real pitch-control gradient
  (`compute_pitch_control_grid`, imported read-only) alongside the
  defensive line and space-behind-line shading -- v1/V2's radar had
  this; it was inadvertently left out of the first final-dashboard
  draft and added back in during prototype QA before the full render.

## QA-driven fixes made during this restyle

- **Label collision found during full-clip QA (t=30s)**: a "RUN" text
  label on one attacker's box visually overlapped the "LEAD ATTACKER"
  chip on a different, nearby attacker's box -- two different
  entities' labels colliding because the players stood close together
  on screen. Fixed by dropping the "RUN" text entirely in favor of a
  colored border only (which cannot collide with text); the lead
  attacker keeps its chip since there is only ever one per frame. The
  full 120s dashboard was **re-rendered** after this fix.
- The per-frame trap-state timeline segment was originally recomputed
  every output frame (3600 one-frame-wide rectangles x 3600 frames =
  ~13M draws) -- refactored to render the static timeline background
  ONCE and redraw only the moving current-time marker per frame.
- The current-time marker's low-contrast-on-bright-background issue
  (same fix as the pressing dashboard) applies here too via the shared
  `dashboard_style.py` module.

## Output

`offside_break/outputs/final_dashboard/offside_dashboard_120s_final.mp4`
(full 120s, 3600 frames, 2304x1204). `offside_break/outputs/v2_120s/
offside_dashboard_120s_v2.mp4` (V2's own layout) is superseded by this
restyle and was deleted once this file passed full QA -- its hash/size
is recorded in the final report for this stage.
