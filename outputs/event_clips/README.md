# Offside-break event-centered clips

**0 trap-break candidates exist on this 120s clip** (see `offside_v2_summary.md`) -- there is nothing to center a genuine 'break' clip on. These clips are instead centered on the longest-duration attacking runs (a disclosed, outcome-independent selection rule -- NOT a claim that these are trap breaks or even trap-adjacent), 3 from the primary direction (team1 attacks team0's line, the dashboard's own direction) and 2 from the other. Cut from the QA'd full-120s V2 dashboard (`offside_break/outputs/v2_120s/offside_dashboard_120s_v2.mp4`), not re-rendered. Each clip = [run_start - 5s, run_end + 5s], clamped to [0, 120s].

| Run | Direction | Type | Duration (s) | Run window (s) | Clip window (s) | File |
|---|---|---|---|---|---|---|
| 14 | team1_attacks_team0 | curved | 14.0 | 32.6-46.6 | 27.6-51.6 | run_014_team1_attacks_team0_curved.mp4 |
| 1 | team1_attacks_team0 | check-run | 12.4 | 26.5-38.9 | 21.5-43.9 | run_001_team1_attacks_team0_check-run.mp4 |
| 110 | team1_attacks_team0 | curved | 12.0 | 29.2-41.2 | 24.2-46.2 | run_110_team1_attacks_team0_curved.mp4 |
| 72 | team0_attacks_team1 | curved | 12.2 | 9.8-21.9 | 4.8-26.9 | run_072_team0_attacks_team1_curved.mp4 |
| 78 | team0_attacks_team1 | curved | 8.5 | 7.7-16.2 | 2.7-21.2 | run_078_team0_attacks_team1_curved.mp4 |
