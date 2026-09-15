"""
Event-centered clips (Part 5 of the V2 brief) for offside-break. There
are 0 trap-break candidates on this clip (see offside_v2_summary.md),
so there is nothing to center a "break" clip on -- inventing one would
violate the project's no-fabrication rule. Instead, per
`offside_break/docs/LONGER_VIDEO_STRATEGY_V2.md`, this cuts clips
around the 3 longest-duration attacking runs in the PRIMARY direction
(team1 attacks team0's line -- the dashboard's own direction) plus the
2 longest in the OTHER direction, from the already-rendered, QA'd
`offside_dashboard_120s_v2.mp4` -- no re-rendering, frame-accurate
OpenCV extraction only. Explicitly labeled "notable runs by duration",
NOT break events.
"""
import os
import sys

import cv2
import polars as pl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from offside_break.analytics.offside import offside_line_series  # noqa: E402
from offside_break.analytics.offside_v2 import detect_attacking_runs  # noqa: E402
from pressing_structure.analytics.cleaned_tracking_view import build_cleaned_view  # noqa: E402

SOURCE = os.path.join(REPO_ROOT, "offside_break/outputs/v2_120s/offside_dashboard_120s_v2.mp4")
OUT_DIR = os.path.join(REPO_ROOT, "offside_break/outputs/event_clips")
PAD_SEC = 5.0
# (direction label, attacking_team, defending_team, how many top-duration runs to take)
SELECTIONS = [("team1_attacks_team0", 1, 0, 3), ("team0_attacks_team1", 0, 1, 2)]


def main():
    tracking = pl.read_parquet(os.path.join(REPO_ROOT, "outputs/tracking/testVideo1_120s/tracking.parquet"))
    cleaned = build_cleaned_view(tracking)
    n_total = int(cleaned["frame"].max()) + 1

    cap = cv2.VideoCapture(SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = []
    for direction, att, deff, top_n in SELECTIONS:
        line_series = offside_line_series(cleaned, deff, 0, n_total - 1)
        runs = detect_attacking_runs(cleaned, att, deff, line_series)
        top_runs = runs.sort("duration_sec", descending=True).head(top_n).to_dicts() if runs.height else []
        for run in top_runs:
            f0 = max(0, run["start_frame"] - int(PAD_SEC * fps))
            f1 = min(n_frames_video - 1, run["end_frame"] + int(PAD_SEC * fps))
            out_path = os.path.join(OUT_DIR, f"run_{run['run_id']:03d}_{direction}_{run['run_type']}.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            for f in range(f0, f1 + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            writer.release()
            manifest.append({"run_id": run["run_id"], "direction": direction, "run_type": run["run_type"],
                              "duration_sec": run["duration_sec"], "run_start_sec": run["start_time_sec"],
                              "run_end_sec": run["end_time_sec"], "clip_start_sec": f0 / fps,
                              "clip_end_sec": f1 / fps, "file": os.path.basename(out_path)})
            print(f"wrote {out_path} ({f0}-{f1})")
    cap.release()

    with open(os.path.join(OUT_DIR, "README.md"), "w") as fh:
        fh.write("# Offside-break event-centered clips\n\n")
        fh.write("**0 trap-break candidates exist on this 120s clip** (see `offside_v2_summary.md`) "
                 "-- there is nothing to center a genuine 'break' clip on. These clips are instead "
                 "centered on the longest-duration attacking runs (a disclosed, outcome-independent "
                 "selection rule -- NOT a claim that these are trap breaks or even trap-adjacent), "
                 "3 from the primary direction (team1 attacks team0's line, the dashboard's own "
                 f"direction) and 2 from the other. Cut from the QA'd full-120s V2 dashboard "
                 f"(`{os.path.relpath(SOURCE, REPO_ROOT)}`), not re-rendered. Each clip = "
                 f"[run_start - {PAD_SEC:.0f}s, run_end + {PAD_SEC:.0f}s], clamped to [0, 120s].\n\n")
        fh.write("| Run | Direction | Type | Duration (s) | Run window (s) | Clip window (s) | File |\n"
                  "|---|---|---|---|---|---|---|\n")
        for m in manifest:
            fh.write(f"| {m['run_id']} | {m['direction']} | {m['run_type']} | {m['duration_sec']:.1f} | "
                      f"{m['run_start_sec']:.1f}-{m['run_end_sec']:.1f} | "
                      f"{m['clip_start_sec']:.1f}-{m['clip_end_sec']:.1f} | {m['file']} |\n")
    print(f"\n{len(manifest)} clips written to {OUT_DIR}")


if __name__ == "__main__":
    main()
