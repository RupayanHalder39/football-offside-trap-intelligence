"""
Static, full-120s-clip summary graphs for the offside-break V2 report
(Part 4 of the V2 brief). Recomputes analytics fresh (cheap, ~15-20s for
both attacking directions) rather than depending on the dashboard
render. Writes PNGs to `offside_break/outputs/v2_120s/graphs/`.

**Both directions computed**: unlike the live dashboard (which must
pick one fixed attacking/defending pairing for a single video), this
report computes BOTH directions (team 1 attacking team 0's line, and
team 0 attacking team 1's line) since a full-clip static report can
show both without a readability tradeoff. Time-series plots (line
depth, trap synchrony) use TWO subplots (one per direction); pooled
distributional plots combine both directions and disclose n per side.

**Sample-size honesty**: this clip produced 0 TRAP_ACTIVE frames and 0
trap-break candidates for EITHER direction (see offside_v2_summary.md
for the measured score distribution). Plots that would normally show
"outcomes by category" are rendered anyway where there is at least
something to show, explicitly labeled exploratory, and plots that
would be empty (e.g. "candidate break outcomes") are replaced with a
plot that states the finding in text rather than fabricating bars.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from offside_break.analytics.offside import offside_line_series  # noqa: E402
from offside_break.analytics.offside_v2 import (classify_trap_break_candidates, detect_attacking_runs,  # noqa: E402
                                                  trap_state_series)
from pressing_structure.analytics.cleaned_tracking_view import build_cleaned_view, load_ball_view  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "offside_break", "outputs", "v2_120s", "graphs")
EXPLORATORY_NOTE = "descriptive/exploratory only -- NOT statistical evidence"
DIR_COLOR = {"team1_attacks_team0": "#1565c0", "team0_attacks_team1": "#c62828"}
TRAP_COLOR = {"NO_TRAP": "#9e9e9e", "TRAP_FORMING": "#ff9800", "TRAP_ACTIVE": "#d32f2f", "UNCERTAIN": "#616161"}
RUN_TYPE_ORDER = ["straight", "diagonal", "lateral", "curved", "check-run",
                   "inside-to-outside", "outside-to-inside", "unknown"]


def _savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    tracking = pl.read_parquet(os.path.join(REPO_ROOT, "outputs/tracking/testVideo1_120s/tracking.parquet"))
    cleaned = build_cleaned_view(tracking)
    ball = load_ball_view(os.path.join(REPO_ROOT, "outputs/analytics/testVideo1_120s_v3"))
    passes = pl.read_parquet(os.path.join(REPO_ROOT, "outputs/analytics/testVideo1_120s_v3/passes.parquet"))
    n_total = int(cleaned["frame"].max()) + 1
    fps = 30.0

    dirs = {"team1_attacks_team0": (1, 0), "team0_attacks_team1": (0, 1)}
    data = {}
    for name, (att, deff) in dirs.items():
        print(f"computing {name} (attacking={att}, defending={deff})...", flush=True)
        line_series = offside_line_series(cleaned, deff, 0, n_total - 1)
        trap_series = trap_state_series(cleaned, deff, 0, n_total - 1)
        runs = detect_attacking_runs(cleaned, att, deff, line_series)
        cands = classify_trap_break_candidates(runs, trap_series, cleaned, ball, passes, att, deff) if runs.height else runs
        data[name] = {"line_series": line_series, "trap_series": trap_series, "runs": runs, "candidates": cands,
                       "attacking_team": att, "defending_team": deff}

    # 1. Defensive-line depth over time (2 subplots, one per direction)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, name in zip(axes, dirs):
        ls = data[name]["line_series"].to_dicts()
        t = [r["frame"] / fps for r in ls]
        d = [r["line_depth_cm_smoothed"] / 100.0 if r["line_depth_cm_smoothed"] is not None else np.nan for r in ls]
        ax.plot(t, d, color=DIR_COLOR[name], linewidth=1)
        ax.set_ylabel("Line depth (m)")
        ax.set_title(f"{name} -- defensive line depth over time", fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Defensive-line depth over the full 120s ({EXPLORATORY_NOTE})")
    fig.tight_layout()
    _savefig(fig, "01_defensive_line_depth_over_time.png")

    # 2. Line velocity distribution (numerical derivative of smoothed depth, pooled both directions)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in dirs:
        ls = data[name]["line_series"]["line_depth_cm_smoothed"].to_list()
        vel = [(ls[i] - ls[i - 1]) * fps / 100.0 for i in range(1, len(ls))
               if ls[i] is not None and ls[i - 1] is not None]
        ax.hist(vel, bins=60, alpha=0.55, label=f"{name} (n={len(vel)})", color=DIR_COLOR[name], range=(-15, 15))
    ax.set_xlabel("Frame-to-frame line velocity (m/s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Defensive-line velocity distribution\n({EXPLORATORY_NOTE})", fontsize=10)
    ax.legend(fontsize=8)
    _savefig(fig, "02_line_velocity_distribution.png")

    # 3. Trap synchrony score over time (2 subplots)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, name in zip(axes, dirs):
        ts = data[name]["trap_series"].to_dicts()
        t = [r["frame"] / fps for r in ts]
        s = [r["score"] if r["score"] is not None else np.nan for r in ts]
        ax.plot(t, s, color=DIR_COLOR[name], linewidth=0.8)
        ax.axhline(0.65, color="red", linestyle="--", linewidth=1, label="TRAP_ACTIVE threshold (0.65)")
        ax.axhline(0.4, color="orange", linestyle="--", linewidth=1, label="TRAP_FORMING threshold (0.4)")
        ax.set_ylabel("Score (0-1)")
        ax.set_title(f"{name} -- trap synchrony score over time (max reached: {np.nanmax(s):.2f})", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Trap synchrony score over the full 120s -- NEVER reaches TRAP_ACTIVE on this clip ({EXPLORATORY_NOTE})")
    fig.tight_layout()
    _savefig(fig, "03_trap_synchrony_over_time.png")

    # 4. Number of trap candidate episodes (trap state frame counts, both directions)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    states = ["NO_TRAP", "TRAP_FORMING", "TRAP_ACTIVE", "UNCERTAIN"]
    width = 0.35
    x = np.arange(len(states))
    for i, name in enumerate(dirs):
        counts_map = {r["trap_state"]: r["count"] for r in data[name]["trap_series"]["trap_state"].value_counts().to_dicts()}
        counts = [counts_map.get(s, 0) for s in states]
        ax.bar(x + (i - 0.5) * width, counts, width, label=name, color=DIR_COLOR[name])
    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.set_ylabel("Frame count (of 3600)")
    ax.set_title(f"Trap-state frame counts by direction\n(0 TRAP_ACTIVE frames on this clip -- {EXPLORATORY_NOTE})", fontsize=10)
    ax.legend(fontsize=8)
    _savefig(fig, "04_trap_state_frame_counts.png")

    # 5. Run-type counts (pooled)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(RUN_TYPE_ORDER))
    width = 0.35
    for i, name in enumerate(dirs):
        runs = data[name]["runs"]
        counts_map = {r["run_type"]: r["count"] for r in runs["run_type"].value_counts().to_dicts()} if runs.height else {}
        counts = [counts_map.get(t, 0) for t in RUN_TYPE_ORDER]
        ax.bar(x + (i - 0.5) * width, counts, width, label=f"{name} (n={runs.height})", color=DIR_COLOR[name])
    ax.set_xticks(x)
    ax.set_xticklabels(RUN_TYPE_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Run episode count")
    ax.set_title(f"Attacking-run type counts by direction\n({EXPLORATORY_NOTE})", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, "05_run_type_counts.png")

    # 6. Attacker-to-line distance distribution (at run end, pooled)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in dirs:
        runs = data[name]["runs"]
        if runs.height:
            vals = [v / 100.0 for v in runs["distance_to_line_end_cm"].to_list() if v is not None]
            ax.hist(vals, bins=30, alpha=0.55, label=f"{name} (n={len(vals)})", color=DIR_COLOR[name])
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="line (0 = at the line)")
    ax.set_xlabel("Distance to line at run end (m, + = ahead of line)")
    ax.set_ylabel("Count")
    ax.set_title(f"Attacker-to-line distance distribution\n({EXPLORATORY_NOTE})", fontsize=10)
    ax.legend(fontsize=8)
    _savefig(fig, "06_attacker_line_distance_distribution.png")

    # 7. Relative (forward) speed distribution during runs (pooled)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in dirs:
        runs = data[name]["runs"]
        if runs.height:
            vals = [v / 100.0 for v in runs["mean_forward_speed_cm_s"].to_list() if v is not None]
            ax.hist(vals, bins=30, alpha=0.55, label=f"{name} (n={len(vals)})", color=DIR_COLOR[name])
    ax.set_xlabel("Mean forward speed during run (m/s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Run forward-speed distribution\n({EXPLORATORY_NOTE})", fontsize=10)
    ax.legend(fontsize=8)
    _savefig(fig, "07_run_forward_speed_distribution.png")

    # 8. Trap-active vs non-trap-active run counts
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels, counts, colors = [], [], []
    for name in dirs:
        runs = data[name]["runs"]
        cands = data[name]["candidates"]
        if runs.height and cands.height:
            trap_active_runs = cands.filter(pl.col("is_trap_break_candidate")).height
            other_runs = runs.height - trap_active_runs
        else:
            trap_active_runs, other_runs = 0, runs.height
        labels += [f"{name}\nTRAP_ACTIVE-linked", f"{name}\nother"]
        counts += [trap_active_runs, other_runs]
        colors += ["#d32f2f", DIR_COLOR[name]]
    ax.bar(range(len(labels)), counts, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Run episode count")
    ax.set_title(f"Runs overlapping a TRAP_ACTIVE frame vs. other runs\n(0 on this clip, both directions -- {EXPLORATORY_NOTE})", fontsize=9)
    fig.tight_layout()
    _savefig(fig, "08_trap_active_vs_other_runs.png")

    # 9. Attacker-depth-vs-line-depth for selected notable runs (no true trap-break
    #    candidates exist on this clip -- illustrating the 3 longest-duration runs
    #    instead, clearly labeled as such, not as trap-break events)
    from offside_break.analytics.offside import _forward_depth, compute_offside_line  # noqa: E402
    primary = "team1_attacks_team0"
    runs_df = data[primary]["runs"]
    top_runs = runs_df.sort("duration_sec", descending=True).head(3).to_dicts() if runs_df.height else []
    fig, axes = plt.subplots(1, max(len(top_runs), 1), figsize=(5 * max(len(top_runs), 1), 4), squeeze=False)
    att, deff = data[primary]["attacking_team"], data[primary]["defending_team"]
    for i, run in enumerate(top_runs):
        ax = axes[0][i]
        frames = list(range(run["start_frame"], run["end_frame"] + 1))
        t = [f / fps for f in frames]
        attacker_rows = cleaned.filter((pl.col("track_id") == run["attacker_track_id"]) &
                                        (pl.col("frame") >= run["start_frame"]) & (pl.col("frame") <= run["end_frame"])).to_dicts()
        arow_by_frame = {r["frame"]: r for r in attacker_rows}
        depth = [(_forward_depth(deff, arow_by_frame[f]["x_pitch"]) / 100.0) if f in arow_by_frame and arow_by_frame[f]["x_pitch"] is not None else np.nan for f in frames]
        line_d = [(compute_offside_line(cleaned, deff, f)["line_depth_cm"] or np.nan) / 100.0 for f in frames]
        ax.plot(t, depth, label="attacker depth", color="#ef6c00")
        ax.plot(t, line_d, label="line depth", color="#c62828")
        ax.set_title(f"run {run['run_id']} (track {run['attacker_track_id']}, {run['run_type']}, {run['duration_sec']:.1f}s)", fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Depth from defending goal (m)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Attacker depth vs. line depth for the 3 longest runs on this clip\n"
                 f"(illustrative -- NO trap-break candidates were found; these are NOT break events, {EXPLORATORY_NOTE})", fontsize=9)
    fig.tight_layout()
    _savefig(fig, "09_attacker_vs_line_depth_selected_runs.png")

    # 10. Candidate break outcomes -- explicitly report the null finding rather than fabricate bars
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    total_runs = sum(data[n]["runs"].height for n in dirs)
    max_score = max(max((v for v in data[n]["trap_series"]["score"].to_list() if v is not None), default=0.0) for n in dirs)
    msg = (f"0 trap-break candidates found on this 120s clip (either direction).\n\n"
           f"Reason: a candidate requires TRAP_ACTIVE to have held during the run "
           f"(score >= 0.65 sustained for >= 0.5s); the trap synchrony score never "
           f"sustained above {max_score:.2f} "
           f"on this clip (see graph 03). {total_runs} attacking-run episodes were detected "
           f"in total across both directions, none of them during a TRAP_ACTIVE window.\n\n"
           f"This is a measured finding, not a detector failure: the thresholds are the same "
           f"simple, disclosed ones used throughout this project (see offside_v2_summary.md).")
    ax.text(0.02, 0.5, msg, fontsize=10, va="center", wrap=True)
    ax.set_title("Candidate break outcomes: no candidates on this clip", fontsize=11)
    _savefig(fig, "10_candidate_break_outcomes.png")

    print(f"\n{sum(data[n]['runs'].height for n in dirs)} total run episodes across both directions, "
          f"{len(os.listdir(OUT_DIR))} graphs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
