"""
2026-09-08 V9 -- back-line MEMBERSHIP EVALUATION against human labels
(item I of the V9 request). Pure metric functions only -- NOTHING here
runs against real data yet, because `offside_manual_validation_v8.csv`'s
`human_backline_label` column is still empty by design (O5/V8): humans
have not labeled it. This module is ready the moment they do; the
accompanying tests exercise it on SYNTHETIC labels only, to prove the
math, not to declare a winner between V6/V7/V8-candidate.

Expected input: a list of row-dicts (e.g. from
`offside_manual_validation_v8.csv`, one row per (frame, team, track_id))
each carrying at least:
    frame, team, track_id, v7_confirmed (bool), human_backline_label
(a truthy/falsy label; blank/None rows are SKIPPED, never treated as
a negative label by default -- an unlabeled row is not evidence).
"""
import numpy as np

TRUTHY_LABELS = {'1', 'true', 'yes', 'y', 'member', 'backline', True, 1}
FALSY_LABELS = {'0', 'false', 'no', 'n', 'not_member', 'none', False, 0}


def _as_bool_label(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s == '':
            return None
        if s in TRUTHY_LABELS:
            return True
        if s in FALSY_LABELS:
            return False
        return None  # an unrecognized string is NOT silently coerced
    if isinstance(v, bool):
        return v
    return None


def _labeled_rows(rows, label_key='human_backline_label'):
    out = []
    for r in rows:
        lab = _as_bool_label(r.get(label_key))
        if lab is None:
            continue  # unlabeled -- excluded, never assumed negative
        out.append((r, lab))
    return out


def precision_recall_f1(rows, predicted_key, label_key='human_backline_label'):
    """`predicted_key`: the row field holding this method's own
    True/False membership call (e.g. 'v7_confirmed'). Only rows with a
    REAL human label are scored -- coverage (below) reports how many
    that actually was."""
    labeled = _labeled_rows(rows, label_key)
    tp = sum(1 for r, lab in labeled if r.get(predicted_key) and lab)
    fp = sum(1 for r, lab in labeled if r.get(predicted_key) and not lab)
    fn = sum(1 for r, lab in labeled if (not r.get(predicted_key)) and lab)
    tn = sum(1 for r, lab in labeled if (not r.get(predicted_key)) and not lab)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, n_labeled=len(labeled))


def per_team_f1(rows, predicted_key, label_key='human_backline_label'):
    teams = sorted({r['team'] for r in rows if 'team' in r})
    return {t: precision_recall_f1([r for r in rows if r.get('team') == t], predicted_key, label_key) for t in teams}


def coverage(rows, label_key='human_backline_label'):
    """Fraction of rows that actually carry a real human label."""
    total = len(rows)
    n_labeled = len(_labeled_rows(rows, label_key))
    return dict(n_total=total, n_labeled=n_labeled, coverage_fraction=(n_labeled / total) if total else None)


def membership_stability(frame_sequences, predicted_key):
    """`frame_sequences`: an ORDERED list of per-frame confirmed-id sets
    (e.g. [{1,2,3,4}, {1,2,3,4}, {1,2,3,5}, ...]) for one team/method.
    Returns one-frame membership CHURN -- the fraction of consecutive
    frame-pairs whose confirmed-id set changed at all -- and the mean
    Jaccard similarity between consecutive frames (1.0 == perfectly
    stable)."""
    if len(frame_sequences) < 2:
        return dict(churn_fraction=None, mean_jaccard=None, n_transitions=0)
    changed = 0
    jaccards = []
    for a, b in zip(frame_sequences, frame_sequences[1:]):
        if a != b:
            changed += 1
        union = a | b
        jaccards.append(len(a & b) / len(union) if union else 1.0)
    n = len(frame_sequences) - 1
    return dict(churn_fraction=changed / n, mean_jaccard=float(np.mean(jaccards)), n_transitions=n)


def line_centre_jump_stats(centre_x_sequence):
    """`centre_x_sequence`: an ORDERED list of per-frame unit-centre x
    values (None where invalid, skipped). Returns the distribution of
    FRAME-TO-FRAME jumps (cm) -- p50/p75/p90/max -- a stability
    diagnostic, not a pass/fail threshold."""
    vals = [v for v in centre_x_sequence if v is not None]
    jumps = [abs(b - a) for a, b in zip(vals, vals[1:])]
    if not jumps:
        return dict(p50=None, p75=None, p90=None, max=None, n=0)
    arr = np.array(jumps)
    return dict(p50=float(np.percentile(arr, 50)), p75=float(np.percentile(arr, 75)),
                p90=float(np.percentile(arr, 90)), max=float(arr.max()), n=len(jumps))


def compare_methods(rows, method_keys=('v6_member', 'v7_confirmed', 'v8_candidate_member'),
                     label_key='human_backline_label'):
    """Side-by-side precision/recall/F1/coverage for each method key
    present in `rows`. Returns a dict keyed by method name -- NO
    "winner" is declared here; that judgment is left to whoever reads
    the human-labeled numbers once they exist."""
    out = {}
    for key in method_keys:
        if not any(key in r for r in rows):
            continue
        out[key] = dict(**precision_recall_f1(rows, key, label_key), coverage=coverage(rows, label_key))
    return out
