#!/usr/bin/env python3
"""Generate the comparison tables from the per-sample result JSONs: SG vs all STFT configs.

Reads  $SG_RESULTS/<model>/*_results.json   (default: results/<model>/)
Writes $SG_TABLES/comparison.csv, comparison.tex (Table 3) and localisation.tex (Table 4)
       (default: tables/)

Significance markers (SG vs each STFT configuration, per metric and domain):
  *   SG significantly better      dag  SG significantly worse
  (Holm-corrected two-sided Mann-Whitney U with |Cliff's delta| >= 0.147)
  Bold: best median per metric and domain.

Usage (repo root)::
    PYTHONPATH=. python shared/generate_comparison_table.py
"""

import csv
import json
import os
import math
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS = Path(os.environ.get("SG_RESULTS", "results"))
TABLES = Path(os.environ.get("SG_TABLES", "tables"))

DOMAINS = [
    {
        "label": "Synt-0",
        "model": "Conv1D",
        "path": str(RESULTS / "synt-setup0"),
        "has_localization": True,
    },
    {
        "label": "Synt-1",
        "model": "Conv1D",
        "path": str(RESULTS / "synt-setup1"),
        "has_localization": True,
    },
    {
        "label": "Synt-2",
        "model": "Conv1D",
        "path": str(RESULTS / "synt-setup2"),
        "has_localization": True,
    },
    {
        "label": "ECG",
        "model": "Conv1D",
        "path": str(RESULTS / "arrhythmia-cnn"),
        "has_localization": False,
    },
    {
        "label": "Speech",
        "model": "Conv1D",
        "path": str(RESULTS / "audio-wavcnn"),
        "has_localization": False,
    },
    {
        "label": "Sleep-MASS",
        "model": "Chambon2018",
        "path": str(RESULTS / "chambon2018"),
        "has_localization": False,
    },
    {
        "label": "Sleep-EDF",
        "model": "TsinalisCNN",
        "path": str(RESULTS / "tsinalis-2016"),
        "has_localization": False,
    },
]

METRICS_STANDARD = [
    ("freq_infidelity", "Freq.Inf", True),    # (key, label, lower_is_better)
    ("freq_complexity", "Freq.Cmplx", True),
    ("time_infidelity", "Time.Inf", True),
    ("time_complexity", "Time.Cmplx", True),
]

METRICS_LOC = [
    ("freq_localization", "Freq.Loc", False),   # higher is better
    ("time_localization", "Time.Loc", False),
]

EXPLAINERS_STFT = [
    ("IG", "time"), ("IG", "bal"), ("IG", "freq"),
    ("Sal", "time"), ("Sal", "bal"), ("Sal", "freq"),
    ("IxG", "time"), ("IxG", "bal"), ("IxG", "freq"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_CACHE = {}


def load_per_sample(path, explainer_name):
    """Load per-sample metric arrays from result JSON (memoised)."""
    ck = (str(path), explainer_name)
    if ck in _CACHE:
        return _CACHE[ck]
    result = _load_per_sample(path, explainer_name)
    _CACHE[ck] = result
    return result


def _load_per_sample(path, explainer_name):
    base = Path(path)
    all_metrics = METRICS_STANDARD + METRICS_LOC
    metric_keys = [m[0] for m in all_metrics]

    # Try standard filenames. Prefer the full-run "<name>_results.json" (synt/audio/arrhythmia
    # naming) over "<name>.json": some result dirs also hold early n=200 smoke files named
    # "<name>.json", which must never feed the tables. Sleep uses "<name>.json" only.
    for pattern in [f"{explainer_name}_results.json", f"{explainer_name}.json"]:
        fpath = base / pattern
        if fpath.exists():
            d = json.load(open(fpath))
            m = d.get("metrics", d.get("results", {}))
            if isinstance(m, dict) and "freq_infidelity" in m:
                return {k: [v for v in m.get(k, []) if v is not None]
                        for k in metric_keys}

    # SG: try SG-shapley-*.json or SG_results.json
    if explainer_name == "SG":
        for f in sorted(base.glob("SG*.json")):
            if "comparison" in f.name or "summary" in f.name or "_chunk" in f.name:
                continue
            d = json.load(open(f))
            m = d.get("metrics", d.get("results", {}))
            if isinstance(m, dict) and "freq_infidelity" in m:
                return {k: [v for v in m.get(k, []) if v is not None]
                        for k in metric_keys}
    return None


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

# A difference counts
# as significant only if it survives Holm-Bonferroni correction across the WHOLE
# family of SG-vs-STFT tests AND has a non-negligible effect size (|Cliff's delta|
# >= DELTA_MIN). This avoids both multiple-comparison false positives and the
# huge-n over-powering artefact of Mann-Whitney U on full test sets.
DELTA_MIN = 0.147   # Cliff's delta: <0.147 = negligible (Romano et al.)


def _cliff_test(sg, st, lower_is_better):
    """Return (p_raw, sg_worse_bool, |cliff_delta|) or None if insufficient data.

    Uses the FULL arrays (unpaired Mann-Whitney U); Cliff's delta is derived
    cheaply from U as delta = 2U/(n1 n2) - 1.
    """
    sg = [v for v in (sg or []) if v is not None]
    st = [v for v in (st or []) if v is not None]
    if len(sg) < 2 or len(st) < 2:
        return None
    try:
        U, p = mannwhitneyu(sg, st, alternative="two-sided")   # U for sg
    except Exception:
        return None
    delta = 2.0 * U / (len(sg) * len(st)) - 1.0                # Cliff's delta, sg vs st
    worse = (delta > 0) if lower_is_better else (delta < 0)    # SG worse than STFT
    return p, worse, abs(delta)


def _holm(pvals):
    """Holm-Bonferroni: boolean reject[] at FWER 0.05, aligned to input order."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= 0.05 / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def compute_corrected_markers():
    """Pre-pass: family-wide Holm + Cliff's-delta significance for every
    SG-vs-STFT (domain, explainer, metric) test.

    Returns dict keyed by (domain_label, "METHOD-window", metric_key) -> verdict
    string in {"better", "worse", ""}.
    """
    tests = []  # (key, p, worse, |delta|, lower)
    for domain in DOMAINS:
        label = domain["label"]
        sg = load_per_sample(domain["path"], "SG")
        if sg is None:
            continue
        metrics = list(METRICS_STANDARD)
        if domain["has_localization"]:
            metrics += list(METRICS_LOC)
        for method, window in EXPLAINERS_STFT:
            ename = f"{method}-{window}"
            st = load_per_sample(domain["path"], ename)
            if st is None:
                continue
            for mkey, _, lower in metrics:
                r = _cliff_test(sg.get(mkey, []), st.get(mkey, []), lower)
                if r is None:
                    continue
                p, worse, adelta = r
                tests.append([(label, ename, mkey), p, worse, adelta])

    reject = _holm([t[1] for t in tests])
    markers = {}
    for t, rej in zip(tests, reject):
        key, p, worse, adelta = t
        if rej and adelta >= DELTA_MIN:
            markers[key] = "worse" if worse else "better"
        else:
            markers[key] = ""
    return markers


def render_marker(verdict):
    """Map a verdict string to the table marker for the current output mode."""
    if verdict == "better":
        return "*"
    if verdict == "worse":
        return "\\dag" if tex_mode else "†"
    return ""


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

tex_mode = False  # toggled when generating tex


def build_rows():
    """Build all rows of the comparison table.

    Returns list of dicts with keys:
      domain, model, explainer, window, + one key per metric
      each metric value is (median, sig_marker, is_best)
    """
    rows = []

    corrected = compute_corrected_markers()

    for domain in DOMAINS:
        label = domain["label"]
        model = domain["model"]
        dpath = domain["path"]
        has_loc = domain["has_localization"]

        metrics = list(METRICS_STANDARD)
        if has_loc:
            metrics += list(METRICS_LOC)

        # Load SG. If SG results are missing (e.g. still running), keep the
        # domain but emit "-" for the SG row and drop SG-vs-STFT significance.
        sg_data = load_per_sample(dpath, "SG")
        sg_missing = sg_data is None
        if sg_missing:
            print(f"  NOTE: SG not found for {label} -> SG row = '-'")
            sg_data = {}

        # Collect all medians to find best per metric
        all_medians = {}  # metric_key -> {explainer_label: median}

        sg_medians = {}
        for mkey, mlabel, lower in metrics:
            vals = sg_data.get(mkey, [])
            sg_medians[mkey] = np.median(vals) if vals else float("nan")
            all_medians.setdefault(mkey, {})["SG"] = sg_medians[mkey]

        stft_data_cache = {}
        for method, window in EXPLAINERS_STFT:
            ename = f"{method}-{window}"
            data = load_per_sample(dpath, ename)
            stft_data_cache[(method, window)] = data
            if data is None:
                continue
            for mkey, mlabel, lower in metrics:
                vals = data.get(mkey, [])
                med = np.median(vals) if vals else float("nan")
                all_medians.setdefault(mkey, {})[ename] = med

        # Find best per metric
        best_per_metric = {}
        for mkey, mlabel, lower in metrics:
            meds = all_medians.get(mkey, {})
            if not meds:
                continue
            valid = {k: v for k, v in meds.items() if not math.isnan(v)}
            if not valid:
                continue
            if lower:
                best_per_metric[mkey] = min(valid, key=valid.get)
            else:
                best_per_metric[mkey] = max(valid, key=valid.get)

        # SG row
        sg_row = {
            "domain": label,
            "model": model,
            "explainer": "SG",
            "window": "---",
        }
        for mkey, mlabel, lower in metrics:
            is_best = best_per_metric.get(mkey) == "SG"
            sg_row[mkey] = (sg_medians[mkey], "", is_best)

        # Fill missing loc metrics with None
        if not has_loc:
            for mkey, _, _ in METRICS_LOC:
                sg_row[mkey] = (None, "", False)

        rows.append(sg_row)

        # STFT rows
        for method, window in EXPLAINERS_STFT:
            ename = f"{method}-{window}"
            data = stft_data_cache[(method, window)]

            row = {
                "domain": "",  # empty for continuation
                "model": "",
                "explainer": method,
                "window": window,
            }

            for mkey, mlabel, lower in metrics:
                if data is None:
                    row[mkey] = (None, "", False)
                    continue
                vals = data.get(mkey, [])
                med = np.median(vals) if vals else float("nan")
                sig = render_marker(corrected.get((label, ename, mkey), ""))
                is_best = best_per_metric.get(mkey) == ename
                row[mkey] = (med, sig, is_best)

            if not has_loc:
                for mkey, _, _ in METRICS_LOC:
                    row[mkey] = (None, "", False)

            rows.append(row)

    return rows


def write_csv(rows, outpath):
    """Write comparison table as CSV."""
    global tex_mode
    tex_mode = False

    all_metrics = METRICS_STANDARD + METRICS_LOC
    fieldnames = ["Domain", "Model", "Explainer", "Window"]
    fieldnames += [m[1] for m in all_metrics]

    outpath.parent.mkdir(parents=True, exist_ok=True)

    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            csv_row = {
                "Domain": row["domain"],
                "Model": row["model"],
                "Explainer": row["explainer"],
                "Window": row["window"],
            }
            for mkey, mlabel, _ in all_metrics:
                med, sig, is_best = row[mkey]
                if med is None or (isinstance(med, float) and math.isnan(med)):
                    csv_row[mlabel] = "---"
                else:
                    val = f"{med:.4f}"
                    if sig:
                        val += sig
                    if is_best:
                        val = f"[{val}]"
                    csv_row[mlabel] = val
            writer.writerow(csv_row)

    print(f"Saved: {outpath}")


def _cell(med, sig, is_best, fmt="{:.4f}"):
    if med is None or (isinstance(med, float) and math.isnan(med)):
        return "---"
    val = fmt.format(med)
    if sig:
        val += f"\\textsuperscript{{{sig}}}"
    if is_best:
        val = f"\\textbf{{{val}}}"
    return val


COMPARISON_CAPTION = ("Comparison of SG against the nine STFT-based configurations (per-sample medians). "
                      "{*}\\,SG significantly better, $^\\dag$\\,SG significantly worse (Holm-corrected two-sided "
                      "Mann-Whitney U with $|\\text{Cliff's }\\delta|\\geq0.147$, Section~\\ref{sec:metrics}). "
                      "\\textbf{Bold}: best median per metric and domain. Localisation on the synthetic setups is "
                      "reported in Table~\\ref{tab:localisation}.")
LOCALISATION_CAPTION = ("Localisation on the three synthetic setups, where ground-truth time-frequency masks are "
                        "available (per-sample medians, higher is better). Markers and bold as in Table~\\ref{tab:comparison}.")


def write_tex(rows, outpath):
    """Write the comparison table (four standard metrics) as LaTeX: Table 3 of the paper."""
    global tex_mode
    tex_mode = True
    rows = build_rows()   # rebuild with tex significance markers
    outpath.parent.mkdir(parents=True, exist_ok=True)

    lines = ["\\begin{landscape}", "\\begin{table}[p]", "\\centering",
             "\\caption{" + COMPARISON_CAPTION + "}", "\\label{tab:comparison}",
             "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{@{}llll" + "c" * len(METRICS_STANDARD) + "@{}}", "\\toprule"]
    header = "\\textbf{Domain} & \\textbf{Model} & \\textbf{Expl.} & \\textbf{Win.}"
    for _, mlabel, lower in METRICS_STANDARD:
        header += f" & \\textbf{{{mlabel}}} " + ("$\\downarrow$" if lower else "$\\uparrow$")
    lines.append(header + " \\\\")
    lines.append("\\midrule")
    prev_domain = None
    for row in rows:
        domain = row["domain"]
        if domain and prev_domain and domain != prev_domain:
            lines.append("\\midrule")
        if domain:
            prev_domain = domain
        cells = [domain, row["model"] if domain else "", row["explainer"], row["window"]]
        cells += [_cell(*row[mkey]) for mkey, _, _ in METRICS_STANDARD]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", "\\end{landscape}"]
    outpath.write_text("\n".join(lines) + "\n")
    print(f"Saved: {outpath}")


def write_localisation_tex(rows, outpath):
    """Write the localisation table (synthetic setups only) as LaTeX: Table 4 of the paper."""
    global tex_mode
    tex_mode = True
    rows = build_rows()
    has_loc = {d["label"]: d["has_localization"] for d in DOMAINS}
    outpath.parent.mkdir(parents=True, exist_ok=True)

    lines = ["\\begin{table}[t]", "\\centering", "\\caption{" + LOCALISATION_CAPTION + "}",
             "\\label{tab:localisation}", "\\begin{tabular}{@{}lllcc@{}}", "\\toprule",
             "\\textbf{Domain} & \\textbf{Expl.} & \\textbf{Win.} & \\textbf{Freq.Loc} $\\uparrow$ & \\textbf{Time.Loc} $\\uparrow$ \\\\",
             "\\midrule"]
    current, prev_domain = None, None
    for row in rows:
        if row["domain"]:
            current = row["domain"]
        if not has_loc.get(current, False):
            continue
        if row["domain"] and prev_domain and row["domain"] != prev_domain:
            lines.append("\\midrule")
        if row["domain"]:
            prev_domain = row["domain"]
        cells = [row["domain"], row["explainer"], row["window"]] + [_cell(*row[mkey]) for mkey, _, _ in METRICS_LOC]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    outpath.write_text("\n".join(lines) + "\n")
    print(f"Saved: {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global tex_mode

    print("Building comparison table...")
    tex_mode = False
    rows = build_rows()

    out_dir = TABLES
    write_csv(rows, out_dir / "comparison.csv")
    write_tex(rows, out_dir / "comparison.tex")
    write_localisation_tex(rows, out_dir / "localisation.tex")

    # Print summary stats: count ALL applicable metric-domain pairs (localization
    # counts only for synthetic domains, matching the paper's metric-domain pairs).
    dom_has_loc = {d["label"]: d["has_localization"] for d in DOMAINS}
    total_metrics = 0
    sg_wins = 0
    for row in rows:
        if row["explainer"] == "SG":
            metrics = list(METRICS_STANDARD)
            if dom_has_loc.get(row["domain"], False):
                metrics += list(METRICS_LOC)
            for mkey, _, _ in metrics:
                med, _, is_best = row[mkey]
                if med is None or (isinstance(med, float) and math.isnan(med)):
                    continue
                total_metrics += 1
                if is_best:
                    sg_wins += 1

    print(f"\nSG is best median on {sg_wins}/{total_metrics} metric-domain pairs")


if __name__ == "__main__":
    main()
