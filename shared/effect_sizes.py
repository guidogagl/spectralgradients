"""Emit tables/effect_sizes.tex (Table 5 of the paper): signed Cliff's delta of SG vs the best STFT
rival per metric-domain, so the corrected-significance decisions (which hinge on
|delta| >= 0.147) are auditable.

Convention: effect = signed Cliff's delta with + FAVOURING SG regardless of metric
direction (for lower-is-better metrics we flip the sign). Marker */dagger reuses the
family-wide Holm + |delta|>=0.147 verdict against that same best rival.

Usage (repo root): PYTHONPATH=. python shared/effect_sizes.py
"""
import math
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from shared.generate_comparison_table import (
    DOMAINS, METRICS_STANDARD, METRICS_LOC, EXPLAINERS_STFT, TABLES,
    load_per_sample, compute_corrected_markers,
)


def med(vals):
    vals = [v for v in (vals or []) if v is not None]
    return float(np.median(vals)) if vals else float("nan")


def signed_delta(sg, rival, lower):
    sg = [v for v in (sg or []) if v is not None]
    rival = [v for v in (rival or []) if v is not None]
    if len(sg) < 2 or len(rival) < 2:
        return None
    U, _ = mannwhitneyu(sg, rival, alternative="two-sided")   # U for sg
    d = 2.0 * U / (len(sg) * len(rival)) - 1.0                # >0 => sg larger
    return (-d if lower else d)                               # + => SG better


def main():
    markers = compute_corrected_markers()
    all_metrics = METRICS_STANDARD + METRICS_LOC
    lines = []
    lines.append("\\begin{table}[!t]")
    lines.append("\\centering")
    lines.append("\\caption{Effect sizes for the decision-relevant comparison: signed "
                 "Cliff's $\\delta$ of SG against the best-median STFT rival on each "
                 "metric (positive favours SG; $|\\delta|\\!<\\!0.147$ is negligible). "
                 "{*}\\,/\\,$^\\dag$ mark SG significantly better\\,/\\,worse under the "
                 "corrected standard (Holm-Bonferroni + $|\\delta|\\geq0.147$). "
                 "Localization applies to synthetic domains only. Three decimals are shown because two cells lie within $0.005$ of the threshold.}")
    lines.append("\\label{tab:effect}")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{@{}l" + "c" * len(all_metrics) + "@{}}")
    lines.append("\\toprule")
    header = "\\textbf{Domain}"
    for _, mlabel, lower in all_metrics:
        header += f" & \\textbf{{{mlabel}}}"
    lines.append(header + " \\\\")
    lines.append("\\midrule")

    for d in DOMAINS:
        label = d["label"]
        sg = load_per_sample(d["path"], "SG")
        metrics = list(METRICS_STANDARD) + (list(METRICS_LOC) if d["has_localization"] else [])
        cells = [label]
        for mkey, mlabel, lower in all_metrics:
            if sg is None or (mkey in (m[0] for m in METRICS_LOC) and not d["has_localization"]):
                cells.append("---"); continue
            # best-median rival among the 9 STFT configs
            best_g, best_med = None, None
            for m, w in EXPLAINERS_STFT:
                dd = load_per_sample(d["path"], f"{m}-{w}")
                if dd is None:
                    continue
                mm = med(dd.get(mkey, []))
                if math.isnan(mm):
                    continue
                if best_med is None or (mm < best_med if lower else mm > best_med):
                    best_med, best_g = mm, f"{m}-{w}"
            if best_g is None:
                cells.append("---"); continue
            rival = load_per_sample(d["path"], best_g)
            eff = signed_delta(sg.get(mkey, []), rival.get(mkey, []), lower)
            if eff is None:
                cells.append("---"); continue
            verdict = markers.get((label, best_g, mkey), "")
            sup = "*" if verdict == "better" else ("\\dag" if verdict == "worse" else "")
            s = f"{eff:+.3f}"
            if sup:
                s += f"\\textsuperscript{{{sup}}}"
            cells.append(s)
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out = TABLES / "effect_sizes.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
