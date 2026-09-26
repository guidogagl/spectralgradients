"""Δf-sensitivity test for Spectral Gradients (paper §5.5).

Paired comparison, on identical samples, of SG run with the benchmark band width against SG run
with a coarser band width (Sleep-MASS/Chambon2018: 0.5 -> 2 Hz on a strided 1/8 chunk;
Synt-1/2: 2 -> 5 Hz on all 5,000 samples), plus the best time-optimised STFT rival on the same
samples. Hypothesis: SG's time complexity (normalised entropy of the time marginal) drops when
1/Δf becomes small relative to the event, because each band component is spread over ~1/Δf.

Usage (repo root): PYTHONPATH=. python shared/dftest_compare.py [--coarse-root results/dftest]
Writes tables/dftest.json and tables/dftest.tex (Table 6 of the paper).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

METS = ["freq_infidelity", "freq_complexity", "time_infidelity", "time_complexity"]
SHORT = {"freq_infidelity": "Freq.Inf", "freq_complexity": "Freq.Cmplx",
         "time_infidelity": "Time.Inf", "time_complexity": "Time.Cmplx"}


def per_sample(path):
    d = json.load(open(path))
    m = d.get("metrics", d.get("results"))
    return {k: np.array(m[k], dtype=float) for k in METS}, d.get("n_samples")


def cliff(a, b):
    """Signed Cliff's delta, negative when a tends to be smaller than b."""
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2.0 * u / (len(a) * len(b)) - 1.0


def paired_block(label, coarse, fine, rival, rival_name, idx=None):
    """coarse/fine/rival: dicts metric->array; idx selects the paired subset of fine/rival."""
    out = {"case": label, "n": int(len(coarse[METS[0]])), "rival": rival_name, "metrics": {}}
    for m in METS:
        f = fine[m] if idx is None else fine[m][idx]
        r = rival[m] if idx is None else rival[m][idx]
        c = coarse[m]
        assert len(f) == len(c) == len(r), (m, len(f), len(c), len(r))
        out["metrics"][m] = {
            "median_fine": float(np.median(f)), "median_coarse": float(np.median(c)),
            "median_rival": float(np.median(r)),
            "delta_coarse_vs_fine": cliff(c, f),      # <0: coarser Δf gives lower values
            "delta_coarse_vs_rival": cliff(c, r),
            "delta_fine_vs_rival": cliff(f, r),
            "p_coarse_vs_fine": float(mannwhitneyu(c, f).pvalue),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.environ.get("SG_RESULTS", "results"), help="results root")
    ap.add_argument("--coarse-root", default=None, help="root of the coarse-band synthetic runs (default: <results>/dftest)")
    ap.add_argument("--tables", default=os.environ.get("SG_TABLES", "tables"), help="output directory")
    args = ap.parse_args()
    R = Path(args.results)
    coarse_root = Path(args.coarse_root) if args.coarse_root else R / "dftest"
    blocks = []

    # ---- Sleep-MASS (Chambon2018): fine 0.5 Hz (full run) vs coarse 2 Hz (chunk 0 of 8) ----
    d = R / "chambon2018"
    fine, n_fine = per_sample(d / "SG-shapley-fs0.5-s10-p20.json")
    coarse_path = d / "SG-shapley-fs2.0-s10-p20_chunk0of8.json"
    if coarse_path.exists():
        coarse, n_c = per_sample(coarse_path)
        idx = np.arange(n_fine)[0::8]
        assert len(idx) == n_c, (len(idx), n_c)
        rival, _ = per_sample(d / "IxG-time.json")
        blocks.append(paired_block("Sleep-MASS", coarse, fine, rival, "IxG-time", idx))
    else:
        print("[skip] MASS coarse run not found:", coarse_path)

    # ---- Synt-1/2: fine 2 Hz (benchmark run) vs coarse 5 Hz (configs/synt/synt-setup{1,2}_sg_df5.yaml) ----
    for s in (1, 2):
        fine, n = per_sample(R / f"synt-setup{s}/SG_results.json")
        cp = coarse_root / f"synt-setup{s}/SG_results.json"
        if not cp.exists():
            print("[skip] synt coarse run not found:", cp); continue
        coarse, nc = per_sample(cp)
        assert nc == n, (nc, n)
        rival, _ = per_sample(R / f"synt-setup{s}/IG-time_results.json")
        blocks.append(paired_block(f"Synt-{s}", coarse, fine, rival, "IG-time"))

    out = Path(args.tables)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dftest.json").write_text(json.dumps(blocks, indent=1))

    # ---- summary ----
    print("\n=== band-width test: paired medians, fine vs coarse ===")
    for b in blocks:
        tc = b["metrics"]["time_complexity"]
        print(f"{b['case']:12s} n={b['n']:5d}  Time.Cmplx fine {tc['median_fine']:.4f} -> coarse {tc['median_coarse']:.4f} "
              f"(delta={tc['delta_coarse_vs_fine']:+.3f}, p={tc['p_coarse_vs_fine']:.1e}); rival {b['rival']} {tc['median_rival']:.4f}")
        for key, name in (("freq_complexity", "Freq.Cmplx"), ("freq_infidelity", "Freq.Inf"), ("time_infidelity", "Time.Inf")):
            m = b["metrics"][key]
            print(f"{'':12s} {name:10s} {m['median_fine']:.3f} -> {m['median_coarse']:.3f} (delta={m['delta_coarse_vs_fine']:+.2f})")

    # ---- LaTeX table ----
    lines = [r"\begin{table}[t]", r"\centering", r"\setlength{\tabcolsep}{2pt}",
             r"\caption{Sensitivity of SG to the band width $\Delta f$, on identical samples (paired). "
             r"``Fine'' is the benchmark setting of Table~\ref{tab:configs}, ``coarse'' a wider band ($0.5 \to 2$~Hz on Sleep-MASS, $2 \to 5$~Hz on Synt-1/2); "
             r"$\delta$ is Cliff's delta of coarse vs.\ fine (negative: lower with the coarser band); the last "
             r"column is the best time-optimised STFT rival on the same samples. Sleep-MASS uses a strided "
             r"$1/8$ subset of the benchmark sample set.}",
             r"\label{tab:dftest}",
             r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"\textbf{Case} & \textbf{Metric} & \textbf{SG fine} & \textbf{SG coarse} & $\delta$ & \textbf{Rival} & \textbf{Rival value} \\",
             r"\midrule"]
    for b in blocks:
        first = True
        for m in METS:
            x = b["metrics"][m]
            case = b["case"].replace("->", r"$\rightarrow$") if first else ""
            mark = r"\textsuperscript{*}" if abs(x["delta_coarse_vs_fine"]) >= 0.147 and x["p_coarse_vs_fine"] < 0.05 / (4 * len(blocks)) else ""
            lines.append(f"{case} & {SHORT[m]} & {x['median_fine']:.4f} & {x['median_coarse']:.4f}{mark} & "
                         f"{x['delta_coarse_vs_fine']:+.2f} & {b['rival'] if first else ''} & {x['median_rival']:.4f} \\\\")
            first = False
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    (out / "dftest.tex").write_text("\n".join(lines) + "\n")
    print(f"[saved] {out / 'dftest.json'}, {out / 'dftest.tex'}")


if __name__ == "__main__":
    main()
