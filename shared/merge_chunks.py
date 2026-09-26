"""Merge per-chunk benchmark result JSONs into a single full-run JSON.

Chunked jobs write `<stem>_chunk{i}of{N}.json`, each containing the per-sample
metric arrays for its strided partition of the shared sample set. This script
concatenates those arrays back into `<stem>.json` with the same schema as a
non-chunked run (order across samples is irrelevant for medians / Mann-Whitney).

Usage (repo root):
    python shared/merge_chunks.py results/tsinalis-2016/SG-shapley-fs0.5-s10-p20 \
                                  results/tsinalis-2016/IG-bal
Each argument is the output stem WITHOUT the `.json` extension (``run.sh merge`` does this
for every chunked stem it finds under ``$SG_RESULTS``).
"""

import argparse
import glob
import json
import re
from pathlib import Path


def merge_one(base: str, keep_partial: bool = False):
    base = Path(base)
    d, stem = base.parent, base.name
    paths = glob.glob(str(d / f"{stem}_chunk*of*.json"))
    if not paths:
        print(f"[skip] no chunks found for {base}")
        return
    def idx(p):
        return int(re.search(r"_chunk(\d+)of\d+\.json$", p).group(1))
    n = int(re.search(r"_chunk\d+of(\d+)\.json$", paths[0]).group(1))
    paths = sorted(paths, key=idx)
    got = [idx(p) for p in paths]
    missing = [i for i in range(n) if i not in got]
    if missing:
        print(f"[WARN] {stem}: missing chunks {missing} of {n} — refusing to merge "
              f"(re-run those chunks first).")
        return

    # Per-sample metric arrays live under "metrics" (sleep) or "results"
    # (synt/audio/arrhythmia). Detect and preserve the same key on output.
    merged, meta, total, tsum, mkey = None, {}, 0, 0.0, None
    for p in paths:
        j = json.load(open(p))
        if mkey is None:
            mkey = "metrics" if "metrics" in j else "results"
        m = j[mkey]
        if merged is None:
            merged = {k: list(v) for k, v in m.items()}
            meta = {k: j[k] for k in ("model", "explainer", "config") if k in j}
        else:
            for k, v in m.items():
                merged[k].extend(v)
        total += int(j.get("n_samples", len(next(iter(m.values())))))
        tsum += float(j.get("time_seconds", j.get("elapsed_sec", 0.0)))

    out = {**meta, "n_samples": total, mkey: merged,
           "time_seconds": tsum, "merged_from_chunks": n}
    outp = d / f"{stem}.json"
    json.dump(out, open(outp, "w"))
    print(f"[ok] merged {n} chunks -> {outp}  (n_samples={total})")
    if not keep_partial:
        for p in paths:
            Path(p).unlink()
        print(f"     removed {n} partial chunk files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bases", nargs="+", help="output stems without .json")
    ap.add_argument("--keep-partial", action="store_true",
                    help="do not delete per-chunk files after merging")
    a = ap.parse_args()
    for b in a.bases:
        merge_one(b, keep_partial=a.keep_partial)
