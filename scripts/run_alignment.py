#!/usr/bin/env python
from __future__ import annotations
import json, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from seqlib.aligner import (
    have_bowtie2, build_bt2_index, run_bowtie2,
    parse_sam_scores, summarize_alignment, count_per_reference, plot_histograms
)

# ---------- NEW: sweep utility ----------
def sweep_params(index_prefix: str,
                 reads_fa: str,
                 sam_dir: Path,
                 base_params: dict,
                 grid: list[dict],
                 mapq_min: int) -> pd.DataFrame:
    """
    Run multiple Bowtie2 parameter combinations and collect summary stats.

    Returns a DataFrame with columns:
      tag, N, L, score_min, pct_unique, pct_multi, pct_unaligned, mean_AS, mean_NM, n_aligned
    """
    rows = []
    sam_dir.mkdir(parents=True, exist_ok=True)
    for over in grid:
        p = {**base_params, **over}
        tag_parts = []
        if "N" in p: tag_parts.append(f"N{p['N']}")
        if "L" in p: tag_parts.append(f"L{p['L']}")
        if "score_min" in p: tag_parts.append(f"SM{p['score_min']}")
        tag = "_".join(tag_parts) if tag_parts else "base"

        sam_out = sam_dir / f"{Path(reads_fa).parent.name}__sweep_{tag}.sam"
        summary = run_bowtie2(index_prefix, reads_fa, str(sam_out), p)
        df = parse_sam_scores(str(sam_out), mapq_min=mapq_min)
        stats = summarize_alignment(df)

        rows.append({
            "tag": tag,
            "N": p.get("N", ""),
            "L": p.get("L", ""),
            "score_min": p.get("score_min", ""),
            "pct_unique": summary.get("pct_unique"),
            "pct_multi": summary.get("pct_multi"),
            "pct_unaligned": summary.get("pct_unaligned"),
            "mean_AS": stats.get("mean_AS"),
            "mean_NM": stats.get("mean_NM"),
            "n_aligned": stats.get("n_aligned")
        })
    return pd.DataFrame(rows)

def _parse_list(arg: str | None, cast):
    """Parse comma-separated list; return [] if arg is None/empty."""
    if not arg:
        return []
    return [cast(x.strip()) for x in arg.split(",") if x.strip()]

# ---------------------------------------

def main():
    print("Starting run_alignment.py...")
    ap = argparse.ArgumentParser(description="Run Bowtie2 on simulated reads and analyze SAM.")
    ap.add_argument("--ref-fasta", required=True, help="Reference FASTA (true library or backbone-derived).")
    ap.add_argument("--reads-fasta", required=True, help="Reads FASTA produced by simulator.")
    ap.add_argument("--params", default="configs/bowtie2_params.json", help="JSON file with bowtie2 params.")
    ap.add_argument("--outdir", default="data/alignments", help="Output root dir.")
    ap.add_argument("--tag", default="", help="Optional tag for single run filenames.")

    # ---------- NEW: sweep CLI ----------
    ap.add_argument("--sweep", action="store_true",
                    help="If set, sweep Bowtie2 parameters (see --sweep-N, --sweep-L, --sweep-score-min).")
    ap.add_argument("--sweep-N", default="", help="Comma list for -N (e.g. '0,1').")
    ap.add_argument("--sweep-L", default="", help="Comma list for -L (e.g. '20,28').")
    ap.add_argument("--sweep-score-min", default="", help="Comma list for --score-min values (e.g. 'L,-0.6,-0.6;L,-1.0,-0.5'). "
                                                          "Use semicolons to separate multiple complex values.")
    # ------------------------------------

    args = ap.parse_args()
    print(f"Arguments parsed: ref_fasta={args.ref_fasta}, reads_fasta={args.reads_fasta}, outdir={args.outdir}")
    
    if not have_bowtie2():
        raise SystemExit("ERROR: bowtie2/bowtie2-build not found on PATH. Install via conda or brew.")
    print("✅ Bowtie2 found")

    params = json.loads(Path(args.params).read_text())
    mapq_min = int(params.get("mapq_min", 0))
    print(f"Parameters loaded: {params}")

    ref_tag   = Path(args.ref_fasta).parent.name
    reads_tag = Path(args.reads_fasta).parent.name
    single_tag = args.tag or f"{reads_tag}"
    print(f"Tags: ref_tag={ref_tag}, reads_tag={reads_tag}, single_tag={single_tag}")

    # 1) build index
    print("Building Bowtie2 index...")
    root = Path(args.outdir) / ref_tag
    idx_dir = root / "bt2_index"
    index_prefix = build_bt2_index(args.ref_fasta, str(idx_dir))
    print(f"Index built at: {index_prefix}")

    # 2) single run (always perform once)
    print("Running Bowtie2 alignment...")
    sam_dir = root
    sam_out = sam_dir / f"{reads_tag}__{single_tag}.sam"
    summary = run_bowtie2(index_prefix, args.reads_fasta, str(sam_out), params)
    print(f"Alignment complete. Summary: {summary}")

    print("Parsing SAM scores...")
    df = parse_sam_scores(str(sam_out), mapq_min=mapq_min)
    stats = summarize_alignment(df)
    counts = count_per_reference(df)
    print(f"Stats: {stats}")

    res_dir = sam_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{single_tag}_align_summary.json").write_text(
        json.dumps({"bowtie2": summary, "stats": stats, "mapq_min": mapq_min}, indent=2)
    )
    counts.to_csv(res_dir / f"{single_tag}_counts_per_ref.csv", index=False)
    df.to_csv(res_dir / f"{single_tag}_per_read_scores.csv", index=False)
    plot_histograms(df, str(res_dir), single_tag)

    print(f"[single] SAM:      {sam_out}")
    print(f"[single] Summary:  {res_dir}/{single_tag}_align_summary.json")
    print(f"[single] Counts:   {res_dir}/{single_tag}_counts_per_ref.csv")
    print(f"[single] Per-read: {res_dir}/{single_tag}_per_read_scores.csv")

    # 3) optional sweep
    if args.sweep:
        # Build grid of overrides
        N_vals = _parse_list(args.sweep_N, int)
        L_vals = _parse_list(args.sweep_L, int)

        score_list_raw = args.sweep_score_min.strip()
        score_vals = []
        if score_list_raw:
            # allow semicolon-separated complex values, e.g. "L,-0.6,-0.6;L,-1.0,-0.5"
            score_vals = [s.strip() for s in score_list_raw.split(";") if s.strip()]

        grid = []
        # If user provided any of N/L/score_min, build Cartesian products appropriately.
        if N_vals or L_vals or score_vals:
            if not N_vals: N_vals = [params.get("N", 0)]
            if not L_vals: L_vals = [params.get("L", 20)]
            if not score_vals: score_vals = [params.get("score_min", "L,-0.6,-0.6")]
            for n in N_vals:
                for l in L_vals:
                    for sm in score_vals:
                        grid.append({"N": n, "L": l, "score_min": sm})
        else:
            # default tiny grid if --sweep used with no specifics
            grid = [{"N":0,"L":20}, {"N":1,"L":20}, {"N":0,"L":28}]

        df_sweep = sweep_params(index_prefix, args.reads_fasta, sam_dir, params, grid, mapq_min)
        df_sweep.to_csv(res_dir / "param_sweep.csv", index=False)
        print(f"[sweep] CSV: {res_dir}/param_sweep.csv")

        # quick comparison plot (unique vs multi)
        if not df_sweep.empty:
            fig, ax = plt.subplots(figsize=(7,3.5))
            x = range(len(df_sweep))
            ax.bar(x, df_sweep["pct_unique"], label="unique (%)")
            ax.bar(x, df_sweep["pct_multi"], bottom=df_sweep["pct_unique"], label="multi (%)")
            ax.set_xticks(x, df_sweep["tag"], rotation=45, ha="right")
            ax.set_ylabel("% of reads")
            ax.set_title("Bowtie2 mapping by parameter set")
            ax.legend()
            fig.tight_layout()
            fig.savefig(res_dir / "param_sweep_mapping.png", dpi=160)
            plt.close(fig)
            print(f"[sweep] Plot: {res_dir}/param_sweep_mapping.png")

    print("✅ Alignment analysis complete!")

if __name__ == "__main__":
    main()