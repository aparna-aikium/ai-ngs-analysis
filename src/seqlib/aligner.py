from __future__ import annotations
import re, json, subprocess, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAM_FLAG_SECONDARY = 0x100
SAM_FLAG_SUPP      = 0x800

def have_bowtie2() -> bool:
    return shutil.which("bowtie2") is not None and shutil.which("bowtie2-build") is not None

def build_bt2_index(ref_fa: str, out_dir: str, prefix: str="bt2_index") -> str:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    idx = out / prefix
    if not (out / f"{prefix}.1.bt2").exists() and not (out / f"{prefix}.1.bt2l").exists():
        cmd = ["bowtie2-build", ref_fa, str(idx)]
        subprocess.run(cmd, check=True)
    return str(idx)

def _bt2_cmd(index_prefix: str, reads_fa: str, sam_out: str, params: dict) -> list[str]:
    # Base
    cmd = ["bowtie2", "-x", index_prefix, "-f", "-U", reads_fa, "-S", sam_out]
    # Preset & knobs
    if params.get("preset"):
        cmd += params["preset"].split()
    if "N" in params: cmd += ["-N", str(params["N"])]
    if "L" in params: cmd += ["-L", str(params["L"])]
    if "i" in params: cmd += ["-i", str(params["i"])]
    if "score_min" in params: cmd += ["--score-min", str(params["score_min"])]
    return cmd

def run_bowtie2(index_prefix: str, reads_fa: str, sam_out: str, params: dict) -> dict:
    Path(sam_out).parent.mkdir(parents=True, exist_ok=True)
    cmd = _bt2_cmd(index_prefix, reads_fa, sam_out, params)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode not in (0, 1):  # 1 can happen when few/none map
        raise RuntimeError(f"bowtie2 failed ({p.returncode}):\n{p.stderr}")
    # Parse the bowtie2 stderr summary
    # Example lines: "10000 reads; of these: ...", "90.00% aligned 0 times", "10.00% aligned exactly 1 time"
    summary = {"stderr": p.stderr, "returncode": p.returncode, "cmd": cmd}
    m = re.search(r"(\d+)\s+reads;", p.stderr)
    if m: summary["reads_total"] = int(m.group(1))
    m = re.search(r"(\d+\.\d+)%\s+aligned 0 times", p.stderr)
    if m: summary["pct_unaligned"] = float(m.group(1))
    m = re.search(r"(\d+\.\d+)%\s+aligned exactly 1 time", p.stderr)
    if m: summary["pct_unique"] = float(m.group(1))
    m = re.search(r"(\d+\.\d+)%\s+aligned >1 times", p.stderr)
    if m: summary["pct_multi"] = float(m.group(1))
    return summary

def parse_sam_scores(sam_path: str, mapq_min: int = 0) -> pd.DataFrame:
    """
    Returns per-read rows with columns:
    read_id, rname (reference id), mapq, AS (alignment score), NM (edit distance), primary (bool).
    Only primary alignments (no 0x100/0x800) are returned.
    """
    rows = []
    with open(sam_path) as fh:
        for line in fh:
            if line.startswith("@"): 
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11: 
                continue
            qname, flag, rname, mapq = parts[0], int(parts[1]), parts[2], int(parts[4])
            if rname == "*" or (flag & SAM_FLAG_SECONDARY) or (flag & SAM_FLAG_SUPP):
                continue
            if mapq < mapq_min:
                continue
            opt = parts[11:]
            AS, NM = None, None
            for f in opt:
                if f.startswith("AS:i:"):
                    AS = int(f.split(":")[-1])
                elif f.startswith("NM:i:"):
                    NM = int(f.split(":")[-1])
            rows.append((qname, rname, mapq, AS, NM))
    df = pd.DataFrame(rows, columns=["read_id","rname","mapq","AS","NM"])
    return df

def summarize_alignment(df: pd.DataFrame) -> dict:
    out = {}
    if df.empty:
        out["n_aligned"] = 0
        return out
    out["n_aligned"] = len(df)
    out["mean_AS"]    = float(df["AS"].dropna().mean()) if "AS" in df else np.nan
    out["mean_NM"]    = float(df["NM"].dropna().mean()) if "NM" in df else np.nan
    out["mapq>=10"]   = int((df["mapq"] >= 10).sum())
    out["unique_refs"] = df["rname"].nunique()
    return out

def count_per_reference(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["rname","reads","frequency"])
    counts = (df.groupby("rname")
                .size()
                .rename("reads")
                .reset_index())
    counts["frequency"] = counts["reads"] / counts["reads"].sum()
    return counts

def plot_histograms(df: pd.DataFrame, out_dir: str, tag: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    # AS
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(df["AS"].dropna(), bins=40)
    ax.set_title(f"Alignment score (AS) — {tag}")
    ax.set_xlabel("AS"); ax.set_ylabel("Reads")
    fig.savefig(Path(out_dir)/f"{tag}_hist_AS.png", bbox_inches="tight")
    plt.close(fig)
    # NM
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(df["NM"].dropna(), bins=40)
    ax.set_title(f"Edit distance (NM) — {tag}")
    ax.set_xlabel("NM"); ax.set_ylabel("Reads")
    fig.savefig(Path(out_dir)/f"{tag}_hist_NM.png", bbox_inches="tight")
    plt.close(fig)