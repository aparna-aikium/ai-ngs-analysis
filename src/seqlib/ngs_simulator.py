from pathlib import Path
import json, random
from collections import Counter
from .utils import timestamp_tag

DNA4 = list("ACGT")
AA20 = list("ARNDCEQGHILKMFPSTWYV")

def _read_library_csv(csv_path: Path):
    rows = []
    with open(csv_path) as fh:
        next(fh)  # header
        for line in fh:
            rid, seq = line.strip().split(",", 1)
            rows.append((rid, seq))
    return rows

def _flip(ch: str, alphabet: list[str], rng: random.Random) -> str:
    choices = [a for a in alphabet if a != ch]
    return rng.choice(choices)

def _corrupt(seq: str, p: float, alpha: list[str], rng: random.Random):
    errs = []
    out = []
    for i, c in enumerate(seq):
        if rng.random() < p:
            nc = _flip(c, alpha, rng)
            out.append(nc); errs.append((i+1, f"{c}>{nc}"))
        else:
            out.append(c)
    return "".join(out), errs

def simulate_reads(
    library_dir: str,
    reads_total: int,
    p_error: float,
    seed_sim: int,
    abundance_mode: str = "uniform",
    lognormal_sigma: float = 1.0
) -> str:
    libdir = Path(library_dir)
    csv_path = next(p for p in libdir.iterdir() if p.name.endswith("_true_library.csv"))
    manifest_path = next(p for p in libdir.iterdir() if p.name.endswith("_manifest.json"))

    manifest = json.loads(Path(manifest_path).read_text())
    seq_type = manifest["seq_type"]
    records = _read_library_csv(csv_path)
    L = len(records[0][1])
    alpha = DNA4 if seq_type == "dna" else AA20

    rng = random.Random(seed_sim)

    # weights
    if abundance_mode == "uniform":
        weights = [1.0] * len(records)
    else:
        # simple lognormal via CLT-ish z
        import math
        vals = []
        for _ in records:
            z = sum(rng.random() for _ in range(12)) - 6.0
            vals.append(math.exp(lognormal_sigma * z))
        weights = vals
    tot = sum(weights)
    probs = [w/tot for w in weights]

    # CDF for sampling
    from bisect import bisect
    cdf = []
    acc = 0.0
    for p in probs:
        acc += p; cdf.append(acc)
    def sample_idx():
        return bisect(cdf, rng.random())

    # generate reads
    reads, cov = [], Counter()
    for i in range(reads_total):
        idx = sample_idx()
        sid, src = records[idx]
        obs, errs = _corrupt(src, p_error, alpha, rng)
        reads.append({
            "read_id": f"read_{i:07d}", "source_id": sid,
            "source_seq": src, "observed_seq": obs,
            "num_errors": len(errs),
            "error_positions": ";".join(str(p) for p,_ in errs),
            "error_changes": ";".join(ch for _,ch in errs)
        })
        cov[sid] += 1

    ts = timestamp_tag()
    outdir = Path("data/reads") / f"{ts}_{seq_type}_from_{libdir.name}"
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "reads.fasta","w") as fh:
        for r in reads:
            fh.write(f">{r['read_id']}\n{r['observed_seq']}\n")
    with open(outdir / "reads.csv","w") as fh:
        cols = ["read_id","source_id","source_seq","observed_seq","num_errors","error_positions","error_changes"]
        fh.write(",".join(cols)+"\n")
        for r in reads:
            fh.write(",".join(str(r[c]) for c in cols)+"\n")
    with open(outdir / "coverage_per_source.csv","w") as fh:
        fh.write("source_id,reads\n")
        for sid, c in cov.items():
            fh.write(f"{sid},{c}\n")
    with open(outdir / "params.json","w") as fh:
        json.dump({
            "timestamp": ts, "library_tag": libdir.name, "reads_total": reads_total,
            "p_error": p_error, "seed_sim": seed_sim, "abundance_mode": abundance_mode,
            "lognormal_sigma": lognormal_sigma
        }, fh, indent=2)
    return str(outdir)
