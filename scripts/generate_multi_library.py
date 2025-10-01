import argparse, json, yaml, random
from pathlib import Path
from datetime import datetime

import pandas as pd

from seqlib.utils import load_fasta_or_txt
from seqlib.degeneracy_schemas import normalize_degeneracy

def _ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _theoretical_max(pos_to_choices):
    m = 1
    for chs in pos_to_choices.values(): m *= len(chs)
    return m

def _gen_from_one(backbone_path, seq_type, posdeg_list, n_sequences, seed):
    print(f"Loading backbone from: {backbone_path}")
    backbone = load_fasta_or_txt(backbone_path)
    print(f"Backbone loaded, length: {len(backbone)}")
    L = len(backbone)
    rng = random.Random(seed)

    # normalize degeneracy
    pos_to_choices = {}
    for d in posdeg_list:
        pos = int(d["pos"])
        if not (1 <= pos <= L):
            raise ValueError(f"Position {pos} out of 1..{L}")
        pos_to_choices[pos] = normalize_degeneracy(seq_type, d["chars"])
    
    print(f"Positions to mutate: {list(pos_to_choices.keys())}")
    print(f"Choices per position: {pos_to_choices}")

    tmax = _theoretical_max(pos_to_choices)
    print(f"Theoretical max combinations: {tmax}")

    # decide mode: full combinatorial vs sampled
    if n_sequences in (None, "all"):
        # enumerate all combos
        from itertools import product
        positions = sorted(pos_to_choices.keys())
        choice_lists = [list(pos_to_choices[p]) for p in positions]
        seqs = []
        for combo in product(*choice_lists):
            s = list(backbone)
            for p, ch in zip(positions, combo):
                s[p-1] = ch
            seqs.append("".join(s))
    else:
        # sample unique
        n = min(int(n_sequences), tmax)
        print(f"Generating {n} sequences")
        seqs_set = set()
        def mutate():
            s = list(backbone)
            for p, choices in pos_to_choices.items():
                s[p-1] = rng.choice(choices)
            return "".join(s)
        while len(seqs_set) < n:
            seqs_set.add(mutate())
        seqs = list(seqs_set)
    
    print(f"Generated {len(seqs)} sequences")
    return seqs, L, pos_to_choices, tmax

def main():
    print("Starting generate_multi_library.py...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/multi_library.yaml")
    args = ap.parse_args()
    print(f"Loading config from: {args.config}")
    
    cfg = yaml.safe_load(open(args.config))
    print(f"Config loaded: {list(cfg.keys())}")

    seed_library = int(cfg.get("seed_library", 12345))
    print(f"Seed library: {seed_library}")

    rows = []
    meta = []
    total_weight = 0.0

    # per-backbone generation
    print(f"Processing {len(cfg['libraries'])} libraries...")
    for idx, spec in enumerate(cfg["libraries"]):
        print(f"\nProcessing library {idx+1}: {spec['backbone_id']}")
        backbone_id = spec["backbone_id"]
        seq_type    = spec["seq_type"]
        backbone    = spec["backbone_path"]
        posdeg      = spec["positions_degeneracy"]
        nseq        = spec.get("n_sequences", None)
        start_w     = float(spec.get("start_weight", 1.0))

        # use a derived seed for reproducibility across backbones
        seed = seed_library + idx * 101

        seqs, L, pos_to_choices, tmax = _gen_from_one(backbone, seq_type, posdeg, nseq, seed)

        for i, s in enumerate(seqs):
            rows.append({
                "seq_id": f"{backbone_id}_{i:06d}",
                "sequence": s,
                "backbone_id": backbone_id,
                "seq_type": seq_type,
                "start_weight": start_w
            })
        meta.append({
            "backbone_id": backbone_id,
            "backbone_path": backbone,
            "length": L,
            "pos_to_choices": {int(k): v for k,v in pos_to_choices.items()},
            "theoretical_max": tmax,
            "n_generated": len(seqs),
            "start_weight": start_w,
            "seed": seed
        })
        total_weight += start_w * len(seqs)

    df = pd.DataFrame(rows)
    print(f"\nTotal sequences generated: {len(df)}")
    if df.empty:
        raise RuntimeError("No sequences generated; check config.")

    # start_frequency: proportional to start_weight per sequence
    # (each sequence from a backbone inherits its backbone's start_weight)
    # normalize over all sequences:
    df["start_frequency"] = df["start_weight"] / (df["start_weight"].sum())

    # output folder
    ts = _ts()
    outdir = Path("data/libraries") / f"{ts}_combined_multi_backbone"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")

    # save combined files
    fasta_path = outdir / "combined_library.fasta"
    with open(fasta_path, "w") as fh:
        for _, r in df.iterrows():
            fh.write(f">{r.seq_id}\n{r.sequence}\n")
    print(f"FASTA saved: {fasta_path}")

    csv_path = outdir / "combined_library.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    manifest = {
        "timestamp": ts,
        "seed_library": seed_library,
        "backbones": meta
    }
    with open(outdir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest saved: {outdir / 'manifest.json'}")

    print(f"\n✅ Library generation complete!")
    print(f"Output directory: {outdir}")

if __name__ == "__main__":
    main()
