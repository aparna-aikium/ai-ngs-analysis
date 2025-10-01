from pathlib import Path
import json, random
from .utils import load_fasta_or_txt, write_fasta, timestamp_tag
from .degeneracy_schemas import normalize_degeneracy

def _theoretical_max(pos_to_choices):
    m = 1
    for chs in pos_to_choices.values():
        m *= len(chs)
    return m

def generate_library(seq_type, backbone_path, positions_degeneracy,
                     n_sequences=10_000, seed_library=0, tag="run") -> str:
    backbone = load_fasta_or_txt(backbone_path)
    L = len(backbone)

    pos_to_choices = {}
    for d in positions_degeneracy:
        pos = int(d["pos"])
        if not (1 <= pos <= L):
            raise ValueError(f"Position {pos} out of range 1..{L}")
        pos_to_choices[pos] = normalize_degeneracy(seq_type, d["chars"])

    theo = _theoretical_max(pos_to_choices)
    if n_sequences > theo:
        n_sequences = theo

    rng = random.Random(seed_library)
    seqs = set()
    def mutate():
        s = list(backbone)
        for pos, choices in pos_to_choices.items():
            s[pos-1] = rng.choice(choices)
        return "".join(s)

    while len(seqs) < n_sequences:
        seqs.add(mutate())

    records = [(f"lib_{i:05d}", s) for i, s in enumerate(seqs)]

    ts = timestamp_tag()
    outdir = Path("data/libraries") / f"{ts}_{seq_type}_L{L}_N{len(records)}_seed{seed_library}_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    write_fasta(records, outdir / f"{outdir.name}_true_library.fasta")
    with open(outdir / f"{outdir.name}_true_library.csv","w") as fh:
        fh.write("id,sequence\n")
        for rid, s in records:
            fh.write(f"{rid},{s}\n")
    manifest = {
        "timestamp": ts, "seq_type": seq_type, "backbone_path": backbone_path,
        "backbone_length": L, "positions": sorted(pos_to_choices.keys()),
        "pos_to_choices": {int(k): v for k,v in pos_to_choices.items()},
        "n_sequences": len(records), "seed_library": seed_library,
        "theoretical_max": theo
    }
    with open(outdir / f"{outdir.name}_manifest.json","w") as fh:
        json.dump(manifest, fh, indent=2)
    return str(outdir)
