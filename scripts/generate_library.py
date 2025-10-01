from pathlib import Path
import yaml, random, json
from seqlib.utils import timestamp_tag

def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    
    # Handle the new experiment.yaml format
    if "library" in cfg:
        # New format with library section
        lib_cfg = cfg["library"]
        mode = lib_cfg.get("mode", "single")
        refs = lib_cfg.get("references", [])
        seed_library = lib_cfg.get("seed_library", 12345)
    else:
        # Old format
        mode = cfg.get("mode", "single")
        refs = cfg.get("references", [])
        seed_library = cfg.get("seed_library", 12345)
    
    if not refs:
        raise ValueError("No references found in configuration")
    
    outdir = Path("data/libraries")
    outdir.mkdir(parents=True, exist_ok=True)

    ts = timestamp_tag()
    lib_tag = f"{ts}_{mode}_library"
    libdir = outdir / lib_tag
    libdir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for ref in refs:
        ref_id = ref["id"]
        seq_type = ref.get("seq_type", "dna")
        backbone_path = ref["backbone_path"]
        positions_degeneracy = ref["positions_degeneracy"]
        n_sequences = ref.get("n_sequences", 5000)
        
        # Generate variants for this reference
        from seqlib.library_generator import generate_library
        ref_outdir = generate_library(
            seq_type=seq_type,
            backbone_path=backbone_path,
            positions_degeneracy=positions_degeneracy,
            n_sequences=n_sequences,
            seed_library=seed_library,
            tag=ref_id
        )
        
        # Read the generated sequences
        ref_csv = Path(ref_outdir) / f"{Path(ref_outdir).name}_true_library.csv"
        if ref_csv.exists():
            import pandas as pd
            df = pd.read_csv(ref_csv)
            for _, row in df.iterrows():
                all_records.append((f"{ref_id}_{row['id']}", row['sequence']))

    # Write combined outputs
    lib_csv = libdir / f"{lib_tag}_true_library.csv"
    with open(lib_csv, "w") as fh:
        fh.write("id,sequence\n")
        for rid, seq in all_records:
            fh.write(f"{rid},{seq}\n")

    manifest = {
        "mode": mode,
        "num_variants": len(all_records),
        "seq_type": "dna",
        "schema": "custom",
        "references": refs,
        "seed_library": seed_library
    }
    with open(libdir / f"{lib_tag}_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"✅ Library created at {libdir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python scripts/generate_library.py configs/experiment.yaml")
    else:
        main(sys.argv[1])
