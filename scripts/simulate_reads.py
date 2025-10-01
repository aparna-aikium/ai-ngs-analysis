import argparse, yaml
from pathlib import Path
import pandas as pd
import random
import json

def simulate_reads_from_csv(csv_path, reads_total, p_error, seed_sim):
    """
    Custom function to simulate reads from a CSV with start_frequency column
    """
    print(f"Reading library from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if "start_frequency" not in df.columns:
        raise ValueError("CSV must contain 'start_frequency' column")
    
    # Normalize frequencies
    frequencies = df["start_frequency"].values
    frequencies = frequencies / frequencies.sum()
    
    # Set up random number generator
    rng = random.Random(seed_sim)
    
    # Create output directory
    outdir = Path(csv_path).parent / f"simulated_reads_{seed_sim}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {reads_total} reads...")
    
    # Generate reads
    reads = []
    coverage = {}
    
    for i in range(reads_total):
        # Sample sequence based on frequency
        idx = rng.choices(range(len(df)), weights=frequencies)[0]
        row = df.iloc[idx]
        
        source_seq = row["sequence"]
        source_id = row.get("seq_id", f"seq_{idx}")
        
        # Add to coverage
        coverage[source_id] = coverage.get(source_id, 0) + 1
        
        # Corrupt sequence with errors
        observed_seq = ""
        errors = []
        for j, char in enumerate(source_seq):
            if rng.random() < p_error:
                # Introduce error - for DNA, randomly change to another base
                if char in "ACGT":
                    new_char = rng.choice([c for c in "ACGT" if c != char])
                else:
                    new_char = char
                observed_seq += new_char
                errors.append((j+1, f"{char}>{new_char}"))
            else:
                observed_seq += char
        
        reads.append({
            "read_id": f"read_{i:07d}",
            "source_id": source_id,
            "source_seq": source_seq,
            "observed_seq": observed_seq,
            "num_errors": len(errors),
            "error_positions": ";".join(str(p) for p, _ in errors),
            "error_changes": ";".join(ch for _, ch in errors)
        })
    
    # Write outputs
    fasta_path = outdir / "reads.fasta"
    with open(fasta_path, "w") as fh:
        for read in reads:
            fh.write(f">{read['read_id']}\n{read['observed_seq']}\n")
    
    csv_path_out = outdir / "reads.csv"
    with open(csv_path_out, "w") as fh:
        cols = ["read_id", "source_id", "source_seq", "observed_seq", "num_errors", "error_positions", "error_changes"]
        fh.write(",".join(cols) + "\n")
        for read in reads:
            fh.write(",".join(str(read[col]) for col in cols) + "\n")
    
    # Write coverage summary
    coverage_path = outdir / "coverage_per_source.csv"
    with open(coverage_path, "w") as fh:
        fh.write("source_id,reads\n")
        for source_id, count in sorted(coverage.items()):
            fh.write(f"{source_id},{count}\n")
    
    # Write parameters
    params_path = outdir / "params.json"
    with open(params_path, "w") as fh:
        json.dump({
            "reads_total": reads_total,
            "p_error": p_error,
            "seed_sim": seed_sim,
            "library_csv": str(csv_path)
        }, fh, indent=2)
    
    print(f"✅ Reads simulated successfully!")
    print(f"   FASTA: {fasta_path}")
    print(f"   CSV: {csv_path_out}")
    print(f"   Coverage: {coverage_path}")
    print(f"   Params: {params_path}")
    
    return str(outdir)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="YAML config for simulation")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    
    # Handle the new experiment.yaml format
    if "ngs" in cfg:
        ngs_cfg = cfg["ngs"]
        reads_total = ngs_cfg.get("reads_total", 100000)
        p_error = float(ngs_cfg.get("p_error", 0.001))
        seed_sim = ngs_cfg.get("seed_sim", 999)
        
        # Find the library CSV file based on source
        source = ngs_cfg.get("source", "true_library")
        if source == "selection_round":
            # Look for selection results
            selection_round = ngs_cfg.get("selection_round_index", 3)
            # This would need to be implemented based on your selection output structure
            raise NotImplementedError("Selection round source not yet implemented")
        else:
            # Use the most recent library directory
            library_dirs = list(Path("data/libraries").glob("*_combined_multi_backbone"))
            if not library_dirs:
                raise FileNotFoundError("No library directories found. Run library generation first.")
            library_dir = max(library_dirs, key=lambda x: x.stat().st_mtime)
            library_csv = library_dir / "combined_library.csv"
    else:
        # Old format
        library_csv = cfg["library_csv"]
        reads_total = cfg["reads_total"]
        p_error = float(cfg["p_error"])
        seed_sim = cfg["seed_sim"]
    
    print(f"Using library CSV: {library_csv}")
    print(f"Simulating {reads_total} reads with error rate {p_error}")
    
    # Simulate reads using our custom function
    outdir = simulate_reads_from_csv(
        csv_path=str(library_csv),
        reads_total=reads_total,
        p_error=p_error,
        seed_sim=seed_sim
    )
    print(f"Reads simulated in: {outdir}")

if __name__ == "__main__":
    main()