#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, 'src')

try:
    from seqlib.library_generator import generate_library
    print("✓ Successfully imported generate_library")
except Exception as e:
    print(f"✗ Failed to import generate_library: {e}")
    sys.exit(1)

try:
    import yaml
    print("✓ Successfully imported yaml")
except Exception as e:
    print(f"✗ Failed to import yaml: {e}")
    sys.exit(1)

# Test config loading
try:
    with open('configs/default_library.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    print("✓ Successfully loaded config")
    print(f"  Config: {cfg}")
except Exception as e:
    print(f"✗ Failed to load config: {e}")
    sys.exit(1)

# Test backbone file reading
try:
    backbone_path = cfg["backbone_path"]
    print(f"✓ Backbone path: {backbone_path}")
    
    if os.path.exists(backbone_path):
        print(f"✓ Backbone file exists")
        with open(backbone_path, 'r') as f:
            content = f.read()
        print(f"  Content length: {len(content)} chars")
    else:
        print(f"✗ Backbone file does not exist: {backbone_path}")
        sys.exit(1)
except Exception as e:
    print(f"✗ Failed to check backbone file: {e}")
    sys.exit(1)

print("\nAttempting to run generate_library...")
try:
    outdir = generate_library(
        seq_type=cfg["seq_type"],
        backbone_path=cfg["backbone_path"],
        positions_degeneracy=cfg["positions_degeneracy"],
        n_sequences=cfg["n_sequences"],
        seed_library=cfg["seed_library"],
        tag=cfg.get("tag","run")
    )
    print(f"✓ Success! Output directory: {outdir}")
except Exception as e:
    print(f"✗ Failed to run generate_library: {e}")
    import traceback
    traceback.print_exc()
