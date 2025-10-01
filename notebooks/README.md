# Jupyter Notebook Examples

This folder contains examples of how to use the `seqlib` package.

## Available Examples

### 1. `01_generate_library.py` - Library Generation
Demonstrates how to:
- Generate DNA/AA sequence libraries with degeneracy
- Use configuration files
- Create custom libraries with different parameters

### 2. `02_simulate_ngs_reads.py` - NGS Read Simulation
Shows how to:
- Simulate NGS reads from generated libraries
- Configure error rates and read counts
- Explore simulation outputs

## How to Use

### Option 1: Run as Python Scripts
```bash
export PYTHONPATH=src
python notebooks/01_generate_library.py
python notebooks/02_simulate_ngs_reads.py
```

### Option 2: Convert to Jupyter Notebooks
1. Copy the content of each `.py` file
2. Create a new Jupyter notebook
3. Paste each "Cell" section into separate notebook cells
4. Run the cells in order

### Option 3: Use the Command Line Scripts
```bash
export PYTHONPATH=src
python scripts/generate_library.py --config configs/default_library.yaml
python scripts/simulate_reads.py --config configs/default_simulation.yaml
```

## Prerequisites
- Python 3.7+
- Required packages: `pyyaml`
- Set `PYTHONPATH=src` to import the `seqlib` package

## File Structure
```
notebooks/
├── README.md                    # This file
├── 01_generate_library.py      # Library generation example
└── 02_simulate_ngs_reads.py    # NGS simulation example
```
