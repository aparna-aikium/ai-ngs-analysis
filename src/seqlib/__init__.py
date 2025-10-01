"""
AI-Enabled NGS Analysis and Visualizations Package

This package provides tools for:
- Generating DNA/AA sequence libraries with degeneracy
- Simulating NGS reads with configurable error rates
- Sequence analysis and visualization utilities
"""

# Import main functions for easy access
from .library_generator import generate_library
from .ngs_simulator import simulate_reads
from .utils import load_fasta_or_txt, write_fasta, timestamp_tag
from .degeneracy_schemas import normalize_degeneracy, IUPAC_DNA, AA20

# Define what gets imported with "from seqlib import *"
__all__ = [
    'generate_library',
    'simulate_reads', 
    'load_fasta_or_txt',
    'write_fasta',
    'timestamp_tag',
    'normalize_degeneracy',
    'IUPAC_DNA',
    'AA20'
]

# Package metadata
__version__ = "0.1.0"
__author__ = "AI NGS Analysis Team"
__description__ = "AI-Enabled NGS Analysis and Visualizations"
