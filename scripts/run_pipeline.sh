#!/bin/bash

# AI-NGS Analysis Pipeline Runner
# This script automatically finds the latest library and runs the complete pipeline

set -e  # Exit on any error

echo "🔍 Finding latest library directory..."

# Find the most recent library directory
LATEST_LIB=$(ls -td data/libraries/*_combined_multi_backbone | head -1)

if [ -z "$LATEST_LIB" ]; then
    echo "❌ No library directories found. Run library generation first."
    exit 1
fi

echo "✅ Found library: $LATEST_LIB"

# Extract the library name for output files
LIB_NAME=$(basename "$LATEST_LIB")
echo "📁 Library name: $LIB_NAME"

# Set environment
export PYTHONPATH=src

echo ""
echo "🚀 Starting AI-NGS Analysis Pipeline..."
echo "========================================"

# Step 1: Run selection
echo ""
echo "📊 Step 1: Running selection..."
python scripts/run_selection.py --library-csv "$LATEST_LIB/combined_library.csv" --config configs/experiment.yaml

# Step 2: Simulate NGS reads
echo ""
echo "🧬 Step 2: Simulating NGS reads..."
python scripts/simulate_reads.py --config configs/experiment.yaml

# Step 3: Run Bowtie2 alignment
echo ""
echo "🎯 Step 3: Running Bowtie2 alignment..."

# Find the simulated reads directory
READS_DIR=$(ls -td "$LATEST_LIB"/simulated_reads_* | head -1)
if [ -z "$READS_DIR" ]; then
    echo "❌ No simulated reads found. Check NGS simulation step."
    exit 1
fi

echo "📖 Using reads from: $READS_DIR"

python scripts/run_alignment.py \
    --ref-fasta "$LATEST_LIB/combined_library.fasta" \
    --reads-fasta "$READS_DIR/reads.fasta" \
    --params configs/bowtie2_params.json \
    --outdir data/alignments

echo ""
echo "🎉 Pipeline completed successfully!"
echo "📊 Results available in: data/alignments/$LIB_NAME/"
echo "📈 Alignment summary: data/alignments/$LIB_NAME/results/"
