from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator, constr, conint, confloat
from typing import List, Dict, Any, Optional, Literal
import tempfile
import os
import json
import pandas as pd
from pathlib import Path
import sys
import shutil
import random
import numpy as np

# Add the ai-ngs-analysis package to the path
sys.path.insert(0, '/Users/aparnaanandkumar/Documents/aikium/ngs_analysis_tool/ai-ngs-analysis/src')

from seqlib import library_generator as libgen
from seqlib import selection as select
from seqlib import ngs_simulator as ngs
from seqlib import analyzer as analyzer

app = FastAPI(title="NGS Analysis API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js frontend (both ports)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI API Models
class AIAskRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    codeSelection: Optional[Dict[str, str]] = None
    currentStep: Optional[str] = None

class AIContextInfo(BaseModel):
    file: str
    selection: str

class AIDataStatus(BaseModel):
    library: bool
    selection: bool
    ngs: bool
    analysis: bool

class AISource(BaseModel):
    file: str
    path: str
    range: Optional[str] = None

class AIResponse(BaseModel):
    summary: str
    keyTakeaways: List[str]
    details: Optional[str] = None
    context: AIContextInfo
    dataStatus: AIDataStatus
    sources: List[AISource]
    quickActions: List[str]

# Constants for validation - DNA IUPAC codes only
# DNA IUPAC nucleotide codes (https://www.bioinformatics.org/sms/iupac.html)
DNA_IUPAC = set("ACGTNRYMKSWBDHV")                 # Complete DNA IUPAC codes
# A=Adenine, C=Cytosine, G=Guanine, T=Thymine
# N=Any base, R=Purine (A/G), Y=Pyrimidine (C/T), M=Amino (A/C), K=Keto (G/T), S=Strong (G/C), W=Weak (A/T)
# B=Not A (C/G/T), D=Not C (A/G/T), H=Not G (A/C/T), V=Not T (A/C/G)

# Temporarily relaxed for debugging
DegStr = str  # Will validate in the validator instead

# Pydantic models for request/response
class BackboneRequest(BaseModel):
    sequence_type: Literal["dna"] = Field(default="dna", description="Template alphabet (DNA only)")
    template: constr(min_length=1)                 # content checked below
    degeneracy: DegStr                             # e.g. "5:N,15:R,25:M"
    weight: confloat(ge=0) = 1.0

    @validator("template")
    def template_alphabet(cls, v, values):
        s = v.upper()
        if not set(s) <= set("ACGT"):
            bad_chars = sorted(set(s) - set("ACGT"))
            raise ValueError(f"DNA template must contain only A/C/G/T. Found invalid characters: {''.join(bad_chars)}")
        return s

    @validator("degeneracy")
    def degeneracy_codes_and_bounds(cls, v, values):
        tmpl = values.get("template", "")
        allowed = DNA_IUPAC

        # Check basic format first
        if not v or not v.strip():
            raise ValueError(f"Degeneracy string cannot be empty")
        
        # Check regex pattern for DNA codes
        import re
        if not re.match(r"^\d+:[A-Z](,\d+:[A-Z])*$", v):
            raise ValueError(f"Degeneracy string '{v}' must match pattern 'position:CODE,position:CODE' (e.g., '5:N,15:R')")

        for pair in v.split(","):
            if ":" not in pair:
                raise ValueError(f"Invalid degeneracy format: '{pair}'. Expected 'position:code'")
            pos_s, code = pair.split(":", 1)
            try:
                pos = int(pos_s)
            except ValueError:
                raise ValueError(f"Invalid position: '{pos_s}'. Must be an integer.")
            if not (1 <= pos <= len(tmpl)):
                raise ValueError(f"Position {pos} exceeds template length ({len(tmpl)}).")
            if code not in allowed:
                allowed_str = ''.join(sorted(allowed))
                raise ValueError(
                    f"Degeneracy code '{code}' not allowed for DNA "
                    f"(allowed: {allowed_str})."
                )
        return v

class LibraryGenerationRequest(BaseModel):
    backbones: List[BackboneRequest]
    library_size: conint(ge=1)
    seed: int = 1337

    class Config:
        extra = "forbid"   # blocks stray fields like degeneracyPositions with a clear error

class SelectionRequest(BaseModel):
    library_csv: str
    rounds: int = 3
    stringency: float = 0.7
    target_c: float = 50.0
    expr_sigma: float = 0.5
    pcr_cycles: int = 10
    pcr_bias: float = 0.3
    mu_log10kd: float = -7.0
    alpha: float = 1.0
    sigma_kd: float = 0.3
    molecules: int = 200000
    abundance_mode: str = "uniform"
    lognormal_sigma: float = 1.0
    seed: int = 4242

class NGSRequest(BaseModel):
    library_dir: str
    reads_per_pool: int = 250000
    read_length: int = 100
    error_rate: float = 0.005
    seed: int = 1337

class AnalysisRequest(BaseModel):
    library_csv: str
    selection_dir: str
    top_n: int = 25

# Helper functions
def create_temp_backbone(template: str) -> str:
    """Create a temporary backbone file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
        f.write(f">backbone\n{template}\n")
        return f.name

def create_temp_selection_config(
    rounds: int, stringency: float, target_c: float, pcr_cycles: int,
    pcr_bias: float, expr_sigma: float, mu_log10kd: float, alpha: float,
    sigma_kd: float, molecules: int, seed: int
) -> str:
    """Create a temporary selection config file"""
    config = {
        "rounds": rounds,
        "mode": "competitive",
        "targets": [
            {
                "name": "T1",
                "concentration_nM": target_c,
                "weight": 1.0
            }
        ],
        "stringency": {
            "S": stringency
        },
        "pcr": {
            "cycles": pcr_cycles,
            "bias_sigma": pcr_bias
        },
        "expression": {
            "enabled": True,
            "sigma": expr_sigma
        },
        "KD_model": {
            "mu_log10KD": mu_log10kd,
            "alpha": alpha,
            "sigma": sigma_kd
        },
        "input_molecules_per_round": molecules,
        "seed_selection": seed
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config, f)
        return f.name

def detect_sequence_type(sequence: str) -> str:
    """Detect if sequence is DNA or amino acid based on character composition"""
    clean_seq = sequence.upper()
    dna_chars = len([c for c in clean_seq if c in 'ATGC'])
    aa_only_chars = len([c for c in clean_seq if c in 'DEFHIKLMNPQRSVWY'])
    
    # If >80% are DNA characters, consider it DNA
    dna_ratio = dna_chars / len(clean_seq) if clean_seq else 0
    aa_only_ratio = aa_only_chars / len(clean_seq) if clean_seq else 0
    
    if dna_ratio > 0.8:
        return "dna"
    elif aa_only_ratio > 0.3:  # AA-specific chars indicate protein
        return "aa"
    else:
        return "dna"  # Default to DNA if ambiguous

def convert_degeneracy_to_lib_format(degeneracy_spec: str, sequence_type: str) -> List[Dict[str, Any]]:
    """Convert validated degeneracy string to library format - validation already done by Pydantic"""
    
    positions_degeneracy = []
    if degeneracy_spec.strip():
        for pair in degeneracy_spec.split(","):
            pos_str, code = pair.split(":", 1)
            pos_int = int(pos_str)
            
            positions_degeneracy.append({
                'pos': pos_int - 1,  # Convert to 0-based indexing
                'chars': code  # Send the actual character code, not the count
            })
    
    return positions_degeneracy

def generate_random_template(seed_offset: int) -> str:
    """Generate a random nucleotide sequence"""
    random.seed(seed_offset)
    length = random.randint(30, 60)
    nucleotides = ['A', 'T', 'G', 'C']
    return ''.join(random.choices(nucleotides, k=length))

# API Endpoints
@app.get("/")
async def root():
    return {"message": "NGS Analysis API", "version": "1.0.0"}

@app.post("/api/library/generate")
async def generate_library(request: LibraryGenerationRequest):
    """Generate a library from multiple backbones - matches Streamlit functionality"""
    try:
        backbone_debug = []
        for b in request.backbones:
            backbone_debug.append({
                'sequence_type': b.sequence_type,
                'template': b.template[:20] + '...' if len(b.template) > 20 else b.template,
                'degeneracy': b.degeneracy,
                'weight': b.weight
            })
        # Validate backbones
        if not request.backbones:
            raise HTTPException(status_code=400, detail="At least one backbone is required")
        
        # Normalize weights to sum to 1.0 (they represent final ratios)
        total_weight = sum(b.weight for b in request.backbones)
        normalized_weights = [b.weight / total_weight for b in request.backbones]
        
        # Calculate theoretical maximum for each backbone
        theoretical_maxima = []
        for i, backbone in enumerate(request.backbones):
            # Calculate theoretical max using character counts
            theoretical_max = 1
            if backbone.degeneracy.strip():
                for pair in backbone.degeneracy.split(","):
                    pos_str, code = pair.split(":", 1)
                    # Get character count for this code
                    if backbone.sequence_type == "dna":
                        CODE_COUNTS = {
                            'A': 1, 'T': 1, 'G': 1, 'C': 1, 'N': 4, 'R': 2, 'Y': 2, 
                            'M': 2, 'K': 2, 'S': 2, 'W': 2, 'B': 3, 'D': 3, 'H': 3, 'V': 3
                        }
                    else:  # aa
                        CODE_COUNTS = {
                            'A': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'K': 1, 'L': 1,
                            'M': 1, 'N': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1, 'T': 1, 'V': 1, 'W': 1, 'Y': 1,
                            'X': 20, 'B': 2, 'Z': 2, 'J': 2
                        }
                    
                    if code in CODE_COUNTS:
                        char_count = CODE_COUNTS[code]
                    else:
                        # Handle custom character sets like "AC", "GT"
                        char_count = len(set(code))
                    
                    theoretical_max *= char_count
            theoretical_maxima.append(theoretical_max)
        
        # Find the limiting factor - which backbone constrains the total library size?
        max_total_sizes = []
        for i, (max_seq, weight) in enumerate(zip(theoretical_maxima, normalized_weights)):
            if weight > 0:
                max_total = max_seq / weight
                max_total_sizes.append(max_total)
            else:
                max_total_sizes.append(float('inf'))
        
        # The limiting factor is the smallest total library size
        limiting_total = min(max_total_sizes)
        limiting_backbone_idx = max_total_sizes.index(limiting_total)
        
        # Calculate sequences for each backbone based on the limiting total
        sequences_per_backbone = []
        for i, weight in enumerate(normalized_weights):
            target_seq = limiting_total * weight
            # Round to nearest integer, but don't exceed theoretical maximum
            n_seq = round(target_seq)
            n_seq = min(n_seq, theoretical_maxima[i])
            sequences_per_backbone.append(n_seq)
        
        # Calculate actual ratios
        total_sequences = sum(sequences_per_backbone)
        
        # Generate libraries for each backbone
        all_libraries = []
        temp_files = []
        
        for i, backbone in enumerate(request.backbones):
            if sequences_per_backbone[i] > 0:
                # Get degeneracy for this backbone
                backbone_deg = convert_degeneracy_to_lib_format(backbone.degeneracy, backbone.sequence_type)
                
                # Use explicit sequence type from request
                seq_type = backbone.sequence_type
                
                # Create temporary backbone file
                backbone_path = create_temp_backbone(backbone.template)
                temp_files.append(backbone_path)
                
                # Generate library for this backbone using pre-calculated sequence count
                
                library_dir = libgen.generate_library(
                    seq_type=seq_type,
                    backbone_path=backbone_path,
                    positions_degeneracy=backbone_deg,
                    n_sequences=sequences_per_backbone[i],
                    seed_library=request.seed + i,
                    tag=f"api_backbone_{i+1}"
                )
                
                
                # Read library data
                library_csv = Path(library_dir) / f"{Path(library_dir).name}_true_library.csv"
                lib_df = pd.read_csv(library_csv)
                lib_df = lib_df.rename(columns={'id': 'seq_id', 'sequence': 'seq'})
                lib_df['sequence'] = lib_df['seq']
                lib_df['backbone_id'] = f'backbone_{i+1}'
                # Start frequency: each sequence gets equal frequency within backbone
                # Total backbone frequency = target ratio
                lib_df['start_frequency'] = normalized_weights[i] / len(lib_df)
                
                all_libraries.append(lib_df)
        
        # Combine all libraries
        library_df = pd.concat(all_libraries, ignore_index=True)
        
        # Create combined library directory
        combined_dir = Path("data/libraries") / f"combined_api_{request.seed}"
        combined_dir.mkdir(parents=True, exist_ok=True)
        
        # Save combined library
        library_csv = combined_dir / "combined_library.csv"
        library_df.to_csv(library_csv, index=False)
        
        # Clean up temporary files
        for temp_file in temp_files:
            os.unlink(temp_file)
        
        # Calculate backbone distribution
        backbone_distribution = library_df['backbone_id'].value_counts().to_dict()
        
        # Return results
        return {
            "success": True,
            "library_dir": str(combined_dir),
            "library_csv": str(library_csv),
            "total_sequences": len(library_df),
            "unique_sequences": library_df['seq'].nunique(),
            "backbone_distribution": backbone_distribution,
            "average_length": float(library_df['seq'].str.len().mean()),
            "library_preview": library_df.to_dict('records'),
            "backbones": [
                {
                    "sequence_type": b.sequence_type,
                    "template": b.template,
                    "degeneracy": b.degeneracy,
                    "weight": b.weight
                } for b in request.backbones
            ],
            "weight_normalization": {
                "raw_weights": [b.weight for b in request.backbones],
                "normalized_weights": normalized_weights,
                "total_weight": total_weight
            },
            "theoretical_maxima": theoretical_maxima,
            "sequences_per_backbone": sequences_per_backbone,
            "limiting_total": limiting_total
        }
        
    except Exception as e:
        # Detailed error logging
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ BACKEND ERROR: {str(e)}")
        print(f"❌ BACKEND TRACEBACK:\n{error_trace}")
        
        raise HTTPException(status_code=500, detail=f"Library generation failed: {str(e)}")

@app.post("/api/library/random-template")
async def generate_random_template_endpoint(seed_offset: int):
    """Generate a random template sequence"""
    try:
        template = generate_random_template(seed_offset)
        return {
            "success": True,
            "template": template,
            "length": len(template)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Random template generation failed: {str(e)}")

@app.post("/api/selection/run")
async def run_selection(request: SelectionRequest):
    """Run selection simulation"""
    try:
        
        # Create selection config
        config_path = create_temp_selection_config(
            request.rounds, request.stringency, request.target_c, request.pcr_cycles,
            request.pcr_bias, request.expr_sigma, request.mu_log10kd, request.alpha,
            request.sigma_kd, request.molecules, request.seed
        )
        
        # Run selection
        selection_dir = select.run_selection(request.library_csv, config_path)
        
        # Read selection results
        selection_results = []
        for r in range(1, request.rounds + 1):
            round_file = Path(selection_dir) / f"round_{r:02d}_pool.csv"
            if round_file.exists():
                round_df = pd.read_csv(round_file)
                round_df['round'] = r
                
                # Calculate KD from capture probability using Langmuir inversion
                # Use the target concentration from the request (convert to nM if needed)
                ligand_conc_nM = request.target_c  # Already in nM
                
                # Langmuir inversion: KD = C * (1 - θ) / θ
                # Where θ is the capture probability and C is ligand concentration
                round_df['KD'] = round_df.apply(lambda row: 
                    ligand_conc_nM * (1 - row['capture_prob']) / max(row['capture_prob'], 1e-10) 
                    if row['capture_prob'] > 1e-10 else float('inf'), axis=1)
                
                # Convert infinite values to a large but finite number for display
                round_df['KD'] = round_df['KD'].replace([float('inf'), -float('inf')], 1e12)
                
                selection_results.append(round_df.to_dict('records'))
        
        # Calculate diversity metrics
        diversity_metrics = []
        for i, round_data in enumerate(selection_results):
            if round_data and 'frequency' in round_data[0]:
                freqs = [row['frequency'] for row in round_data if row['frequency'] > 0]
                if freqs:
                    freqs = np.array(freqs)
                    shannon_h = -np.sum(freqs * np.log(freqs))
                    diversity_metrics.append({
                        'round': i + 1,
                        'shannon_h': float(shannon_h),
                        'pool_size': len(round_data)
                    })
        
        # Clean up
        os.unlink(config_path)
        
        return {
            "success": True,
            "selection_dir": selection_dir,
            "rounds": request.rounds,
            "final_pool_size": len(selection_results[-1]) if selection_results else 0,
            "selection_results": selection_results,
            "diversity_metrics": diversity_metrics
        }
        
    except Exception as e:
        import traceback
        print(f"❌ SELECTION ERROR: {str(e)}")
        print(f"❌ SELECTION TRACEBACK:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Selection failed: {str(e)}")

@app.post("/api/ngs/simulate")
async def simulate_ngs(request: NGSRequest):
    """Simulate NGS reads"""
    try:
        # Check if library directory exists and what files are in it
        from pathlib import Path
        lib_path = Path(request.library_dir)
        if lib_path.exists():
            files = list(lib_path.iterdir())
            csv_files = [f for f in files if f.name.endswith("_true_library.csv")]
            json_files = [f for f in files if f.name.endswith("_manifest.json")]
        
        # Check if we have the expected files, if not create them from combined_library.csv
        lib_path = Path(request.library_dir)
        csv_files = [f for f in lib_path.iterdir() if f.name.endswith("_true_library.csv")]
        json_files = [f for f in lib_path.iterdir() if f.name.endswith("_manifest.json")]
        
        if not csv_files or not json_files:
            # Read the combined library
            combined_csv = lib_path / "combined_library.csv"
            if combined_csv.exists():
                df = pd.read_csv(combined_csv)
                
                # Create _true_library.csv in the expected format (id, sequence)
                true_lib_csv = lib_path / f"{lib_path.name}_true_library.csv"
                true_lib_df = df[['seq_id', 'seq']].rename(columns={'seq_id': 'id', 'seq': 'sequence'})
                true_lib_df.to_csv(true_lib_csv, index=False)
                
                # Create _manifest.json with basic info
                manifest = {
                    "seq_type": "dna",  # Assume DNA for now
                    "n_sequences": len(df),
                    "backbone_length": len(df['seq'].iloc[0]) if len(df) > 0 else 39
                }
                manifest_json = lib_path / f"{lib_path.name}_manifest.json"
                with open(manifest_json, 'w') as f:
                    import json
                    json.dump(manifest, f, indent=2)
                
        
        # Simulate reads from library
        reads_dir = ngs.simulate_reads(
            library_dir=request.library_dir,
            reads_total=request.reads_per_pool,
            p_error=request.error_rate,
            seed_sim=request.seed,
            abundance_mode="uniform",  # Use uniform since selection already set frequencies
            lognormal_sigma=1.0
        )
        
        
        # Read coverage data
        coverage_file = Path(reads_dir) / "coverage_per_source.csv"
        if coverage_file.exists():
            coverage_df = pd.read_csv(coverage_file)
            coverage_stats = {
                "unique_sequences": len(coverage_df),
                "total_reads": int(coverage_df['reads'].sum()),
                "average_coverage": float(coverage_df['reads'].mean()),
                "max_coverage": int(coverage_df['reads'].max()),
                "min_coverage": int(coverage_df['reads'].min())
            }
        else:
            coverage_stats = {}
        
        return {
            "success": True,
            "reads_dir": reads_dir,
            "coverage_stats": coverage_stats
        }
        
    except Exception as e:
        import traceback
        print(f"❌ NGS ERROR: {str(e)}")
        print(f"❌ NGS TRACEBACK:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"NGS simulation failed: {str(e)}")

@app.post("/api/analysis/run")
async def run_analysis(request: AnalysisRequest):
    """Run enrichment analysis"""
    try:
        # Read library data
        library_df = pd.read_csv(request.library_csv)
        
        # Standardize column names
        if 'id' in library_df.columns and 'seq_id' not in library_df.columns:
            library_df['seq_id'] = library_df['id']
        if 'seq' not in library_df.columns and 'sequence' in library_df.columns:
            library_df['seq'] = library_df['sequence']
        
        # Pre-selection: uniform distribution (each sequence has equal frequency)
        total_sequences = len(library_df)
        pre_frequency = 1.0 / total_sequences  # uniform distribution
        
        # Post-selection: read the final round results
        selection_path = Path(request.selection_dir)
        
        # Find the final round file (highest numbered round)
        round_files = list(selection_path.glob("round_*_pool.csv"))
        if not round_files:
            raise ValueError(f"No round files found in {request.selection_dir}")
        
        # Get the last round (highest number)
        final_round_file = max(round_files, key=lambda x: int(x.stem.split('_')[1]))
        final_round_df = pd.read_csv(final_round_file)
        
        # Merge library with final round results (include all selection data)
        selection_cols = ['seq_id', 'frequency', 'KD_summary', 'capture_prob']
        # Add backbone_id if available in selection results
        if 'backbone_id' in final_round_df.columns:
            selection_cols.append('backbone_id')
        # Add any KD target columns
        kd_cols = [col for col in final_round_df.columns if col.startswith('KD_') and col != 'KD_summary']
        selection_cols.extend(kd_cols)
        
        enrichment_df = library_df.merge(
            final_round_df[selection_cols], 
            on='seq_id', 
            how='left',
            suffixes=('_lib', '_sel')
        )
        
        # Debug: Print column information
        print(f"Library columns: {library_df.columns.tolist()}")
        print(f"Selection columns: {final_round_df.columns.tolist()}")
        print(f"Enrichment columns after merge: {enrichment_df.columns.tolist()}")
        
        # Fix backbone_id column - use selection data (more complete)
        if 'backbone_id_sel' in enrichment_df.columns:
            enrichment_df['backbone_id'] = enrichment_df['backbone_id_sel']
            # Drop the suffixed columns
            enrichment_df = enrichment_df.drop(columns=['backbone_id_lib', 'backbone_id_sel'], errors='ignore')
        elif 'backbone_id_lib' in enrichment_df.columns:
            enrichment_df['backbone_id'] = enrichment_df['backbone_id_lib']
            enrichment_df = enrichment_df.drop(columns=['backbone_id_lib'], errors='ignore')
        elif 'backbone_id' not in enrichment_df.columns:
            # If no backbone_id available anywhere, create placeholder
            enrichment_df['backbone_id'] = 'unknown'
            
        print(f"First few backbone_id values: {enrichment_df['backbone_id'].head().tolist() if 'backbone_id' in enrichment_df.columns else 'No backbone_id column'}")
        
        # Fill missing sequences with 0 frequency (they were eliminated)
        enrichment_df['frequency'] = enrichment_df['frequency'].fillna(0)
        enrichment_df['KD_summary'] = enrichment_df['KD_summary'].fillna(1e6)  # high KD for eliminated
        enrichment_df['capture_prob'] = enrichment_df['capture_prob'].fillna(0)
        
        # Calculate enrichment metrics
        enrichment_df['pre_frequency'] = pre_frequency
        enrichment_df['post_frequency'] = enrichment_df['frequency']
        enrichment_df['enrichment'] = (enrichment_df['post_frequency'] + 1e-10) / (enrichment_df['pre_frequency'] + 1e-10)
        
        # Calculate Log2 Fold Change
        enrichment_df['log2_fc'] = np.log2(enrichment_df['enrichment'])
        
        # Calculate P-values using binomial test approximation
        # For simplicity, use chi-square test based on expected vs observed frequencies
        import scipy.stats as stats
        total_reads = 1000000  # Assume 1M total reads for p-value calculation
        enrichment_df['pre_reads'] = enrichment_df['pre_frequency'] * total_reads
        enrichment_df['post_reads'] = enrichment_df['post_frequency'] * total_reads
        
        # One-tailed enrichment test for each variant
        p_values = []
        for _, row in enrichment_df.iterrows():
            expected = row['pre_reads']
            observed = row['post_reads']
            if expected > 0:
                # One-tailed test: is observed significantly GREATER than expected?
                if observed > expected:
                    # Use chi-square for enrichment (observed > expected)
                    chi2_stat = ((observed - expected) ** 2) / expected
                    p_val = 1 - stats.chi2.cdf(chi2_stat, df=1)
                else:
                    # If depleted (observed <= expected), p-value should be high (not significant)
                    p_val = 1.0
            else:
                p_val = 1.0
            p_values.append(p_val)
        
        enrichment_df['p_value'] = p_values
        
        # Debug p-values
        print(f"P-values calculated: {len(p_values)}")
        print(f"Sample p-values: {p_values[:5]}")
        print(f"P-value column in enrichment_df: {'p_value' in enrichment_df.columns}")
        print(f"Sample enrichment_df p_values: {enrichment_df['p_value'].head().tolist()}")
        
        # Sort by enrichment and add rank
        enrichment_df = enrichment_df.sort_values('enrichment', ascending=False).reset_index(drop=True)
        enrichment_df['rank'] = range(1, len(enrichment_df) + 1)
        
        # Get top N results with all calculated fields
        top_results = enrichment_df.head(request.top_n)
        
        # Prepare result columns - always include backbone_id
        result_cols = ['seq_id', 'seq', 'pre_frequency', 'post_frequency', 'enrichment', 
                      'log2_fc', 'p_value', 'KD_summary', 'rank', 'pre_reads', 'post_reads', 'backbone_id']
        
        # Select only available columns
        available_cols = [col for col in result_cols if col in enrichment_df.columns]
        
        return {
            "success": True,
            "total_sequences": len(enrichment_df),
            "final_round_file": str(final_round_file),
            "sequences_surviving": int((enrichment_df['post_frequency'] > 0).sum()),
            "top_results": top_results[available_cols].to_dict('records'),
            "enrichment_stats": {
                "max_enrichment": float(enrichment_df['enrichment'].max()),
                "min_enrichment": float(enrichment_df['enrichment'].min()),
                "mean_enrichment": float(enrichment_df['enrichment'].mean()),
                "median_enrichment": float(enrichment_df['enrichment'].median())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/ai/ask", response_model=AIResponse)
async def ai_ask(request: AIAskRequest):
    """AI Assistant endpoint that returns structured, clean responses"""
    
    def generate_clean_response(message: str, code_context=None, current_step=None) -> AIResponse:
        """Generate a clean, structured AI response"""
        
        # Determine context info
        context_file = "Interface"
        context_selection = "General"
        
        if code_context:
            context_file = code_context.get('file', 'Unknown')
            context_selection = code_context.get('code', '')[:30] + "..." if len(code_context.get('code', '')) > 30 else code_context.get('code', '')
        
        # Mock data status (in real implementation, get from session/database)
        data_status = AIDataStatus(
            library=True,   # Would check actual state
            selection=True,
            ngs=False,
            analysis=False
        )
        
        # Code-specific responses
        if code_context:
            code = code_context.get('code', '').lower()
            
            if 'enrichment' in code:
                return AIResponse(
                    summary="Calculates enrichment ratios by comparing post-selection to pre-selection frequencies for variant analysis.",
                    keyTakeaways=[
                        "Measures variant enrichment or depletion during selection",
                        "Calculates fold-change: post_frequency / pre_frequency", 
                        "Applies log2 transformation for better visualization",
                        "Handles edge cases where frequencies are zero"
                    ],
                    details="This calculation is central to understanding which variants performed better or worse during the selection process. The log2 transformation helps normalize the data and makes fold-changes easier to interpret visually. Values greater than 0 indicate enrichment, while values less than 0 indicate depletion.\n\nThe algorithm also includes safeguards for edge cases, such as when pre-selection frequencies are zero or very small, which could cause division errors or unrealistic fold-change values.",
                    context=AIContextInfo(file=context_file, selection=context_selection),
                    dataStatus=data_status,
                    sources=[
                        AISource(file="analyzer.py", path="ai-ngs-analysis/src/seqlib/analyzer.py", range="45-67")
                    ],
                    quickActions=["Show example", "View code", "Related functions"]
                )
            
            elif 'p_value' in code or 'p-value' in code:
                return AIResponse(
                    summary="Statistical test using one-tailed chi-square to determine if observed enrichment is significant.",
                    keyTakeaways=[
                        "Uses one-tailed chi-square test for significance",
                        "H₀: No enrichment (observed ≤ expected)",
                        "H₁: Significant enrichment (observed > expected)",
                        "p < 0.05 suggests statistically significant enrichment"
                    ],
                    details="The chi-square test calculates: χ² = (observed - expected)² / expected\n\nThe p-value is derived from the chi-square distribution. If observed counts are less than or equal to expected counts, the p-value is set to 1.0, indicating no evidence of enrichment.\n\nThis approach is specifically designed for enrichment analysis where we're testing for positive selection effects.",
                    context=AIContextInfo(file=context_file, selection=context_selection),
                    dataStatus=data_status,
                    sources=[
                        AISource(file="analyzer.py", path="ai-ngs-analysis/src/seqlib/analyzer.py", range="120-145")
                    ],
                    quickActions=["Show calculation", "Statistical details", "More about chi-square"]
                )
            
            elif 'kd' in code:
                return AIResponse(
                    summary="Calculates dissociation constant (KD) using Langmuir inversion from binding simulation data.",
                    keyTakeaways=[
                        "Formula: KD = C × (1 - θ) / θ",
                        "C: ligand concentration (simulation parameter)",
                        "θ: capture probability (from binding simulation)",
                        "Lower KD indicates stronger binding affinity"
                    ],
                    details="The KD value represents the concentration at which 50% binding occupancy occurs. This calculation uses the Langmuir binding model, which assumes:\n\n- Single binding site per molecule\n- No cooperative binding effects\n- Equilibrium conditions\n\nThe capture probability θ comes from the selection simulation, while the ligand concentration C is a simulation parameter that can be adjusted.",
                    context=AIContextInfo(file=context_file, selection=context_selection),
                    dataStatus=data_status,
                    sources=[
                        AISource(file="selection.py", path="ai-ngs-analysis/src/seqlib/selection.py", range="200-225")
                    ],
                    quickActions=["Binding theory", "Show parameters", "Related metrics"]
                )
            
            else:
                # Generic code explanation
                return AIResponse(
                    summary=f"Code from {context_file} handles data processing and calculations for the NGS analysis pipeline.",
                    keyTakeaways=[
                        "Part of the core data processing workflow",
                        "Handles calculations for downstream analysis",
                        "Integrates with the backend API system",
                        "Supports the visualization pipeline"
                    ],
                    details="This code snippet is part of the larger NGS analysis system that processes sequencing data through multiple stages. The pipeline includes library generation, selection simulation, NGS simulation, and statistical analysis.\n\nThe component you've selected plays a role in transforming raw data into meaningful insights that can be visualized and interpreted by researchers.",
                    context=AIContextInfo(file=context_file, selection=context_selection),
                    dataStatus=data_status,
                    sources=[
                        AISource(file=context_file, path=f"components/{context_file}", range="1-50")
                    ],
                    quickActions=["Show workflow", "Related components", "Data flow"]
                )
        
        # General responses based on message content
        if any(word in message.lower() for word in ['plot', 'chart', 'visualiz', 'graph']):
            return AIResponse(
                summary="I can generate custom visualizations from your NGS analysis data with corresponding TSX code.",
                keyTakeaways=[
                    "Creates interactive charts using Recharts library",
                    "Generates production-ready TSX components", 
                    "Supports multiple plot types (scatter, bar, line, pie)",
                    "Includes proper data transformation and styling"
                ],
                details="The plot generation system can create various types of visualizations:\n\n- Enrichment vs p-value scatter plots\n- Top variants bar charts\n- KD distribution histograms\n- Selection round comparisons\n\nEach plot comes with complete TSX source code that you can copy and use in your own components.",
                context=AIContextInfo(file="AI Assistant", selection="Plot generation"),
                dataStatus=data_status,
                sources=[
                    AISource(file="AIAssistant.tsx", path="components/AIAssistant.tsx", range="300-450")
                ],
                quickActions=["Create scatter plot", "Show bar chart", "Generate TSX code"]
            )
        
        # Default response
        return AIResponse(
            summary="Your NGS analysis pipeline processes sequence data through library generation, selection simulation, and statistical analysis.",
            keyTakeaways=[
                "Pipeline includes four main stages: Library, Selection, NGS, Analysis",
                "Each stage builds on the previous one's output",
                "Statistical methods ensure reliable enrichment detection",
                "Results include both individual variants and summary statistics"
            ],
            details="The analysis workflow is designed to simulate realistic experimental conditions:\n\n1. Library Generation: Creates diverse sequence variants with specified degeneracy\n2. Selection Simulation: Models binding-based enrichment using biophysical parameters\n3. NGS Simulation: Adds realistic sequencing artifacts and coverage patterns\n4. Analysis: Calculates statistical significance and ranks variants\n\nThis approach helps researchers understand which sequences might perform best in actual experiments.",
            context=AIContextInfo(file="Analysis Pipeline", selection="Overview"),
            dataStatus=data_status,
            sources=[
                AISource(file="pipeline.py", path="backend/pipeline.py", range="1-100")
            ],
            quickActions=["Show pipeline", "Explain workflow", "View results"]
        )
    
    try:
        response = generate_clean_response(
            request.message, 
            request.codeSelection, 
            request.currentStep
        )
        return response
    
    except Exception as e:
        # Return error response in the same format
        return AIResponse(
            summary=f"I encountered an error processing your request: {str(e)}",
            keyTakeaways=["Please try rephrasing your question", "Check if all required data is available"],
            context=AIContextInfo(file="Error", selection="System error"),
            dataStatus=AIDataStatus(library=False, selection=False, ngs=False, analysis=False),
            sources=[],
            quickActions=["Try again", "Contact support"]
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)