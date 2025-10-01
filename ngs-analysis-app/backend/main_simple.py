from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, constr, conint, confloat
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
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import structlog
from datetime import datetime
import time

# Load environment variables from .env file
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Add the ai-ngs-analysis package to the path
sys.path.insert(0, '/Users/aparnaanandkumar/Documents/aikium/ngs_analysis_tool/ai-ngs-analysis/src')

from seqlib import library_generator as libgen
from seqlib import selection as select
from seqlib import ngs_simulator as ngs
from seqlib import analyzer as analyzer

# Initialize OpenAI client
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

app = FastAPI(
    title="NGS Analysis Chat API", 
    version="2.0.0",
    description="OpenAI-powered assistant for NGS analysis with streaming responses"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:54112"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for chat
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: Optional[Dict[str, Any]] = None
    stream: bool = True

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def build_enhanced_system_context(context: Optional[Dict[str, Any]] = None) -> str:
    """Build enhanced system context for OpenAI"""
    
    base_context = """You are an expert AI assistant specializing in Next-Generation Sequencing (NGS) analysis and protein engineering. You help researchers with:

🧬 **NGS Analysis Pipeline:**
- Library generation with degeneracy patterns and backbone sequences
- Selection simulation with binding affinity modeling
- NGS read simulation with realistic coverage and error patterns  
- Statistical analysis of enrichment and variant identification

🔬 **Technical Expertise:**
- Protein-protein interactions and binding kinetics
- PCR amplification bias and coverage optimization
- Statistical significance testing (p-values, enrichment ratios)
- Experimental design and parameter optimization

📊 **Data Analysis:**
- Sequence diversity calculations and library complexity
- Coverage requirements and statistical power analysis
- Variant calling and enrichment analysis
- Results interpretation and experimental recommendations

**Current Pipeline Context:**"""

    if context:
        current_step = context.get('currentStep', 'setup')
        
        if current_step == 'library':
            base_context += "\n🧪 **Library Generation Phase** - Designing template sequences with degeneracy patterns"
            if context.get('hasLibrary'):
                base_context += "\n✅ Library generated successfully"
        elif current_step == 'selection':
            base_context += "\n🎯 **Selection Phase** - Simulating binding-based selection rounds"
            if context.get('hasSelection'):
                base_context += "\n✅ Selection completed"
        elif current_step == 'ngs':
            base_context += "\n🧬 **NGS Simulation Phase** - Generating realistic sequencing reads"
            if context.get('hasNGS'):
                base_context += "\n✅ NGS simulation completed"
        elif current_step == 'analysis':
            base_context += "\n📊 **Analysis Phase** - Statistical analysis of enrichment patterns"
            if context.get('hasAnalysis'):
                base_context += "\n✅ Analysis completed"
                
                # Add analysis results context
                analysis_data = context.get('analysisData')
                if analysis_data:
                    base_context += f"\n📈 **Current Results:**"
                    base_context += f"\n- Total sequences: {analysis_data.get('totalSequences', 'N/A'):,}"
                    base_context += f"\n- Surviving sequences: {analysis_data.get('survivingSequences', 'N/A'):,}"
                    
                    top_variants = analysis_data.get('topVariants', [])
                    if top_variants:
                        base_context += f"\n- Top variants found: {len(top_variants)}"
                        for i, variant in enumerate(top_variants[:3], 1):
                            enrichment = variant.get('enrichment', 0)
                            p_value = variant.get('p_value', 1)
                            base_context += f"\n  {i}. Enrichment: {enrichment:.2f}x, p-value: {p_value:.2e}"

    base_context += """

**Response Guidelines:**
- Provide specific, actionable advice based on the current pipeline step
- Explain technical concepts clearly with relevant context
- Include quantitative recommendations when appropriate
- Reference best practices in protein engineering and NGS analysis
- Be concise but comprehensive in explanations"""

    return base_context

@app.post("/api/ai/chat")
async def ai_chat_simple(request: ChatRequest):
    """Simplified chat endpoint for frontend compatibility (no auth required for testing)"""
    
    # Validate OpenAI API key
    if not openai_client.api_key:
        raise HTTPException(
            status_code=500, 
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        )
    
    try:
        # Build enhanced system context
        system_context = build_enhanced_system_context(request.context)
        
        # Prepare OpenAI messages
        openai_messages = [{"role": "system", "content": system_context}]
        
        # Add user messages
        for msg in request.messages:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        if request.stream:
            return StreamingResponse(
                stream_simple_response(openai_messages),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Non-streaming response
            response = await openai_client.chat.completions.create(
                    model="gpt-4o",
                messages=openai_messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            return {
                "message": response.choices[0].message.content,
                "usage": response.usage.dict() if response.usage else None
            }
            
    except Exception as e:
        logger.error("simple_chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

async def stream_simple_response(messages: List[Dict[str, str]]):
    """Simple streaming response without full enterprise features"""
    try:
        # Create OpenAI stream
        stream = await openai_client.chat.completions.create(
                    model="gpt-4o",
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1500
        )
        
        # Stream response
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                # Send chunk to client
                yield f"data: {json.dumps({'content': content})}\n\n"
                
        # Send completion signal
        yield f"data: {json.dumps({'done': True})}\n\n"
        
    except Exception as e:
        error_msg = f"AI Error: {str(e)}"
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

# All the existing NGS pipeline endpoints remain the same...
# (I'll include the key ones but truncate for brevity)

class LibraryRequest(BaseModel):
    library_size: conint(ge=1000, le=1000000) = Field(default=50000, description="Number of sequences to generate")
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    backbone_length: conint(ge=10, le=200) = Field(default=39, description="Length of backbone sequences")
    sequence_type: Literal["dna", "protein"] = Field(default="dna", description="Type of sequences to generate")
    degeneracy_positions: List[Dict[str, Any]] = Field(default_factory=list, description="Positions with degeneracy")
    backbones: List[Dict[str, Any]] = Field(default_factory=list, description="Backbone sequences with weights")

@app.post("/api/library/generate")
async def generate_library(request: LibraryRequest):
    """Generate a combinatorial library with degeneracy patterns"""
    try:
        # Create temporary directory for this generation
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        temp_dir = f"/tmp/library_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Set up parameters
        params = {
            'library_size': request.library_size,
            'random_seed': request.random_seed or random.randint(1, 10000),
            'backbone_length': request.backbone_length,
            'sequence_type': request.sequence_type,
            'output_dir': temp_dir
        }
        
        # Process backbones and degeneracy
        backbones_data = []
        for backbone in request.backbones:
            backbones_data.append({
                'sequence': backbone.get('sequence', ''),
                'weight': backbone.get('weight', 1.0)
            })
        
        degeneracy_data = []
        for deg in request.degeneracy_positions:
            degeneracy_data.append({
                'position': deg.get('position', 0),
                'code': deg.get('code', 'N')
            })
        
        # Generate library
        result = libgen.generate_combinatorial_library(
            backbones=backbones_data,
            degeneracy_positions=degeneracy_data,
            **params
        )
        
        return {
            "status": "success",
            "library_csv": result.get("library_csv"),
            "manifest": result.get("manifest"),
            "stats": result.get("stats", {}),
            "output_dir": temp_dir
        }
        
    except Exception as e:
        logger.error("library_generation_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Library generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
