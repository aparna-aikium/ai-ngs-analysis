"""
Retrieval-Augmented Generation system for NGS analysis data
"""
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import os
from pathlib import Path
import structlog
from models import KnowledgeDocument, User, UserRole
from sqlalchemy.orm import Session

logger = structlog.get_logger()

class NGSKnowledgeBase:
    """Knowledge base for NGS analysis information"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create collections
        self.collections = {
            "protocols": self.client.get_or_create_collection("ngs_protocols"),
            "papers": self.client.get_or_create_collection("research_papers"),
            "faqs": self.client.get_or_create_collection("faqs"),
            "manuals": self.client.get_or_create_collection("user_manuals"),
            "analysis_results": self.client.get_or_create_collection("analysis_results")
        }
    
    def add_document(
        self, 
        content: str, 
        doc_type: str, 
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None
    ) -> str:
        """Add document to knowledge base"""
        try:
            if doc_id is None:
                import uuid
                doc_id = str(uuid.uuid4())
            
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            # Add to appropriate collection
            collection = self.collections.get(doc_type, self.collections["manuals"])
            collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info(
                "document_added_to_kb",
                doc_id=doc_id,
                doc_type=doc_type,
                content_length=len(content)
            )
            
            return doc_id
            
        except Exception as e:
            logger.error("failed_to_add_document", error=str(e))
            raise
    
    def search_documents(
        self, 
        query: str, 
        doc_types: Optional[List[str]] = None,
        n_results: int = 5,
        user_role: UserRole = UserRole.VIEWER
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            all_results = []
            collections_to_search = doc_types or list(self.collections.keys())
            
            for doc_type in collections_to_search:
                if doc_type not in self.collections:
                    continue
                
                collection = self.collections[doc_type]
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, 10),
                    include=["documents", "metadatas", "distances"]
                )
                
                # Process results
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    
                    # Check access permissions
                    required_role = metadata.get("required_role", UserRole.VIEWER)
                    if not self._check_access(user_role, required_role):
                        continue
                    
                    all_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "doc_type": doc_type,
                        "similarity": 1 - distance,  # Convert distance to similarity
                        "source": metadata.get("source", "Unknown")
                    })
            
            # Sort by similarity and return top results
            all_results.sort(key=lambda x: x["similarity"], reverse=True)
            return all_results[:n_results]
            
        except Exception as e:
            logger.error("document_search_failed", error=str(e))
            return []
    
    def _check_access(self, user_role: UserRole, required_role: UserRole) -> bool:
        """Check if user has access to document based on role"""
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.VIEWER: 1,
            UserRole.RESEARCHER: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 1)
        
        return user_level >= required_level

class NGSDataRetriever:
    """Retriever for NGS analysis results and experimental data"""
    
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        self.knowledge_base = NGSKnowledgeBase()
    
    def index_analysis_results(self, results_dir: str):
        """Index NGS analysis results for retrieval"""
        results_path = Path(results_dir)
        
        if not results_path.exists():
            logger.warning("results_directory_not_found", path=results_dir)
            return
        
        # Index CSV files with analysis results
        for csv_file in results_path.glob("**/*.csv"):
            try:
                df = pd.read_csv(csv_file)
                
                # Create searchable content from CSV
                content = self._create_searchable_content(df, csv_file.name)
                
                metadata = {
                    "file_path": str(csv_file),
                    "file_name": csv_file.name,
                    "file_type": "analysis_results",
                    "row_count": len(df),
                    "columns": list(df.columns),
                    "required_role": UserRole.VIEWER
                }
                
                self.knowledge_base.add_document(
                    content=content,
                    doc_type="analysis_results",
                    metadata=metadata,
                    doc_id=f"analysis_{csv_file.stem}"
                )
                
            except Exception as e:
                logger.error("failed_to_index_csv", file=str(csv_file), error=str(e))
    
    def _create_searchable_content(self, df: pd.DataFrame, filename: str) -> str:
        """Create searchable text content from DataFrame"""
        content_parts = [
            f"Analysis results from {filename}",
            f"Contains {len(df)} sequences with {len(df.columns)} data columns",
            f"Columns: {', '.join(df.columns)}"
        ]
        
        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col in df.columns:
                stats = df[col].describe()
                content_parts.append(
                    f"{col}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                    f"min={stats['min']:.3f}, max={stats['max']:.3f}"
                )
        
        # Add top sequences if available
        if 'seq_id' in df.columns and 'enrichment' in df.columns:
            top_sequences = df.nlargest(5, 'enrichment')
            content_parts.append("Top enriched sequences:")
            for _, row in top_sequences.iterrows():
                content_parts.append(f"- {row['seq_id']}: enrichment={row['enrichment']:.2f}")
        
        return "\n".join(content_parts)
    
    def get_relevant_context(
        self, 
        query: str, 
        user_role: UserRole,
        current_analysis_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get relevant context for user query"""
        context = {
            "documents": [],
            "current_data": None,
            "suggestions": []
        }
        
        # Search knowledge base
        documents = self.knowledge_base.search_documents(
            query=query,
            user_role=user_role,
            n_results=5
        )
        context["documents"] = documents
        
        # Add current analysis context if available
        if current_analysis_data:
            context["current_data"] = self._format_current_data(current_analysis_data)
        
        # Generate suggestions based on query
        context["suggestions"] = self._generate_suggestions(query, documents)
        
        return context
    
    def _format_current_data(self, analysis_data: Dict[str, Any]) -> str:
        """Format current analysis data for context"""
        if not analysis_data:
            return ""
        
        parts = []
        
        if "total_sequences" in analysis_data:
            parts.append(f"Current analysis contains {analysis_data['total_sequences']} sequences")
        
        if "sequences_surviving" in analysis_data:
            parts.append(f"{analysis_data['sequences_surviving']} sequences survived selection")
        
        if "top_results" in analysis_data and analysis_data["top_results"]:
            top_result = analysis_data["top_results"][0]
            parts.append(f"Top variant: {top_result.get('seq_id', 'Unknown')} with enrichment {top_result.get('enrichment', 'N/A')}")
        
        return ". ".join(parts)
    
    def _generate_suggestions(self, query: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Generate helpful suggestions based on query and retrieved documents"""
        suggestions = []
        
        query_lower = query.lower()
        
        # Query-based suggestions
        if "enrichment" in query_lower:
            suggestions.append("Try asking about statistical significance or p-values")
            suggestions.append("Consider exploring binding affinity (KD) relationships")
        
        if "selection" in query_lower:
            suggestions.append("Ask about stringency parameters and their effects")
            suggestions.append("Explore PCR bias and amplification effects")
        
        if "library" in query_lower:
            suggestions.append("Consider degeneracy patterns and complexity")
            suggestions.append("Ask about computational requirements for different library sizes")
        
        # Document-based suggestions
        doc_types = set(doc["doc_type"] for doc in documents)
        if "protocols" in doc_types:
            suggestions.append("Check the related protocols for experimental details")
        
        if "papers" in doc_types:
            suggestions.append("Review the research papers for theoretical background")
        
        return suggestions[:3]  # Limit to 3 suggestions

def initialize_knowledge_base(db: Session, data_directory: str) -> NGSDataRetriever:
    """Initialize knowledge base with default content"""
    retriever = NGSDataRetriever(data_directory)
    
    # Add default NGS knowledge
    default_docs = [
        {
            "content": """
            NGS Library Generation Best Practices:
            
            1. Library Complexity: Ensure sufficient diversity to avoid PCR bias
            2. Degeneracy Design: Use appropriate IUPAC codes (N, R, Y, M, K, S, W, B, D, H, V)
            3. Quality Control: Verify library distribution before selection
            4. Computational Requirements: 10^4 sequences = ~500MB, 10^5 = ~5GB, 10^6 = ~50GB
            5. Coverage Planning: Aim for 100x coverage minimum for statistical power
            """,
            "doc_type": "protocols",
            "metadata": {
                "title": "NGS Library Generation Best Practices",
                "source": "Internal Protocol",
                "required_role": UserRole.VIEWER
            }
        },
        {
            "content": """
            Selection Simulation Parameters:
            
            - Stringency (S): Controls selection pressure (0.1 = low, 0.9 = high)
            - Target Concentration: Affects binding equilibrium (typically 1-100 nM)
            - KD Model: μ_log10KD = -7.0 ± 0.3 for typical antibody-antigen interactions
            - PCR Cycles: 10-15 cycles typical, higher cycles increase bias
            - Expression Sigma: 0.5 represents moderate expression variation
            """,
            "doc_type": "protocols",
            "metadata": {
                "title": "Selection Simulation Parameters",
                "source": "User Manual",
                "required_role": UserRole.RESEARCHER
            }
        },
        {
            "content": """
            Statistical Analysis Interpretation:
            
            - Enrichment Ratio: post_frequency / pre_frequency
            - Log2 Fold Change: log2(enrichment_ratio)
            - P-value: Chi-square test for significance (p < 0.05 = significant)
            - KD Calculation: Langmuir model inversion from capture probability
            - False Discovery Rate: Consider multiple testing correction
            """,
            "doc_type": "manuals",
            "metadata": {
                "title": "Statistical Analysis Guide",
                "source": "Analysis Manual",
                "required_role": UserRole.VIEWER
            }
        }
    ]
    
    for doc in default_docs:
        retriever.knowledge_base.add_document(
            content=doc["content"],
            doc_type=doc["doc_type"],
            metadata=doc["metadata"]
        )
    
    # Index existing analysis results
    analysis_dirs = [
        "data/libraries",
        "data/selection", 
        "data/reads"
    ]
    
    for dir_path in analysis_dirs:
        full_path = Path(data_directory) / dir_path
        if full_path.exists():
            retriever.index_analysis_results(str(full_path))
    
    logger.info("knowledge_base_initialized", doc_count=len(default_docs))
    return retriever

# Global retriever instance
_retriever: Optional[NGSDataRetriever] = None

def get_retriever() -> NGSDataRetriever:
    """Get global retriever instance"""
    global _retriever
    if _retriever is None:
        data_dir = os.getenv("NGS_DATA_DIRECTORY", "./data")
        _retriever = NGSDataRetriever(data_dir)
    return _retriever
