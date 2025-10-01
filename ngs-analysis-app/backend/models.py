"""
Database models for the enterprise chat assistant
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid

Base = declarative_base()

class UserRole(str, Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    VIEWER = "viewer"
    GUEST = "guest"

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    role = Column(String, nullable=False, default=UserRole.VIEWER)
    is_active = Column(Boolean, default=True)
    sso_provider = Column(String, nullable=False)  # google, microsoft, okta, etc.
    sso_user_id = Column(String, nullable=False)
    organization = Column(String, nullable=True)
    department = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Rate limiting fields
    daily_message_count = Column(Integer, default=0)
    monthly_token_usage = Column(Integer, default=0)
    last_reset_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user")
    chat_messages = relationship("ChatMessage", back_populates="user")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Session metadata
    total_messages = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    context_type = Column(String, nullable=True)  # library, selection, ngs, analysis
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Message content
    role = Column(String, nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    content_redacted = Column(Text, nullable=True)  # PII-redacted version for logging
    
    # Metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    model_used = Column(String, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    
    # Context and retrieval
    context_data = Column(JSON, nullable=True)  # NGS analysis context
    retrieved_documents = Column(JSON, nullable=True)  # RAG documents used
    
    # Security and compliance
    contains_pii = Column(Boolean, default=False)
    pii_entities = Column(JSON, nullable=True)  # Detected PII entities
    
    # Performance metrics
    response_time_ms = Column(Float, nullable=True)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    user = relationship("User", back_populates="chat_messages")

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)  # login, logout, chat_message, admin_action
    resource = Column(String, nullable=True)  # session_id, user_id, etc.
    details = Column(JSON, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    success = Column(Boolean, default=True)
    error_message = Column(String, nullable=True)

class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    document_type = Column(String, nullable=False)  # manual, faq, protocol, paper
    source_file = Column(String, nullable=True)
    
    # Vector search
    embedding_model = Column(String, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String, ForeignKey("users.id"), nullable=True)
    
    # Access control
    required_role = Column(String, default=UserRole.VIEWER)
    is_active = Column(Boolean, default=True)

class RateLimitRule(Base):
    __tablename__ = "rate_limit_rules"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    role = Column(String, nullable=False)
    
    # Limits
    messages_per_hour = Column(Integer, default=100)
    messages_per_day = Column(Integer, default=500)
    tokens_per_month = Column(Integer, default=100000)
    
    # Model restrictions
    allowed_models = Column(JSON, nullable=True)  # List of allowed OpenAI models
    max_tokens_per_request = Column(Integer, default=4000)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
