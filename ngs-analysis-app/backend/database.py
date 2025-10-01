"""
Database configuration and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import os
from models import Base
import structlog

logger = structlog.get_logger()

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./ngs_chat.db"  # Default to SQLite for development
)

# For production, use PostgreSQL:
# DATABASE_URL = "postgresql://user:password@localhost/ngs_chat"

# Create engine
if DATABASE_URL.startswith("sqlite"):
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=0,
        pool_pre_ping=True,
        echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("database_tables_created")
    except Exception as e:
        logger.error("failed_to_create_tables", error=str(e))
        raise

def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database with default data"""
    from models import RateLimitRule, UserRole
    
    create_tables()
    
    # Create default rate limit rules
    db = SessionLocal()
    try:
        # Check if rules already exist
        existing_rules = db.query(RateLimitRule).count()
        if existing_rules == 0:
            default_rules = [
                RateLimitRule(
                    role=UserRole.GUEST,
                    messages_per_hour=10,
                    messages_per_day=50,
                    tokens_per_month=10000,
                    max_tokens_per_request=1000,
                    allowed_models=["gpt-3.5-turbo"]
                ),
                RateLimitRule(
                    role=UserRole.VIEWER,
                    messages_per_hour=50,
                    messages_per_day=200,
                    tokens_per_month=50000,
                    max_tokens_per_request=2000,
                    allowed_models=["gpt-3.5-turbo", "gpt-4-turbo-preview"]
                ),
                RateLimitRule(
                    role=UserRole.RESEARCHER,
                    messages_per_hour=100,
                    messages_per_day=500,
                    tokens_per_month=200000,
                    max_tokens_per_request=4000,
                    allowed_models=["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4"]
                ),
                RateLimitRule(
                    role=UserRole.ADMIN,
                    messages_per_hour=500,
                    messages_per_day=2000,
                    tokens_per_month=1000000,
                    max_tokens_per_request=8000,
                    allowed_models=["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4", "gpt-4-32k"]
                )
            ]
            
            for rule in default_rules:
                db.add(rule)
            
            db.commit()
            logger.info("default_rate_limit_rules_created", count=len(default_rules))
        
    except Exception as e:
        logger.error("failed_to_init_database", error=str(e))
        db.rollback()
        raise
    finally:
        db.close()
