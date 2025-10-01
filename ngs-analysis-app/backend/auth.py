"""
Authentication and authorization system with SSO support
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from models import User, UserRole, AuditLog
from database import get_db
import httpx
import structlog
import os

logger = structlog.get_logger()

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# SSO Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID")
OKTA_DOMAIN = os.getenv("OKTA_DOMAIN")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class AuthenticationError(Exception):
    pass

class AuthorizationError(Exception):
    pass

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Invalid token")
        return payload
    except JWTError:
        raise AuthenticationError("Invalid token")

async def get_current_user(
    request: Request,
    token_payload: dict = Depends(verify_token),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    user_id = token_payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    
    if user is None:
        await log_audit_event(
            db, None, "authentication_failed", 
            details={"reason": "user_not_found", "user_id": user_id},
            request=request, success=False
        )
        raise AuthenticationError("User not found")
    
    if not user.is_active:
        await log_audit_event(
            db, user.id, "authentication_failed",
            details={"reason": "user_inactive"},
            request=request, success=False
        )
        raise AuthenticationError("User account is inactive")
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    return user

def require_role(required_role: UserRole):
    """Decorator to require specific user role"""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.VIEWER: 1,
            UserRole.RESEARCHER: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(required_role, 3)
        
        if user_level < required_level:
            raise AuthorizationError(f"Insufficient permissions. Required: {required_role}")
        
        return current_user
    
    return role_checker

async def verify_google_token(token: str) -> Dict[str, Any]:
    """Verify Google OAuth token"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://www.googleapis.com/oauth2/v1/tokeninfo?access_token={token}"
        )
        if response.status_code != 200:
            raise AuthenticationError("Invalid Google token")
        
        token_info = response.json()
        if token_info.get("audience") != GOOGLE_CLIENT_ID:
            raise AuthenticationError("Invalid Google token audience")
        
        # Get user info
        user_response = await client.get(
            f"https://www.googleapis.com/oauth2/v2/userinfo?access_token={token}"
        )
        if user_response.status_code != 200:
            raise AuthenticationError("Failed to get Google user info")
        
        return user_response.json()

async def verify_microsoft_token(token: str) -> Dict[str, Any]:
    """Verify Microsoft OAuth token"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://graph.microsoft.com/v1.0/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.status_code != 200:
            raise AuthenticationError("Invalid Microsoft token")
        
        return response.json()

async def create_or_update_user(
    db: Session, 
    sso_data: Dict[str, Any], 
    provider: str
) -> User:
    """Create or update user from SSO data"""
    email = sso_data.get("email")
    if not email:
        raise AuthenticationError("Email not provided by SSO provider")
    
    # Check if user exists
    user = db.query(User).filter(User.email == email).first()
    
    if user:
        # Update existing user
        user.full_name = sso_data.get("name", user.full_name)
        user.last_login = datetime.utcnow()
        user.sso_provider = provider
        user.sso_user_id = sso_data.get("id", sso_data.get("sub"))
    else:
        # Create new user
        user = User(
            email=email,
            full_name=sso_data.get("name", ""),
            role=UserRole.VIEWER,  # Default role
            sso_provider=provider,
            sso_user_id=sso_data.get("id", sso_data.get("sub")),
            organization=sso_data.get("hd"),  # Google hosted domain
            is_active=True,
            last_login=datetime.utcnow()
        )
        db.add(user)
    
    db.commit()
    db.refresh(user)
    return user

async def log_audit_event(
    db: Session,
    user_id: Optional[str],
    action: str,
    resource: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None,
    success: bool = True,
    error_message: Optional[str] = None
):
    """Log audit event"""
    audit_log = AuditLog(
        user_id=user_id,
        action=action,
        resource=resource,
        details=details,
        ip_address=request.client.host if request else None,
        user_agent=request.headers.get("user-agent") if request else None,
        success=success,
        error_message=error_message
    )
    
    db.add(audit_log)
    db.commit()
    
    # Also log to structured logger
    logger.info(
        "audit_event",
        user_id=user_id,
        action=action,
        resource=resource,
        success=success,
        error_message=error_message,
        **details if details else {}
    )

# Exception handlers
def auth_exception_handler(request: Request, exc: AuthenticationError):
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=str(exc),
        headers={"WWW-Authenticate": "Bearer"},
    )

def authz_exception_handler(request: Request, exc: AuthorizationError):
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=str(exc),
    )
