"""
Rate limiting system for chat API
"""
from typing import Dict, Optional
from datetime import datetime, timedelta
import redis
from fastapi import HTTPException, Request
from models import User, UserRole, RateLimitRule
from sqlalchemy.orm import Session
from security import security_logger
import structlog
import json
import os

logger = structlog.get_logger()

class RateLimiter:
    """Redis-based rate limiter with role-based limits"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
        except Exception as e:
            logger.warning("redis_connection_failed", error=str(e))
            self.redis_client = None
    
    def check_rate_limit(
        self, 
        user: User, 
        db: Session,
        request_type: str = "message"
    ) -> Dict[str, any]:
        """
        Check if user has exceeded rate limits
        Returns: {allowed: bool, remaining: int, reset_time: datetime, limit_type: str}
        """
        if not self.redis_client:
            # Fallback to database-only rate limiting
            return self._check_db_rate_limit(user, db, request_type)
        
        # Get rate limit rules for user role
        rules = self._get_rate_limit_rules(user.role, db)
        
        current_time = datetime.utcnow()
        user_key = f"rate_limit:{user.id}"
        
        # Check hourly limit
        hourly_result = self._check_time_window(
            user_key, "hourly", rules["messages_per_hour"], 3600, current_time
        )
        
        if not hourly_result["allowed"]:
            security_logger.log_rate_limit_exceeded(
                user.id, "hourly", hourly_result["current"], rules["messages_per_hour"]
            )
            return hourly_result
        
        # Check daily limit
        daily_result = self._check_time_window(
            user_key, "daily", rules["messages_per_day"], 86400, current_time
        )
        
        if not daily_result["allowed"]:
            security_logger.log_rate_limit_exceeded(
                user.id, "daily", daily_result["current"], rules["messages_per_day"]
            )
            return daily_result
        
        # Check monthly token limit (stored in database)
        monthly_result = self._check_monthly_tokens(user, db, rules["tokens_per_month"])
        
        if not monthly_result["allowed"]:
            security_logger.log_rate_limit_exceeded(
                user.id, "monthly_tokens", monthly_result["current"], rules["tokens_per_month"]
            )
            return monthly_result
        
        # All checks passed
        return {
            "allowed": True,
            "remaining": min(hourly_result["remaining"], daily_result["remaining"]),
            "reset_time": min(hourly_result["reset_time"], daily_result["reset_time"]),
            "limit_type": "none"
        }
    
    def _check_time_window(
        self, 
        user_key: str, 
        window_type: str, 
        limit: int, 
        window_seconds: int,
        current_time: datetime
    ) -> Dict[str, any]:
        """Check rate limit for a specific time window"""
        window_key = f"{user_key}:{window_type}"
        
        try:
            # Use Redis sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            cutoff_time = current_time - timedelta(seconds=window_seconds)
            pipe.zremrangebyscore(window_key, 0, cutoff_time.timestamp())
            
            # Count current requests
            pipe.zcard(window_key)
            
            # Add current request
            pipe.zadd(window_key, {str(current_time.timestamp()): current_time.timestamp()})
            
            # Set expiration
            pipe.expire(window_key, window_seconds)
            
            results = pipe.execute()
            current_count = results[1]
            
            if current_count >= limit:
                # Remove the request we just added since it's not allowed
                self.redis_client.zrem(window_key, str(current_time.timestamp()))
                
                # Calculate reset time
                oldest_request = self.redis_client.zrange(window_key, 0, 0, withscores=True)
                if oldest_request:
                    reset_time = datetime.fromtimestamp(oldest_request[0][1]) + timedelta(seconds=window_seconds)
                else:
                    reset_time = current_time + timedelta(seconds=window_seconds)
                
                return {
                    "allowed": False,
                    "remaining": 0,
                    "current": current_count,
                    "limit": limit,
                    "reset_time": reset_time,
                    "limit_type": window_type
                }
            
            return {
                "allowed": True,
                "remaining": limit - current_count - 1,
                "current": current_count + 1,
                "limit": limit,
                "reset_time": current_time + timedelta(seconds=window_seconds),
                "limit_type": window_type
            }
            
        except Exception as e:
            logger.error("rate_limit_check_failed", error=str(e))
            # Fail open - allow request if Redis is down
            return {
                "allowed": True,
                "remaining": limit,
                "current": 0,
                "limit": limit,
                "reset_time": current_time + timedelta(seconds=window_seconds),
                "limit_type": window_type
            }
    
    def _check_monthly_tokens(self, user: User, db: Session, token_limit: int) -> Dict[str, any]:
        """Check monthly token usage limit"""
        current_time = datetime.utcnow()
        
        # Reset monthly counter if needed
        if user.last_reset_date is None or \
           (current_time - user.last_reset_date).days >= 30:
            user.monthly_token_usage = 0
            user.last_reset_date = current_time
            db.commit()
        
        if user.monthly_token_usage >= token_limit:
            reset_time = user.last_reset_date + timedelta(days=30)
            return {
                "allowed": False,
                "remaining": 0,
                "current": user.monthly_token_usage,
                "limit": token_limit,
                "reset_time": reset_time,
                "limit_type": "monthly_tokens"
            }
        
        return {
            "allowed": True,
            "remaining": token_limit - user.monthly_token_usage,
            "current": user.monthly_token_usage,
            "limit": token_limit,
            "reset_time": user.last_reset_date + timedelta(days=30),
            "limit_type": "monthly_tokens"
        }
    
    def _check_db_rate_limit(self, user: User, db: Session, request_type: str) -> Dict[str, any]:
        """Fallback database-only rate limiting"""
        current_time = datetime.utcnow()
        
        # Reset daily counter if needed
        if user.last_reset_date is None or \
           (current_time - user.last_reset_date).days >= 1:
            user.daily_message_count = 0
            user.last_reset_date = current_time
            db.commit()
        
        rules = self._get_rate_limit_rules(user.role, db)
        daily_limit = rules["messages_per_day"]
        
        if user.daily_message_count >= daily_limit:
            reset_time = user.last_reset_date + timedelta(days=1)
            return {
                "allowed": False,
                "remaining": 0,
                "current": user.daily_message_count,
                "limit": daily_limit,
                "reset_time": reset_time,
                "limit_type": "daily"
            }
        
        return {
            "allowed": True,
            "remaining": daily_limit - user.daily_message_count,
            "current": user.daily_message_count,
            "limit": daily_limit,
            "reset_time": user.last_reset_date + timedelta(days=1),
            "limit_type": "daily"
        }
    
    def _get_rate_limit_rules(self, role: UserRole, db: Session) -> Dict[str, int]:
        """Get rate limit rules for user role"""
        # Try to get from database first
        rule = db.query(RateLimitRule).filter(
            RateLimitRule.role == role,
            RateLimitRule.is_active == True
        ).first()
        
        if rule:
            return {
                "messages_per_hour": rule.messages_per_hour,
                "messages_per_day": rule.messages_per_day,
                "tokens_per_month": rule.tokens_per_month,
                "max_tokens_per_request": rule.max_tokens_per_request,
                "allowed_models": rule.allowed_models or ["gpt-4-turbo-preview"]
            }
        
        # Default limits by role
        default_limits = {
            UserRole.GUEST: {
                "messages_per_hour": 10,
                "messages_per_day": 50,
                "tokens_per_month": 10000,
                "max_tokens_per_request": 1000,
                "allowed_models": ["gpt-3.5-turbo"]
            },
            UserRole.VIEWER: {
                "messages_per_hour": 50,
                "messages_per_day": 200,
                "tokens_per_month": 50000,
                "max_tokens_per_request": 2000,
                "allowed_models": ["gpt-3.5-turbo", "gpt-4-turbo-preview"]
            },
            UserRole.RESEARCHER: {
                "messages_per_hour": 100,
                "messages_per_day": 500,
                "tokens_per_month": 200000,
                "max_tokens_per_request": 4000,
                "allowed_models": ["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4"]
            },
            UserRole.ADMIN: {
                "messages_per_hour": 500,
                "messages_per_day": 2000,
                "tokens_per_month": 1000000,
                "max_tokens_per_request": 8000,
                "allowed_models": ["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4", "gpt-4-32k"]
            }
        }
        
        return default_limits.get(role, default_limits[UserRole.VIEWER])
    
    def increment_usage(self, user: User, db: Session, tokens_used: int = 0):
        """Increment user's usage counters"""
        # Update database counters
        user.daily_message_count += 1
        user.monthly_token_usage += tokens_used
        db.commit()
        
        logger.info(
            "usage_incremented",
            user_id=user.id,
            daily_messages=user.daily_message_count,
            monthly_tokens=user.monthly_token_usage,
            tokens_added=tokens_used
        )
    
    def get_usage_stats(self, user: User, db: Session) -> Dict[str, any]:
        """Get current usage statistics for user"""
        rules = self._get_rate_limit_rules(user.role, db)
        
        return {
            "daily_messages": {
                "used": user.daily_message_count,
                "limit": rules["messages_per_day"],
                "remaining": max(0, rules["messages_per_day"] - user.daily_message_count)
            },
            "monthly_tokens": {
                "used": user.monthly_token_usage,
                "limit": rules["tokens_per_month"],
                "remaining": max(0, rules["tokens_per_month"] - user.monthly_token_usage)
            },
            "role": user.role,
            "allowed_models": rules["allowed_models"],
            "max_tokens_per_request": rules["max_tokens_per_request"]
        }

def create_rate_limit_exception(limit_result: Dict[str, any]) -> HTTPException:
    """Create appropriate HTTP exception for rate limit"""
    headers = {
        "X-RateLimit-Limit": str(limit_result["limit"]),
        "X-RateLimit-Remaining": str(limit_result["remaining"]),
        "X-RateLimit-Reset": str(int(limit_result["reset_time"].timestamp())),
        "X-RateLimit-Type": limit_result["limit_type"]
    }
    
    if limit_result["limit_type"] == "hourly":
        message = f"Hourly message limit exceeded. Try again in {(limit_result['reset_time'] - datetime.utcnow()).seconds // 60} minutes."
    elif limit_result["limit_type"] == "daily":
        message = f"Daily message limit exceeded. Resets at {limit_result['reset_time'].strftime('%H:%M UTC')}."
    elif limit_result["limit_type"] == "monthly_tokens":
        message = f"Monthly token limit exceeded. Resets on {limit_result['reset_time'].strftime('%Y-%m-%d')}."
    else:
        message = "Rate limit exceeded. Please try again later."
    
    return HTTPException(
        status_code=429,
        detail=message,
        headers=headers
    )

# Global rate limiter instance
rate_limiter = RateLimiter()
