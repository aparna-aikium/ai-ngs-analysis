"""
Security utilities including PII detection and content filtering
"""
from typing import List, Dict, Any, Tuple
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
import re
import structlog

logger = structlog.get_logger()

class PIIDetector:
    """PII detection and anonymization using Microsoft Presidio"""
    
    def __init__(self):
        # Initialize Presidio engines
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Custom patterns for scientific/research data
        self.custom_patterns = {
            "SAMPLE_ID": r"\b[A-Z]{2,4}\d{4,8}\b",  # Sample IDs like AB123456
            "SEQUENCE_ID": r"\b(seq|SEQ)_\d+\b",     # Sequence IDs
            "GENE_ID": r"\b[A-Z]{3,5}\d{1,4}\b",     # Gene IDs
            "PROTEIN_ID": r"\bP\d{5}\b",             # Protein IDs
        }
        
        # Sensitive patterns that should be flagged
        self.sensitive_patterns = {
            "PATIENT_ID": r"\b(patient|subject|participant)[-_]?\d+\b",
            "MEDICAL_RECORD": r"\bMR\d{6,}\b",
            "SOCIAL_SECURITY": r"\b\d{3}-\d{2}-\d{4}\b",
        }
    
    def detect_pii(self, text: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Detect PII in text
        Returns: (contains_pii, list_of_entities)
        """
        try:
            # Use Presidio for standard PII detection
            results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=[
                    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
                    "CREDIT_CARD", "US_SSN", "US_PASSPORT",
                    "IP_ADDRESS", "DATE_TIME", "LOCATION"
                ]
            )
            
            entities = []
            for result in results:
                entities.append({
                    "type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "text": text[result.start:result.end]
                })
            
            # Check custom patterns
            custom_entities = self._check_custom_patterns(text)
            entities.extend(custom_entities)
            
            contains_pii = len(entities) > 0
            
            if contains_pii:
                logger.warning(
                    "pii_detected",
                    entity_count=len(entities),
                    entity_types=[e["type"] for e in entities]
                )
            
            return contains_pii, entities
            
        except Exception as e:
            logger.error("pii_detection_error", error=str(e))
            return False, []
    
    def _check_custom_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Check for custom patterns in text"""
        entities = []
        
        for pattern_name, pattern in {**self.custom_patterns, **self.sensitive_patterns}.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "type": pattern_name,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.9,  # High confidence for regex matches
                    "text": match.group()
                })
        
        return entities
    
    def anonymize_text(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Anonymize PII in text"""
        try:
            # Convert our entity format to Presidio format
            presidio_results = []
            for entity in entities:
                from presidio_analyzer import RecognizerResult
                result = RecognizerResult(
                    entity_type=entity["type"],
                    start=entity["start"],
                    end=entity["end"],
                    score=entity["score"]
                )
                presidio_results.append(result)
            
            # Anonymize the text
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=presidio_results,
                operators={
                    "DEFAULT": {"type": "replace", "new_value": "[REDACTED]"},
                    "PERSON": {"type": "replace", "new_value": "[PERSON]"},
                    "EMAIL_ADDRESS": {"type": "replace", "new_value": "[EMAIL]"},
                    "PHONE_NUMBER": {"type": "replace", "new_value": "[PHONE]"},
                    "SAMPLE_ID": {"type": "replace", "new_value": "[SAMPLE_ID]"},
                    "SEQUENCE_ID": {"type": "replace", "new_value": "[SEQ_ID]"},
                }
            )
            
            return anonymized.text
            
        except Exception as e:
            logger.error("anonymization_error", error=str(e))
            return text  # Return original text if anonymization fails

class ContentFilter:
    """Content filtering for inappropriate or dangerous content"""
    
    def __init__(self):
        # Patterns for potentially harmful content
        self.harmful_patterns = [
            r"\b(hack|exploit|vulnerability|backdoor)\b",
            r"\b(password|credential|secret|token)\s*[:=]\s*\S+",
            r"\b(sql\s+injection|xss|csrf)\b",
        ]
        
        # Patterns for off-topic content (non-scientific)
        self.off_topic_patterns = [
            r"\b(politics|political|election|vote)\b",
            r"\b(religion|religious|god|allah|buddha)\b",
            r"\b(cryptocurrency|bitcoin|trading|investment)\b",
        ]
    
    def check_content(self, text: str) -> Dict[str, Any]:
        """
        Check content for policy violations
        Returns: {is_safe: bool, violations: List[str], warnings: List[str]}
        """
        violations = []
        warnings = []
        
        text_lower = text.lower()
        
        # Check for harmful content
        for pattern in self.harmful_patterns:
            if re.search(pattern, text_lower):
                violations.append(f"Potentially harmful content detected: {pattern}")
        
        # Check for off-topic content
        for pattern in self.off_topic_patterns:
            if re.search(pattern, text_lower):
                warnings.append(f"Off-topic content detected: {pattern}")
        
        # Check message length
        if len(text) > 10000:
            warnings.append("Message is very long")
        
        is_safe = len(violations) == 0
        
        return {
            "is_safe": is_safe,
            "violations": violations,
            "warnings": warnings
        }

class SecurityLogger:
    """Centralized security event logging"""
    
    def __init__(self):
        self.logger = structlog.get_logger("security")
    
    def log_pii_detection(self, user_id: str, session_id: str, entities: List[Dict[str, Any]]):
        """Log PII detection event"""
        self.logger.warning(
            "pii_detected_in_chat",
            user_id=user_id,
            session_id=session_id,
            entity_count=len(entities),
            entity_types=[e["type"] for e in entities]
        )
    
    def log_content_violation(self, user_id: str, session_id: str, violations: List[str]):
        """Log content policy violation"""
        self.logger.error(
            "content_policy_violation",
            user_id=user_id,
            session_id=session_id,
            violations=violations
        )
    
    def log_rate_limit_exceeded(self, user_id: str, limit_type: str, current_usage: int, limit: int):
        """Log rate limit exceeded"""
        self.logger.warning(
            "rate_limit_exceeded",
            user_id=user_id,
            limit_type=limit_type,
            current_usage=current_usage,
            limit=limit
        )
    
    def log_unauthorized_access(self, user_id: str, resource: str, required_role: str, user_role: str):
        """Log unauthorized access attempt"""
        self.logger.error(
            "unauthorized_access_attempt",
            user_id=user_id,
            resource=resource,
            required_role=required_role,
            user_role=user_role
        )

# Global instances
pii_detector = PIIDetector()
content_filter = ContentFilter()
security_logger = SecurityLogger()

def process_message_security(text: str, user_id: str, session_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Process message through security pipeline
    Returns: (processed_text, security_metadata)
    """
    security_metadata = {
        "contains_pii": False,
        "pii_entities": [],
        "content_safe": True,
        "content_violations": [],
        "content_warnings": [],
        "redacted_text": None
    }
    
    # 1. PII Detection
    contains_pii, pii_entities = pii_detector.detect_pii(text)
    security_metadata["contains_pii"] = contains_pii
    security_metadata["pii_entities"] = pii_entities
    
    # 2. Content Filtering
    content_check = content_filter.check_content(text)
    security_metadata["content_safe"] = content_check["is_safe"]
    security_metadata["content_violations"] = content_check["violations"]
    security_metadata["content_warnings"] = content_check["warnings"]
    
    # 3. Create redacted version for logging
    if contains_pii:
        redacted_text = pii_detector.anonymize_text(text, pii_entities)
        security_metadata["redacted_text"] = redacted_text
        security_logger.log_pii_detection(user_id, session_id, pii_entities)
    
    # 4. Log violations
    if not content_check["is_safe"]:
        security_logger.log_content_violation(user_id, session_id, content_check["violations"])
    
    return text, security_metadata
