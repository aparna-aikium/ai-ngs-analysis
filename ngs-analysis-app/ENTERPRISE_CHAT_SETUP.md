# Enterprise ChatGPT-Style Assistant Setup Guide

This guide covers the complete setup of the enterprise-grade ChatGPT-style assistant with SSO authentication, role-based access, full logging, RAG capabilities, and security features.

## 🏗️ Architecture Overview

The system includes:
- **SSO Authentication**: Google, Microsoft, Okta integration
- **Role-Based Access Control**: Admin, Researcher, Viewer, Guest roles
- **Rate Limiting**: Per-user, per-role limits with Redis caching
- **Full Logging**: All prompts/completions logged with PII redaction
- **RAG System**: Retrieval over NGS analysis data and documentation
- **Security**: PII detection, content filtering, audit trails
- **Streaming UI**: Real-time ChatGPT-style interface

## 📋 Prerequisites

1. **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com)
2. **SSO Provider**: Google OAuth, Microsoft Azure AD, or Okta
3. **Database**: PostgreSQL (recommended) or SQLite (development)
4. **Redis**: For rate limiting and caching (optional but recommended)
5. **Python 3.8+**: With pip and virtual environment

## 🔧 Installation

### 1. Install Dependencies

```bash
cd ngs-analysis-app/backend
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `.env` file in the backend directory:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here

# JWT Security
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/ngs_chat
# Or for development: DATABASE_URL=sqlite:///./ngs_chat.db

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379

# SSO Configuration
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
MICROSOFT_CLIENT_ID=your-microsoft-client-id
OKTA_DOMAIN=your-domain.okta.com

# Data Directory
NGS_DATA_DIRECTORY=./data

# Logging
SQL_DEBUG=false
LOG_LEVEL=INFO
```

### 3. Database Setup

#### PostgreSQL (Recommended for Production)

```bash
# Install PostgreSQL
brew install postgresql  # macOS
# or apt-get install postgresql  # Ubuntu

# Create database
createdb ngs_chat

# Update DATABASE_URL in .env
DATABASE_URL=postgresql://username:password@localhost/ngs_chat
```

#### SQLite (Development Only)

```bash
# SQLite will be created automatically
DATABASE_URL=sqlite:///./ngs_chat.db
```

### 4. Redis Setup (Optional but Recommended)

```bash
# Install Redis
brew install redis  # macOS
# or apt-get install redis-server  # Ubuntu

# Start Redis
redis-server

# Update REDIS_URL in .env
REDIS_URL=redis://localhost:6379
```

## 🔐 SSO Configuration

### Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add authorized redirect URIs:
   - `http://localhost:3000/auth/callback/google` (development)
   - `https://yourdomain.com/auth/callback/google` (production)
6. Copy Client ID to `.env` file

### Microsoft Azure AD Setup

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to Azure Active Directory > App registrations
3. Create new registration
4. Add redirect URIs:
   - `http://localhost:3000/auth/callback/microsoft` (development)
   - `https://yourdomain.com/auth/callback/microsoft` (production)
5. Copy Application (client) ID to `.env` file

### Okta Setup

1. Go to [Okta Developer Console](https://developer.okta.com)
2. Create new application
3. Choose "Single-Page App" (SPA)
4. Add redirect URIs
5. Copy Client ID and domain to `.env` file

## 🚀 Running the System

### 1. Start Backend

```bash
cd ngs-analysis-app/backend
python main.py
```

The backend will:
- Initialize database tables
- Set up knowledge base with NGS documentation
- Start API server on http://localhost:8000

### 2. API Documentation

Visit http://localhost:8000/docs for interactive API documentation.

## 👥 User Management

### Default Roles

- **Guest**: 10 messages/hour, 50/day, 10K tokens/month
- **Viewer**: 50 messages/hour, 200/day, 50K tokens/month  
- **Researcher**: 100 messages/hour, 500/day, 200K tokens/month
- **Admin**: 500 messages/hour, 2000/day, 1M tokens/month

### Role Assignment

Users are automatically assigned "Viewer" role on first login. Admins can update roles via database:

```sql
UPDATE users SET role = 'researcher' WHERE email = 'user@example.com';
```

### Rate Limit Customization

Update rate limits in database:

```sql
UPDATE rate_limit_rules 
SET messages_per_hour = 200, tokens_per_month = 500000 
WHERE role = 'researcher';
```

## 🔍 Monitoring & Logging

### Structured Logging

All events are logged in JSON format:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "info",
  "event": "chat_response_completed",
  "user_id": "user-123",
  "session_id": "session-456",
  "tokens_used": 150,
  "response_time_ms": 1200,
  "model": "gpt-4-turbo-preview"
}
```

### Security Events

- PII detection: `pii_detected_in_chat`
- Rate limiting: `rate_limit_exceeded`
- Authentication: `login_success`, `login_failed`
- Authorization: `unauthorized_access_attempt`

### Database Audit Trail

All user actions are logged in the `audit_logs` table:

```sql
SELECT * FROM audit_logs 
WHERE user_id = 'user-123' 
ORDER BY timestamp DESC;
```

## 🛡️ Security Features

### PII Detection & Redaction

Automatically detects and redacts:
- Personal names, emails, phone numbers
- Social security numbers, credit cards
- IP addresses, locations
- Custom scientific identifiers (sample IDs, sequence IDs)

### Content Filtering

Blocks:
- Potentially harmful content (hacking, exploits)
- Off-topic content (politics, religion)
- Inappropriate requests

### Rate Limiting

Multi-tier rate limiting:
- Hourly message limits
- Daily message limits  
- Monthly token limits
- Per-request token limits

## 📚 Knowledge Base (RAG)

### Default Content

The system includes built-in knowledge about:
- NGS library generation best practices
- Selection simulation parameters
- Statistical analysis interpretation
- Experimental design guidelines

### Adding Custom Content

```python
from rag_system import get_retriever

retriever = get_retriever()
retriever.knowledge_base.add_document(
    content="Your custom NGS protocol...",
    doc_type="protocols",
    metadata={
        "title": "Custom Protocol",
        "source": "Lab Manual",
        "required_role": "researcher"
    }
)
```

### Indexing Analysis Results

The system automatically indexes:
- CSV files with analysis results
- Library generation outputs
- Selection simulation results
- NGS simulation data

## 🔧 API Usage Examples

### Authentication

```javascript
// Login with Google
const response = await fetch('/api/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    provider: 'google',
    token: 'google-oauth-token'
  })
});

const { access_token } = await response.json();
```

### Sending Chat Messages

```javascript
// Send message with streaming
const response = await fetch('/api/chat/message', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${access_token}`
  },
  body: JSON.stringify({
    message: "What does this enrichment analysis mean?",
    session_id: "session-123",
    use_rag: true,
    context: {
      currentStep: "analysis",
      hasAnalysis: true,
      analysisData: { /* analysis results */ }
    }
  })
});

// Handle streaming response
const reader = response.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = new TextDecoder().decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.content) {
        console.log(data.content); // Stream content
      }
    }
  }
}
```

### Session Management

```javascript
// Get user sessions
const sessions = await fetch('/api/chat/sessions', {
  headers: { 'Authorization': `Bearer ${access_token}` }
});

// Create new session
const newSession = await fetch('/api/chat/sessions', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${access_token}` }
});
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check DATABASE_URL format
   - Ensure PostgreSQL is running
   - Verify credentials

2. **OpenAI API Errors**
   - Verify API key is valid
   - Check billing limits
   - Ensure model access permissions

3. **SSO Authentication Failed**
   - Verify client IDs and secrets
   - Check redirect URI configuration
   - Ensure proper scopes are requested

4. **Rate Limiting Too Strict**
   - Update rate_limit_rules table
   - Check Redis connection
   - Verify user role assignments

5. **PII Detection False Positives**
   - Adjust detection thresholds in security.py
   - Add custom patterns for scientific data
   - Review anonymization rules

### Logs Analysis

```bash
# View recent logs
tail -f logs/app.log | jq '.'

# Filter security events
grep "pii_detected\|rate_limit\|unauthorized" logs/app.log | jq '.'

# Monitor API performance
grep "chat_response_completed" logs/app.log | jq '.response_time_ms'
```

## 🔄 Production Deployment

### Environment Variables

```bash
# Production settings
DATABASE_URL=postgresql://user:pass@prod-db:5432/ngs_chat
REDIS_URL=redis://prod-redis:6379
JWT_SECRET_KEY=super-secure-production-key
LOG_LEVEL=WARNING
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Security Checklist

- [ ] Change default JWT secret key
- [ ] Use HTTPS in production
- [ ] Set up proper CORS origins
- [ ] Enable database SSL
- [ ] Configure log rotation
- [ ] Set up monitoring alerts
- [ ] Regular security updates
- [ ] Backup database regularly

## 📊 Monitoring Dashboard

Key metrics to monitor:
- Active users and sessions
- Message volume and token usage
- Response times and error rates
- Rate limit violations
- PII detection events
- Model usage and costs

The system provides comprehensive enterprise-grade chat capabilities with full security, compliance, and monitoring features suitable for production deployment.
