# AI Assistant Configuration

The NGS Analysis Tool now includes a first-class OpenAI-powered AI assistant with streaming responses. Here's how to configure it:

## Prerequisites

1. **OpenAI Account**: Sign up at [platform.openai.com](https://platform.openai.com)
2. **API Key**: Get your API key from [API Keys page](https://platform.openai.com/account/api-keys)
3. **Billing**: Ensure you have billing set up (API usage is charged per token)

## Configuration Steps

### Option 1: Environment Variable (Recommended)

1. **Create a .env file** in the backend directory:
   ```bash
   cd backend
   touch .env
   ```

2. **Add your API key** to the .env file:
   ```bash
   echo "OPENAI_API_KEY=sk-your-actual-api-key-here" >> .env
   ```

3. **Start the application**:
   ```bash
   cd ..
   ./start.sh
   ```

### Option 2: Export Environment Variable

```bash
export OPENAI_API_KEY="sk-your-actual-api-key-here"
./start.sh
```

### Option 3: Add to Shell Profile (Permanent)

Add to your `~/.zshrc` or `~/.bash_profile`:
```bash
export OPENAI_API_KEY="sk-your-actual-api-key-here"
```

Then reload:
```bash
source ~/.zshrc
```

## Testing the Configuration

1. **Start the application**:
   ```bash
   ./start.sh
   ```

2. **Check the logs** - you should see:
   ```
   INFO:     Started server process [xxxxx]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   ```

3. **Test the AI endpoint**:
   ```bash
   curl -X POST http://localhost:8000/api/ai/chat \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Hello, what can you help me with?"}], "stream": false}'
   ```

## Features

The AI assistant now provides:

- **Real-time streaming responses** (like ChatGPT)
- **Context-aware answers** based on your current pipeline step
- **Expert knowledge** in NGS analysis, computational biology
- **Chart explanations** when you ask "what does this chart mean"
- **Code analysis** with biological context
- **Experimental design recommendations**

## Cost Information

- Model: GPT-4 Turbo Preview
- Approximate cost: $0.01-0.03 per message (depending on length)
- Streaming responses don't increase cost
- Context awareness adds ~200-500 tokens per request

## Troubleshooting

### "OpenAI API key not configured" Error
- Ensure the environment variable is set correctly
- Restart the backend after setting the key
- Check the .env file is in the correct directory (`backend/.env`)

### "Failed to get AI response" Error
- Check your OpenAI account has billing enabled
- Verify the API key is valid and active
- Check your usage limits haven't been exceeded

### Streaming Not Working
- Ensure your frontend is updated to handle streaming responses
- Check browser console for JavaScript errors
- Try the non-streaming test endpoint first

## Security Notes

- **Never commit your API key** to version control
- The .env file is already in .gitignore
- Use different API keys for development and production
- Consider using OpenAI's usage monitoring and alerts
