#!/bin/bash

# NGS Analysis Pipeline Startup Script

echo "🧬 Starting NGS Analysis Pipeline..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
npm run install:all

# Start the application
echo "🚀 Starting development servers..."
echo "   - Backend API: http://localhost:8000"
echo "   - Frontend: http://localhost:3000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

npm run dev
