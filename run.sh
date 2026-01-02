#!/bin/bash

# Infinite Memory Chat - Setup and Run Script
# Supports OpenAI and MongoDB vector backends

set -e  # Exit on any error

echo "🧠 Infinite Memory Chat - Setup & Launch 3T version"
echo "========================================"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3.14 -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

echo ""
echo "🔧 Configuration Check:"

# Load .env file if it exists
if [ -f ".env" ]; then
    echo "   Loading .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "   No .env file found, using environment variables"
fi

# Check for required API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "❌ OPENAI_API_KEY not set!"
    echo "   This is required for both OpenAI and MongoDB backends"
    echo "   (OpenAI is used for chat responses and embeddings)"
    echo ""
    echo "   Option 1: Create a .env file (recommended):"
    echo "   cp env.example .env"
    echo "   # Then edit .env with your API key"
    echo ""
    echo "   Option 2: Set environment variable:"
    echo "   export OPENAI_API_KEY='your-key-here'"
    echo ""
    exit 1
fi

# Display backend configuration
BACKEND=${VECTOR_BACKEND:-openai}
echo "   Vector Backend: $BACKEND"

if [ "$BACKEND" = "mongodb" ]; then
    if [ -z "$MONGODB_CONNECTION_STRING" ]; then
        echo ""
        echo "⚠️  MongoDB backend selected but MONGODB_CONNECTION_STRING not set!"
        echo "   Add to .env file or set environment variable:"
        echo "   export MONGODB_CONNECTION_STRING='mongodb+srv://user:pass@cluster.mongodb.net/'"
        echo ""
        echo "   Falling back to OpenAI backend..."
        export VECTOR_BACKEND=openai
        BACKEND=openai
    else
        echo "   MongoDB Database: ${MONGODB_DATABASE:-infinite_memory_chat}"
    fi
fi

echo "   Chat Model: ${CHAT_MODEL:-gpt-4o-mini}"
echo "   Embedding Model: ${EMBEDDING_MODEL:-text-embedding-3-small}"

echo ""
echo "🚀 Launching application with $BACKEND backend..."
echo "========================================"
echo ""

# Run the chat
python3.14 infinite_memory_chat.py
