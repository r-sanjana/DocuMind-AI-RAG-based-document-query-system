#!/bin/bash
# DocuMind AI – One-command setup script

set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║     🧠 DocuMind AI – Setup           ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED="3.11"
echo "✅ Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate
echo "✅ Virtual environment activated"

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Set up .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  Please edit .env and set your GEMINI_API_KEY:"
    echo "    https://aistudio.google.com/app/apikey"
    echo ""
else
    echo "✅ .env file found"
fi

# Create data directories
mkdir -p data/uploads
mkdir -p backend/vectorstore
echo "✅ Directories created"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  🚀 Setup complete! Start with:      ║"
echo "║                                      ║"
echo "║  Terminal 1 (backend):               ║"
echo "║  cd backend && uvicorn main:app      ║"
echo "║  --reload --port 8000                ║"
echo "║                                      ║"
echo "║  Terminal 2 (frontend):              ║"
echo "║  cd frontend && streamlit run app.py ║"
echo "╚══════════════════════════════════════╝"
echo ""
