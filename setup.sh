#!/bin/bash
#
# Foresight Setup Script
# Installs all dependencies for local transcription and insight extraction
#

set -e

echo "=========================================="
echo "  Foresight - Local Interview Processor"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success() { echo -e "${GREEN}✓${NC} $1"; }
warning() { echo -e "${YELLOW}!${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }
info() { echo -e "  $1"; }

# Check OS
OS="$(uname -s)"
if [[ "$OS" != "Darwin" && "$OS" != "Linux" ]]; then
    error "Unsupported OS: $OS"
    exit 1
fi

echo "Step 1/5: Checking system requirements..."
echo ""

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    success "Python $PYTHON_VERSION"
else
    error "Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
    success "pip installed"
else
    error "pip not found. Please install pip"
    exit 1
fi

echo ""
echo "Step 2/5: Installing Ollama..."
echo ""

# Install Ollama
if command -v ollama &> /dev/null; then
    success "Ollama already installed"
else
    if [[ "$OS" == "Darwin" ]]; then
        if command -v brew &> /dev/null; then
            info "Installing via Homebrew..."
            brew install ollama
            success "Ollama installed"
        else
            info "Installing via curl..."
            curl -fsSL https://ollama.com/install.sh | sh
            success "Ollama installed"
        fi
    else
        info "Installing via curl..."
        curl -fsSL https://ollama.com/install.sh | sh
        success "Ollama installed"
    fi
fi

echo ""
echo "Step 3/5: Installing ffmpeg (for audio processing)..."
echo ""

# Install ffmpeg
if command -v ffmpeg &> /dev/null; then
    success "ffmpeg already installed"
else
    if [[ "$OS" == "Darwin" ]]; then
        if command -v brew &> /dev/null; then
            brew install ffmpeg
            success "ffmpeg installed"
        else
            error "Please install Homebrew first: https://brew.sh"
            exit 1
        fi
    else
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y ffmpeg
            success "ffmpeg installed"
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y ffmpeg
            success "ffmpeg installed"
        else
            error "Please install ffmpeg manually"
            exit 1
        fi
    fi
fi

echo ""
echo "Step 4/5: Installing Python packages..."
echo ""

# Install Python dependencies
pip3 install --upgrade pip
pip3 install openai-whisper watchdog pyyaml

success "Python packages installed"

echo ""
echo "Step 5/5: Downloading AI models..."
echo ""

# Start Ollama service
info "Starting Ollama service..."
if [[ "$OS" == "Darwin" ]]; then
    # On macOS, ollama runs as an app or service
    ollama serve &> /dev/null &
    sleep 3
else
    # On Linux, start as background process
    ollama serve &> /dev/null &
    sleep 3
fi

# Pull Mistral model (~4GB)
info "Downloading Mistral model (~4GB, this may take a few minutes)..."
ollama pull mistral
success "Mistral model downloaded"

# Whisper will download models on first use, but we can pre-download
info "Pre-downloading Whisper medium model (~1.5GB)..."
python3 -c "import whisper; whisper.load_model('medium')" 2>/dev/null || true
success "Whisper model ready"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Test your installation:"
echo "  python3 interview_processor.py --test"
echo ""
echo "Process your first file:"
echo "  python3 interview_processor.py --file ~/path/to/audio.m4a"
echo ""
echo "Start watching for new files:"
echo "  python3 interview_processor.py --watch"
echo ""
