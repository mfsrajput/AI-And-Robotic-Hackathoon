#!/bin/bash

# VLA Environment Setup Script
# Sets up the environment for Vision-Language-Action system development
# Ensures Python 3.11+ compatibility across all components

set -e  # Exit on any error

echo "==========================================="
echo "VLA System Environment Setup Script"
echo "==========================================="

# Function to print status messages
print_status() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Check if running on the correct OS
print_status "Checking operating system..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi
print_success "Operating system: $OS"

# Check Python version
print_status "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    print_error "Python 3.11 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

print_success "Python version: $PYTHON_VERSION (compatible)"

# Check if pip is available
print_status "Checking pip availability..."
if ! command -v pip3 &> /dev/null; then
    print_warning "pip3 not found, attempting to install..."
    python3 -m ensurepip --upgrade
fi

if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not available and could not be installed"
    exit 1
fi

print_success "pip3 is available"

# Check if git is available
print_status "Checking git availability..."
if ! command -v git &> /dev/null; then
    print_error "git is not installed"
    exit 1
fi
print_success "git is available"

# Create virtual environment if it doesn't exist
VENV_NAME="vla_env"
if [ ! -d "$VENV_NAME" ]; then
    print_status "Creating virtual environment: $VENV_NAME"
    python3 -m venv "$VENV_NAME"
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists: $VENV_NAME"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_NAME/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip3 install --upgrade pip
print_success "pip upgraded"

# Install system dependencies based on OS
print_status "Installing system dependencies..."

if [ "$OS" = "linux" ]; then
    # For Linux systems
    if command -v apt-get &> /dev/null; then
        # Ubuntu/Debian
        print_status "Installing system dependencies for Ubuntu/Debian..."
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            pkg-config \
            libavcodec-dev \
            libavformat-dev \
            libswscale-dev \
            libv4l-dev \
            libxvidcore-dev \
            libx264-dev \
            libjpeg-dev \
            libpng-dev \
            libtiff-dev \
            gfortran \
            openexr \
            libatlas-base-dev \
            python3-dev \
            python3-tk \
            libportaudio2 \
            libportaudiocpp0 \
            portaudio19-dev
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        print_status "Installing system dependencies for CentOS/RHEL..."
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y \
            cmake \
            pkgconfig \
            libavcodec-devel \
            libavformat-devel \
            libswscale-devel \
            libv4l-devel \
            libxvidcore-devel \
            libx264-devel \
            libjpeg-turbo-devel \
            libpng-devel \
            libtiff-devel \
            openexr-devel \
            libatlas-devel \
            portaudio-devel
    fi
elif [ "$OS" = "macos" ]; then
    # For macOS systems
    if command -v brew &> /dev/null; then
        print_status "Installing system dependencies for macOS..."
        brew install \
            cmake \
            pkg-config \
            ffmpeg \
            portaudio
    else
        print_warning "Homebrew not found. Please install Homebrew or install dependencies manually."
    fi
fi

print_success "System dependencies installed"

# Install Python packages from requirements
print_status "Installing Python packages..."

# Create requirements.txt if it doesn't exist
if [ ! -f "requirements-vla.txt" ]; then
    print_status "Creating requirements file..."
    cat > requirements-vla.txt << EOF
# VLA System Requirements
# Vision-Language-Action system dependencies

# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0

# Audio processing for voice commands
sounddevice>=0.4.0
pyaudio>=0.2.11
speechrecognition>=3.8.1
webrtcvad>=2.0.10

# ROS 2 Python libraries (if available)
rclpy>=3.0.0

# Machine learning and AI
torch>=1.12.0
torchvision>=0.13.0
transformers>=4.20.0
sentencepiece>=0.1.96

# Computer vision
opencv-python>=4.5.0
Pillow>=8.3.0

# Web and networking
requests>=2.25.0
flask>=2.0.0
websockets>=10.0

# Utilities
pyyaml>=6.0
tqdm>=4.62.0
click>=8.0.0
dataclasses-json>=0.5.0
typing-extensions>=4.0.0

# Testing
pytest>=7.0.0
pytest-cov>=3.0.0

# Async programming
asyncio-mqtt>=0.11.0
aiofiles>=0.8.0

# Development tools
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
jupyter>=1.0.0
EOF
    print_success "requirements-vla.txt created"
fi

# Install the packages
pip3 install -r requirements-vla.txt
print_success "Python packages installed"

# Install Whisper.cpp Python bindings if not already installed
print_status "Installing Whisper.cpp Python bindings..."
if ! python3 -c "import whisper_cpp" &> /dev/null; then
    # Clone and build Whisper.cpp if needed
    if [ ! -d "whisper.cpp" ]; then
        print_status "Cloning Whisper.cpp..."
        git clone https://github.com/ggerganov/whisper.cpp.git
        cd whisper.cpp
        make
        cd ..
    fi

    # Install the Python bindings
    pip3 install git+https://github.com/ahmedkhalf/whisper-cpp-pybind.git
    print_success "Whisper.cpp Python bindings installed"
else
    print_status "Whisper.cpp Python bindings already installed"
fi

# Install Ollama if not already installed
print_status "Checking for Ollama installation..."
if ! command -v ollama &> /dev/null; then
    print_status "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    print_success "Ollama installed"
else
    print_status "Ollama is already installed"
fi

# Start Ollama service
print_status "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!
sleep 5  # Wait for Ollama to start

# Download a default model for testing
print_status "Downloading default LLM model (this may take a few minutes)..."
ollama pull llama2
print_success "Llama2 model downloaded"

# Check if ROS 2 is available
print_status "Checking for ROS 2 installation..."
if python3 -c "import rclpy" &> /dev/null; then
    print_success "ROS 2 Python libraries are available"
else
    print_warning "ROS 2 Python libraries not found. ROS integration will be limited."
    print_status "To install ROS 2 Humble Hawksbill, visit: https://docs.ros.org/en/humble/Installation.html"
fi

# Create necessary directories if they don't exist
print_status "Creating project directories..."
mkdir -p docs/04-vla
mkdir -p launch
mkdir -p config
mkdir -p scripts/voice_control
mkdir -p scripts/llm_task_planning
mkdir -p scripts/multi_modal
mkdir -p scripts/capstone_project
mkdir -p assets/diagrams
mkdir -p assets/code-examples/voice_control
mkdir -p assets/code-examples/llm_task_planning
mkdir -p assets/code-examples/multi_modal_integration
mkdir -p assets/code-examples/capstone_project

print_success "Project directories created"

# Create a basic configuration file
print_status "Creating basic configuration..."
cat > config/vla_pipeline_config.yaml << EOF
# VLA Pipeline Configuration
# Vision-Language-Action system configuration file

voice_processing:
  whisper_model: "base"  # Options: tiny, base, small, medium, large
  audio_sample_rate: 16000
  audio_channels: 1
  noise_suppression: true
  silence_threshold: 0.01

llm_integration:
  model_name: "llama2"
  temperature: 0.7
  max_tokens: 512
  top_p: 0.9
  presence_penalty: 0.0
  frequency_penalty: 0.0

multi_modal_fusion:
  visual_confidence_threshold: 0.7
  fusion_algorithm: "attention"
  context_window_size: 10

ros_integration:
  navigation_action_server: "/navigate_to_pose"
  manipulation_action_server: "/grasp_object"
  perception_action_server: "/detect_objects"
  action_timeout: 30.0

performance:
  max_response_time: 2.0  # seconds
  min_success_rate: 0.8
  battery_threshold: 20.0  # percent

privacy:
  local_processing_only: true
  data_retention_days: 7
  encryption_enabled: true
EOF

print_success "Configuration file created: config/vla_pipeline_config.yaml"

# Create a setup completion marker
print_status "Marking setup as complete..."
touch .vla_setup_complete
print_success "Setup completion marker created"

# Final summary
print_success "==========================================="
print_success "VLA Environment Setup Complete!"
print_success "==========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source $VENV_NAME/bin/activate"
echo "2. Run the test suite: python scripts/test_vla_components.py"
echo "3. Start the Ollama service: ollama serve (in a separate terminal)"
echo "4. Try the voice demo: python assets/code-examples/voice_control/voice_demo.py"
echo ""
echo "The environment is ready for VLA system development!"
echo "Python version: $PYTHON_VERSION"
echo "Virtual environment: $VENV_NAME"
echo "Setup completed at: $(date)"
echo ""

# Deactivate virtual environment
deactivate
print_status "Virtual environment deactivated. Remember to activate it before working on the project."