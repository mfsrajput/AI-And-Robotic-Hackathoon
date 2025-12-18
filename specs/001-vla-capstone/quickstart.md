# Quickstart Guide: Module 4 - Vision-Language-Action (VLA) + Capstone

**Date**: 2025-12-10
**Feature**: 001-vla-capstone

## Overview

This guide provides a quick setup and initial run of the Vision-Language-Action (VLA) system for the Physical AI & Humanoid Robotics textbook. The VLA system enables voice-controlled humanoid robots with multi-modal perception, LLM-based task planning, and explicit ROS 2 action server integration.

## Prerequisites

### System Requirements
- Ubuntu 22.04 LTS (recommended for ROS 2 Humble)
- Python 3.11+
- At least 8GB RAM (16GB recommended)
- 50GB free disk space
- Microphone for voice input
- Internet connection (for initial setup only - all processing runs locally)

### Software Dependencies
1. **ROS 2 Humble Hawksbill** (or ROS 2 Jazzy Jalisco)
2. **Ollama** (for local LLM hosting)
3. **Whisper.cpp** (for local voice-to-text)
4. **Git**

## Installation

### 1. Install ROS 2
```bash
# Follow official ROS 2 Humble installation guide:
# https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html

# Or for a quick setup:
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list'
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo rosdep init
rosdep update
source /opt/ros/humble/setup.bash
```

### 2. Install Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 3. Install Whisper.cpp
```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
make
cd ..
```

### 4. Set up Python Environment
```bash
# Create virtual environment
python3 -m venv vla_env
source vla_env/bin/activate
pip install --upgrade pip

# Install Python dependencies
pip install numpy scipy pyaudio sounddevice openai-whisper
pip install rclpy  # ROS 2 Python client library
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # Use appropriate URL for your hardware
pip install opencv-python
```

## Initial Setup

### 1. Download LLM Model
```bash
# Pull the recommended model for educational purposes
ollama pull llama3.1:8b
```

### 2. Clone the Textbook Repository
```bash
git clone <repository-url>
cd <repository-directory>
source /opt/ros/humble/setup.bash
source install/setup.bash  # If using ROS workspace
```

### 3. Configure the VLA System
```bash
# Navigate to the VLA configuration directory
cd config/

# The system is pre-configured for local processing, privacy compliance, and educational use
# Review the configuration in llm_config.yaml, whisper_config.yaml, and vla_pipeline_config.yaml
# New file for LLM prompt templates per FR-006: llm_prompt_templates.yaml
```

## Running the VLA System

### 1. Start Ollama Server
```bash
ollama serve
```

### 2. Launch the VLA System
```bash
# In a new terminal, source ROS environment
source /opt/ros/humble/setup.bash
cd <workspace-directory>

# Launch the complete VLA system with integration to previous modules
ros2 launch launch/vla_capstone.launch.py
```

### 3. Test Voice Command
```bash
# In another terminal, run the voice command test
source /opt/ros/humble/setup.bash
source vla_env/bin/activate
python3 scripts/voice_control/voice_processor.py --test
```

## Basic Usage Examples

### Example 1: Simple Navigation with ROS Action Server
1. Say: "Please navigate to the kitchen"
2. The system will:
   - Convert speech to text using Whisper
   - Process with LLM to generate navigation plan using prompt templates
   - Execute navigation using ROS 2 NavigateToPose action server
   - Provide feedback on completion

### Example 2: Object Manipulation with ROS Action Server
1. Say: "Go to the table and pick up the red cup"
2. The system will:
   - Process the multi-step command using LLM with navigation/manipulation templates
   - Generate action sequence: navigate → detect object → grasp → confirm
   - Execute the sequence using appropriate ROS action servers
   - Include safety checks and error recovery

### Example 3: Multi-Modal Command with "Go to the blue box" Scenario
1. Combine voice with visual input
2. Say: "Go to the blue box" while the robot can see the environment
3. The system will:
   - Process voice command and visual input
   - Fuse modalities to identify the blue box
   - Navigate to the identified blue box using ROS action servers
   - This demonstrates the specific acceptance scenario from User Story 3

## Educational Components

### 1. Code Examples
The system includes runnable code examples in the `scripts/` directory:
- `voice_control/` - Voice processing and Whisper integration
- `llm_task_planning/` - LLM-based task planning with prompt templates
- `multi_modal/` - Vision-language fusion with ROS action interfaces
- `capstone_project/` - Complete autonomous system

### 2. ROS 2 Action Server Integration
The system implements explicit ROS 2 action servers:
- **Navigation**: nav2_msgs/action/NavigateToPose for navigation tasks
- **Manipulation**: Custom action servers for grasp, place, and manipulation tasks
- **Task Planning**: Action servers for high-level task execution

### 3. LLM Prompt Templates
Located in `config/llm_prompt_templates.yaml`, these templates provide:
- Reusable, version-controlled prompts for robotics tasks
- Examples for navigation, manipulation, perception, and multi-modal tasks
- Educational examples of prompt engineering best practices

### 4. Observability Tools
Monitor system behavior with:
- RViz panels showing internal states
- Detailed logging with educational annotations
- Performance metrics and debugging tools
- ROS 2 action server feedback mechanisms

### 5. Assessment Materials
The capstone project includes:
- Project rubrics with clear evaluation criteria
- Tiered assessments (beginner to advanced)
- Integration with previous modules (ROS 2, Digital Twin, Isaac Sim)

## Deep Integration with Previous Modules

### Module 1 (ROS 2) Integration
- Uses ROS 2 Humble/Jazzy action servers as designed in Module 1
- Leverages existing topic structures where applicable
- Builds upon Module 1's error handling patterns

### Module 2 (Digital Twin) Integration
- Interfaces for simulating VLA capabilities in digital twin environment
- Simulation for safe testing of complex action sequences
- Consistency with Module 2's sensor simulation patterns

### Module 3 (Isaac Sim) Integration
- Isaac Sim for realistic physics simulation of humanoid actions
- Isaac Sim's vision sensors for multi-modal training
- Consistency with Module 3's simulation workflows

## Troubleshooting

### Common Issues

1. **"No audio input detected"**
   - Check microphone permissions
   - Verify audio device selection in config

2. **"LLM response taking too long"**
   - Ensure Ollama server is running
   - Check model is properly loaded

3. **"Action sequence fails"**
   - Verify ROS 2 network configuration
   - Check robot simulation environment (if using simulation)
   - Confirm ROS action server names are valid

### Performance Optimization
- Use quantized LLM models for faster inference
- Adjust Whisper model size based on performance needs
- Configure timeout values in config files appropriately

## Accessibility Compliance (WCAG 2.1 AA)

The system includes:
- Alt-text for all diagrams and code examples
- Semantic Markdown structure
- Color contrast verification for diagrams
- Keyboard navigation for interactive elements
- Proper heading hierarchy in documentation

## Next Steps

1. Complete the hands-on exercises in `docs/04-vla/`
2. Experiment with different voice commands
3. Modify action sequences to understand the planning process
4. Integrate with Isaac Sim for physics-based simulation
5. Work on the capstone autonomous humanoid project
6. Verify all 10 constitutional pedagogical principles are met
7. Test the "Go to the blue box" scenario as a concrete worked example

## Support

- Check the documentation in `docs/04-vla/` for detailed explanations
- Review the educational examples and worked solutions
- Use the discussion forums for questions about implementation
- Refer to the constitutional pedagogical principles for learning objectives