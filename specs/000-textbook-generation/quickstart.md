# Quickstart Guide: Physical AI & Humanoid Robotics Textbook Generation

**Feature**: 001-textbook-generation
**Date**: 2025-12-09
**Phase**: 1 (Design & Architecture)
**Input**: Data model from `/specs/001-textbook-generation/data-model.md`

## Overview

This quickstart guide provides the essential information needed to begin development on the Physical AI & Humanoid Robotics textbook generation system. It covers environment setup, key development workflows, and the primary components you'll interact with during implementation.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 4090 or equivalent (for Isaac Sim and synthetic data generation)
- **RAM**: 32GB minimum (64GB recommended for Isaac Sim)
- **Storage**: 500GB SSD (1TB recommended for synthetic datasets)
- **CPU**: 16-core processor (AMD Ryzen 9 or Intel i9 series recommended)

### Software Requirements
- **OS**: Ubuntu 22.04 LTS
- **ROS 2**: Humble Hawksbill (with Isaac ROS GEMs)
- **Isaac Sim**: 2023.2.0 or later
- **Unity**: 2022.3 LTS or later (with ROS TCP Connector)
- **CUDA**: 12.0 or later
- **Python**: 3.11 with pip, virtualenv
- **Docker**: 20.10 or later
- **Git**: 2.30 or later

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/physical-ai-humanoid-textbook.git
cd physical-ai-humanoid-textbook
git checkout 001-textbook-generation
```

### 2. Install ROS 2 Humble with Isaac Extensions

```bash
# Add ROS 2 repository
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install ros-humble-isaac-*  # Isaac ROS packages

# Install colcon and other tools
sudo apt install python3-colcon-common-extensions python3-rosdep python3-vcstool
```

### 3. Set Up Isaac Sim

```bash
# Download Isaac Sim from NVIDIA Developer portal
# Follow installation instructions for your system

# Set up Isaac Sim environment
export ISAACSIM_PATH="/path/to/isaac-sim"
export ISAACSIM_PYTHON_PATH="$ISAACSIM_PATH/python.sh"

# Verify Isaac Sim installation
$ISAACSIM_PYTHON_PATH -c "import omni; print('Isaac Sim Python environment OK')"
```

### 4. Configure Unity-ROS Integration

```bash
# Install Unity Hub and Unity 2022.3 LTS
# In Unity Hub, install the ROS TCP Connector package
# Import the Isaac ROS Unity plugins

# Set up Unity project structure
mkdir -p unity_projects/humanoid_textbook
cd unity_projects/humanoid_textbook
# Initialize new Unity project here
```

### 5. Create Python Virtual Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

## Key Development Workflows

### 1. Generating a New Chapter

To create a new chapter following pedagogical principles:

```bash
# Use the chapter generator script
./scripts/generate_chapter.py \
  --module "module-1-ros2" \
  --chapter-number 1 \
  --title "ROS 2 and Embodied Control" \
  --learning-objectives "objective1,objective2,objective3" \
  --topic "ros2-basics"
```

### 2. Creating Synthetic Data with Isaac Sim

To generate synthetic training data:

```bash
# Navigate to Isaac Sim
cd $ISAACSIM_PATH

# Run synthetic data generation
./python.sh -m scripts.synthetic_data_gen \
  --config configs/humanoid_scenes.yaml \
  --output datasets/humanoid_training \
  --samples 10000 \
  --domain-randomization true
```

### 3. Validating Chapter Content

To validate a chapter against pedagogical requirements:

```bash
# Validate chapter structure and content
./scripts/validate_chapter.py \
  --chapter-path textbook/module-1-ros2/chapter-1.md \
  --validate-pedagogy true \
  --validate-accessibility true \
  --validate-code-examples true
```

### 4. Running Performance Benchmarks

To benchmark code example execution times:

```bash
# Run performance tests
./scripts/benchmark_code_examples.py \
  --module-path textbook/module-1-ros2/ \
  --timeout 2.0 \
  --benchmark-type "execution_time"
```

## Primary Components

### 1. Textbook Generation Pipeline

Located in `textbook_generation/`, this component handles:
- Chapter content generation following pedagogical templates
- Code example validation and testing
- Diagram generation from Mermaid specifications
- Accessibility compliance checking

### 2. Synthetic Data Generator

Located in `synthetic_data/`, this component:
- Interfaces with Isaac Sim for synthetic dataset generation
- Implements domain randomization techniques
- Validates synthetic data quality
- Manages dataset versioning and storage

### 3. Content Validation System

Located in `validation/`, this system:
- Checks chapters against 10 Constitution pedagogical principles
- Validates code examples for syntax and execution
- Ensures accessibility compliance (WCAG 2.1 AA)
- Performs cross-module consistency checks

### 4. Isaac Sim Integration

Located in `isaac_integration/`, this module:
- Manages Isaac Sim scene configuration
- Controls synthetic data generation parameters
- Handles USD scene creation and management
- Interfaces with Isaac ROS GEMs

## Configuration Files

### 1. Main Configuration

`config/textbook_generation.yaml` contains:
- Global generation parameters
- Pedagogical principle enforcement settings
- Performance targets and constraints
- Accessibility compliance levels

### 2. Isaac Sim Configuration

`config/isaac_sim.yaml` includes:
- Scene generation parameters
- Domain randomization settings
- Sensor simulation configurations
- Hardware acceleration options

### 3. Module-Specific Configurations

Each module has its own configuration:
- `config/modules/module-1-ros2.yaml`
- `config/modules/module-2-digital-twin.yaml`
- `config/modules/module-3-ai-brain.yaml`
- `config/modules/module-4-vla-capstone.yaml`

## Common Commands

### Development Commands

```bash
# Generate a complete module
./scripts/generate_module.py --module-id 1 --output-dir textbook/

# Run all validation checks
./scripts/run_validation.sh

# Generate synthetic datasets
./scripts/generate_synthetic_data.sh --scene humanoid_indoor --samples 5000

# Build textbook for publication
./scripts/build_textbook.sh --format pdf --output physical_ai_humanoid_robotics.pdf
```

### Testing Commands

```bash
# Run unit tests
python -m pytest tests/unit/

# Run integration tests
python -m pytest tests/integration/

# Run performance tests
python -m pytest tests/performance/

# Run accessibility tests
./scripts/test_accessibility.sh
```

### Deployment Commands

```bash
# Deploy to online platform
./scripts/deploy_online.sh --target production

# Generate static website
./scripts/generate_static_site.sh --output docs_build/

# Package for distribution
./scripts/package_distribution.sh --format all
```

## Troubleshooting

### Common Issues

1. **Isaac Sim won't start**: Ensure GPU drivers are up to date and CUDA version matches Isaac Sim requirements
2. **Code examples fail**: Check ROS 2 environment is sourced and dependencies are installed
3. **Performance issues**: Verify Isaac Sim is using GPU acceleration and domain randomization isn't too aggressive
4. **Accessibility validation fails**: Ensure all diagrams have alt-text and proper color contrast ratios

### Performance Tips

- Use SSD storage for Isaac Sim cache and synthetic datasets
- Close unnecessary applications when running Isaac Sim
- Use CUDA-enabled hardware for synthetic data generation
- Consider containerized environments for consistency across development machines

## Getting Started Checklist

- [ ] Environment setup complete (ROS 2, Isaac Sim, Unity)
- [ ] Python virtual environment created and activated
- [ ] Clone repository and checkout feature branch
- [ ] Run `./scripts/validate_environment.sh` to verify setup
- [ ] Generate a test chapter to verify pipeline
- [ ] Run validation scripts to confirm compliance
- [ ] Review pedagogical template in `templates/chapter_template.md`

## Next Steps

1. **Review the architecture**: Examine `specs/001-textbook-generation/architecture/` for system design details
2. **Explore the data model**: Study `specs/001-textbook-generation/data-model.md` for entity relationships
3. **Start with Module 1**: Begin with ROS 2 fundamentals in `textbook/module-1-ros2/`
4. **Follow the pedagogical template**: Use the chapter template to ensure Constitution compliance
5. **Run validation frequently**: Use validation scripts to catch issues early

This quickstart guide provides the foundation for productive development on the Physical AI & Humanoid Robotics textbook project. Refer to the detailed architecture documentation for deeper implementation details.