#!/usr/bin/env python3
"""
Python compatibility verification script for VLA system
Checks Python version and library compatibility for educational purposes
"""

import sys
import platform
import importlib
from typing import Dict, List, Tuple
import subprocess
import os


class PythonCompatibilityChecker:
    """
    Checks Python version and library compatibility for VLA system
    """
    
    def __init__(self):
        self.required_version = (3, 11)
        self.recommended_version = (3, 11, 0)
        self.check_results = {}
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Check if Python version meets requirements"""
        current_version = sys.version_info[:2]
        current_full_version = sys.version_info[:3]
        
        if current_version < self.required_version:
            message = (f"Python version {current_full_version} is below minimum requirement "
                      f"of Python {self.required_version[0]}.{self.required_version[1]}")
            return False, message
        else:
            message = f"Python version {current_full_version} meets requirements"
            return True, message
    
    def check_os_compatibility(self) -> Tuple[bool, str]:
        """Check if operating system is compatible"""
        system = platform.system().lower()
        
        if system in ['linux', 'darwin']:  # Linux and macOS
            return True, f"Operating system {system} is supported"
        elif system == 'windows':
            return True, f"Operating system {system} is supported (with WSL recommended)"
        else:
            return False, f"Operating system {system} may have compatibility issues"
    
    def check_required_libraries(self) -> Dict[str, Tuple[bool, str]]:
        """Check if required libraries are available"""
        required_libs = [
            'rclpy',           # ROS 2 Python client library
            'numpy',           # Numerical computing
            'scipy',           # Scientific computing
            'cv2',             # OpenCV for computer vision
            'torch',           # PyTorch for ML (if using)
            'transformers',    # Hugging Face transformers
            'openai',          # OpenAI API (if using)
            'ollama',          # Ollama client
            'speech_recognition',  # Speech recognition
            'pyaudio',         # Audio processing
            'yaml',            # YAML parsing
            'dataclasses',     # Data class utilities
            'typing',          # Type hints
            'asyncio',         # Async programming
            'functools',       # Functional programming tools
        ]
        
        results = {}
        
        for lib in required_libs:
            try:
                importlib.import_module(lib)
                results[lib] = (True, f"Library '{lib}' is available")
            except ImportError as e:
                results[lib] = (False, f"Library '{lib}' is not available: {str(e)}")
        
        return results
    
    def check_optional_libraries(self) -> Dict[str, Tuple[bool, str]]:
        """Check if optional libraries are available"""
        optional_libs = [
            'whisper',         # OpenAI Whisper
            'pyttsx3',         # Text-to-speech
            'playsound',       # Audio playback
            'mediapipe',       # MediaPipe for pose estimation
            'gym',             # OpenAI Gym for RL
            'stable_baselines3', # RL library
        ]
        
        results = {}
        
        for lib in optional_libs:
            try:
                importlib.import_module(lib)
                results[lib] = (True, f"Optional library '{lib}' is available")
            except ImportError as e:
                results[lib] = (False, f"Optional library '{lib}' is not available: {str(e)}")
        
        return results
    
    def check_c_compiler(self) -> Tuple[bool, str]:
        """Check if C compiler is available (needed for some packages)"""
        try:
            result = subprocess.run(['gcc', '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            if result.returncode == 0:
                return True, "C compiler (gcc) is available"
            else:
                return False, "C compiler (gcc) is not available or not working"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False, "C compiler (gcc) is not available"
    
    def check_gpu_support(self) -> Tuple[bool, str]:
        """Check if GPU support is available (for accelerated processing)"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_gpu = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_gpu)
                return True, f"GPU support available: {gpu_count} GPU(s), current: {gpu_name}"
            else:
                return False, "No GPU available, will use CPU"
        except ImportError:
            return False, "PyTorch not available to check GPU support"
        except Exception as e:
            return False, f"Error checking GPU support: {str(e)}"
    
    def run_all_checks(self) -> Dict[str, any]:
        """Run all compatibility checks"""
        print("Running Python compatibility checks for VLA system...")
        print("="*60)
        
        # Check Python version
        version_ok, version_msg = self.check_python_version()
        print(f"Python Version Check: {'✓' if version_ok else '✗'} {version_msg}")
        self.check_results['python_version'] = (version_ok, version_msg)
        
        # Check OS compatibility
        os_ok, os_msg = self.check_os_compatibility()
        print(f"OS Compatibility: {'✓' if os_ok else '✗'} {os_msg}")
        self.check_results['os_compatibility'] = (os_ok, os_msg)
        
        # Check C compiler
        compiler_ok, compiler_msg = self.check_c_compiler()
        print(f"C Compiler: {'✓' if compiler_ok else '✗'} {compiler_msg}")
        self.check_results['c_compiler'] = (compiler_ok, compiler_msg)
        
        # Check required libraries
        print("\nRequired Libraries:")
        required_results = self.check_required_libraries()
        all_required_ok = True
        for lib, (ok, msg) in required_results.items():
            print(f"  {lib}: {'✓' if ok else '✗'} {msg}")
            if not ok:
                all_required_ok = False
        self.check_results['required_libraries'] = (all_required_ok, required_results)
        
        # Check optional libraries
        print("\nOptional Libraries:")
        optional_results = self.check_optional_libraries()
        for lib, (ok, msg) in optional_results.items():
            print(f"  {lib}: {'✓' if ok else '✗'} {msg}")
        self.check_results['optional_libraries'] = (True, optional_results)  # Optional, so always OK
        
        # Check GPU support
        gpu_ok, gpu_msg = self.check_gpu_support()
        print(f"\nGPU Support: {'✓' if gpu_ok else '✗'} {gpu_msg}")
        self.check_results['gpu_support'] = (gpu_ok, gpu_msg)
        
        # Overall result
        overall_ok = version_ok and os_ok and all_required_ok
        print(f"\n{'='*60}")
        print(f"Overall Compatibility: {'✓ COMPATIBLE' if overall_ok else '✗ INCOMPATIBLE'}")
        
        if not overall_ok:
            print("\nRecommendations:")
            if not version_ok:
                print("- Upgrade Python to version 3.11 or higher")
            if not all_required_ok:
                print("- Install missing required libraries using pip")
            print("\nFor setup instructions, see: docs/04-vla/setup-instructions.md")
        
        return self.check_results
    
    def generate_requirements_file(self, filename: str = "requirements-vla.txt"):
        """Generate a requirements file for VLA system"""
        requirements = [
            "# VLA System Requirements",
            "# Generated by PythonCompatibilityChecker",
            "",
            "# Core dependencies",
            "rclpy>=3.0.0",
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "",
            "# Audio processing",
            "pyaudio>=0.2.11",
            "speech-recognition>=3.8.1",
            "whisper==1.0",  # Placeholder - actual Whisper package
            "",
            "# AI/ML libraries",
            "torch>=1.12.0",
            "transformers>=4.20.0",
            "ollama>=0.1.0",
            "",
            "# Computer vision",
            "opencv-python>=4.5.0",
            "mediapipe>=0.8.0",
            "",
            "# Utilities",
            "pyyaml>=6.0",
            "dataclasses-json>=0.5.0",
            "requests>=2.25.0",
            "",
            "# Development tools",
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
        
        with open(filename, 'w') as f:
            f.write('\n'.join(requirements))
        
        print(f"\nRequirements file generated: {filename}")
    
    def print_environment_info(self):
        """Print detailed environment information"""
        print("\nEnvironment Information:")
        print(f"  Python Version: {platform.python_version()}")
        print(f"  Python Implementation: {platform.python_implementation()}")
        print(f"  Platform: {platform.platform()}")
        print(f"  System: {platform.system()} {platform.release()}")
        print(f"  Architecture: {platform.architecture()[0]}")
        print(f"  Machine: {platform.machine()}")
        print(f"  Processor: {platform.processor()}")
        print(f"  Working Directory: {os.getcwd()}")


def main():
    """Main function to run compatibility checks"""
    print("VLA System - Python Compatibility Verification")
    print("="*60)
    
    checker = PythonCompatibilityChecker()
    
    # Print environment info
    checker.print_environment_info()
    
    # Run all checks
    results = checker.run_all_checks()
    
    # Generate requirements file
    checker.generate_requirements_file()
    
    # Return exit code based on compatibility
    version_ok, _ = results.get('python_version', (False, ""))
    required_ok, _ = results.get('required_libraries', (False, {}))
    
    if version_ok and required_ok[0]:  # required_ok is (all_ok, details_dict)
        print("\n✓ All compatibility checks passed!")
        return 0
    else:
        print("\n✗ Some compatibility checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
