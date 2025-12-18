"""
Whisper Service for VLA System
Local Whisper model integration using Whisper.cpp for privacy-compliant processing
"""

import os
import sys
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import threading
from queue import Queue, Empty
import subprocess
import json
import tempfile

from common_data_models import VoiceCommand
from observability_utils import EducationalLogger, MetricsCollector, DebuggingHelper
from error_handling_utils import EducationalErrorHandler, InputValidator


@dataclass
class WhisperConfig:
    """Configuration for Whisper model integration"""
    model_path: str = "models/ggml-tiny.en.bin"  # Default English model
    language: str = "en"
    translate: bool = False
    beam_size: int = 5
    temperature: float = 0.0
    patience: float = 1.0
    length_penalty: float = 1.0
    max_tokens: int = 448
    offset: int = 0
    duration: int = 0
    token_timestamps: bool = False
    thold_pt: float = 0.01
    thold_ptsum: float = 0.01
    max_len: int = 0
    split_on_word: bool = False
    max_tokens_suppress: int = 1
    print_special: bool = False
    print_progress: bool = True
    print_realtime: bool = False
    print_timestamps: bool = True
    timestamp_sample_rate: int = 1
    speed_up: bool = False
    threads: int = 4  # Number of threads for processing
    processors: int = 1  # Number of processors to use


class WhisperService:
    """
    Service class for Whisper model integration using Whisper.cpp
    Provides local, privacy-compliant speech-to-text functionality
    """
    
    def __init__(self, config: WhisperConfig):
        self.config = config
        self.logger = EducationalLogger("whisper_service")
        self.metrics = MetricsCollector(self.logger)
        self.debug_helper = DebuggingHelper(self.logger)
        self.error_handler = EducationalErrorHandler()
        self.validator = InputValidator(self.error_handler)
        
        # Validate configuration
        self._validate_config()
        
        # Check if whisper.cpp is available
        self.whisper_cpp_available = self._check_whisper_cpp_availability()
        
        # Initialize processing queue
        self.processing_queue = Queue()
        self.result_queue = Queue()
        self.is_processing = False
        
        # Processing statistics
        self.total_processed = 0
        self.total_errors = 0
        
        self.logger.info("WhisperService", "__init__", "Whisper service initialized", {
            'model_path': self.config.model_path,
            'language': self.config.language,
            'whisper_cpp_available': self.whisper_cpp_available
        })
    
    def _validate_config(self):
        """Validate Whisper configuration"""
        try:
            # Validate model path exists
            if not os.path.exists(self.config.model_path):
                self.logger.warning("WhisperService", "_validate_config", 
                                  f"Model file not found: {self.config.model_path}")
            
            # Validate numeric parameters
            self.validator.validate_range(self.config.beam_size, 1, 10, "beam_size")
            self.validator.validate_range(self.config.temperature, 0.0, 1.0, "temperature")
            self.validator.validate_range(self.config.patience, 0.0, 5.0, "patience")
            self.validator.validate_range(self.config.threads, 1, 16, "threads")
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="config_validation",
                suggested_fix="Check Whisper configuration parameters",
                learning_objective="Understand configuration validation for ML models"
            )
    
    def _check_whisper_cpp_availability(self) -> bool:
        """Check if whisper.cpp is available and properly installed"""
        try:
            # Check if whisper.cpp executable exists in path
            result = subprocess.run(['which', 'whisper'], capture_output=True, text=True)
            if result.returncode == 0:
                return True
            
            # Check if whisper.cpp directory exists in common locations
            common_paths = [
                "./whisper.cpp/main",
                "../whisper.cpp/main",
                "/opt/whisper.cpp/main",
                os.path.expanduser("~/whisper.cpp/main")
            ]
            
            for path in common_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    return True
            
            self.logger.warning("WhisperService", "_check_whisper_cpp_availability", 
                              "Whisper.cpp not found in expected locations")
            return False
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="whisper_cpp_check",
                suggested_fix="Install whisper.cpp from https://github.com/ggerganov/whisper.cpp",
                learning_objective="Learn about external dependency management"
            )
            return False
    
    def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Transcribe audio data using Whisper model
        This is the main method for voice-to-text conversion
        """
        start_time = time.time()
        
        try:
            self.logger.debug("WhisperService", "transcribe_audio", "Starting transcription", {
                'audio_data_size': len(audio_data),
                'sample_rate': sample_rate
            })
            
            # Validate inputs
            if not self.validator.validate_not_empty(audio_data, "audio_data"):
                return self._create_error_result("Audio data is empty")
            
            if not self.validator.validate_range(sample_rate, 8000, 48000, "sample_rate"):
                return self._create_error_result("Invalid sample rate")
            
            # Save audio data to temporary file for Whisper.cpp processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                # Convert raw audio bytes to WAV format
                # This is a simplified conversion - in practice, use proper WAV writing
                wav_data = self._convert_to_wav(audio_data, sample_rate)
                temp_audio.write(wav_data)
                temp_audio_path = temp_audio.name
            
            try:
                # Call Whisper.cpp for transcription
                if self.whisper_cpp_available:
                    result = self._call_whisper_cpp(temp_audio_path)
                else:
                    # Fallback: simulate transcription for educational purposes
                    result = self._simulate_transcription(audio_data)
                
                # Calculate processing metrics
                processing_time = time.time() - start_time
                self.metrics.record_metric('transcription_time', processing_time)
                self.metrics.record_metric('audio_size_processed', len(audio_data))
                
                # Add processing statistics
                result['processing_time'] = processing_time
                result['audio_size'] = len(audio_data)
                
                self.total_processed += 1
                self.logger.info("WhisperService", "transcribe_audio", "Transcription completed", {
                    'transcript_length': len(result.get('transcript', '')),
                    'confidence': result.get('confidence', 0),
                    'processing_time': processing_time
                })
                
                return result
                
            finally:
                # Clean up temporary file
                os.unlink(temp_audio_path)
                
        except Exception as e:
            self.total_errors += 1
            processing_time = time.time() - start_time
            self.metrics.record_metric('transcription_error_time', processing_time)
            
            self.error_handler.handle_error(
                e,
                context="transcribe_audio",
                suggested_fix="Check audio format and Whisper model availability",
                learning_objective="Understand error handling in ML inference"
            )
            
            return self._create_error_result(str(e))
    
    def _convert_to_wav(self, audio_data: bytes, sample_rate: int) -> bytes:
        """
        Convert raw audio bytes to WAV format for Whisper.cpp
        This is a simplified implementation for educational purposes
        """
        try:
            import wave
            import io
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_array.tobytes())
            
            return wav_buffer.getvalue()
            
        except ImportError:
            # If wave module not available, return original data
            self.logger.warning("WhisperService", "_convert_to_wav", 
                              "wave module not available, returning raw data")
            return audio_data
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="convert_to_wav",
                suggested_fix="Install required audio processing libraries",
                learning_objective="Understand audio format conversion"
            )
            return audio_data
    
    def _call_whisper_cpp(self, audio_file_path: str) -> Dict[str, Any]:
        """Call Whisper.cpp for actual transcription"""
        try:
            # Build Whisper.cpp command
            cmd = [
                'whisper',  # This assumes whisper is in PATH
                audio_file_path,
                '--model', self.config.model_path,
                '--language', self.config.language,
                '--threads', str(self.config.threads),
                '--max-tokens', str(self.config.max_tokens),
                '--offset', str(self.config.offset),
                '--duration', str(self.config.duration),
                '--beam-size', str(self.config.beam_size),
                '--temperature', str(self.config.temperature),
                '--no-timestamps'  # Simplify output for this implementation
            ]
            
            if self.config.translate:
                cmd.append('--translate')
            if not self.config.print_progress:
                cmd.append('--no-print-progress')
            
            # Execute Whisper.cpp
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 1-minute timeout
            )
            
            if result.returncode != 0:
                self.logger.error("WhisperService", "_call_whisper_cpp", 
                                f"Whisper.cpp failed: {result.stderr}")
                return self._create_error_result(f"Whisper.cpp error: {result.stderr}")
            
            # Parse Whisper.cpp output
            transcript = self._parse_whisper_output(result.stdout)
            
            # Calculate confidence (simplified approach)
            confidence = self._calculate_confidence(transcript, result.stdout)
            
            return {
                'transcript': transcript,
                'confidence': confidence,
                'raw_output': result.stdout,
                'success': True,
                'model_used': os.path.basename(self.config.model_path)
            }
            
        except subprocess.TimeoutExpired:
            return self._create_error_result("Whisper transcription timed out")
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="call_whisper_cpp",
                suggested_fix="Check Whisper.cpp installation and model file",
                learning_objective="Handle subprocess execution errors"
            )
            return self._create_error_result(f"Subprocess error: {str(e)}")
    
    def _parse_whisper_output(self, output: str) -> str:
        """Parse Whisper.cpp output to extract transcript"""
        # This is a simplified parser - Whisper.cpp output format may vary
        # In practice, you'd want a more robust parser
        lines = output.strip().split('\n')
        
        # Look for lines that contain transcriptions
        transcript_parts = []
        for line in lines:
            # Remove timestamp prefixes like [00:00:00.000 --> 00:00:01.000]
            import re
            clean_line = re.sub(r'\[\d\d:\d\d:\d\d\.\d\d\d --> \d\d:\d\d:\d\d\.\d\d\d\]\s*', '', line).strip()
            
            # Skip empty lines and metadata
            if clean_line and not clean_line.startswith('whisper_'):
                transcript_parts.append(clean_line)
        
        return ' '.join(transcript_parts).strip()
    
    def _calculate_confidence(self, transcript: str, raw_output: str) -> float:
        """Calculate confidence score for transcription"""
        if not transcript.strip():
            return 0.0
        
        # Simple confidence calculation based on transcript characteristics
        # In practice, Whisper models provide confidence scores directly
        length_score = min(len(transcript) / 100.0, 0.5)  # Longer transcripts get higher score up to 0.5
        word_count = len(transcript.split())
        word_score = min(word_count / 20.0, 0.5)  # More words get higher score up to 0.5
        
        # Combine scores
        confidence = length_score + word_score
        
        # Ensure in [0, 1] range
        return max(0.0, min(1.0, confidence))
    
    def _simulate_transcription(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Simulate transcription when Whisper.cpp is not available
        This maintains functionality for educational purposes
        """
        self.logger.warning("WhisperService", "_simulate_transcription", 
                          "Whisper.cpp not available, using simulation")
        
        # Calculate audio characteristics for simulation
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        energy = np.mean(np.abs(audio_array))
        
        # Generate simulated transcript based on audio energy
        if energy > 0.05:  # High energy = likely speech
            if energy > 0.2:
                transcript = "this is a test voice command for the robot"
            else:
                transcript = "voice command detected"
        else:
            transcript = ""  # No significant audio
        
        confidence = min(energy * 5, 0.8)  # Map energy to confidence with max 0.8 for simulation
        
        return {
            'transcript': transcript,
            'confidence': confidence,
            'success': transcript != "",
            'model_used': 'simulation',
            'note': 'Whisper.cpp not available - using simulation'
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create a standardized error result"""
        return {
            'transcript': '',
            'confidence': 0.0,
            'success': False,
            'error': error_message,
            'model_used': 'none'
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current status of the Whisper service"""
        return {
            'whisper_cpp_available': self.whisper_cpp_available,
            'config': {
                'model_path': self.config.model_path,
                'language': self.config.language,
                'threads': self.config.threads
            },
            'statistics': {
                'total_processed': self.total_processed,
                'total_errors': self.total_errors,
                'error_rate': self.total_errors / max(1, self.total_processed) if self.total_processed > 0 else 0
            },
            'metrics': self.metrics.get_statistics(),
            'queue_sizes': {
                'processing': self.processing_queue.qsize(),
                'results': self.result_queue.qsize()
            }
        }
    
    def load_model(self) -> bool:
        """Load Whisper model into memory (if supported by the backend)"""
        # This would be implemented if using a Python binding for Whisper.cpp
        # For now, we just verify the model file exists
        try:
            if os.path.exists(self.config.model_path):
                self.logger.info("WhisperService", "load_model", "Model verification successful", {
                    'model_path': self.config.model_path,
                    'model_size': os.path.getsize(self.config.model_path)
                })
                return True
            else:
                self.logger.error("WhisperService", "load_model", "Model file not found", {
                    'model_path': self.config.model_path
                })
                return False
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="load_model",
                suggested_fix="Verify model file path and permissions",
                learning_objective="Learn about model loading and verification"
            )
            return False


def main():
    """Main function for testing Whisper service"""
    print("Whisper Service - Local Speech Recognition")
    print("="*50)
    
    # Create Whisper configuration
    config = WhisperConfig(
        model_path="models/ggml-tiny.en.bin",  # Default model path
        language="en",
        threads=2
    )
    
    # Initialize Whisper service
    service = WhisperService(config)
    status = service.get_service_status()
    
    print(f"Whisper service initialized:")
    print(f"  Available: {status['whisper_cpp_available']}")
    print(f"  Model: {status['config']['model_path']}")
    print(f"  Language: {status['config']['language']}")
    
    if not status['whisper_cpp_available']:
        print("\n⚠️  Whisper.cpp not found. This is expected in development environment.")
        print("   Install from: https://github.com/ggerganov/whisper.cpp")
    
    # Test with simulated audio data (for demonstration)
    print("\nTesting with simulated audio data...")
    dummy_audio = b'\x00' * 32000  # 1 second of silence at 16kHz, 16-bit
    result = service.transcribe_audio(dummy_audio)
    
    print(f"Test result: {result}")
    print(f"Transcript: '{result.get('transcript', '')}'")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    
    print(f"\nService status: {service.get_service_status()}")


if __name__ == "__main__":
    main()
