"""
Audio Input Handler for VLA System
Manages audio capture, preprocessing, and real-time processing for voice commands
"""

import pyaudio
import numpy as np
import threading
import queue
import time
from typing import Dict, Any, Callable, Optional
import struct
import collections
import math

from common_data_models import VoiceCommand
from observability_utils import EducationalLogger, MetricsCollector, DebuggingHelper
from error_handling_utils import EducationalErrorHandler, InputValidator


class AudioInputHandler:
    """
    Handles audio input from microphone with preprocessing for voice command recognition
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 callback: Optional[Callable[[bytes], None]] = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.callback = callback
        
        # Initialize educational utilities
        self.logger = EducationalLogger("audio_input")
        self.metrics = MetricsCollector(self.logger)
        self.debug_helper = DebuggingHelper(self.logger)
        self.error_handler = EducationalErrorHandler()
        self.validator = InputValidator(self.error_handler)
        
        # Audio processing parameters
        self.audio_format = pyaudio.paInt16
        self.frame_size = chunk_size
        self.buffer_size = chunk_size * 4  # 4x chunk size for buffering
        
        # Initialize PyAudio
        self.pyaudio_instance = pyaudio.PyAudio()
        
        # Audio processing variables
        self.stream = None
        self.is_recording = False
        self.recording_thread = None
        self.audio_queue = queue.Queue()
        self.energy_history = collections.deque(maxlen=100)  # Keep last 100 energy readings
        
        # Voice activity detection parameters
        self.voice_threshold = 500  # Initial threshold for voice detection
        self.noise_floor = 100      # Estimated noise floor
        self.speech_buffer = b""    # Buffer to store speech segments
        
        # Preprocessing parameters
        self.noise_reduction_enabled = True
        self.agc_enabled = True     # Automatic gain control
        self.vad_enabled = True     # Voice activity detection
        
        self.logger.info("AudioInputHandler", "__init__", "Audio input handler initialized", {
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'chunk_size': self.chunk_size
        })
    
    def initialize_stream(self):
        """Initialize the audio input stream"""
        try:
            self.stream = self.pyaudio_instance.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback if self.callback else None
            )
            
            self.logger.info("AudioInputHandler", "initialize_stream", "Audio stream initialized", {
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'format': 'Int16'
            })
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="initialize_stream",
                suggested_fix="Check microphone availability and audio drivers",
                learning_objective="Understand audio stream initialization"
            )
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Internal audio callback for real-time processing"""
        if self.vad_enabled:
            # Perform voice activity detection
            is_speech = self._detect_voice_activity(in_data)
            
            if is_speech:
                # Add to speech buffer
                self.speech_buffer += in_data
                
                # If buffer is large enough, process it
                if len(self.speech_buffer) >= self.chunk_size * 8:  # 8 chunks worth
                    # Process the speech buffer
                    processed_audio = self._preprocess_audio(self.speech_buffer)
                    
                    # Call the user callback
                    if self.callback:
                        self.callback(processed_audio)
                    
                    # Clear the buffer
                    self.speech_buffer = b""
            else:
                # If we have accumulated speech and now detect silence, process it
                if len(self.speech_buffer) >= self.chunk_size:
                    processed_audio = self._preprocess_audio(self.speech_buffer)
                    
                    if self.callback:
                        self.callback(processed_audio)
                    
                    self.speech_buffer = b""
        else:
            # No VAD, just preprocess and pass through
            processed_audio = self._preprocess_audio(in_data)
            if self.callback:
                self.callback(processed_audio)
        
        return (None, pyaudio.paContinue)
    
    def _detect_voice_activity(self, audio_data: bytes) -> bool:
        """Detect voice activity in audio data"""
        try:
            # Convert bytes to numpy array for processing
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Calculate energy (RMS)
            energy = np.sqrt(np.mean(audio_array ** 2))
            
            # Update energy history
            self.energy_history.append(energy)
            
            # Update noise floor estimate (slow adaptation)
            if energy < self.noise_floor:
                self.noise_floor = self.noise_floor * 0.99 + energy * 0.01
            elif energy > self.noise_floor * 3:  # If significantly louder than noise
                self.noise_floor = self.noise_floor * 0.9 + energy * 0.1
            
            # Calculate adaptive threshold
            adaptive_threshold = self.noise_floor * 3  # 3x noise floor
            
            # Update voice threshold based on recent history
            if self.energy_history:
                recent_max = max(list(self.energy_history)[-10:]) if len(self.energy_history) >= 10 else self.noise_floor * 2
                self.voice_threshold = max(self.noise_floor * 2, min(adaptive_threshold, recent_max * 0.3))
            
            # Voice activity if energy exceeds threshold
            is_voice = energy > self.voice_threshold
            
            # Record metrics
            self.metrics.record_metric('audio_energy', energy)
            self.metrics.record_metric('voice_threshold', self.voice_threshold)
            self.metrics.record_metric('noise_floor', self.noise_floor)
            
            return is_voice
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="detect_voice_activity",
                suggested_fix="Check audio data format and processing",
                learning_objective="Understand voice activity detection algorithms"
            )
            return False
    
    def _preprocess_audio(self, audio_data: bytes) -> bytes:
        """Apply preprocessing to audio data"""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Apply noise reduction if enabled
            if self.noise_reduction_enabled:
                audio_array = self._apply_noise_reduction(audio_array)
            
            # Apply automatic gain control if enabled
            if self.agc_enabled:
                audio_array = self._apply_agc(audio_array)
            
            # Convert back to bytes
            processed_audio = (audio_array * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
            
            # Record preprocessing metrics
            self.metrics.record_metric('preprocessing_applied', 1)
            
            return processed_audio
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="preprocess_audio",
                suggested_fix="Check audio data format and preprocessing parameters",
                learning_objective="Learn about audio signal preprocessing"
            )
            return audio_data  # Return original if preprocessing fails
    
    def _apply_noise_reduction(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction to audio array"""
        try:
            # Simple spectral subtraction approach
            # Calculate short-term average for noise estimation
            frame_size = 256
            if len(audio_array) >= frame_size:
                # Estimate noise profile from low-energy segments
                energy = np.abs(audio_array)
                noise_mask = energy < np.mean(energy) * 0.5  # Assume low-energy parts are noise
                
                if np.any(noise_mask):
                    noise_profile = np.mean(audio_array[noise_mask])
                    
                    # Apply noise reduction
                    reduced = np.copy(audio_array)
                    for i in range(0, len(reduced), frame_size):
                        frame = reduced[i:i+frame_size]
                        frame_energy = np.mean(np.abs(frame))
                        
                        # Only reduce noise in high-energy frames
                        if frame_energy > abs(noise_profile) * 2:
                            reduced[i:i+frame_size] = np.sign(frame) * np.maximum(
                                np.abs(frame) - abs(noise_profile) * 0.3,  # Reduce noise level
                                0  # Ensure non-negative
                            )
                    
                    return reduced
            
            return audio_array
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="apply_noise_reduction",
                suggested_fix="Check noise reduction algorithm implementation",
                learning_objective="Understand audio noise reduction techniques"
            )
            return audio_array
    
    def _apply_agc(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply automatic gain control to normalize audio levels"""
        try:
            # Calculate current RMS level
            current_rms = np.sqrt(np.mean(audio_array ** 2))
            
            if current_rms == 0:
                return audio_array
            
            # Target RMS level (adjustable)
            target_rms = 0.1  # Moderate level to avoid clipping
            
            # Calculate gain
            gain = target_rms / current_rms
            
            # Apply gain with clipping prevention
            adjusted_audio = audio_array * min(gain, 1.0)  # Only reduce, don't amplify too much
            
            # Record AGC metrics
            self.metrics.record_metric('agc_gain_applied', gain)
            
            return adjusted_audio
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="apply_agc",
                suggested_fix="Check automatic gain control implementation",
                learning_objective="Learn about audio level normalization"
            )
            return audio_array
    
    def start_recording(self):
        """Start audio recording"""
        try:
            if self.stream is None:
                self.initialize_stream()
            
            self.is_recording = True
            
            # If no callback was provided during initialization, use our own processing loop
            if self.callback is None:
                self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
                self.recording_thread.start()
            
            self.stream.start_stream()
            
            self.logger.info("AudioInputHandler", "start_recording", "Audio recording started")
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="start_recording",
                suggested_fix="Check audio stream initialization",
                learning_objective="Understand audio recording lifecycle"
            )
    
    def _recording_loop(self):
        """Recording loop for when no callback is provided"""
        while self.is_recording and self.stream.is_active():
            try:
                # Read audio data
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Process the audio data
                processed_audio = self._preprocess_audio(audio_data)
                
                # Add to queue for external processing
                self.audio_queue.put(processed_audio)
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
                
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context="recording_loop",
                    suggested_fix="Check audio reading and processing",
                    learning_objective="Handle real-time audio processing errors"
                )
                break
    
    def stop_recording(self):
        """Stop audio recording"""
        try:
            self.is_recording = False
            
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
            
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            
            self.logger.info("AudioInputHandler", "stop_recording", "Audio recording stopped")
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="stop_recording",
                suggested_fix="Check audio stream cleanup",
                learning_objective="Understand proper resource cleanup"
            )
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[bytes]:
        """Get an audio chunk from the queue (non-blocking)"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def calibrate_noise_floor(self, duration: float = 3.0) -> float:
        """Calibrate noise floor by measuring ambient noise"""
        self.logger.info("AudioInputHandler", "calibrate_noise_floor", f"Starting {duration}s noise floor calibration")
        
        original_vad = self.vad_enabled
        self.vad_enabled = False  # Disable VAD during calibration
        
        try:
            # Collect audio samples for calibration
            samples = []
            start_time = time.time()
            
            while time.time() - start_time < duration:
                if self.stream:
                    audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    energy = np.sqrt(np.mean(audio_array ** 2))
                    samples.append(energy)
                else:
                    time.sleep(0.01)
            
            if samples:
                # Calculate noise floor as 10th percentile of energy
                self.noise_floor = np.percentile(samples, 10)
                self.voice_threshold = self.noise_floor * 3  # Set threshold to 3x noise floor
                
                self.logger.info("AudioInputHandler", "calibrate_noise_floor", "Calibration completed", {
                    'noise_floor': self.noise_floor,
                    'voice_threshold': self.voice_threshold
                })
                
                return self.noise_floor
            else:
                return self.noise_floor
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="calibrate_noise_floor",
                suggested_fix="Check audio stream during calibration",
                learning_objective="Understand audio calibration techniques"
            )
            return self.noise_floor
        finally:
            self.vad_enabled = original_vad  # Restore original VAD setting
    
    def get_audio_stats(self) -> Dict[str, Any]:
        """Get current audio statistics"""
        return {
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'chunk_size': self.chunk_size,
            'is_recording': self.is_recording,
            'queue_size': self.audio_queue.qsize(),
            'noise_floor': self.noise_floor,
            'voice_threshold': self.voice_threshold,
            'energy_history': list(self.energy_history)[-10:] if self.energy_history else [],
            'metrics': self.metrics.get_statistics()
        }
    
    def set_callback(self, callback: Callable[[bytes], None]):
        """Set the callback function for processed audio data"""
        self.callback = callback


def main():
    """Main function for testing audio input handler"""
    print("Audio Input Handler - Voice Command Processing")
    print("="*55)
    
    # Create audio input handler
    def audio_callback(audio_data):
        print(f"Received audio chunk of size: {len(audio_data)}")
    
    handler = AudioInputHandler(callback=audio_callback)
    
    print("Audio input handler initialized with parameters:")
    print(f"  Sample Rate: {handler.sample_rate} Hz")
    print(f"  Channels: {handler.channels}")
    print(f"  Chunk Size: {handler.chunk_size}")
    
    # Calibrate noise floor
    print("\nCalibrating noise floor (speak as little as possible)...")
    noise_floor = handler.calibrate_noise_floor(2.0)  # 2 second calibration
    print(f"Calibrated noise floor: {noise_floor:.6f}")
    
    # Show initial stats
    stats = handler.get_audio_stats()
    print(f"\nInitial audio stats: {stats}")
    
    try:
        # Start recording
        handler.start_recording()
        print("\nRecording started. Speak into the microphone...")
        print("Press Ctrl+C to stop recording.")
        
        # Record for a while
        start_time = time.time()
        while time.time() - start_time < 10:  # Record for 10 seconds
            time.sleep(0.1)
            
            # Show updated stats periodically
            if int(time.time() - start_time) % 3 == 0:
                stats = handler.get_audio_stats()
                print(f"Energy: {stats.get('energy_history', ['N/A'])[-1] if stats.get('energy_history') else 'N/A'}")
    
    except KeyboardInterrupt:
        print("\nStopping recording...")
    
    finally:
        handler.stop_recording()
        print("Audio input handler stopped.")
        
        # Show final metrics
        final_stats = handler.get_audio_stats()
        print(f"\nFinal audio stats: {final_stats}")


if __name__ == "__main__":
    main()
