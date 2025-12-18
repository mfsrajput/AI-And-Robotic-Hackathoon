"""
Noise Reduction and Preprocessing Utilities for VLA System
Advanced audio preprocessing techniques for voice command enhancement
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, wiener
import librosa
from typing import Tuple, Optional, Dict, Any
import warnings

from observability_utils import EducationalLogger, MetricsCollector, DebuggingHelper
from error_handling_utils import EducationalErrorHandler, InputValidator


class NoiseReductionProcessor:
    """
    Advanced noise reduction processor using multiple techniques
    """
    
    def __init__(self):
        self.logger = EducationalLogger("noise_reduction")
        self.metrics = MetricsCollector(self.logger)
        self.debug_helper = DebuggingHelper(self.logger)
        self.error_handler = EducationalErrorHandler()
        self.validator = InputValidator(self.error_handler)
        
        # Initialize processing parameters
        self.noise_profile = None
        self.noise_profile_size = 2048  # Size of noise profile buffer
        self.fft_size = 1024
        self.hop_length = 256
        
        self.logger.info("NoiseReductionProcessor", "__init__", "Noise reduction processor initialized")
    
    def estimate_noise_profile(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Estimate noise profile from a segment of audio that contains only noise
        """
        try:
            self.logger.debug("NoiseReductionProcessor", "estimate_noise_profile", "Estimating noise profile", {
                'audio_length': len(audio_data),
                'sample_rate': sample_rate
            })
            
            # Ensure audio data is in the right format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Apply windowing to reduce spectral leakage
            windowed = audio_data * signal.windows.hann(len(audio_data))
            
            # Compute FFT
            fft = np.fft.rfft(windowed)
            magnitude = np.abs(fft)
            
            # Store as noise profile
            self.noise_profile = magnitude
            
            self.logger.info("NoiseReductionProcessor", "estimate_noise_profile", "Noise profile estimated", {
                'profile_length': len(magnitude)
            })
            
            return magnitude
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="estimate_noise_profile",
                suggested_fix="Check audio data format and length",
                learning_objective="Understand noise profile estimation"
            )
            return np.zeros(self.noise_profile_size // 2 + 1)
    
    def spectral_subtraction(self, audio_data: np.ndarray, noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply spectral subtraction noise reduction
        """
        try:
            start_time = self.metrics.start_timer('spectral_subtraction')
            
            if noise_profile is None:
                noise_profile = self.noise_profile
            
            if noise_profile is None:
                self.logger.warning("NoiseReductionProcessor", "spectral_subtraction", 
                                  "No noise profile available, returning original audio")
                return audio_data
            
            # Ensure audio data is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Process in frames
            frame_size = self.fft_size
            hop_size = self.hop_length
            
            # Pad audio to ensure we can process all frames
            pad_length = (frame_size - len(audio_data) % frame_size) % frame_size
            padded_audio = np.pad(audio_data, (0, pad_length), mode='constant')
            
            # Initialize output array
            output_audio = np.zeros_like(padded_audio)
            
            # Process each frame
            for i in range(0, len(padded_audio) - frame_size, hop_size):
                frame = padded_audio[i:i+frame_size]
                
                # Apply window
                windowed_frame = frame * signal.windows.hann(frame_size)
                
                # Compute FFT
                fft = np.fft.rfft(windowed_frame)
                magnitude = np.abs(fft)
                phase = np.angle(fft)
                
                # Apply spectral subtraction
                # Ensure noise profile is the same length as magnitude
                if len(noise_profile) != len(magnitude):
                    resized_noise = np.interp(
                        np.linspace(0, len(noise_profile)-1, len(magnitude)),
                        np.arange(len(noise_profile)),
                        noise_profile
                    )
                else:
                    resized_noise = noise_profile
                
                # Subtract noise (with flooring to prevent negative magnitudes)
                enhanced_magnitude = np.maximum(magnitude - resized_noise * 0.5, magnitude * 0.1)
                
                # Reconstruct the frame
                enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
                enhanced_frame = np.fft.irfft(enhanced_fft)
                
                # Apply window again for overlap-add
                enhanced_frame = enhanced_frame * signal.windows.hann(frame_size)
                
                # Overlap-add
                output_audio[i:i+frame_size] += enhanced_frame
            
            # Remove padding
            output_audio = output_audio[:len(audio_data)]
            
            processing_time = self.metrics.stop_timer('spectral_subtraction')
            self.metrics.record_metric('spectral_subtraction_time', processing_time)
            
            self.logger.debug("NoiseReductionProcessor", "spectral_subtraction", "Spectral subtraction completed", {
                'processing_time': processing_time
            })
            
            return output_audio
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="spectral_subtraction",
                suggested_fix="Check audio data format and noise profile",
                learning_objective="Handle spectral subtraction errors"
            )
            return audio_data
    
    def wiener_filtering(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply Wiener filtering for noise reduction
        """
        try:
            start_time = self.metrics.start_timer('wiener_filtering')
            
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Apply Wiener filter
            # This is a simplified approach - in practice, you'd estimate signal and noise PSDs
            filtered_audio = wiener(audio_data)
            
            processing_time = self.metrics.stop_timer('wiener_filtering')
            self.metrics.record_metric('wiener_filtering_time', processing_time)
            
            self.logger.debug("NoiseReductionProcessor", "wiener_filtering", "Wiener filtering completed", {
                'processing_time': processing_time
            })
            
            return filtered_audio
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="wiener_filtering",
                suggested_fix="Check audio data format for Wiener filtering",
                learning_objective="Handle Wiener filtering errors"
            )
            return audio_data
    
    def adaptive_filtering(self, audio_data: np.ndarray, reference_noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply adaptive filtering if reference noise is available
        """
        try:
            start_time = self.metrics.start_timer('adaptive_filtering')
            
            if reference_noise is None:
                self.logger.warning("NoiseReductionProcessor", "adaptive_filtering", 
                                  "No reference noise available for adaptive filtering")
                return audio_data
            
            # Ensure both signals are the same length
            min_length = min(len(audio_data), len(reference_noise))
            audio_data = audio_data[:min_length]
            reference_noise = reference_noise[:min_length]
            
            # Simple adaptive filtering using least mean squares approach
            # In practice, you'd use more sophisticated adaptive algorithms
            from scipy.signal import lfilter
            
            # Design a simple adaptive filter (simplified implementation)
            # This is a basic approach - real adaptive filtering is more complex
            output = audio_data - 0.1 * reference_noise  # Simple noise cancellation
            
            # Clip to prevent overflow
            output = np.clip(output, -1.0, 1.0)
            
            processing_time = self.metrics.stop_timer('adaptive_filtering')
            self.metrics.record_metric('adaptive_filtering_time', processing_time)
            
            self.logger.debug("NoiseReductionProcessor", "adaptive_filtering", "Adaptive filtering completed", {
                'processing_time': processing_time
            })
            
            return output
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="adaptive_filtering",
                suggested_fix="Check reference noise and audio data compatibility",
                learning_objective="Handle adaptive filtering errors"
            )
            return audio_data


class AudioPreprocessor:
    """
    Comprehensive audio preprocessor with multiple enhancement techniques
    """
    
    def __init__(self):
        self.logger = EducationalLogger("audio_preprocessor")
        self.metrics = MetricsCollector(self.logger)
        self.debug_helper = DebuggingHelper(self.logger)
        self.error_handler = EducationalErrorHandler()
        self.validator = InputValidator(self.error_handler)
        
        # Initialize noise reduction processor
        self.noise_reducer = NoiseReductionProcessor()
        
        # Initialize other processors
        self.sample_rate = 16000
        self.frame_size = 1024
        self.hop_length = 256
        
        # Preprocessing parameters
        self.preemphasis_coeff = 0.97  # For pre-emphasis filtering
        self.agc_target_level = 0.1    # Target level for automatic gain control
        self.normalization_enabled = True
        
        self.logger.info("AudioPreprocessor", "__init__", "Audio preprocessor initialized")
    
    def preprocess_audio(self, 
                        audio_data: np.ndarray, 
                        sample_rate: int = 16000,
                        apply_noise_reduction: bool = True,
                        apply_preemphasis: bool = True,
                        apply_agc: bool = True,
                        apply_normalization: bool = True) -> np.ndarray:
        """
        Apply comprehensive preprocessing to audio data
        """
        try:
            self.metrics.start_timer('total_preprocessing')
            
            # Validate inputs
            if len(audio_data) == 0:
                self.logger.warning("AudioPreprocessor", "preprocess_audio", "Empty audio data provided")
                return audio_data
            
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            original_audio = audio_data.copy()
            
            self.logger.debug("AudioPreprocessor", "preprocess_audio", "Starting preprocessing", {
                'audio_length': len(audio_data),
                'sample_rate': sample_rate,
                'original_rms': np.sqrt(np.mean(audio_data ** 2))
            })
            
            # Apply noise reduction if enabled
            if apply_noise_reduction:
                audio_data = self.noise_reducer.spectral_subtraction(audio_data)
            
            # Apply pre-emphasis filtering to boost high frequencies
            if apply_preemphasis:
                audio_data = self._apply_preemphasis(audio_data)
            
            # Apply automatic gain control
            if apply_agc:
                audio_data = self._apply_agc(audio_data)
            
            # Apply normalization
            if apply_normalization and self.normalization_enabled:
                audio_data = self._apply_normalization(audio_data)
            
            # Update metrics
            processing_time = self.metrics.stop_timer('total_preprocessing')
            self.metrics.record_metric('total_preprocessing_time', processing_time)
            self.metrics.record_metric('audio_energy_before', np.sqrt(np.mean(original_audio ** 2)))
            self.metrics.record_metric('audio_energy_after', np.sqrt(np.mean(audio_data ** 2)))
            
            self.logger.debug("AudioPreprocessor", "preprocess_audio", "Preprocessing completed", {
                'processing_time': processing_time,
                'energy_change': (np.sqrt(np.mean(audio_data ** 2)) - np.sqrt(np.mean(original_audio ** 2)))
            })
            
            return audio_data
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="preprocess_audio",
                suggested_fix="Check audio data format and preprocessing parameters",
                learning_objective="Handle comprehensive audio preprocessing"
            )
            return audio_data  # Return original if preprocessing fails
    
    def _apply_preemphasis(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis filter to boost high frequencies"""
        try:
            # Simple first-order pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]
            preemphasized = np.zeros_like(audio_data)
            preemphasized[0] = audio_data[0]
            
            for i in range(1, len(audio_data)):
                preemphasized[i] = audio_data[i] - self.preemphasis_coeff * audio_data[i-1]
            
            return preemphasized
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="apply_preemphasis",
                suggested_fix="Check pre-emphasis filter implementation",
                learning_objective="Understand pre-emphasis filtering"
            )
            return audio_data
    
    def _apply_agc(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply automatic gain control to normalize audio level"""
        try:
            # Calculate current RMS level
            current_rms = np.sqrt(np.mean(audio_data ** 2))
            
            if current_rms == 0:
                return audio_data
            
            # Calculate required gain
            gain = self.agc_target_level / current_rms
            
            # Apply gain with limits to prevent excessive amplification
            gain = np.clip(gain, 0.1, 10.0)  # Limit gain to reasonable range
            
            adjusted_audio = audio_data * gain
            
            # Check for clipping
            max_val = np.max(np.abs(adjusted_audio))
            if max_val > 1.0:
                adjusted_audio = adjusted_audio / max_val * 0.99  # Prevent clipping
            
            return adjusted_audio
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="apply_agc",
                suggested_fix="Check automatic gain control implementation",
                learning_objective="Handle AGC errors"
            )
            return audio_data
    
    def _apply_normalization(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply normalization to audio data"""
        try:
            # Peak normalization
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                normalized_audio = audio_data / max_val
                # Scale to target range (e.g., 80% of maximum to avoid clipping)
                normalized_audio = normalized_audio * 0.8
            else:
                normalized_audio = audio_data
            
            return normalized_audio
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="apply_normalization",
                suggested_fix="Check normalization implementation",
                learning_objective="Handle normalization errors"
            )
            return audio_data
    
    def estimate_noise_profile_from_silence(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Estimate noise profile by analyzing low-energy segments (assumed to be silence)
        """
        try:
            self.logger.info("AudioPreprocessor", "estimate_noise_profile_from_silence", 
                           "Estimating noise profile from silence", {
                               'audio_length': len(audio_data)
                           })
            
            # Find low-energy segments (potential silence)
            frame_length = 1024
            hop_length = 512
            
            # Calculate energy for each frame
            energies = []
            frames = []
            
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i+frame_length]
                energy = np.mean(frame ** 2)
                energies.append(energy)
                frames.append(frame)
            
            if not energies:
                self.logger.warning("AudioPreprocessor", "estimate_noise_profile_from_silence", 
                                  "No frames found for noise estimation")
                return False
            
            # Find frames with low energy (bottom 25%)
            energy_threshold = np.percentile(energies, 25)
            low_energy_frames = [frames[i] for i in range(len(frames)) if energies[i] <= energy_threshold]
            
            if not low_energy_frames:
                self.logger.warning("AudioPreprocessor", "estimate_noise_profile_from_silence", 
                                  "No low-energy frames found for noise estimation")
                return False
            
            # Combine low-energy frames for noise profile
            combined_noise = np.concatenate(low_energy_frames)
            
            # Estimate noise profile
            noise_profile = self.noise_reducer.estimate_noise_profile(combined_noise, sample_rate)
            
            self.logger.info("AudioPreprocessor", "estimate_noise_profile_from_silence", 
                           "Noise profile estimated from silence", {
                               'low_energy_frames': len(low_energy_frames),
                               'profile_length': len(noise_profile)
                           })
            
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="estimate_noise_profile_from_silence",
                suggested_fix="Check silence detection and noise estimation",
                learning_objective="Handle noise profile estimation from silence"
            )
            return False
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get statistics about preprocessing performance"""
        return {
            'metrics': self.metrics.get_statistics(),
            'noise_reducer_stats': {
                # Add any specific noise reducer stats if available
            },
            'parameters': {
                'sample_rate': self.sample_rate,
                'frame_size': self.frame_size,
                'hop_length': self.hop_length,
                'preemphasis_coeff': self.preemphasis_coeff,
                'agc_target_level': self.agc_target_level
            }
        }


class VoiceActivityDetector:
    """
    Voice Activity Detection (VAD) for identifying speech segments
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.logger = EducationalLogger("vad")
        self.metrics = MetricsCollector(self.logger)
        self.debug_helper = DebuggingHelper(self.logger)
        self.error_handler = EducationalErrorHandler()
        self.validator = InputValidator(self.error_handler)
        
        # VAD parameters
        self.energy_threshold = 0.01  # Minimum energy for speech
        self.zero_crossing_threshold = 0.05  # Zero crossing rate threshold
        self.speech_padding = 0.1  # Seconds to pad detected speech
        self.min_speech_duration = 0.2  # Minimum speech duration in seconds
        
        # Adaptive parameters
        self.adaptive = True
        self.noise_floor = 0.001
        self.energy_history = []
        self.history_size = 100
        
        self.logger.info("VoiceActivityDetector", "__init__", "VAD initialized", {
            'sample_rate': self.sample_rate
        })
    
    def detect_voice_activity(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Detect voice activity in audio data
        Returns boolean array where True indicates speech
        """
        try:
            self.metrics.start_timer('vad_detection')
            
            # Calculate frame-based features
            frame_length = 512  # About 32ms at 16kHz
            hop_length = 256    # About 16ms at 16kHz
            
            # Calculate energy for each frame
            frames = []
            energies = []
            zero_crossing_rates = []
            
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i+frame_length]
                frames.append(frame)
                
                # Calculate energy (RMS)
                energy = np.sqrt(np.mean(frame ** 2))
                energies.append(energy)
                
                # Calculate zero crossing rate
                zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
                zero_crossing_rates.append(zcr)
            
            # Convert to numpy arrays
            energies = np.array(energies)
            zero_crossing_rates = np.array(zero_crossing_rates)
            
            # Apply adaptive thresholding if enabled
            if self.adaptive:
                # Update noise floor estimate
                if len(self.energy_history) < self.history_size:
                    self.energy_history.extend(energies)
                else:
                    self.energy_history = list(energies[-self.history_size//2:]) + self.energy_history[:self.history_size//2]
                
                # Calculate adaptive threshold based on recent history
                if self.energy_history:
                    recent_min = np.percentile(self.energy_history, 10)
                    recent_mean = np.mean(self.energy_history)
                    self.noise_floor = recent_min * 0.5 + recent_mean * 0.5
                    self.energy_threshold = self.noise_floor * 3  # 3x noise floor
            
            # Create speech detection mask
            energy_speech = energies > self.energy_threshold
            zcr_speech = zero_crossing_rates > self.zero_crossing_threshold
            
            # Combine criteria
            speech_mask = energy_speech & zcr_speech
            
            # Convert frame-level detection to sample-level detection
            sample_mask = np.zeros(len(audio_data), dtype=bool)
            
            for i, is_speech in enumerate(speech_mask):
                start_idx = i * hop_length
                end_idx = min(start_idx + frame_length, len(audio_data))
                if is_speech:
                    sample_mask[start_idx:end_idx] = True
            
            # Apply smoothing to remove short non-speech segments
            sample_mask = self._smooth_detection(sample_mask)
            
            processing_time = self.metrics.stop_timer('vad_detection')
            self.metrics.record_metric('vad_processing_time', processing_time)
            
            self.logger.debug("VoiceActivityDetector", "detect_voice_activity", "VAD completed", {
                'processing_time': processing_time,
                'speech_frames': np.sum(speech_mask),
                'total_frames': len(speech_mask)
            })
            
            return sample_mask
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="detect_voice_activity",
                suggested_fix="Check audio data format and VAD parameters",
                learning_objective="Handle VAD processing errors"
            )
            # Return all False if detection fails
            return np.zeros(len(audio_data), dtype=bool)
    
    def _smooth_detection(self, detection_mask: np.ndarray) -> np.ndarray:
        """Apply smoothing to voice activity detection"""
        try:
            # Remove short non-speech segments (less than 20ms)
            min_silence_length = int(0.02 * self.sample_rate / (self.hop_length))  # 20ms
            if min_silence_length < 1:
                min_silence_length = 1
            
            # Find runs of speech and non-speech
            runs = []
            current_run = {'start': 0, 'is_speech': detection_mask[0], 'length': 1}
            
            for i in range(1, len(detection_mask)):
                if detection_mask[i] == current_run['is_speech']:
                    current_run['length'] += 1
                else:
                    runs.append(current_run)
                    current_run = {'start': i, 'is_speech': detection_mask[i], 'length': 1}
            
            runs.append(current_run)
            
            # Create smoothed mask
            smoothed_mask = detection_mask.copy()
            
            for i, run in enumerate(runs):
                if not run['is_speech'] and run['length'] < min_silence_length:
                    # If short non-speech segment, convert to speech
                    start_idx = run['start']
                    end_idx = start_idx + run['length']
                    smoothed_mask[start_idx:end_idx] = True
            
            return smoothed_mask
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="smooth_detection",
                suggested_fix="Check detection smoothing implementation",
                learning_objective="Handle VAD smoothing errors"
            )
            return detection_mask


def main():
    """Main function for testing preprocessing utilities"""
    print("Noise Reduction and Preprocessing Utilities - VLA System")
    print("="*65)
    
    # Create preprocessor
    preprocessor = AudioPreprocessor()
    vad = VoiceActivityDetector()
    
    print("Preprocessing utilities initialized.")
    print(f"  Sample rate: {preprocessor.sample_rate} Hz")
    print(f"  Frame size: {preprocessor.frame_size}")
    
    # Create test audio (simulated)
    print("\nGenerating test audio for demonstration...")
    
    # Generate a simple test signal with some noise
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(preprocessor.sample_rate * duration))
    
    # Create a simple signal with voice-like characteristics
    signal_freq = 200  # Hz
    voice_signal = 0.3 * np.sin(2 * np.pi * signal_freq * t)
    
    # Add some noise
    noise = 0.1 * np.random.normal(0, 1, len(t))
    noisy_audio = voice_signal + noise
    
    print(f"Test audio generated: {len(noisy_audio)} samples")
    
    # Apply preprocessing
    print("\nApplying preprocessing...")
    processed_audio = preprocessor.preprocess_audio(
        noisy_audio, 
        sample_rate=preprocessor.sample_rate
    )
    
    print("Preprocessing completed.")
    print(f"  Original RMS: {np.sqrt(np.mean(noisy_audio ** 2)):.6f}")
    print(f"  Processed RMS: {np.sqrt(np.mean(processed_audio ** 2)):.6f}")
    
    # Apply VAD
    print("\nApplying Voice Activity Detection...")
    vad_result = vad.detect_voice_activity(noisy_audio)
    speech_ratio = np.sum(vad_result) / len(vad_result)
    
    print(f"VAD completed. Speech detected in {speech_ratio:.2%} of audio.")
    
    # Show preprocessing statistics
    stats = preprocessor.get_preprocessing_stats()
    print(f"\nPreprocessing statistics: {stats['metrics']}")
    
    print("\nPreprocessing utilities test completed.")


if __name__ == "__main__":
    main()
