"""
Voice-to-Text Conversion Pipeline for VLA System
Integrates audio input, preprocessing, Whisper model, and output processing
"""

import threading
import queue
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import numpy as np

from common_data_models import VoiceCommand
from audio_input import AudioInputHandler
from whisper_service import WhisperService, WhisperConfig
from observability_utils import EducationalLogger, MetricsCollector, DebuggingHelper
from error_handling_utils import EducationalErrorHandler, InputValidator


class VoiceToTextPipeline:
    """
    Complete voice-to-text conversion pipeline integrating all components
    """
    
    def __init__(self, 
                 whisper_config: WhisperConfig,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024):
        self.whisper_config = whisper_config
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        
        # Initialize educational utilities
        self.logger = EducationalLogger("voice_to_text_pipeline")
        self.metrics = MetricsCollector(self.logger)
        self.debug_helper = DebuggingHelper(self.logger)
        self.error_handler = EducationalErrorHandler()
        self.validator = InputValidator(self.error_handler)
        
        # Initialize components
        self.audio_handler = AudioInputHandler(
            sample_rate=sample_rate,
            channels=channels,
            chunk_size=chunk_size,
            callback=self._on_audio_chunk
        )
        self.whisper_service = WhisperService(whisper_config)
        
        # Pipeline state
        self.is_running = False
        self.pipeline_thread = None
        self.result_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Processing buffers
        self.speech_buffer = b""
        self.speech_segments = []
        
        # Callbacks
        self.transcript_callback: Optional[Callable[[VoiceCommand], None]] = None
        self.error_callback: Optional[Callable[[Exception], None]] = None
        
        self.logger.info("VoiceToTextPipeline", "__init__", "Voice-to-text pipeline initialized", {
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'whisper_model': self.whisper_config.model_path
        })
    
    def _on_audio_chunk(self, audio_data: bytes):
        """Callback for when audio chunk is received from audio handler"""
        try:
            # Add to speech buffer
            self.speech_buffer += audio_data
            
            # If buffer is large enough, process it
            if len(self.speech_buffer) >= self.chunk_size * 8:  # 8 chunks worth
                # Process the speech buffer
                self._process_speech_buffer()
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="on_audio_chunk",
                suggested_fix="Check audio data processing",
                learning_objective="Handle audio callback errors"
            )
    
    def _process_speech_buffer(self):
        """Process accumulated speech buffer through the pipeline"""
        if not self.speech_buffer:
            return
        
        try:
            self.metrics.start_timer('speech_buffer_processing')
            
            # Preprocess audio if needed (additional preprocessing beyond audio handler)
            processed_audio = self._additional_preprocessing(self.speech_buffer)
            
            # Transcribe using Whisper service
            result = self.whisper_service.transcribe_audio(processed_audio, self.sample_rate)
            
            processing_time = self.metrics.stop_timer('speech_buffer_processing')
            
            if result.get('success') and result.get('transcript', '').strip():
                # Create VoiceCommand object
                voice_command = VoiceCommand(
                    audio_data=self.speech_buffer,  # Store original audio
                    transcript=result['transcript'],
                    timestamp=datetime.now(),
                    confidence=result.get('confidence', 0.0),
                    status='completed',
                    language=self.whisper_config.language
                )
                
                # Add processing metrics
                voice_command_dict = voice_command.to_dict()
                voice_command_dict['processing_time'] = processing_time
                voice_command_dict['audio_size'] = len(self.speech_buffer)
                
                # Update metrics
                self.metrics.record_metric('successful_transcriptions', 1)
                self.metrics.record_metric('transcription_time', processing_time)
                self.metrics.record_metric('transcript_length', len(result['transcript']))
                
                # Call transcript callback if available
                if self.transcript_callback:
                    try:
                        self.transcript_callback(voice_command)
                    except Exception as e:
                        self.error_handler.handle_error(
                            e,
                            context="transcript_callback",
                            suggested_fix="Check transcript callback implementation",
                            learning_objective="Handle callback execution errors"
                        )
                
                # Add to result queue
                self.result_queue.put(voice_command)
                
                self.logger.info("VoiceToTextPipeline", "_process_speech_buffer", "Successfully processed speech buffer", {
                    'transcript': result['transcript'][:50] + "..." if len(result['transcript']) > 50 else result['transcript'],
                    'confidence': result.get('confidence', 0),
                    'processing_time': processing_time
                })
            else:
                # Handle failed transcription
                self.metrics.record_metric('failed_transcriptions', 1)
                
                if result.get('error'):
                    self.logger.warning("VoiceToTextPipeline", "_process_speech_buffer", "Transcription failed", {
                        'error': result['error'],
                        'processing_time': processing_time
                    })
            
            # Clear the buffer
            self.speech_buffer = b""
            
        except Exception as e:
            processing_time = self.metrics.stop_timer('speech_buffer_processing')
            self.error_handler.handle_error(
                e,
                context="process_speech_buffer",
                suggested_fix="Check speech buffer processing logic",
                learning_objective="Handle speech processing errors"
            )
            
            # Call error callback if available
            if self.error_callback:
                try:
                    self.error_callback(e)
                except Exception as cb_error:
                    self.logger.error("VoiceToTextPipeline", "_process_speech_buffer", "Error in error callback", {
                        'callback_error': str(cb_error)
                    })
    
    def _additional_preprocessing(self, audio_data: bytes) -> bytes:
        """Additional preprocessing specific to the pipeline"""
        try:
            # Convert to numpy array for processing
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Apply additional noise reduction if needed
            # This could include more sophisticated techniques than the audio handler
            if len(audio_array) > 0:
                # Normalize audio
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    audio_array = audio_array / max_val * 0.8  # Reduce to 80% to prevent clipping
            
            # Convert back to bytes
            processed_audio = (audio_array * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
            
            return processed_audio
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="additional_preprocessing",
                suggested_fix="Check audio preprocessing implementation",
                learning_objective="Handle audio signal processing errors"
            )
            return audio_data  # Return original if preprocessing fails
    
    def start_pipeline(self):
        """Start the voice-to-text pipeline"""
        try:
            self.is_running = True
            
            # Calibrate noise floor
            self.logger.info("VoiceToTextPipeline", "start_pipeline", "Calibrating noise floor...")
            self.audio_handler.calibrate_noise_floor(2.0)
            
            # Start audio input
            self.audio_handler.start_recording()
            
            self.logger.info("VoiceToTextPipeline", "start_pipeline", "Voice-to-text pipeline started")
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="start_pipeline",
                suggested_fix="Check pipeline initialization",
                learning_objective="Handle pipeline startup errors"
            )
            self.is_running = False
    
    def stop_pipeline(self):
        """Stop the voice-to-text pipeline"""
        try:
            self.is_running = False
            
            # Stop audio handler
            self.audio_handler.stop_recording()
            
            self.logger.info("VoiceToTextPipeline", "stop_pipeline", "Voice-to-text pipeline stopped")
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="stop_pipeline",
                suggested_fix="Check pipeline shutdown",
                learning_objective="Handle pipeline cleanup properly"
            )
    
    def set_transcript_callback(self, callback: Callable[[VoiceCommand], None]):
        """Set callback for when a transcript is available"""
        self.transcript_callback = callback
    
    def set_error_callback(self, callback: Callable[[Exception], None]):
        """Set callback for when an error occurs"""
        self.error_callback = callback
    
    def get_next_transcript(self, timeout: float = 1.0) -> Optional[VoiceCommand]:
        """Get the next transcript from the queue (non-blocking with timeout)"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current status of the pipeline"""
        return {
            'is_running': self.is_running,
            'audio_handler_status': self.audio_handler.get_audio_stats(),
            'whisper_service_status': self.whisper_service.get_service_status(),
            'result_queue_size': self.result_queue.qsize(),
            'command_queue_size': self.command_queue.qsize(),
            'speech_buffer_size': len(self.speech_buffer),
            'metrics': self.metrics.get_statistics()
        }
    
    def process_audio_file(self, file_path: str) -> VoiceCommand:
        """Process audio from a file (for batch processing)"""
        try:
            self.logger.info("VoiceToTextPipeline", "process_audio_file", "Processing audio file", {
                'file_path': file_path
            })
            
            # Read audio file (simplified - in practice, use proper audio libraries)
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            # Transcribe the audio
            result = self.whisper_service.transcribe_audio(audio_data, self.sample_rate)
            
            if result.get('success') and result.get('transcript', '').strip():
                voice_command = VoiceCommand(
                    audio_data=audio_data,
                    transcript=result['transcript'],
                    timestamp=datetime.now(),
                    confidence=result.get('confidence', 0.0),
                    status='completed',
                    language=self.whisper_config.language
                )
                
                self.logger.info("VoiceToTextPipeline", "process_audio_file", "File processed successfully", {
                    'transcript': result['transcript'][:100] + "..." if len(result['transcript']) > 100 else result['transcript']
                })
                
                return voice_command
            else:
                self.logger.warning("VoiceToTextPipeline", "process_audio_file", "File processing failed", {
                    'error': result.get('error', 'No transcript generated')
                })
                return None
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="process_audio_file",
                suggested_fix="Check audio file format and processing",
                learning_objective="Handle file-based audio processing"
            )
            return None


def main():
    """Main function for testing the voice-to-text pipeline"""
    print("Voice-to-Text Pipeline - VLA System")
    print("="*45)
    
    # Create Whisper configuration
    whisper_config = WhisperConfig(
        model_path="models/ggml-tiny.en.bin",  # Default model
        language="en",
        threads=2
    )
    
    # Create pipeline
    pipeline = VoiceToTextPipeline(whisper_config)
    
    print("Pipeline initialized with configuration:")
    print(f"  Whisper model: {whisper_config.model_path}")
    print(f"  Language: {whisper_config.language}")
    print(f"  Sample rate: {pipeline.sample_rate} Hz")
    
    # Define transcript callback
    def on_transcript(voice_command: VoiceCommand):
        print(f"\n🎤 Transcribed: '{voice_command.transcript}'")
        print(f"   Confidence: {voice_command.confidence:.2f}")
        print(f"   Timestamp: {voice_command.timestamp}")
    
    # Set the callback
    pipeline.set_transcript_callback(on_transcript)
    
    # Show initial status
    status = pipeline.get_pipeline_status()
    print(f"\nInitial pipeline status: {status}")
    
    try:
        # Start the pipeline
        print("\nStarting voice-to-text pipeline...")
        pipeline.start_pipeline()
        
        print("Pipeline running. Speak into the microphone...")
        print("Press Ctrl+C to stop.")
        
        # Run for a while to collect transcripts
        start_time = time.time()
        while time.time() - start_time < 15:  # Run for 15 seconds
            # Check for new transcripts
            transcript = pipeline.get_next_transcript(timeout=0.1)
            if transcript:
                print(f"  Received transcript: {transcript.transcript}")
            
            # Show status periodically
            if int(time.time() - start_time) % 5 == 0:
                status = pipeline.get_pipeline_status()
                print(f"  Pipeline status - Audio queue: {status['audio_handler_status']['queue_size']}, "
                      f"Results: {status['result_queue_size']}")
    
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    
    finally:
        # Stop the pipeline
        pipeline.stop_pipeline()
        print("Pipeline stopped.")
        
        # Show final status
        final_status = pipeline.get_pipeline_status()
        print(f"\nFinal pipeline status: {final_status}")
        
        # Show metrics summary
        print(f"\nProcessing metrics:")
        metrics_summary = pipeline.metrics.get_statistics()
        for metric_name, stats in metrics_summary.items():
            print(f"  {metric_name}: {stats}")


if __name__ == "__main__":
    main()
