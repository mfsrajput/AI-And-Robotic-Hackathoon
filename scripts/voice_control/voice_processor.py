#!/usr/bin/env python3
"""
Voice Command Processor Node for VLA System
ROS 2 node that processes voice commands using local Whisper model
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import Twist
import numpy as np
import threading
import queue
import time
from typing import Optional

# Import our custom modules
from common_data_models import VoiceCommand
from observability_utils import EducationalLogger, MetricsCollector, DebuggingHelper
from error_handling_utils import EducationalErrorHandler, InputValidator


class VoiceProcessorNode(Node):
    """
    ROS 2 node for processing voice commands using local Whisper model
    """
    
    def __init__(self):
        super().__init__('voice_processor_node')
        
        # Initialize educational utilities
        self.logger = EducationalLogger("voice_processor")
        self.metrics = MetricsCollector(self.logger)
        self.debug_helper = DebuggingHelper(self.logger)
        self.error_handler = EducationalErrorHandler()
        self.validator = InputValidator(self.error_handler)
        
        # Configuration parameters
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('chunk_size', 1024)
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('enable_noise_reduction', True)
        
        self.sample_rate = self.get_parameter('sample_rate').value
        self.channels = self.get_parameter('channels').value
        self.chunk_size = self.get_parameter('chunk_size').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.enable_noise_reduction = self.get_parameter('enable_noise_reduction').value
        
        # Initialize processing components
        self.audio_buffer = queue.Queue()
        self.is_processing = False
        self.active_voice_command: Optional[VoiceCommand] = None
        
        # Create callback group for reentrant callbacks
        callback_group = ReentrantCallbackGroup()
        
        # Create subscribers
        self.audio_subscriber = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10,
            callback_group=callback_group
        )
        
        self.command_subscriber = self.create_subscription(
            String,
            'voice_commands',
            self.command_callback,
            10,
            callback_group=callback_group
        )
        
        # Create publishers
        self.transcript_publisher = self.create_publisher(
            String,
            'voice_transcripts',
            10
        )
        
        self.processed_command_publisher = self.create_publisher(
            String,
            'processed_voice_commands',
            10
        )
        
        # Create action server for voice commands
        self.voice_action_server = ActionServer(
            self,
            # Using a generic action interface since we're not creating the full ROS 2 action definition here
            self.execute_voice_command,
            'voice_command',
            callback_group=callback_group
        )
        
        # Start audio processing thread
        self.processing_thread = threading.Thread(target=self.process_audio_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("VoiceProcessorNode", "__init__", "Voice processor node initialized", {
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'confidence_threshold': self.confidence_threshold
        })
    
    def audio_callback(self, msg: AudioData):
        """Handle incoming audio data"""
        try:
            self.logger.debug("VoiceProcessorNode", "audio_callback", "Received audio data", {
                'data_size': len(msg.data),
                'timestamp': str(msg.header.stamp)
            })
            
            # Add audio data to processing queue
            self.audio_buffer.put(msg.data)
            
            # Update metrics
            self.metrics.record_metric('audio_data_size', len(msg.data))
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="audio_callback",
                suggested_fix="Check audio input format and buffer handling",
                learning_objective="Understand audio data handling in ROS 2"
            )
    
    def command_callback(self, msg: String):
        """Handle incoming voice commands"""
        try:
            self.logger.info("VoiceProcessorNode", "command_callback", "Received voice command", {
                'command': msg.data
            })
            
            # Process the command
            processed_command = self.process_voice_command(msg.data)
            
            if processed_command:
                # Publish processed command
                processed_msg = String()
                processed_msg.data = processed_command
                self.processed_command_publisher.publish(processed_msg)
                
                self.logger.info("VoiceProcessorNode", "command_callback", "Published processed command", {
                    'command': processed_command
                })
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="command_callback",
                suggested_fix="Check command processing logic",
                learning_objective="Understand command processing patterns"
            )
    
    def process_audio_loop(self):
        """Main audio processing loop running in separate thread"""
        self.logger.info("VoiceProcessorNode", "process_audio_loop", "Starting audio processing loop")
        
        while rclpy.ok():
            try:
                # Get audio data from queue with timeout
                try:
                    audio_data = self.audio_buffer.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process audio data
                self.metrics.start_timer('audio_processing_cycle')
                
                result = self.process_audio_chunk(audio_data)
                
                processing_time = self.metrics.stop_timer('audio_processing_cycle')
                
                if result and result.get('transcript') and result.get('confidence', 0) >= self.confidence_threshold:
                    # Publish transcript
                    transcript_msg = String()
                    transcript_msg.data = result['transcript']
                    self.transcript_publisher.publish(transcript_msg)
                    
                    self.logger.info("VoiceProcessorNode", "process_audio_loop", "Published transcript", {
                        'transcript': result['transcript'],
                        'confidence': result['confidence'],
                        'processing_time': processing_time
                    })
                
                # Update metrics
                self.metrics.record_metric('processing_cycle_time', processing_time)
                
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context="process_audio_loop",
                    suggested_fix="Check audio processing logic and threading",
                    learning_objective="Understand real-time audio processing challenges"
                )
    
    def process_audio_chunk(self, audio_data: bytes) -> dict:
        """Process a chunk of audio data"""
        try:
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32768.0  # Normalize to [-1, 1]
            
            # Apply noise reduction if enabled
            if self.enable_noise_reduction:
                audio_array = self.apply_noise_reduction(audio_array)
            
            # Check if there's significant audio energy (voice activity detection)
            energy = np.mean(np.abs(audio_array))
            if energy < 0.01:  # Threshold for voice activity
                return {}  # No significant audio, return empty result
            
            # In a real implementation, this would call the Whisper service
            # For this educational example, we'll simulate the transcription
            transcript = self.simulate_transcription(audio_array)
            
            # Calculate confidence based on various factors
            confidence = self.calculate_confidence(audio_array, transcript)
            
            return {
                'transcript': transcript,
                'confidence': confidence,
                'audio_energy': energy,
                'processing_method': 'simulated'
            }
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="process_audio_chunk",
                suggested_fix="Check audio data format and processing steps",
                learning_objective="Understand audio signal processing fundamentals"
            )
            return {}
    
    def apply_noise_reduction(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction to audio array"""
        try:
            # Simple noise reduction using spectral subtraction approach
            # In practice, more sophisticated methods would be used
            
            # Calculate short-term average for noise estimation
            frame_size = 256
            if len(audio_array) >= frame_size:
                # Estimate noise profile from first few frames
                noise_profile = np.mean(np.abs(audio_array[:frame_size*2]))
                
                # Apply simple noise reduction
                reduced = np.copy(audio_array)
                for i in range(0, len(reduced), frame_size):
                    frame = reduced[i:i+frame_size]
                    frame_energy = np.mean(np.abs(frame))
                    
                    if frame_energy > noise_profile * 1.5:  # Only process frames with significant signal
                        reduced[i:i+frame_size] = np.maximum(
                            frame - noise_profile * 0.5,  # Reduce noise level
                            np.minimum(frame, -noise_profile * 0.5)  # Handle negative values
                        )
                
                return reduced
            else:
                return audio_array
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="apply_noise_reduction",
                suggested_fix="Check noise reduction algorithm implementation",
                learning_objective="Learn about audio noise reduction techniques"
            )
            return audio_array
    
    def simulate_transcription(self, audio_array: np.ndarray) -> str:
        """Simulate Whisper transcription (in real implementation, this would call Whisper service)"""
        # This is a placeholder for the actual Whisper integration
        # In a real system, this would call the Whisper service
        
        # For simulation, we'll create a simple transcription based on audio characteristics
        energy = np.mean(np.abs(audio_array))
        
        if energy > 0.1:  # High energy = likely speech
            # Simulate different commands based on energy patterns
            if len(audio_array) > 1000:  # Longer audio = more complex command
                return "move forward one meter"
            else:
                return "hello robot"
        else:
            return ""  # No significant audio
    
    def calculate_confidence(self, audio_array: np.ndarray, transcript: str) -> float:
        """Calculate confidence score for transcription"""
        if not transcript.strip():
            return 0.0
        
        # Calculate confidence based on multiple factors
        energy = np.mean(np.abs(audio_array))
        length = len(transcript)
        
        # Base confidence on audio energy and transcript length
        energy_factor = min(energy * 10, 1.0)  # Cap at 1.0
        length_factor = min(length / 20.0, 0.5)  # Longer transcripts get higher confidence up to a point
        
        # Combine factors
        confidence = (energy_factor * 0.7) + (length_factor * 0.3)
        
        # Ensure confidence is in [0, 1] range
        return max(0.0, min(1.0, confidence))
    
    def process_voice_command(self, command_text: str) -> str:
        """Process a voice command and return processed result"""
        try:
            # Validate input
            if not self.validator.validate_not_empty(command_text, "command_text"):
                return ""
            
            # Log the processing
            self.logger.debug("VoiceProcessorNode", "process_voice_command", "Processing command", {
                'original_command': command_text
            })
            
            # Simple command processing - in real system this would involve NLP
            processed = command_text.lower().strip()
            
            # Update metrics
            self.metrics.record_metric('commands_processed', 1)
            
            return processed
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="process_voice_command",
                suggested_fix="Check command processing logic",
                learning_objective="Understand natural language command processing"
            )
            return ""
    
    def execute_voice_command(self, goal_handle):
        """Execute voice command action"""
        self.logger.info("VoiceProcessorNode", "execute_voice_command", "Executing voice command goal")
        
        try:
            # In a real implementation, this would process the goal and return results
            result = type('VoiceCommandResult', (), {})()  # Placeholder result
            result.success = True
            result.message = "Voice command processed successfully"
            result.transcript = "Sample transcript"
            result.confidence = 0.9
            
            goal_handle.succeed()
            return result
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="execute_voice_command",
                suggested_fix="Check action execution logic",
                learning_objective="Understand ROS 2 action execution patterns"
            )
            result = type('VoiceCommandResult', (), {})()
            result.success = False
            result.message = f"Error processing voice command: {str(e)}"
            result.transcript = ""
            result.confidence = 0.0
            
            goal_handle.abort()
            return result
    
    def get_node_status(self) -> dict:
        """Get current status of the voice processor node"""
        return {
            'node_name': self.get_name(),
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'confidence_threshold': self.confidence_threshold,
            'audio_buffer_size': self.audio_buffer.qsize(),
            'is_processing': self.is_processing,
            'active_command': self.active_voice_command.transcript if self.active_voice_command else None,
            'metrics_summary': self.metrics.get_statistics()
        }


def main(args=None):
    """Main function to run the voice processor node"""
    rclpy.init(args=args)
    
    # Create the voice processor node
    voice_processor = VoiceProcessorNode()
    
    # Use MultiThreadedExecutor to handle callbacks in separate threads
    executor = MultiThreadedExecutor()
    executor.add_node(voice_processor)
    
    try:
        voice_processor.logger.info("VoiceProcessorNode", "main", "Starting voice processor node")
        executor.spin()
    except KeyboardInterrupt:
        voice_processor.logger.info("VoiceProcessorNode", "main", "Interrupted by user")
    except Exception as e:
        voice_processor.error_handler.handle_error(
            e,
            context="main",
            suggested_fix="Check node initialization and execution",
            learning_objective="Understand ROS 2 node lifecycle management"
        )
    finally:
        voice_processor.logger.info("VoiceProcessorNode", "main", "Shutting down voice processor node")
        voice_processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
