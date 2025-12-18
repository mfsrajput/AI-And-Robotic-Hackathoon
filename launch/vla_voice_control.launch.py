#!/usr/bin/env python3
"""
Launch file for VLA Voice Control System
Launches all necessary nodes for voice command processing with local Whisper model
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart, OnProcessIO
import os


def generate_launch_description():
    """Generate the launch description for VLA voice control system"""
    
    # Declare launch arguments
    whisper_model_arg = DeclareLaunchArgument(
        'whisper_model_path',
        default_value='models/ggml-tiny.en.bin',
        description='Path to Whisper model file for local processing'
    )
    
    language_arg = DeclareLaunchArgument(
        'language',
        default_value='en',
        description='Language for speech recognition'
    )
    
    sample_rate_arg = DeclareLaunchArgument(
        'sample_rate',
        default_value='16000',
        description='Audio sample rate for processing'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.7',
        description='Minimum confidence threshold for accepting transcriptions'
    )
    
    enable_noise_reduction_arg = DeclareLaunchArgument(
        'enable_noise_reduction',
        default_value='true',
        description='Enable noise reduction preprocessing'
    )
    
    # Get launch configurations
    whisper_model_path = LaunchConfiguration('whisper_model_path')
    language = LaunchConfiguration('language')
    sample_rate = LaunchConfiguration('sample_rate')
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    enable_noise_reduction = LaunchConfiguration('enable_noise_reduction')
    
    # Create the voice processor node
    voice_processor_node = Node(
        package='std_msgs',  # Using std_msgs as placeholder - in real implementation would be vla_system
        executable='voice_processor',  # This would be the executable name
        name='voice_processor_node',
        parameters=[
            {
                'whisper_model_path': whisper_model_path,
                'language': language,
                'sample_rate': sample_rate,
                'confidence_threshold': confidence_threshold,
                'enable_noise_reduction': enable_noise_reduction,
                'channels': 1,
                'chunk_size': 1024,
            }
        ],
        remappings=[
            ('audio_input', '/audio_input'),
            ('voice_commands', '/voice_commands'),
            ('voice_transcripts', '/voice_transcripts'),
            ('processed_voice_commands', '/processed_voice_commands'),
        ],
        output='screen',
        # In a real implementation, we would have the actual executable
        # For now, we'll just log that this would launch the node
    )
    
    # Create an audio input node (placeholder)
    audio_input_node = Node(
        package='sensor_msgs',  # Placeholder package
        executable='audio_input',  # Placeholder executable
        name='audio_input_node',
        parameters=[
            {
                'sample_rate': sample_rate,
                'channels': 1,
                'device_index': -1,  # Default audio device
            }
        ],
        remappings=[
            ('audio_raw', '/audio_input'),
        ],
        output='screen',
    )
    
    # Create a logging node to show system status
    system_status_node = Node(
        package='std_msgs',  # Placeholder package
        executable='system_status',  # Placeholder executable
        name='vla_system_status',
        parameters=[
            {
                'system_name': 'VLA_Voice_Control',
                'check_nodes': ['voice_processor_node', 'audio_input_node'],
            }
        ],
        output='screen',
    )
    
    # Log configuration info
    config_logger = LogInfo(
        msg=[
            '*** VLA Voice Control System Configuration ***',
            '\nWhisper Model: ', whisper_model_path,
            '\nLanguage: ', language,
            '\nSample Rate: ', sample_rate,
            '\nConfidence Threshold: ', confidence_threshold,
            '\nNoise Reduction: ', enable_noise_reduction,
            '\n*** Starting Voice Control System ***'
        ]
    )
    
    # Create the launch description
    ld = LaunchDescription()
    
    # Add the launch arguments
    ld.add_action(whisper_model_arg)
    ld.add_action(language_arg)
    ld.add_action(sample_rate_arg)
    ld.add_action(confidence_threshold_arg)
    ld.add_action(enable_noise_reduction_arg)
    
    # Add configuration logger
    ld.add_action(config_logger)
    
    # Add the nodes
    ld.add_action(voice_processor_node)
    ld.add_action(audio_input_node)
    ld.add_action(system_status_node)
    
    return ld


# Additional Python implementation for voice control system launch
# This provides a more complete implementation that could be used with ROS 2

def create_voice_control_launch_nodes():
    """
    Alternative implementation using Python directly
    This creates the nodes that would be launched by the launch file
    """
    import rclpy
    from rclpy.node import Node
    import threading
    import time
    
    from voice_control.voice_processor import VoiceProcessorNode
    from voice_control.whisper_service import WhisperConfig
    from privacy_compliance import PrivacyComplianceMiddleware
    
    class VoiceControlSystem:
        """
        Complete voice control system manager
        """
        def __init__(self):
            self.logger = __import__('observability_utils').EducationalLogger("voice_control_system")
            self.error_handler = __import__('error_handling_utils').EducationalErrorHandler()
            
            # Initialize privacy compliance
            self.privacy_middleware = PrivacyComplianceMiddleware()
            
            # Default configuration
            self.config = {
                'whisper_model_path': 'models/ggml-tiny.en.bin',
                'language': 'en',
                'sample_rate': 16000,
                'confidence_threshold': 0.7,
                'enable_noise_reduction': True,
                'channels': 1,
                'chunk_size': 1024
            }
            
            self.nodes = []
            self.is_running = False
            
            self.logger.info("VoiceControlSystem", "__init__", "Voice control system initialized")
        
        def configure_from_params(self, **kwargs):
            """Configure the system from parameters"""
            for key, value in kwargs.items():
                if key in self.config:
                    self.config[key] = value
                    self.logger.debug("VoiceControlSystem", "configure_from_params", 
                                    f"Configured {key} = {value}")
        
        def start_system(self):
            """Start the complete voice control system"""
            try:
                self.logger.info("VoiceControlSystem", "start_system", "Starting voice control system")
                
                # Validate privacy compliance
                if not self.privacy_middleware.validate_component_privacy("whisper", 
                                                                        type('Config', (), self.config)()):
                    raise Exception("Privacy compliance validation failed")
                
                # Initialize ROS 2
                rclpy.init()
                
                # Create and start voice processor node
                self.voice_processor = VoiceProcessorNode()
                self.nodes.append(self.voice_processor)
                
                # Start all nodes in separate threads
                for node in self.nodes:
                    node_thread = threading.Thread(target=self._run_node, args=(node,), daemon=True)
                    node_thread.start()
                
                self.is_running = True
                self.logger.info("VoiceControlSystem", "start_system", "Voice control system started successfully")
                
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context="start_system",
                    suggested_fix="Check system configuration and dependencies",
                    learning_objective="Handle voice control system startup errors"
                )
                self.stop_system()
        
        def _run_node(self, node):
            """Run a single ROS 2 node"""
            try:
                rclpy.spin(node)
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context=f"run_node: {node.get_name()}",
                    suggested_fix="Check node implementation",
                    learning_objective="Handle node execution errors"
                )
        
        def stop_system(self):
            """Stop the voice control system"""
            try:
                self.logger.info("VoiceControlSystem", "stop_system", "Stopping voice control system")
                
                # Shutdown all nodes
                for node in self.nodes:
                    try:
                        node.destroy_node()
                    except:
                        pass  # Node might already be destroyed
                
                # Shutdown ROS 2
                rclpy.shutdown()
                
                self.is_running = False
                self.logger.info("VoiceControlSystem", "stop_system", "Voice control system stopped")
                
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context="stop_system",
                    suggested_fix="Check system shutdown procedure",
                    learning_objective="Handle system shutdown errors"
                )
    
    return VoiceControlSystem()


def main():
    """Main function to demonstrate the launch system"""
    print("VLA Voice Control System Launch")
    print("="*35)
    
    print("Launch file: vla_voice_control.launch.py")
    print("This launch file would start the following nodes:")
    print("  - voice_processor_node: Processes voice commands using local Whisper")
    print("  - audio_input_node: Captures audio from microphone")
    print("  - vla_system_status: Monitors system status")
    print("")
    print("Launch arguments:")
    print("  whisper_model_path: Path to Whisper model file")
    print("  language: Language for speech recognition")
    print("  sample_rate: Audio sample rate")
    print("  confidence_threshold: Minimum confidence for transcriptions")
    print("  enable_noise_reduction: Enable/disable noise reduction")
    print("")
    print("All processing occurs locally with no external API calls.")
    print("Privacy compliance is enforced throughout the system.")
    
    # Show how to launch from command line
    print("\nTo launch the system, run:")
    print("  ros2 launch vla_system vla_voice_control.launch.py")
    print("\nWith custom parameters:")
    print("  ros2 launch vla_system vla_voice_control.launch.py whisper_model_path:=models/ggml-base.en.bin language:=en")


if __name__ == '__main__':
    main()
