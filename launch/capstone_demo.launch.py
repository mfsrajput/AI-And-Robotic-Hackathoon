#!/usr/bin/env python3
"""
Launch file for Complete Capstone Autonomous Humanoid Demonstration
Launches all necessary nodes for the complete autonomous humanoid system
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart, OnProcessIO
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate the launch description for the complete capstone demonstration"""
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('vla_system'),
            'config',
            'vla_pipeline_config.yaml'
        ]),
        description='Path to main VLA pipeline configuration file'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'whisper_model_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('vla_system'),
            'models',
            'ggml-tiny.en.bin'
        ]),
        description='Path to Whisper model file for local processing'
    )
    
    enable_voice_arg = DeclareLaunchArgument(
        'enable_voice_processing',
        default_value='true',
        description='Enable voice processing capabilities'
    )
    
    enable_vision_arg = DeclareLaunchArgument(
        'enable_vision_processing', 
        default_value='true',
        description='Enable vision processing capabilities'
    )
    
    enable_llm_arg = DeclareLaunchArgument(
        'enable_llm_planning',
        default_value='true', 
        description='Enable LLM-based task planning'
    )
    
    demo_mode_arg = DeclareLaunchArgument(
        'demo_mode',
        default_value='true',
        description='Enable demonstration mode with pre-defined scenarios'
    )
    
    # Get launch configurations
    config_file = LaunchConfiguration('config_file')
    whisper_model_path = LaunchConfiguration('whisper_model_path')
    enable_voice = LaunchConfiguration('enable_voice_processing')
    enable_vision = LaunchConfiguration('enable_vision_processing')
    enable_llm = LaunchConfiguration('enable_llm_planning')
    demo_mode = LaunchConfiguration('demo_mode')
    
    # Create capstone orchestrator node
    capstone_orchestrator_node = Node(
        package='vla_system',  # This would be the actual package name in a real ROS 2 system
        executable='capstone_orchestrator',  # This would be the executable name
        name='capstone_orchestrator',
        parameters=[
            config_file,
            {
                'whisper_model_path': whisper_model_path,
                'enable_voice_processing': enable_voice,
                'enable_vision_processing': enable_vision,
                'enable_llm_planning': enable_llm,
                'demo_mode': demo_mode,
                'sample_rate': 16000,
                'channels': 1,
                'confidence_threshold': 0.7,
                'enable_noise_reduction': True,
            }
        ],
        remappings=[
            ('/audio_input', '/microphone/audio_raw'),
            ('/voice_commands', '/capstone/voice_commands'),
            ('/processed_commands', '/capstone/processed_commands'),
            ('/action_plans', '/capstone/action_plans'),
            ('/execution_feedback', '/capstone/execution_feedback'),
        ],
        output='screen',
        # Note: In a real implementation, the executable would need to exist
        # For this educational example, we'll just log what would be launched
    )
    
    # Create voice processing node
    voice_processor_node = Node(
        package='vla_system',
        executable='voice_processor',
        name='voice_processor',
        parameters=[
            config_file,
            {
                'whisper_model_path': whisper_model_path,
                'enable_noise_reduction': True,
                'sample_rate': 16000,
                'channels': 1,
                'confidence_threshold': 0.7,
            }
        ],
        remappings=[
            ('/audio_input', '/microphone/audio_raw'),
            ('/voice_transcripts', '/voice_processor/transcripts'),
        ],
        output='screen',
        condition=IfCondition(enable_voice)
    )
    
    # Create LLM planning node
    llm_planner_node = Node(
        package='vla_system', 
        executable='llm_planner',
        name='llm_planner',
        parameters=[
            config_file,
            {
                'enable_local_llm': True,
                'model_name': 'llama3.1:8b',
                'temperature': 0.3,
                'max_tokens': 2048,
            }
        ],
        remappings=[
            ('/natural_language_commands', '/llm_planner/commands'),
            ('/action_sequences', '/llm_planner/action_sequences'),
        ],
        output='screen',
        condition=IfCondition(enable_llm)
    )
    
    # Create execution orchestrator node
    execution_orchestrator_node = Node(
        package='vla_system',
        executable='execution_orchestrator', 
        name='execution_orchestrator',
        parameters=[
            config_file,
            {
                'max_concurrent_executions': 3,
                'default_timeout': 60.0,
                'retry_attempts': 3,
            }
        ],
        remappings=[
            ('/action_sequences', '/execution_orchestrator/action_sequences'),
            ('/execution_results', '/execution_orchestrator/results'),
        ],
        output='screen'
    )
    
    # Create state manager node
    state_manager_node = Node(
        package='vla_system',
        executable='state_manager',
        name='state_manager',
        parameters=[
            config_file,
            {
                'enable_persistence': True,
                'persistence_file': 'capstone_state_backup.json',
                'enable_safety_constraints': True,
            }
        ],
        remappings=[
            ('/system_state', '/state_manager/current_state'),
            ('/safety_status', '/state_manager/safety_status'),
        ],
        output='screen'
    )
    
    # Create system monitor node
    system_monitor_node = Node(
        package='vla_system',
        executable='system_monitor',
        name='system_monitor',
        parameters=[
            config_file,
            {
                'enable_validation': True,
                'enable_assessment': True,
                'validation_frequency': 30,  # seconds
                'enable_metrics_collection': True,
            }
        ],
        remappings=[
            ('/system_metrics', '/system_monitor/metrics'),
            ('/validation_results', '/system_monitor/validation_results'),
        ],
        output='screen'
    )
    
    # Create demonstration node (for demo scenarios)
    demo_node = Node(
        package='vla_system',
        executable='capstone_demo',
        name='capstone_demo',
        parameters=[
            config_file,
            {
                'demo_mode': demo_mode,
                'demo_scenarios': ['voice_navigation', 'object_interaction', 'multi_step_task'],
                'enable_assessment': True,
            }
        ],
        remappings=[
            ('/demo_commands', '/capstone_demo/commands'),
            ('/demo_results', '/capstone_demo/results'),
        ],
        output='screen',
        condition=IfCondition(demo_mode)
    )
    
    # Create logging node to show system status
    system_status_logger = LogInfo(
        msg=["*** Autonomous Humanoid Capstone System Starting ***",
             "\\nConfig File: ", config_file,
             "\\nWhisper Model: ", whisper_model_path,
             "\\nVoice Processing: ", enable_voice,
             "\\nVision Processing: ", enable_vision,
             "\\nLLM Planning: ", enable_llm,
             "\\nDemo Mode: ", demo_mode,
             "\\n*** Launching Complete System ***"]
    )
    
    # Create the launch description
    ld = LaunchDescription()
    
    # Add the launch arguments
    ld.add_action(config_file_arg)
    ld.add_action(model_path_arg)
    ld.add_action(enable_voice_arg)
    ld.add_action(enable_vision_arg)
    ld.add_action(enable_llm_arg)
    ld.add_action(demo_mode_arg)
    
    # Add system status logger
    ld.add_action(system_status_logger)
    
    # Add all the nodes
    ld.add_action(capstone_orchestrator_node)
    ld.add_action(voice_processor_node)
    ld.add_action(llm_planner_node)
    ld.add_action(execution_orchestrator_node)
    ld.add_action(state_manager_node)
    ld.add_action(system_monitor_node)
    ld.add_action(demo_node)
    
    return ld


# Additional Python implementation for the launch system
# This provides a more complete implementation that could be used with ROS 2

def create_capstone_demo_launcher():
    """
    Alternative implementation using Python directly
    This creates the nodes that would be launched by the launch file
    """
    import threading
    import time
    
    from capstone_project.capstone_system import CapstoneSystemOrchestrator
    from capstone_project.execution_orchestrator import ExecutionOrchestrator
    from capstone_project.state_manager import StateManager
    from capstone_validator import CapstoneValidator
    
    class CapstoneDemoLauncher:
        """
        Complete capstone demonstration launcher
        """
        def __init__(self):
            self.logger = __import__('observability_utils').EducationalLogger("capstone_launcher")
            self.error_handler = __import__('error_handling_utils').EducationalErrorHandler()
            
            # Initialize system components
            self.orchestrator = None
            self.execution_orchestrator = None
            self.state_manager = None
            self.validator = None
            
            # System configuration
            self.config = {
                'config_file': 'config/vla_pipeline_config.yaml',
                'whisper_model_path': 'models/ggml-tiny.en.bin',
                'enable_voice_processing': True,
                'enable_vision_processing': True,
                'enable_llm_planning': True,
                'demo_mode': True,
                'sample_rate': 16000,
                'channels': 1,
                'confidence_threshold': 0.7
            }
            
            self.nodes = []
            self.is_running = False
            
            self.logger.info("CapstoneDemoLauncher", "__init__", "Capstone demo launcher initialized")
        
        def configure_from_params(self, **kwargs):
            """Configure the system from parameters"""
            for key, value in kwargs.items():
                if key in self.config:
                    self.config[key] = value
                    self.logger.debug("CapstoneDemoLauncher", "configure_from_params", 
                                    f"Configured {key} = {value}")
        
        def start_system(self):
            """Start the complete capstone demonstration system"""
            try:
                self.logger.info("CapstoneDemoLauncher", "start_system", "Starting capstone demonstration system")
                
                # Initialize all system components
                self.orchestrator = CapstoneSystemOrchestrator()
                self.execution_orchestrator = ExecutionOrchestrator()
                self.state_manager = StateManager()
                self.validator = CapstoneValidator()
                
                # Start all components
                self.execution_orchestrator.start_execution_engine()
                self.orchestrator.start_system()
                
                # Add to nodes list for tracking
                self.nodes.extend([
                    self.orchestrator,
                    self.execution_orchestrator,
                    self.state_manager,
                    self.validator
                ])
                
                self.is_running = True
                self.logger.info("CapstoneDemoLauncher", "start_system", "Capstone demonstration system started successfully")
                
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context="start_system",
                    suggested_fix="Check system configuration and dependencies",
                    learning_objective="Handle capstone system startup errors"
                )
                self.stop_system()
        
        def stop_system(self):
            """Stop the capstone demonstration system"""
            try:
                self.logger.info("CapstoneDemoLauncher", "stop_system", "Stopping capstone demonstration system")
                
                # Stop all components in reverse order
                if hasattr(self, 'execution_orchestrator') and self.execution_orchestrator:
                    self.execution_orchestrator.stop_execution_engine()
                
                if hasattr(self, 'orchestrator') and self.orchestrator:
                    self.orchestrator.stop_system()
                
                self.is_running = False
                self.logger.info("CapstoneDemoLauncher", "stop_system", "Capstone demonstration system stopped")
                
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context="stop_system",
                    suggested_fix="Check system shutdown procedure",
                    learning_objective="Handle capstone system shutdown errors"
                )
        
        def run_demo_scenario(self, scenario_name: str, parameters: dict = None):
            """Run a specific demonstration scenario"""
            try:
                if not self.is_running:
                    raise Exception("System not running, cannot execute scenario")
                
                self.logger.info("CapstoneDemoLauncher", "run_demo_scenario", 
                               f"Running demonstration scenario: {scenario_name}")
                
                # Execute scenario through orchestrator
                if self.orchestrator:
                    result = self.orchestrator.execute_custom_scenario(scenario_name, parameters or {})
                    return result
                else:
                    raise Exception("Orchestrator not initialized")
                    
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context=f"run_demo_scenario: {scenario_name}",
                    suggested_fix="Check scenario execution parameters",
                    learning_objective="Handle scenario execution errors"
                )
                return {"error": str(e), "status": "failed"}
        
        def get_system_status(self):
            """Get status of all system components"""
            status = {
                'is_running': self.is_running,
                'config': self.config,
                'orchestrator_status': None,
                'execution_orchestrator_status': None,
                'state_manager_status': None,
                'validator_status': None
            }
            
            if self.orchestrator:
                status['orchestrator_status'] = self.orchestrator.get_system_status()
            
            if self.execution_orchestrator:
                status['execution_orchestrator_status'] = {
                    'is_running': self.execution_orchestrator.is_running,
                    'active_executions': len(self.execution_orchestrator.active_executions),
                    'queue_size': self.execution_orchestrator.execution_queue.qsize()
                }
            
            if self.state_manager:
                status['state_manager_status'] = self.state_manager.get_system_status()
            
            if self.validator:
                status['validator_status'] = self.validator.generate_validation_report()
            
            return status


def main():
    """Main function to demonstrate the launch system"""
    print("Autonomous Humanoid Capstone Demo Launch System")
    print("="*50)
    
    print("Launch file: capstone_demo.launch.py")
    print("This launch file would start the following nodes:")
    print("  - capstone_orchestrator: Main system orchestrator")
    print("  - voice_processor: Voice command processing")
    print("  - llm_planner: LLM-based task planning")
    print("  - execution_orchestrator: Action sequence execution")
    print("  - state_manager: System state management")
    print("  - system_monitor: System monitoring and validation")
    print("  - capstone_demo: Demonstration scenarios")
    print("")
    print("Launch arguments:")
    print("  config_file: Path to main configuration file")
    print("  whisper_model_path: Path to Whisper model")
    print("  enable_voice_processing: Enable/disable voice processing")
    print("  enable_vision_processing: Enable/disable vision processing")
    print("  enable_llm_planning: Enable/disable LLM planning")
    print("  demo_mode: Enable demonstration mode")
    print("")
    print("All processing occurs locally with no external API calls.")
    print("Privacy compliance is enforced throughout the system.")
    
    # Show how to launch from command line
    print("\nTo launch the capstone demonstration, run:")
    print("  ros2 launch vla_system capstone_demo.launch.py")
    print("\nWith custom parameters:")
    print("  ros2 launch vla_system capstone_demo.launch.py demo_mode:=true enable_voice_processing:=true")
    
    # Demonstrate the Python launcher
    print("\nDemonstrating Python-based launcher...")
    launcher = CapstoneDemoLauncher()
    
    print(f"Launcher initialized with config: {launcher.config}")
    print(f"System status: {launcher.get_system_status()['is_running']}")


if __name__ == '__main__':
    main()
