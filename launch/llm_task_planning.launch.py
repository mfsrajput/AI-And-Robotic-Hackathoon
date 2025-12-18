#!/usr/bin/env python3
"""
Launch file for LLM Task Planning System
Launches all necessary nodes for the LLM-based task planning system in the VLA pipeline
Part of the Vision-Language-Action (VLA) pipeline for autonomous humanoid systems
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
    """Generate the launch description for the LLM task planning system"""

    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('vla_system'),
            'config',
            'llm_config.yaml'
        ]),
        description='Path to LLM configuration file'
    )

    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='llama3.1:8b',
        description='Name of the LLM model to use with Ollama'
    )

    enable_local_llm_arg = DeclareLaunchArgument(
        'enable_local_llm',
        default_value='true',
        description='Enable local LLM processing (no external API calls)'
    )

    temperature_arg = DeclareLaunchArgument(
        'temperature',
        default_value='0.3',
        description='LLM temperature parameter for response creativity'
    )

    max_tokens_arg = DeclareLaunchArgument(
        'max_tokens',
        default_value='2048',
        description='Maximum tokens for LLM responses'
    )

    # Get launch configurations
    config_file = LaunchConfiguration('config_file')
    model_name = LaunchConfiguration('model_name')
    enable_local_llm = LaunchConfiguration('enable_local_llm')
    temperature = LaunchConfiguration('temperature')
    max_tokens = LaunchConfiguration('max_tokens')

    # Create LLM task planning node
    llm_task_planning_node = Node(
        package='vla_system',
        executable='llm_task_planner',  # This would be the executable name in a real ROS 2 system
        name='llm_task_planner',
        parameters=[
            config_file,
            {
                'model_name': model_name,
                'enable_local_llm': enable_local_llm,
                'temperature': LaunchConfiguration('temperature'),
                'max_tokens': LaunchConfiguration('max_tokens'),
                'response_timeout': 30.0,
                'retry_attempts': 3,
                'enable_caching': True,
                'cache_duration': 300,  # 5 minutes
                'privacy_compliance': True,  # Ensure no external data transmission
            }
        ],
        remappings=[
            ('/natural_language_commands', '/llm_planner/commands'),
            ('/action_sequences', '/llm_planner/action_sequences'),
            ('/task_plans', '/llm_planner/task_plans'),
            ('/execution_feedback', '/llm_planner/execution_feedback'),
        ],
        output='screen',
        condition=IfCondition(enable_local_llm)
    )

    # Create action sequence generator node
    action_sequence_generator_node = Node(
        package='vla_system',
        executable='action_sequence_generator',
        name='action_sequence_generator',
        parameters=[
            config_file,
            {
                'default_timeout': 300.0,
                'max_sequence_steps': 50,
                'enable_validation': True,
                'validation_timeout': 10.0,
                'enable_optimization': True,
            }
        ],
        remappings=[
            ('/raw_action_sequences', '/llm_planner/raw_sequences'),
            ('/validated_sequences', '/action_generator/validated_sequences'),
            ('/optimized_sequences', '/action_generator/optimized_sequences'),
        ],
        output='screen'
    )

    # Create command processor node
    command_processor_node = Node(
        package='vla_system',
        executable='command_processor',
        name='command_processor',
        parameters=[
            config_file,
            {
                'command_timeout': 60.0,
                'max_command_history': 100,
                'enable_nlu': True,  # Natural Language Understanding
                'nlu_model_path': 'models/nlu_model.pkl',
                'enable_intent_classification': True,
                'intent_confidence_threshold': 0.7,
            }
        ],
        remappings=[
            ('/raw_commands', '/command_processor/raw_commands'),
            ('/processed_commands', '/command_processor/processed_commands'),
            ('/command_intents', '/command_processor/intents'),
        ],
        output='screen'
    )

    # Create safety validator node
    safety_validator_node = Node(
        package='vla_system',
        executable='safety_validator',
        name='safety_validator',
        parameters=[
            config_file,
            {
                'enable_safety_checking': True,
                'safety_timeout': 5.0,
                'max_force_threshold': 50.0,  # Newtons
                'max_speed_threshold': 1.0,   # m/s
                'enable_collision_checking': True,
                'collision_checking_range': 2.0,  # meters
            }
        ],
        remappings=[
            ('/action_sequences_to_validate', '/safety_validator/input_sequences'),
            ('/validated_sequences', '/safety_validator/validated_sequences'),
            ('/safety_violations', '/safety_validator/violations'),
        ],
        output='screen'
    )

    # Create prompt engineering node
    prompt_engineering_node = Node(
        package='vla_system',
        executable='prompt_engineer',
        name='prompt_engineer',
        parameters=[
            config_file,
            {
                'enable_dynamic_prompts': True,
                'default_temperature': 0.3,
                'enable_context_awareness': True,
                'context_history_size': 10,
                'enable_prompt_optimization': True,
            }
        ],
        remappings=[
            ('/command_context', '/prompt_engineer/context'),
            ('/optimized_prompts', '/prompt_engineer/optimized_prompts'),
        ],
        output='screen'
    )

    # Create LLM response parser node
    response_parser_node = Node(
        package='vla_system',
        executable='response_parser',
        name='response_parser',
        parameters=[
            config_file,
            {
                'enable_json_validation': True,
                'enable_safety_checking': True,
                'parsing_timeout': 5.0,
                'enable_entity_extraction': True,
                'enable_response_sanitization': True,
            }
        ],
        remappings=[
            ('/llm_responses', '/response_parser/raw_responses'),
            ('/parsed_responses', '/response_parser/parsed_responses'),
            ('/validation_results', '/response_parser/validation_results'),
        ],
        output='screen'
    )

    # Create system monitor node for the LLM integration
    llm_system_monitor = Node(
        package='vla_system',
        executable='llm_system_monitor',
        name='llm_system_monitor',
        parameters=[
            config_file,
            {
                'enable_metrics_collection': True,
                'metrics_collection_interval': 10,  # seconds
                'enable_performance_monitoring': True,
                'enable_error_tracking': True,
                'log_level_threshold': 'INFO',
            }
        ],
        remappings=[
            ('/llm_metrics', '/llm_monitor/metrics'),
            ('/llm_performance', '/llm_monitor/performance'),
            ('/llm_errors', '/llm_monitor/errors'),
        ],
        output='screen'
    )

    # Create logging node to show system status
    system_status_logger = LogInfo(
        msg=["*** LLM Task Planning System Starting ***",
             "\nConfig File: ", config_file,
             "\nModel Name: ", model_name,
             "\nLocal LLM Enabled: ", enable_local_llm,
             "\nTemperature: ", temperature,
             "\nMax Tokens: ", max_tokens,
             "\n*** Launching LLM Integration Nodes ***"]
    )

    # Create the launch description
    ld = LaunchDescription()

    # Add the launch arguments
    ld.add_action(config_file_arg)
    ld.add_action(model_name_arg)
    ld.add_action(enable_local_llm_arg)
    ld.add_action(temperature_arg)
    ld.add_action(max_tokens_arg)

    # Add system status logger
    ld.add_action(system_status_logger)

    # Add all the nodes
    ld.add_action(llm_task_planning_node)
    ld.add_action(action_sequence_generator_node)
    ld.add_action(command_processor_node)
    ld.add_action(safety_validator_node)
    ld.add_action(prompt_engineering_node)
    ld.add_action(response_parser_node)
    ld.add_action(llm_system_monitor)

    return ld


# Additional Python implementation for the launch system
# This provides a more complete implementation that could be used with ROS 2

def create_llm_task_planning_launcher():
    """
    Alternative implementation using Python directly
    This creates the nodes that would be launched by the launch file
    """
    import threading
    import time

    from vla_system.llm_task_planning.llm_service import LLMService
    from vla_system.llm_task_planning.action_generator import ActionSequenceGenerator
    from vla_system.llm_task_planning.command_model import CommandFactory
    from vla_system.llm_task_planning.llm_response_parser import ResponseValidationPipeline
    from vla_system.llm_task_planning.prompt_engineering_utils import PromptEngineeringUtilities

    class LLMTaskPlanningLauncher:
        """
        Complete LLM task planning launcher
        """
        def __init__(self):
            self.logger = __import__('vla_system.observability_utils').EducationalLogger("llm_launcher")
            self.error_handler = __import__('vla_system.error_handling_utils').EducationalErrorHandler()

            # Initialize system components
            self.llm_service = None
            self.action_generator = None
            self.command_factory = None
            self.response_parser = None
            self.prompt_engineer = None

            # System configuration
            self.config = {
                'config_file': 'config/llm_config.yaml',
                'model_name': 'llama3.1:8b',
                'enable_local_llm': True,
                'temperature': 0.3,
                'max_tokens': 2048,
                'response_timeout': 30.0,
                'retry_attempts': 3,
                'enable_caching': True,
                'privacy_compliance': True,
            }

            self.nodes = []
            self.is_running = False

            self.logger.info("LLMTaskPlanningLauncher", "__init__", "LLM task planning launcher initialized")

        def configure_from_params(self, **kwargs):
            """Configure the system from parameters"""
            for key, value in kwargs.items():
                if key in self.config:
                    self.config[key] = value
                    self.logger.debug("LLMTaskPlanningLauncher", "configure_from_params",
                                    f"Configured {key} = {value}")

        def start_system(self):
            """Start the LLM task planning system"""
            try:
                self.logger.info("LLMTaskPlanningLauncher", "start_system", "Starting LLM task planning system")

                # Initialize all system components
                self.llm_service = LLMService()
                self.action_generator = ActionSequenceGenerator(self.llm_service)
                self.command_factory = CommandFactory()
                self.response_parser = ResponseValidationPipeline()
                self.prompt_engineer = PromptEngineeringUtilities()

                # Start all components
                self.logger.info("LLMTaskPlanningLauncher", "start_system", "LLM task planning system started successfully")

                self.is_running = True

            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context="start_system",
                    suggested_fix="Check system configuration and dependencies",
                    learning_objective="Handle LLM system startup errors"
                )
                self.stop_system()

        def stop_system(self):
            """Stop the LLM task planning system"""
            try:
                self.logger.info("LLMTaskPlanningLauncher", "stop_system", "Stopping LLM task planning system")

                # Clean up components
                self.llm_service = None
                self.action_generator = None
                self.command_factory = None
                self.response_parser = None
                self.prompt_engineer = None

                self.is_running = False
                self.logger.info("LLMTaskPlanningLauncher", "stop_system", "LLM task planning system stopped")

            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context="stop_system",
                    suggested_fix="Check system shutdown procedure",
                    learning_objective="Handle LLM system shutdown errors"
                )

        def process_command(self, command_text: str, context: dict = None):
            """Process a natural language command through the LLM system"""
            try:
                if not self.is_running:
                    raise Exception("System not running, cannot process command")

                self.logger.info("LLMTaskPlanningLauncher", "process_command",
                               f"Processing command: {command_text}")

                # Create command
                command = self.command_factory.create_text_command(command_text)

                # Generate action sequence
                action_sequence = self.action_generator.generate_action_sequence(
                    command.text,
                    context=context or {}
                )

                if action_sequence:
                    # Validate the sequence
                    validation_results = self.response_parser.process_response(
                        action_sequence.to_json(),
                        expected_format="action_sequence"
                    )

                    if validation_results['validation_success']:
                        self.logger.info("LLMTaskPlanningLauncher", "process_command",
                                       f"Successfully processed command with {len(action_sequence.steps)} steps")
                        return {
                            'success': True,
                            'action_sequence': action_sequence,
                            'validation_results': validation_results
                        }
                    else:
                        self.logger.error("LLMTaskPlanningLauncher", "process_command",
                                        f"Validation failed: {validation_results['validation_errors']}")
                        return {
                            'success': False,
                            'error': 'Validation failed',
                            'validation_results': validation_results
                        }
                else:
                    self.logger.error("LLMTaskPlanningLauncher", "process_command",
                                    "Failed to generate action sequence")
                    return {
                        'success': False,
                        'error': 'Failed to generate action sequence'
                    }

            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context="process_command",
                    suggested_fix="Check command processing parameters",
                    learning_objective="Handle command processing errors"
                )
                return {'success': False, 'error': str(e)}

        def get_system_status(self):
            """Get status of all system components"""
            status = {
                'is_running': self.is_running,
                'config': self.config,
                'llm_service_status': self.llm_service is not None,
                'action_generator_status': self.action_generator is not None,
                'command_factory_status': self.command_factory is not None,
                'response_parser_status': self.response_parser is not None,
                'prompt_engineer_status': self.prompt_engineer is not None
            }

            return status


def main():
    """Main function to demonstrate the launch system"""
    print("LLM Task Planning System Launch File")
    print("="*50)

    print("Launch file: llm_task_planning.launch.py")
    print("This launch file would start the following nodes:")
    print("  - llm_task_planner: Main LLM task planning node")
    print("  - action_sequence_generator: Action sequence generation and validation")
    print("  - command_processor: Natural language command processing")
    print("  - safety_validator: Safety validation for action sequences")
    print("  - prompt_engineer: Dynamic prompt generation and optimization")
    print("  - response_parser: LLM response parsing and validation")
    print("  - llm_system_monitor: System monitoring and metrics")
    print("")

    print("Launch arguments:")
    print("  config_file: Path to LLM configuration file")
    print("  model_name: Name of the LLM model to use")
    print("  enable_local_llm: Enable/disable local LLM processing")
    print("  temperature: LLM temperature parameter")
    print("  max_tokens: Maximum tokens for LLM responses")
    print("")

    print("All processing occurs locally with no external API calls.")
    print("Privacy compliance is enforced throughout the system.")
    print("")

    # Show how to launch from command line
    print("To launch the LLM task planning system, run:")
    print("  ros2 launch vla_system llm_task_planning.launch.py")
    print("")
    print("With custom parameters:")
    print("  ros2 launch vla_system llm_task_planning.launch.py model_name:=llama3.1:8b enable_local_llm:=true")

    # Demonstrate the Python launcher
    print("")
    print("Demonstrating Python-based launcher...")
    launcher = LLMTaskPlanningLauncher()

    print(f"Launcher initialized with config: {launcher.config}")
    print(f"System status: {launcher.get_system_status()['is_running']}")


if __name__ == '__main__':
    main()