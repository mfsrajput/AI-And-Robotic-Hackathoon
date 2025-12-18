#!/usr/bin/env python3
"""
ROS 2 Action Server Interfaces for VLA System
Defines ROS 2 action interfaces for task planning and execution
Part of the Vision-Language-Action (VLA) pipeline for autonomous humanoid systems
"""

import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy

import threading
from typing import Optional, Dict, Any
import json
from datetime import datetime

# Import our custom data models
from .action_generator import ActionSequenceGenerator, ActionSequence
from .command_model import Command, CommandFactory
from .llm_service import LLMService
from ..observability_utils import EducationalLogger


# Define action interfaces (these would typically be in a separate .action file in a real ROS 2 package)
# For this implementation, we'll define them as Python classes that mimic the structure

class TaskPlan:
    """Mimics the structure of a ROS 2 action interface for task planning"""
    class Goal:
        def __init__(self):
            self.command_text = ""
            self.context = {}

    class Result:
        def __init__(self):
            self.success = False
            self.action_sequence = {}
            self.error_message = ""

    class Feedback:
        def __init__(self):
            self.status = ""
            self.progress = 0.0


class ExecuteAction:
    """Mimics the structure of a ROS 2 action interface for action execution"""
    class Goal:
        def __init__(self):
            self.action_sequence = {}
            self.timeout = 0.0

    class Result:
        def __init__(self):
            self.success = False
            self.execution_log = []
            self.error_message = ""

    class Feedback:
        def __init__(self):
            self.current_action = ""
            self.progress = 0.0
            self.status = ""


class LLMTaskPlanningActionServer(Node):
    """
    ROS 2 Action Server for LLM-based task planning
    Converts natural language commands to action sequences using LLM
    """

    def __init__(self):
        super().__init__('llm_task_planning_server')

        # Initialize components
        self.llm_service = LLMService()
        self.action_generator = ActionSequenceGenerator(self.llm_service)
        self.command_factory = CommandFactory()
        self.logger = EducationalLogger("llm_task_planning_server")

        # Initialize action servers
        self.task_plan_server = ActionServer(
            self,
            TaskPlan,  # In a real implementation, this would be the actual action type
            'plan_task',
            execute_callback=self.execute_task_plan_callback,
            goal_callback=self.goal_task_plan_callback,
            cancel_callback=self.cancel_task_plan_callback,
            callback_group=ReentrantCallbackGroup()
        )

        self.logger.info("LLMTaskPlanningActionServer", "__init__",
                        "LLM Task Planning Action Server initialized")

    def goal_task_plan_callback(self, goal_request):
        """Handle incoming task planning goal requests"""
        self.get_logger().info(f'Received task planning request: {goal_request.command_text}')
        return GoalResponse.ACCEPT

    def cancel_task_plan_callback(self, goal_handle):
        """Handle task planning goal cancellation requests"""
        self.get_logger().info('Received task planning cancel request')
        return CancelResponse.ACCEPT

    async def execute_task_plan_callback(self, goal_handle):
        """Execute the task planning action"""
        self.get_logger().info('Executing task planning...')

        feedback_msg = TaskPlan.Feedback()
        result = TaskPlan.Result()

        try:
            # Create command from the goal request
            command = self.command_factory.create_text_command(
                goal_handle.request.command_text,
                user_id="ros2_user"
            )

            # Add context to command
            if goal_handle.request.context:
                command.context = goal_handle.request.context

            # Generate action sequence using LLM
            action_sequence = self.action_generator.generate_action_sequence(
                command.text,
                context=command.context
            )

            if action_sequence:
                # Validate the action sequence
                validation = self.action_generator.validate_action_sequence(action_sequence)
                if validation['is_valid']:
                    # Convert action sequence to dictionary for result
                    result.action_sequence = action_sequence.to_dict()
                    result.success = True
                    result.error_message = ""

                    self.get_logger().info(f'Generated action sequence with {len(action_sequence.steps)} steps')
                else:
                    result.success = False
                    result.error_message = f"Invalid action sequence: {validation['errors']}"
                    self.get_logger().error(f'Invalid action sequence: {validation["errors"]}')
            else:
                result.success = False
                result.error_message = "Failed to generate action sequence"
                self.get_logger().error('Failed to generate action sequence')

        except Exception as e:
            result.success = False
            result.error_message = f"Error during task planning: {str(e)}"
            self.get_logger().error(f'Error during task planning: {str(e)}')

        goal_handle.succeed()
        return result


class ActionExecutionActionServer(Node):
    """
    ROS 2 Action Server for executing action sequences
    Executes the action sequences generated by the task planning server
    """

    def __init__(self):
        super().__init__('action_execution_server')

        # Initialize components
        self.logger = EducationalLogger("action_execution_server")

        # Initialize action server
        self.execute_action_server = ActionServer(
            self,
            ExecuteAction,  # In a real implementation, this would be the actual action type
            'execute_action_sequence',
            execute_callback=self.execute_action_callback,
            goal_callback=self.goal_execute_callback,
            cancel_callback=self.cancel_execute_callback,
            callback_group=ReentrantCallbackGroup()
        )

        self.logger.info("ActionExecutionActionServer", "__init__",
                        "Action Execution Action Server initialized")

    def goal_execute_callback(self, goal_request):
        """Handle incoming action execution goal requests"""
        self.get_logger().info('Received action execution request')
        return GoalResponse.ACCEPT

    def cancel_execute_callback(self, goal_handle):
        """Handle action execution goal cancellation requests"""
        self.get_logger().info('Received action execution cancel request')
        return CancelResponse.ACCEPT

    async def execute_action_callback(self, goal_handle):
        """Execute the action sequence"""
        self.get_logger().info('Executing action sequence...')

        feedback_msg = ExecuteAction.Feedback()
        result = ExecuteAction.Result()

        try:
            # In a real implementation, this would execute the action sequence
            # For this educational example, we'll simulate execution

            action_sequence_data = goal_handle.request.action_sequence
            if not action_sequence_data:
                result.success = False
                result.error_message = "No action sequence provided"
                goal_handle.succeed()
                return result

            # Simulate execution progress
            total_steps = len(action_sequence_data.get('steps', []))
            for i in range(total_steps):
                # Update feedback
                feedback_msg.current_action = f"Step {i+1}/{total_steps}"
                feedback_msg.progress = (i + 1) / total_steps
                feedback_msg.status = f"Executing step {i+1}"
                goal_handle.publish_feedback(feedback_msg)

                # Simulate step execution (in real implementation, this would call actual robot actions)
                import time
                time.sleep(0.5)  # Simulate processing time

                # Add to execution log
                result.execution_log.append({
                    'step_index': i,
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat()
                })

            result.success = True
            result.error_message = ""

            self.get_logger().info(f'Action sequence executed successfully with {total_steps} steps')

        except Exception as e:
            result.success = False
            result.error_message = f"Error during action execution: {str(e)}"
            self.get_logger().error(f'Error during action execution: {str(e)}')

        goal_handle.succeed()
        return result


class VLATaskPlanningNode(Node):
    """
    Main VLA Task Planning Node that combines both action servers
    Provides a complete interface for natural language to robot action mapping
    """

    def __init__(self):
        super().__init__('vla_task_planning_node')

        # Initialize action servers
        self.task_planning_server = LLMTaskPlanningActionServer()
        self.action_execution_server = ActionExecutionActionServer()

        # Initialize components
        self.llm_service = LLMService()
        self.action_generator = ActionSequenceGenerator(self.llm_service)
        self.command_factory = CommandFactory()
        self.logger = EducationalLogger("vla_task_planning_node")

        self.logger.info("VLATaskPlanningNode", "__init__",
                        "VLA Task Planning Node initialized with both action servers")

    def get_executor(self):
        """Get a multi-threaded executor for both action servers"""
        executor = MultiThreadedExecutor()
        executor.add_node(self.task_planning_server)
        executor.add_node(self.action_execution_server)
        return executor


def main(args=None):
    """Main function to run the VLA Task Planning ROS 2 node"""
    print("VLA Task Planning ROS 2 Action Server Demo")
    print("="*60)

    # This would typically be run as a ROS 2 node
    print("This module defines ROS 2 action servers for:")
    print("- Task planning (natural language to action sequence)")
    print("- Action execution (executing action sequences)")
    print("\nIn a real ROS 2 system, this would be launched with:")
    print("  ros2 run vla_system vla_task_planning_node")

    # Show the structure of the action interfaces
    print("\nAction Interface Structures:")
    print("\nTaskPlan Action:")
    print("  Goal: {command_text: string, context: dict}")
    print("  Result: {success: bool, action_sequence: dict, error_message: string}")
    print("  Feedback: {status: string, progress: float}")

    print("\nExecuteAction Action:")
    print("  Goal: {action_sequence: dict, timeout: float}")
    print("  Result: {success: bool, execution_log: list, error_message: string}")
    print("  Feedback: {current_action: string, progress: float, status: string}")

    # In a real ROS 2 environment, we would initialize and spin the node
    if args is None:
        rclpy.init()

        try:
            # Create the main node
            vla_node = VLATaskPlanningNode()

            # Get the executor and spin
            executor = vla_node.get_executor()
            print("\nVLA Task Planning Node ready to accept action requests...")
            print("Use 'ros2 action send_goal' to interact with the action servers")

            # Spin the executor (this would run indefinitely in a real system)
            # For demo purposes, we'll just show what would happen
            # executor.spin()

        except KeyboardInterrupt:
            print("\nShutting down VLA Task Planning Node...")
        finally:
            rclpy.shutdown()


# Additional utility functions for working with the ROS 2 interfaces
def create_task_plan_goal(command_text: str, context: Dict[str, Any] = None) -> TaskPlan.Goal:
    """Create a task plan goal for the action server"""
    goal = TaskPlan.Goal()
    goal.command_text = command_text
    goal.context = context or {}
    return goal


def create_execute_action_goal(action_sequence: Dict[str, Any], timeout: float = 300.0) -> ExecuteAction.Goal:
    """Create an execute action goal for the action server"""
    goal = ExecuteAction.Goal()
    goal.action_sequence = action_sequence
    goal.timeout = timeout
    return goal


def parse_task_plan_result(result) -> Dict[str, Any]:
    """Parse the result from a task plan action"""
    return {
        'success': result.success,
        'action_sequence': result.action_sequence,
        'error_message': result.error_message
    }


def parse_execute_action_result(result) -> Dict[str, Any]:
    """Parse the result from an execute action action"""
    return {
        'success': result.success,
        'execution_log': result.execution_log,
        'error_message': result.error_message
    }


if __name__ == '__main__':
    main()