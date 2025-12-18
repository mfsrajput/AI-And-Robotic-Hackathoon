#!/usr/bin/env python3
"""
LLM Task Planning Demo for VLA System

This script demonstrates the LLM-based task and motion planning for the Vision-Language-Action system.
It shows how natural language commands are converted to sequences of ROS actions.
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import yaml

# Import our models
from scripts.voice_control.voice_command_model import VoiceCommand, CommandTypes, VoiceCommandStatus, Location, Orientation, Environment, Context, Obstacle


@dataclass
class ActionStep:
    """
    Represents a single action within an action sequence.
    """
    id: str
    action_type: str
    parameters: Dict[str, Any]
    description: str
    priority: int = 3
    timeout: float = 10.0
    retry_attempts: int = 1
    dependencies: List[str] = None
    preconditions: List[Dict[str, Any]] = None
    postconditions: List[Dict[str, Any]] = None
    safety_constraints: List[str] = None
    ros_action_server: Optional[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.preconditions is None:
            self.preconditions = []
        if self.postconditions is None:
            self.postconditions = []
        if self.safety_constraints is None:
            self.safety_constraints = []


@dataclass
class ActionSequence:
    """
    Represents a series of ROS 2 actions generated from natural language commands.
    """
    id: str
    name: str
    description: str
    steps: List[ActionStep]
    created_by: str
    created_at: datetime
    updated_at: datetime
    command_id: str
    estimated_duration: float = 0.0
    success_threshold: float = 0.8
    error_recovery_enabled: bool = True
    execution_constraints: Dict[str, Any] = None
    tags: List[str] = None
    version: str = "1.0"
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.execution_constraints is None:
            self.execution_constraints = {}
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


class MockOllamaService:
    """
    Mock service to simulate Ollama/Llama-based LLM integration.
    In a real implementation, this would interface with Ollama
    """

    def __init__(self):
        self.command_mappings = {
            "go to the kitchen": {
                "name": "Navigate to Kitchen",
                "description": "Move the robot to the kitchen area",
                "steps": [
                    {
                        "action_type": "navigate_to",
                        "parameters": {"target_location": "kitchen", "navigation_mode": "path_planning"},
                        "description": "Navigate to the kitchen area",
                        "ros_action_server": "/navigate_to_pose"
                    }
                ]
            },
            "pick up the red cup": {
                "name": "Grasp Red Cup",
                "description": "Detect and grasp the red cup",
                "steps": [
                    {
                        "action_type": "detect_object",
                        "parameters": {"object_type": "red cup", "search_area": "table"},
                        "description": "Detect the red cup on the table",
                        "ros_action_server": "/detect_objects"
                    },
                    {
                        "action_type": "navigate_to",
                        "parameters": {"target_location": "red cup location", "navigation_mode": "precise"},
                        "description": "Move to the location of the red cup",
                        "ros_action_server": "/navigate_to_pose"
                    },
                    {
                        "action_type": "grasp_object",
                        "parameters": {"object_type": "red cup", "grasp_type": "top_grasp"},
                        "description": "Grasp the red cup",
                        "ros_action_server": "/grasp_object"
                    }
                ]
            },
            "find the blue box": {
                "name": "Detect Blue Box",
                "description": "Locate the blue box in the environment",
                "steps": [
                    {
                        "action_type": "detect_object",
                        "parameters": {"object_type": "blue box", "search_area": "room"},
                        "description": "Scan the room for the blue box",
                        "ros_action_server": "/detect_objects"
                    },
                    {
                        "action_type": "report_location",
                        "parameters": {"object_type": "blue box"},
                        "description": "Report the location of the blue box",
                        "ros_action_server": "/report_location"
                    }
                ]
            },
            "go to the blue box": {
                "name": "Navigate to Blue Box",
                "description": "Find and navigate to the blue box",
                "steps": [
                    {
                        "action_type": "detect_object",
                        "parameters": {"object_type": "blue box", "search_area": "room"},
                        "description": "Scan the room for the blue box",
                        "ros_action_server": "/detect_objects"
                    },
                    {
                        "action_type": "navigate_to",
                        "parameters": {"target_location": "blue box location", "navigation_mode": "path_planning"},
                        "description": "Navigate to the blue box location",
                        "ros_action_server": "/navigate_to_pose"
                    }
                ]
            }
        }

    def generate_action_sequence(self, command: str, context: Dict[str, Any]) -> ActionSequence:
        """
        Generate an action sequence from a natural language command using LLM simulation.

        Args:
            command: The natural language command
            context: Contextual information for the command

        Returns:
            ActionSequence object
        """
        print(f"   Generating action sequence for command: '{command}'")

        # Simulate processing time
        time.sleep(0.8)

        # Get the mapping for this command
        mapping = self.command_mappings.get(command.lower())
        if not mapping:
            # Default mapping for unknown commands
            mapping = {
                "name": "Unknown Command",
                "description": f"Unknown command: {command}",
                "steps": [
                    {
                        "action_type": "unknown",
                        "parameters": {"command": command},
                        "description": f"Unknown command: {command}",
                        "ros_action_server": None
                    }
                ]
            }

        # Create action steps from the mapping
        steps = []
        for i, step_data in enumerate(mapping["steps"]):
            step = ActionStep(
                id=f"step_{i+1:02d}",
                action_type=step_data["action_type"],
                parameters=step_data["parameters"],
                description=step_data["description"],
                ros_action_server=step_data["ros_action_server"]
            )
            steps.append(step)

        # Calculate estimated duration based on number of steps
        estimated_duration = len(steps) * 2.5  # 2.5 seconds per step on average

        # Create and return the action sequence
        sequence = ActionSequence(
            id=f"seq_{int(time.time())}",
            name=mapping["name"],
            description=mapping["description"],
            steps=steps,
            created_by="LLM_Service",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            command_id=context.get("command_id", "unknown"),
            estimated_duration=estimated_duration,
            success_threshold=0.8,
            error_recovery_enabled=True
        )

        print(f"   Generated action sequence with {len(steps)} steps")
        return sequence


class MockPromptEngineering:
    """
    Mock service for prompt engineering utilities for robotics task planning.
    """

    def __init__(self):
        self.templates = {
            "navigation": {
                "template": "You are an expert robotics task planner. Given the command '{command}' in context '{context}', generate a navigation plan.",
                "parameters": ["command", "context"]
            },
            "manipulation": {
                "template": "You are an expert robotics task planner. Given the command '{command}' in context '{context}', generate a manipulation plan.",
                "parameters": ["command", "context"]
            },
            "perception": {
                "template": "You are an expert robotics task planner. Given the command '{command}' in context '{context}', generate a perception plan.",
                "parameters": ["command", "context"]
            }
        }

    def apply_template(self, template_name: str, **kwargs) -> str:
        """
        Apply a prompt template with the given parameters.

        Args:
            template_name: Name of the template to use
            **kwargs: Parameters to substitute in the template

        Returns:
            Formatted prompt string
        """
        if template_name not in self.templates:
            return f"Template '{template_name}' not found"

        template = self.templates[template_name]["template"]
        params = self.templates[template_name]["parameters"]

        # Check that all required parameters are provided
        for param in params:
            if param not in kwargs:
                return f"Missing required parameter '{param}' for template '{template_name}'"

        # Format the template with the provided parameters
        try:
            formatted_prompt = template.format(**kwargs)
            return formatted_prompt
        except KeyError as e:
            return f"Error formatting template: {e}"


class LLMTaskPlanningDemo:
    """
    Demo class that demonstrates the LLM-based task planning pipeline.
    """

    def __init__(self):
        self.ollama_service = MockOllamaService()
        self.prompt_engineering = MockPromptEngineering()

    def plan_task(self, command: str, context: Dict[str, Any]) -> Optional[ActionSequence]:
        """
        Plan a task based on a natural language command.

        Args:
            command: The natural language command
            context: Contextual information

        Returns:
            ActionSequence object if successful, None otherwise
        """
        print(f"\n=== Planning Task for Command: '{command}' ===")

        # Step 1: Apply prompt engineering
        print("\n1. Applying prompt engineering...")
        command_type = self._classify_command(command)
        prompt = self.prompt_engineering.apply_template(
            command_type,
            command=command,
            context=json.dumps(context)
        )
        print(f"   Generated prompt: {prompt[:100]}...")

        # Step 2: Generate action sequence with LLM
        print("\n2. Generating action sequence with LLM...")
        sequence = self.ollama_service.generate_action_sequence(command, context)
        print(f"   Sequence ID: {sequence.id}")
        print(f"   Sequence name: {sequence.name}")
        print(f"   Estimated duration: {sequence.estimated_duration:.1f}s")

        # Step 3: Validate the action sequence
        print("\n3. Validating action sequence...")
        if self._validate_sequence(sequence):
            print("   Sequence validation: PASSED")
        else:
            print("   Sequence validation: FAILED")
            return None

        # Step 4: Return the action sequence
        print(f"\nAction sequence generated successfully with {len(sequence.steps)} steps")
        return sequence

    def _classify_command(self, command: str) -> str:
        """
        Classify the command type for prompt selection.

        Args:
            command: The command to classify

        Returns:
            Command type string
        """
        command_lower = command.lower()
        if any(word in command_lower for word in ["go to", "navigate", "move to", "walk to"]):
            return "navigation"
        elif any(word in command_lower for word in ["pick up", "grasp", "grab", "take", "lift"]):
            return "manipulation"
        elif any(word in command_lower for word in ["find", "detect", "locate", "see", "look for"]):
            return "perception"
        else:
            return "navigation"  # Default to navigation

    def _validate_sequence(self, sequence: ActionSequence) -> bool:
        """
        Validate the action sequence according to rules.

        Args:
            sequence: The action sequence to validate

        Returns:
            True if valid, False otherwise
        """
        # Check that sequence has at least one step
        if len(sequence.steps) == 0:
            print("   Error: Sequence has no steps")
            return False

        # Check that all steps have valid action types
        for step in sequence.steps:
            if not step.action_type or step.action_type == "unknown":
                print(f"   Error: Step has unknown action type")
                return False

        # Check that estimated duration is positive
        if sequence.estimated_duration <= 0:
            print("   Error: Estimated duration must be positive")
            return False

        # Check success threshold is between 0.0 and 1.0
        if not 0.0 <= sequence.success_threshold <= 1.0:
            print("   Error: Success threshold must be between 0.0 and 1.0")
            return False

        return True

    def execute_sequence(self, sequence: ActionSequence) -> Dict[str, Any]:
        """
        Execute the action sequence (simulation).

        Args:
            sequence: The action sequence to execute

        Returns:
            Execution result dictionary
        """
        print(f"\n=== Executing Action Sequence: {sequence.name} ===")

        results = {
            "sequence_id": sequence.id,
            "executed_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "total_steps": len(sequence.steps),
            "start_time": datetime.now(),
            "end_time": None,
            "success": False,
            "error": None
        }

        for i, step in enumerate(sequence.steps):
            print(f"\n   Executing step {i+1}/{len(sequence.steps)}: {step.description}")
            print(f"     Action: {step.action_type}")
            print(f"     Parameters: {step.parameters}")

            # Simulate step execution
            time.sleep(0.5)  # Simulate processing time

            # Determine if step succeeds (with some randomness for demo)
            import random
            step_success = random.random() > 0.1  # 90% success rate for demo

            results["executed_steps"] += 1
            if step_success:
                results["successful_steps"] += 1
                print(f"     Result: SUCCESS")
            else:
                results["failed_steps"] += 1
                print(f"     Result: FAILED")
                # For demo purposes, we'll continue execution even if a step fails
                # In real implementation, you might want to handle failures differently

        results["end_time"] = datetime.now()
        results["execution_time"] = (results["end_time"] - results["start_time"]).total_seconds()
        results["success"] = results["successful_steps"] == results["total_steps"]

        print(f"\nExecution Summary:")
        print(f"   Total steps: {results['total_steps']}")
        print(f"   Successful: {results['successful_steps']}")
        print(f"   Failed: {results['failed_steps']}")
        print(f"   Execution time: {results['execution_time']:.2f}s")
        print(f"   Overall success: {results['success']}")

        return results

    def run_demo(self):
        """
        Run the complete LLM task planning demo.
        """
        print("Starting LLM Task Planning Demo for VLA System")
        print("=" * 60)

        # Define test commands and contexts
        test_cases = [
            {
                "command": "Go to the kitchen",
                "context": {
                    "command_id": "cmd_001",
                    "robot_location": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "environment": "living_room",
                    "robot_capabilities": ["navigation", "basic_manipulation"]
                }
            },
            {
                "command": "Find the blue box",
                "context": {
                    "command_id": "cmd_002",
                    "robot_location": {"x": 1.0, "y": 1.0, "z": 0.0},
                    "environment": "living_room",
                    "robot_capabilities": ["perception", "navigation"]
                }
            },
            {
                "command": "Go to the blue box",
                "context": {
                    "command_id": "cmd_003",
                    "robot_location": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "environment": "living_room",
                    "robot_capabilities": ["navigation", "perception"]
                }
            }
        ]

        # Process each test case
        for i, test_case in enumerate(test_cases):
            print(f"\n--- Test Case {i+1} ---")
            command = test_case["command"]
            context = test_case["context"]

            # Plan the task
            sequence = self.plan_task(command, context)

            if sequence:
                # Execute the sequence
                result = self.execute_sequence(sequence)

                # Print sequence details
                print(f"\nSequence Details:")
                for j, step in enumerate(sequence.steps):
                    print(f"   Step {j+1}: {step.action_type} - {step.description}")
                    if step.ros_action_server:
                        print(f"     ROS Server: {step.ros_action_server}")

                print(f"\nCommand '{command}' processed successfully!")
            else:
                print(f"\nCommand '{command}' failed to generate a valid sequence.")

            # Wait between test cases
            if i < len(test_cases) - 1:
                print(f"\nWaiting before next command...")
                time.sleep(1)

        print("\n" + "=" * 60)
        print("LLM Task Planning Demo completed!")


def main():
    """
    Main function to run the LLM task planning demo.
    """
    demo = LLMTaskPlanningDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()