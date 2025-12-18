#!/usr/bin/env python3
"""
Action Sequence Generator for VLA System
Generates executable action sequences from natural language commands using LLM
Part of the Vision-Language-Action (VLA) pipeline for autonomous humanoid systems
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime

from .llm_service import LLMService
from ..observability_utils import EducationalLogger


class ActionType(Enum):
    """Enumeration of supported action types for robotics"""
    MOVE_TO = "move_to"
    GRASP_OBJECT = "grasp_object"
    PLACE_OBJECT = "place_object"
    NAVIGATE_TO = "navigate_to"
    SPEAK = "speak"
    DETECT_OBJECT = "detect_object"
    WAIT = "wait"
    FOLLOW_PATH = "follow_path"
    MANIPULATE_OBJECT = "manipulate_object"
    PERFORM_ACTION = "perform_action"
    STOP = "stop"
    RESET = "reset"


@dataclass
class ActionStep:
    """Represents a single action step in an action sequence"""
    action_type: ActionType
    parameters: Dict[str, Any]
    description: str
    priority: int = 1  # 1 (low) to 5 (high)
    timeout: float = 30.0  # seconds
    retry_attempts: int = 3
    safety_constraints: List[str] = None

    def __post_init__(self):
        if self.safety_constraints is None:
            self.safety_constraints = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert action step to dictionary representation"""
        result = asdict(self)
        result['action_type'] = self.action_type.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionStep':
        """Create action step from dictionary representation"""
        data = data.copy()
        data['action_type'] = ActionType(data['action_type'])
        if 'safety_constraints' not in data:
            data['safety_constraints'] = []
        return cls(**data)


@dataclass
class ActionSequence:
    """Represents a complete action sequence with metadata"""
    id: str
    steps: List[ActionStep]
    original_command: str
    generated_by: str
    timestamp: datetime
    estimated_duration: float = 0.0
    success_threshold: float = 0.95
    error_recovery_enabled: bool = True
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert action sequence to dictionary representation"""
        return {
            'id': self.id,
            'steps': [step.to_dict() for step in self.steps],
            'original_command': self.original_command,
            'generated_by': self.generated_by,
            'timestamp': self.timestamp.isoformat(),
            'estimated_duration': self.estimated_duration,
            'success_threshold': self.success_threshold,
            'error_recovery_enabled': self.error_recovery_enabled,
            'dependencies': self.dependencies
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionSequence':
        """Create action sequence from dictionary representation"""
        data = data.copy()
        data['steps'] = [ActionStep.from_dict(step) for step in data['steps']]
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'dependencies' not in data:
            data['dependencies'] = []
        return cls(**data)


class ActionSequenceGenerator:
    """
    Generates executable action sequences from natural language commands
    Uses LLM to interpret commands and create structured action plans
    """

    def __init__(self, llm_service: LLMService, config: Optional[Dict] = None):
        """
        Initialize the action sequence generator

        Args:
            llm_service: LLM service for natural language processing
            config: Optional configuration dictionary
        """
        self.llm_service = llm_service
        self.config = config or {}
        self.logger = EducationalLogger("action_generator")

        # Initialize action templates and constraints
        self.action_templates = self._load_action_templates()
        self.system_context = self._create_system_context()

        self.logger.info("ActionSequenceGenerator", "__init__",
                        "Action sequence generator initialized successfully")

    def _load_action_templates(self) -> Dict[str, Any]:
        """Load action templates and constraints for robotics tasks"""
        return {
            "navigation": {
                "action_type": ActionType.NAVIGATE_TO,
                "required_params": ["target_location"],
                "optional_params": ["speed", "avoid_obstacles", "max_deviation"],
                "description": "Move the robot to a specified location"
            },
            "manipulation": {
                "action_type": ActionType.GRASP_OBJECT,
                "required_params": ["object_name", "object_position"],
                "optional_params": ["gripper_force", "approach_angle", "grasp_type"],
                "description": "Grasp an object with the robot's manipulator"
            },
            "placement": {
                "action_type": ActionType.PLACE_OBJECT,
                "required_params": ["target_position"],
                "optional_params": ["orientation", "placement_force"],
                "description": "Place an object at a specified location"
            },
            "detection": {
                "action_type": ActionType.DETECT_OBJECT,
                "required_params": ["object_type"],
                "optional_params": ["search_area", "confidence_threshold", "max_objects"],
                "description": "Detect objects of a specific type in the environment"
            },
            "speech": {
                "action_type": ActionType.SPEAK,
                "required_params": ["text"],
                "optional_params": ["voice_type", "volume", "language"],
                "description": "Speak text using the robot's speech system"
            }
        }

    def _create_system_context(self) -> str:
        """Create system context for LLM to understand action generation requirements"""
        return f"""
You are an expert robotics action sequence generator. Your task is to convert natural language commands into structured action sequences for an autonomous humanoid robot.

Available action types:
{chr(10).join([f"- {action_type.value}: {template['description']}" for action_type, template in [(ActionType.NAVIGATE_TO, self.action_templates['navigation']), (ActionType.GRASP_OBJECT, self.action_templates['manipulation']), (ActionType.PLACE_OBJECT, self.action_templates['placement']), (ActionType.DETECT_OBJECT, self.action_templates['detection']), (ActionType.SPEAK, self.action_templates['speech'])]])}

Each action step should include:
- action_type: The type of action to perform
- parameters: Required and optional parameters for the action
- description: A human-readable description of what the action does
- priority: Importance level (1-5)
- timeout: Maximum time to wait for action completion
- safety_constraints: Any safety constraints that apply

Generate action sequences that are safe, executable, and appropriate for the given command.
"""

    def generate_action_sequence(self, command: str, context: Optional[Dict] = None) -> Optional[ActionSequence]:
        """
        Generate an action sequence from a natural language command

        Args:
            command: Natural language command to convert
            context: Additional context information (environment, robot state, etc.)

        Returns:
            ActionSequence object or None if generation fails
        """
        try:
            self.logger.info("ActionSequenceGenerator", "generate_action_sequence",
                           f"Generating action sequence for command: {command}")

            # Create prompt for LLM
            prompt = self._create_generation_prompt(command, context or {})

            # Generate action sequence using LLM
            response = self.llm_service.generate_response(
                prompt=prompt,
                system_context=self.system_context,
                temperature=0.3,
                max_tokens=2048
            )

            if not response:
                self.logger.error("ActionSequenceGenerator", "generate_action_sequence",
                                "LLM returned empty response")
                return None

            # Parse and validate the response
            action_sequence = self._parse_llm_response(response, command)

            if action_sequence:
                self.logger.info("ActionSequenceGenerator", "generate_action_sequence",
                               f"Successfully generated action sequence with {len(action_sequence.steps)} steps")
                return action_sequence
            else:
                self.logger.error("ActionSequenceGenerator", "generate_action_sequence",
                                "Failed to parse LLM response into valid action sequence")
                return None

        except Exception as e:
            self.logger.error("ActionSequenceGenerator", "generate_action_sequence",
                            f"Error generating action sequence: {str(e)}")
            return None

    def _create_generation_prompt(self, command: str, context: Dict) -> str:
        """Create a detailed prompt for the LLM to generate action sequences"""
        context_str = json.dumps(context, indent=2) if context else "{}"

        return f"""
Generate a detailed action sequence for the following command:

Command: "{command}"

Context: {context_str}

Please provide the action sequence in JSON format with the following structure:
{{
    "id": "unique_sequence_id",
    "steps": [
        {{
            "action_type": "action_type_value",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "description": "Human-readable description of the action",
            "priority": 3,
            "timeout": 30.0,
            "retry_attempts": 3,
            "safety_constraints": ["constraint1", "constraint2"]
        }}
    ],
    "original_command": "{command}",
    "estimated_duration": 120.0,
    "success_threshold": 0.95,
    "error_recovery_enabled": true
}}

Available action types: {', '.join([at.value for at in ActionType])}

The action sequence should be safe, executable, and appropriate for the given command.
"""

    def _parse_llm_response(self, response: str, original_command: str) -> Optional[ActionSequence]:
        """Parse the LLM response into an ActionSequence object"""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                self.logger.error("ActionSequenceGenerator", "_parse_llm_response",
                                "No JSON found in LLM response")
                return None

            json_str = json_match.group(0)
            data = json.loads(json_str)

            # Validate required fields
            required_fields = ['steps', 'original_command']
            for field in required_fields:
                if field not in data:
                    self.logger.error("ActionSequenceGenerator", "_parse_llm_response",
                                    f"Missing required field: {field}")
                    return None

            # Validate and convert steps
            steps = []
            for step_data in data['steps']:
                try:
                    # Validate action type
                    if 'action_type' not in step_data:
                        self.logger.warning("ActionSequenceGenerator", "_parse_llm_response",
                                          "Step missing action_type, skipping")
                        continue

                    # Convert action type string to enum
                    try:
                        action_type = ActionType(step_data['action_type'])
                    except ValueError:
                        self.logger.warning("ActionSequenceGenerator", "_parse_llm_response",
                                          f"Invalid action type: {step_data['action_type']}, skipping")
                        continue

                    # Create action step
                    step = ActionStep(
                        action_type=action_type,
                        parameters=step_data.get('parameters', {}),
                        description=step_data.get('description', ''),
                        priority=step_data.get('priority', 1),
                        timeout=step_data.get('timeout', 30.0),
                        retry_attempts=step_data.get('retry_attempts', 3),
                        safety_constraints=step_data.get('safety_constraints', [])
                    )
                    steps.append(step)

                except Exception as e:
                    self.logger.warning("ActionSequenceGenerator", "_parse_llm_response",
                                      f"Error creating action step: {str(e)}, skipping")
                    continue

            if not steps:
                self.logger.error("ActionSequenceGenerator", "_parse_llm_response",
                                "No valid action steps generated")
                return None

            # Create action sequence
            sequence = ActionSequence(
                id=data.get('id', f"seq_{int(datetime.now().timestamp())}"),
                steps=steps,
                original_command=data.get('original_command', original_command),
                generated_by="LLM_Action_Generator",
                timestamp=datetime.now(),
                estimated_duration=data.get('estimated_duration', 0.0),
                success_threshold=data.get('success_threshold', 0.95),
                error_recovery_enabled=data.get('error_recovery_enabled', True),
                dependencies=data.get('dependencies', [])
            )

            return sequence

        except json.JSONDecodeError as e:
            self.logger.error("ActionSequenceGenerator", "_parse_llm_response",
                            f"JSON decode error: {str(e)}")
            return None
        except Exception as e:
            self.logger.error("ActionSequenceGenerator", "_parse_llm_response",
                            f"Error parsing LLM response: {str(e)}")
            return None

    def validate_action_sequence(self, sequence: ActionSequence) -> Dict[str, Any]:
        """
        Validate an action sequence for safety and executability

        Args:
            sequence: Action sequence to validate

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'safety_violations': [],
            'optimization_suggestions': []
        }

        try:
            # Check for empty sequence
            if not sequence.steps:
                validation_results['is_valid'] = False
                validation_results['errors'].append("Action sequence is empty")
                return validation_results

            # Validate each step
            for i, step in enumerate(sequence.steps):
                # Check action type validity
                if not isinstance(step.action_type, ActionType):
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"Invalid action type in step {i}")

                # Check parameters
                if not isinstance(step.parameters, dict):
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"Invalid parameters in step {i}")

                # Check safety constraints
                if not isinstance(step.safety_constraints, list):
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"Invalid safety constraints in step {i}")

                # Check timeout validity
                if step.timeout <= 0:
                    validation_results['warnings'].append(f"Non-positive timeout in step {i}")

                # Check priority validity
                if not (1 <= step.priority <= 5):
                    validation_results['warnings'].append(f"Priority out of range in step {i}")

        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")

        return validation_results

    def optimize_action_sequence(self, sequence: ActionSequence) -> ActionSequence:
        """
        Optimize an action sequence for efficiency and safety

        Args:
            sequence: Action sequence to optimize

        Returns:
            Optimized action sequence
        """
        try:
            # Create a copy of the sequence
            optimized_steps = []

            for step in sequence.steps:
                # Apply optimizations based on action type
                optimized_step = self._optimize_action_step(step)
                optimized_steps.append(optimized_step)

            # Create new sequence with optimized steps
            optimized_sequence = ActionSequence(
                id=f"optimized_{sequence.id}",
                steps=optimized_steps,
                original_command=sequence.original_command,
                generated_by=f"{sequence.generated_by}_optimized",
                timestamp=datetime.now(),
                estimated_duration=sequence.estimated_duration,
                success_threshold=sequence.success_threshold,
                error_recovery_enabled=sequence.error_recovery_enabled,
                dependencies=sequence.dependencies
            )

            return optimized_sequence

        except Exception as e:
            self.logger.error("ActionSequenceGenerator", "optimize_action_sequence",
                            f"Error optimizing action sequence: {str(e)}")
            return sequence  # Return original if optimization fails

    def _optimize_action_step(self, step: ActionStep) -> ActionStep:
        """Apply optimizations to a single action step"""
        # Example optimizations:
        # - Adjust timeouts based on action complexity
        # - Add safety constraints based on action type
        # - Validate parameters

        optimized_step = ActionStep(
            action_type=step.action_type,
            parameters=step.parameters.copy(),
            description=step.description,
            priority=step.priority,
            timeout=step.timeout,
            retry_attempts=step.retry_attempts,
            safety_constraints=step.safety_constraints.copy()
        )

        # Add default safety constraints based on action type
        if step.action_type == ActionType.NAVIGATE_TO and "collision_avoidance" not in step.safety_constraints:
            optimized_step.safety_constraints.append("collision_avoidance")

        elif step.action_type == ActionType.GRASP_OBJECT and "force_limiting" not in step.safety_constraints:
            optimized_step.safety_constraints.append("force_limiting")

        # Adjust timeout based on action type
        if step.action_type in [ActionType.NAVIGATE_TO, ActionType.FOLLOW_PATH]:
            if optimized_step.timeout < 60.0:  # Minimum 60 seconds for navigation
                optimized_step.timeout = 60.0

        return optimized_step


def main():
    """Main function to demonstrate the action sequence generator"""
    print("Action Sequence Generator Demo")
    print("="*40)

    # This would typically be initialized with a real LLM service
    # For demo purposes, we'll show the structure
    print("Action Sequence Generator provides:")
    print("- Natural language to action sequence conversion")
    print("- Action validation and safety checking")
    print("- Sequence optimization")
    print("- Error recovery capabilities")

    # Show example action types
    print(f"\nAvailable action types: {len(ActionType)}")
    for action_type in ActionType:
        print(f"  - {action_type.value}")


if __name__ == "__main__":
    main()