#!/usr/bin/env python3
"""
LLM Response Parser and Validator for VLA System
Parses and validates LLM responses for robotics task planning
Part of the Vision-Language-Action (VLA) pipeline for autonomous humanoid systems
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

from .action_generator import ActionSequence, ActionStep, ActionType
from .command_model import Command
from ..observability_utils import EducationalLogger


@dataclass
class ParsedResponse:
    """Represents a parsed and validated LLM response"""
    success: bool
    parsed_data: Optional[Dict[str, Any]]
    validation_errors: List[str]
    validation_warnings: List[str]
    parsing_time: float
    original_response: str


class LLMResponseParser:
    """
    Parses and validates LLM responses for robotics applications
    Ensures responses conform to expected formats and are safe for execution
    """

    def __init__(self):
        self.logger = EducationalLogger("llm_response_parser")

    def parse_and_validate(self, response: str, expected_format: str = "action_sequence") -> ParsedResponse:
        """
        Parse and validate an LLM response

        Args:
            response: Raw LLM response string
            expected_format: Expected format type ('action_sequence', 'task_plan', 'command', etc.)

        Returns:
            ParsedResponse object with validation results
        """
        start_time = datetime.now()

        # Extract JSON from response (in case it's wrapped in text)
        json_content = self._extract_json(response)
        if not json_content:
            return ParsedResponse(
                success=False,
                parsed_data=None,
                validation_errors=["No valid JSON found in response"],
                validation_warnings=[],
                parsing_time=(datetime.now() - start_time).total_seconds(),
                original_response=response
            )

        try:
            parsed_data = json.loads(json_content)
        except json.JSONDecodeError as e:
            return ParsedResponse(
                success=False,
                parsed_data=None,
                validation_errors=[f"JSON decode error: {str(e)}"],
                validation_warnings=[],
                parsing_time=(datetime.now() - start_time).total_seconds(),
                original_response=response
            )

        # Validate based on expected format
        validation_errors, validation_warnings = self._validate_by_format(parsed_data, expected_format)

        # Check for safety violations
        safety_errors = self._check_safety_violations(parsed_data)
        validation_errors.extend(safety_errors)

        success = len(validation_errors) == 0

        parsing_time = (datetime.now() - start_time).total_seconds()

        return ParsedResponse(
            success=success,
            parsed_data=parsed_data if success else None,
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
            parsing_time=parsing_time,
            original_response=response
        )

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON content from text that may contain additional content"""
        # Look for JSON objects in the text
        json_matches = re.findall(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
        if json_matches:
            # Return the first complete JSON object found
            return json_matches[0]

        # If no matches found, try to find array as well
        array_matches = re.findall(r'\[(?:[^\[\]]|(?R))*\]', text, re.DOTALL)
        if array_matches:
            return array_matches[0]

        return None

    def _validate_by_format(self, data: Dict[str, Any], expected_format: str) -> Tuple[List[str], List[str]]:
        """Validate parsed data based on expected format"""
        errors = []
        warnings = []

        if expected_format == "action_sequence":
            errors, warnings = self._validate_action_sequence(data)
        elif expected_format == "task_plan":
            errors, warnings = self._validate_task_plan(data)
        elif expected_format == "command":
            errors, warnings = self._validate_command(data)
        else:
            errors.append(f"Unknown format type: {expected_format}")

        return errors, warnings

    def _validate_action_sequence(self, data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate action sequence format"""
        errors = []
        warnings = []

        # Required fields for action sequence
        required_fields = ['steps', 'original_command']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        if 'steps' in data:
            if not isinstance(data['steps'], list):
                errors.append("Steps must be a list")
            else:
                for i, step in enumerate(data['steps']):
                    if not isinstance(step, dict):
                        errors.append(f"Step {i} is not a dictionary")
                        continue

                    # Validate step structure
                    step_errors, step_warnings = self._validate_action_step(step, i)
                    errors.extend(step_errors)
                    warnings.extend(step_warnings)

        # Validate other fields if present
        if 'estimated_duration' in data and not isinstance(data['estimated_duration'], (int, float)):
            errors.append("estimated_duration must be a number")

        if 'success_threshold' in data and not isinstance(data['success_threshold'], (int, float)):
            errors.append("success_threshold must be a number")

        if 'success_threshold' in data and (data['success_threshold'] < 0 or data['success_threshold'] > 1):
            errors.append("success_threshold must be between 0 and 1")

        return errors, warnings

    def _validate_action_step(self, step: Dict[str, Any], index: int) -> Tuple[List[str], List[str]]:
        """Validate individual action step"""
        errors = []
        warnings = []

        required_fields = ['action_type', 'parameters', 'description']
        for field in required_fields:
            if field not in step:
                errors.append(f"Missing required field in step {index}: {field}")

        if 'action_type' in step:
            # Validate action type is one of the allowed types
            try:
                ActionType(step['action_type'])
            except ValueError:
                errors.append(f"Invalid action type in step {index}: {step['action_type']}")

        if 'parameters' in step and not isinstance(step['parameters'], dict):
            errors.append(f"Parameters in step {index} must be a dictionary")

        if 'priority' in step:
            priority = step['priority']
            if not isinstance(priority, int) or priority < 1 or priority > 5:
                errors.append(f"Priority in step {index} must be an integer between 1 and 5")

        if 'timeout' in step:
            timeout = step['timeout']
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors.append(f"Timeout in step {index} must be a positive number")

        if 'retry_attempts' in step:
            retry_attempts = step['retry_attempts']
            if not isinstance(retry_attempts, int) or retry_attempts < 0:
                errors.append(f"Retry attempts in step {index} must be a non-negative integer")

        return errors, warnings

    def _validate_task_plan(self, data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate task plan format"""
        errors = []
        warnings = []

        # Required fields for task plan
        required_fields = ['plan']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        if 'plan' in data:
            if not isinstance(data['plan'], list):
                errors.append("Plan must be a list")
            else:
                for i, step in enumerate(data['plan']):
                    if not isinstance(step, dict):
                        errors.append(f"Plan step {i} is not a dictionary")
                        continue

                    # Validate step structure
                    required_step_fields = ['action_type', 'parameters', 'description']
                    for field in required_step_fields:
                        if field not in step:
                            errors.append(f"Missing required field in plan step {i}: {field}")

        return errors, warnings

    def _validate_command(self, data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate command format"""
        errors = []
        warnings = []

        # Required fields for command
        required_fields = ['text', 'command_type']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        if 'command_type' in data:
            # Validate command type is one of the allowed types
            valid_types = ['navigation', 'manipulation', 'speech', 'detection', 'motion', 'composite', 'query', 'configuration']
            if data['command_type'] not in valid_types:
                errors.append(f"Invalid command type: {data['command_type']}")

        if 'priority' in data:
            priority = data['priority']
            if not isinstance(priority, int) or priority < 1 or priority > 5:
                errors.append("Priority must be an integer between 1 and 5")

        if 'timeout' in data:
            timeout = data['timeout']
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors.append("Timeout must be a positive number")

        return errors, warnings

    def _check_safety_violations(self, data: Dict[str, Any]) -> List[str]:
        """Check for potential safety violations in the response"""
        safety_violations = []

        # Check for unsafe action types or parameters
        if 'steps' in data:
            for i, step in enumerate(data['steps']):
                if isinstance(step, dict):
                    # Check for potentially unsafe action types
                    action_type = step.get('action_type', '').lower()
                    if action_type in ['shutdown', 'power_off', 'disable_safety', 'bypass_constraints']:
                        safety_violations.append(f"Potentially unsafe action type in step {i}: {action_type}")

                    # Check for unsafe parameters
                    parameters = step.get('parameters', {})
                    if isinstance(parameters, dict):
                        for param_name, param_value in parameters.items():
                            if 'force' in param_name.lower() and isinstance(param_value, (int, float)):
                                if param_value > 100:  # Arbitrary high force threshold
                                    safety_violations.append(f"Excessive force parameter in step {i}: {param_name}={param_value}")
                            elif 'speed' in param_name.lower() and isinstance(param_value, (int, float)):
                                if param_value > 2.0:  # Arbitrary high speed threshold for safety
                                    safety_violations.append(f"Excessive speed parameter in step {i}: {param_name}={param_value}")

        # Check for unsafe command types
        if 'command_type' in data:
            command_type = data['command_type'].lower()
            if command_type in ['shutdown', 'power_off', 'disable_safety']:
                safety_violations.append(f"Potentially unsafe command type: {command_type}")

        return safety_violations

    def parse_action_sequence(self, response: str) -> Optional[ActionSequence]:
        """Parse LLM response into an ActionSequence object"""
        parsed_response = self.parse_and_validate(response, "action_sequence")

        if not parsed_response.success:
            self.logger.error("LLMResponseParser", "parse_action_sequence",
                            f"Failed to parse action sequence: {parsed_response.validation_errors}")
            return None

        try:
            # Convert parsed data to ActionSequence object
            data = parsed_response.parsed_data

            # Convert steps to ActionStep objects
            steps = []
            for step_data in data.get('steps', []):
                # Convert action type string to enum
                try:
                    action_type = ActionType(step_data['action_type'])
                except ValueError:
                    self.logger.error("LLMResponseParser", "parse_action_sequence",
                                    f"Invalid action type: {step_data['action_type']}")
                    continue

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

            # Create ActionSequence object
            sequence = ActionSequence(
                id=data.get('id', f"seq_{int(datetime.now().timestamp())}"),
                steps=steps,
                original_command=data.get('original_command', ''),
                generated_by="LLM_Response_Parser",
                timestamp=datetime.now(),
                estimated_duration=data.get('estimated_duration', 0.0),
                success_threshold=data.get('success_threshold', 0.95),
                error_recovery_enabled=data.get('error_recovery_enabled', True)
            )

            return sequence

        except Exception as e:
            self.logger.error("LLMResponseParser", "parse_action_sequence",
                            f"Error creating ActionSequence object: {str(e)}")
            return None

    def validate_action_sequence_safety(self, sequence: ActionSequence) -> Dict[str, Any]:
        """Perform safety validation on an action sequence"""
        safety_results = {
            'is_safe': True,
            'violations': [],
            'warnings': [],
            'risk_level': 'low'  # low, medium, high
        }

        high_risk_actions = []
        medium_risk_actions = []

        for i, step in enumerate(sequence.steps):
            # Check for high-risk actions
            if step.action_type in [ActionType.GRASP_OBJECT, ActionType.MANIPULATE_OBJECT]:
                # Check force parameters
                force_param = step.parameters.get('gripper_force', 0)
                if force_param > 50:  # High force threshold
                    high_risk_actions.append(f"Step {i}: High gripper force ({force_param})")
                elif force_param > 20:  # Medium force threshold
                    medium_risk_actions.append(f"Step {i}: Medium gripper force ({force_param})")

            # Check for navigation risks
            elif step.action_type == ActionType.NAVIGATE_TO:
                speed_param = step.parameters.get('speed', 1.0)
                if speed_param > 1.0:  # High speed threshold
                    medium_risk_actions.append(f"Step {i}: High navigation speed ({speed_param})")

            # Check safety constraints
            if not step.safety_constraints:
                safety_results['warnings'].append(f"Step {i} has no safety constraints")

        if high_risk_actions:
            safety_results['is_safe'] = False
            safety_results['violations'].extend(high_risk_actions)
            safety_results['risk_level'] = 'high'
        elif medium_risk_actions:
            safety_results['warnings'].extend(medium_risk_actions)
            safety_results['risk_level'] = 'medium'

        return safety_results

    def sanitize_response(self, response: str) -> str:
        """Sanitize LLM response by removing potentially harmful content"""
        # Remove potentially dangerous commands
        sanitized = re.sub(r'\b(shutdown|power off|disable safety|bypass constraints)\b',
                          '[REDACTED]', response, flags=re.IGNORECASE)

        # Remove potentially harmful code blocks
        sanitized = re.sub(r'```.*?```', '[CODE_BLOCK_REMOVED]', sanitized, flags=re.DOTALL)

        return sanitized

    def extract_entities(self, response: str) -> List[Dict[str, Any]]:
        """Extract entities from LLM response"""
        entities = []

        # Look for location entities
        location_pattern = r'"?location"?\s*:\s*{[^}]*"x"\s*:\s*([+-]?\d*\.?\d+)\s*,\s*"y"\s*:\s*([+-]?\d*\.?\d+)(?:\s*,\s*"z"\s*:\s*([+-]?\d*\.?\d+))?'
        for match in re.finditer(location_pattern, response):
            entity = {
                'type': 'location',
                'value': {
                    'x': float(match.group(1)),
                    'y': float(match.group(2)),
                    'z': float(match.group(3)) if match.group(3) else 0.0
                },
                'confidence': 0.9
            }
            entities.append(entity)

        # Look for object entities
        object_pattern = r'"?object"?\s*:\s*"([^"]+)"'
        for match in re.finditer(object_pattern, response):
            entity = {
                'type': 'object',
                'value': match.group(1),
                'confidence': 0.8
            }
            entities.append(entity)

        return entities


class ResponseValidationPipeline:
    """
    A pipeline that combines parsing, validation, and safety checking
    """

    def __init__(self):
        self.parser = LLMResponseParser()
        self.logger = EducationalLogger("response_validation_pipeline")

    def process_response(self, response: str, expected_format: str = "action_sequence") -> Dict[str, Any]:
        """
        Process an LLM response through the full validation pipeline

        Args:
            response: Raw LLM response
            expected_format: Expected format type

        Returns:
            Dictionary with processing results
        """
        results = {
            'original_response': response,
            'parsing_success': False,
            'validation_success': False,
            'safety_check_passed': True,
            'parsed_object': None,
            'validation_errors': [],
            'validation_warnings': [],
            'safety_violations': [],
            'processing_time': 0.0
        }

        start_time = datetime.now()

        # Step 1: Parse and validate format
        parsed_response = self.parser.parse_and_validate(response, expected_format)
        results['parsing_success'] = parsed_response.success
        results['validation_errors'] = parsed_response.validation_errors
        results['validation_warnings'] = parsed_response.validation_warnings
        results['validation_success'] = parsed_response.success

        if not parsed_response.success:
            results['processing_time'] = (datetime.now() - start_time).total_seconds()
            return results

        # Step 2: Create appropriate object based on format
        if expected_format == "action_sequence":
            action_sequence = self.parser.parse_action_sequence(response)
            if action_sequence:
                results['parsed_object'] = action_sequence

                # Step 3: Perform safety validation
                safety_results = self.parser.validate_action_sequence_safety(action_sequence)
                results['safety_check_passed'] = safety_results['is_safe']
                results['safety_violations'] = safety_results['violations']
                results['safety_warnings'] = safety_results['warnings']
                results['risk_level'] = safety_results['risk_level']

        results['processing_time'] = (datetime.now() - start_time).total_seconds()

        return results


def main():
    """Main function to demonstrate the LLM response parsing and validation"""
    print("LLM Response Parser and Validator Demo")
    print("="*60)

    # Initialize parser
    parser = LLMResponseParser()
    pipeline = ResponseValidationPipeline()

    # Example LLM response
    sample_response = '''
    {
        "id": "seq_12345",
        "steps": [
            {
                "action_type": "navigate_to",
                "parameters": {
                    "target_location": {"x": 1.5, "y": 2.0, "z": 0.0},
                    "speed": 0.5
                },
                "description": "Navigate to the kitchen",
                "priority": 3,
                "timeout": 60.0,
                "retry_attempts": 3,
                "safety_constraints": ["collision_avoidance"]
            },
            {
                "action_type": "grasp_object",
                "parameters": {
                    "object_name": "red cup",
                    "gripper_force": 15.0
                },
                "description": "Grasp the red cup",
                "priority": 4,
                "timeout": 30.0,
                "retry_attempts": 2,
                "safety_constraints": ["force_limiting"]
            }
        ],
        "original_command": "Go to the kitchen and pick up the red cup",
        "estimated_duration": 120.0,
        "success_threshold": 0.95,
        "error_recovery_enabled": true
    }
    '''

    print("Sample LLM Response:")
    print(sample_response[:200] + "...")
    print()

    # Parse and validate the response
    parsed_response = parser.parse_and_validate(sample_response, "action_sequence")
    print(f"Parsing Success: {parsed_response.success}")
    print(f"Validation Errors: {len(parsed_response.validation_errors)}")
    print(f"Validation Warnings: {len(parsed_response.validation_warnings)}")
    print(f"Parsing Time: {parsed_response.parsing_time:.4f}s")

    if parsed_response.validation_errors:
        print("Errors:")
        for error in parsed_response.validation_errors:
            print(f"  - {error}")

    if parsed_response.validation_warnings:
        print("Warnings:")
        for warning in parsed_response.validation_warnings:
            print(f"  - {warning}")

    # Process through the full pipeline
    print("\nProcessing through validation pipeline...")
    pipeline_results = pipeline.process_response(sample_response, "action_sequence")

    print(f"Pipeline Results:")
    print(f"  Parsing Success: {pipeline_results['parsing_success']}")
    print(f"  Validation Success: {pipeline_results['validation_success']}")
    print(f"  Safety Check Passed: {pipeline_results['safety_check_passed']}")
    print(f"  Processing Time: {pipeline_results['processing_time']:.4f}s")

    if pipeline_results['safety_violations']:
        print("Safety Violations:")
        for violation in pipeline_results['safety_violations']:
            print(f"  - {violation}")

    if pipeline_results['safety_warnings']:
        print("Safety Warnings:")
        for warning in pipeline_results['safety_warnings']:
            print(f"  - {warning}")

    # Test with an unsafe response
    print("\nTesting with potentially unsafe response...")
    unsafe_response = '''
    {
        "id": "seq_unsafe",
        "steps": [
            {
                "action_type": "grasp_object",
                "parameters": {"gripper_force": 1000.0},  // Very high force
                "description": "Grasp with maximum force",
                "priority": 5,
                "timeout": 10.0
            }
        ],
        "original_command": "shutdown system",  // Unsafe command
        "estimated_duration": 5.0
    }
    '''

    unsafe_results = pipeline.process_response(unsafe_response, "action_sequence")
    print(f"Unsafe Response - Safety Check Passed: {unsafe_results['safety_check_passed']}")
    print(f"Risk Level: {unsafe_results['risk_level']}")
    print(f"Safety Violations: {len(unsafe_results['safety_violations'])}")


if __name__ == "__main__":
    main()