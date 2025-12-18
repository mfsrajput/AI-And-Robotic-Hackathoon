#!/usr/bin/env python3
"""
Prompt Engineering Utilities for VLA System
Utilities for creating effective prompts for LLMs in robotics task planning
Part of the Vision-Language-Action (VLA) pipeline for autonomous humanoid systems
"""

import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import re
from .observability_utils import EducationalLogger


class PromptTemplateType(Enum):
    """Enumeration of different prompt template types"""
    TASK_PLANNING = "task_planning"
    ACTION_GENERATION = "action_generation"
    OBJECT_DETECTION = "object_detection"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    SPEECH_SYNTHESIS = "speech_synthesis"
    SAFETY_CHECK = "safety_check"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class PromptTemplate:
    """Represents a prompt template with variables and constraints"""
    name: str
    template_type: PromptTemplateType
    template: str
    variables: List[str]
    description: str = ""
    examples: List[str] = None
    constraints: List[str] = None
    output_format: str = "json"
    temperature: float = 0.3
    max_tokens: int = 1024

    def __post_init__(self):
        if self.examples is None:
            self.examples = []
        if self.constraints is None:
            self.constraints = []


class PromptEngineeringUtilities:
    """
    Utilities for creating and managing prompts for robotics task planning
    Includes template management, variable substitution, and response validation
    """

    def __init__(self):
        self.logger = EducationalLogger("prompt_engineering_utils")
        self.templates = self._initialize_templates()
        self.logger.info("PromptEngineeringUtilities", "__init__",
                        "Prompt engineering utilities initialized with templates")

    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize default prompt templates for robotics applications"""
        templates = {}

        # Task Planning Template
        templates['task_planning'] = PromptTemplate(
            name="task_planning",
            template_type=PromptTemplateType.TASK_PLANNING,
            template="""
You are an expert robotics task planner. Given the following command and context, generate a detailed plan of actions.

Command: {command}
Context: {context}

The plan should include:
1. A sequence of actions to accomplish the goal
2. Required parameters for each action
3. Safety constraints to consider
4. Estimated time for each action

Available action types: {available_actions}

Provide your response in JSON format with the following structure:
{{
    "plan": [
        {{
            "action_type": "action_type_value",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "description": "Human-readable description",
            "priority": 3,
            "timeout": 30.0,
            "safety_constraints": ["constraint1", "constraint2"]
        }}
    ],
    "estimated_duration": 120.0,
    "success_criteria": ["criteria1", "criteria2"]
}}

Ensure the plan is safe, executable, and appropriate for the given command.
""",
            variables=["command", "context", "available_actions"],
            description="Template for generating task plans from natural language commands",
            output_format="json",
            temperature=0.3
        )

        # Action Generation Template
        templates['action_generation'] = PromptTemplate(
            name="action_generation",
            template_type=PromptTemplateType.ACTION_GENERATION,
            template="""
Convert the following natural language command into a structured action sequence:

Command: {command}
Robot Capabilities: {capabilities}
Environment Context: {environment_context}

The action sequence should:
- Be safe and executable by the robot
- Consider environmental constraints
- Include appropriate error handling
- Follow safety protocols

Available Actions:
{available_actions}

Provide the action sequence in JSON format:
{{
    "id": "unique_sequence_id",
    "steps": [
        {{
            "action_type": "action_type_value",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "description": "What this action does",
            "priority": 3,
            "timeout": 30.0,
            "retry_attempts": 3,
            "safety_constraints": ["constraint1", "constraint2"]
        }}
    ],
    "original_command": "{command}",
    "estimated_duration": 120.0
}}

Ensure the action sequence is appropriate for the given command and environment.
""",
            variables=["command", "capabilities", "environment_context", "available_actions"],
            description="Template for generating action sequences from natural language commands",
            output_format="json",
            temperature=0.3
        )

        # Navigation Template
        templates['navigation'] = PromptTemplate(
            name="navigation",
            template_type=PromptTemplateType.NAVIGATION,
            template="""
Generate navigation instructions for the robot to reach the target location.

Current Position: {current_position}
Target Position: {target_position}
Environment Map: {environment_map}
Obstacles: {obstacles}

The navigation plan should:
- Avoid all known obstacles
- Follow safe pathways
- Consider robot dimensions
- Include safety margins

Provide the navigation plan in JSON format:
{{
    "path": [
        {{"x": 1.0, "y": 2.0, "z": 0.0}},
        {{"x": 1.5, "y": 2.0, "z": 0.0}},
        {{"x": 2.0, "y": 2.0, "z": 0.0}}
    ],
    "safety_margins": {{"left": 0.5, "right": 0.5, "front": 0.3, "back": 0.2}},
    "estimated_time": 60.0,
    "max_speed": 0.5
}}

Ensure the path is collision-free and safe for the robot.
""",
            variables=["current_position", "target_position", "environment_map", "obstacles"],
            description="Template for generating navigation plans",
            output_format="json",
            temperature=0.2
        )

        # Manipulation Template
        templates['manipulation'] = PromptTemplate(
            name="manipulation",
            template_type=PromptTemplateType.MANIPULATION,
            template="""
Generate manipulation instructions for the robot to interact with an object.

Object: {object_name}
Object Properties: {object_properties}
Target Action: {target_action}
Current Robot State: {robot_state}

The manipulation plan should:
- Consider object properties (size, weight, fragility)
- Use appropriate gripper force
- Include approach angles
- Follow safety protocols

Provide the manipulation plan in JSON format:
{{
    "approach": {{
        "position": {{"x": 1.0, "y": 2.0, "z": 0.5}},
        "orientation": {{"roll": 0.0, "pitch": 0.0, "yaw": 0.0}},
        "gripper_width": 0.05
    }},
    "grasp": {{
        "force": 10.0,
        "approach_angle": 45.0,
        "grasp_type": "pinch"
    }},
    "manipulation_steps": [
        {{
            "action": "move_to_object",
            "parameters": {{"speed": 0.1}}
        }},
        {{
            "action": "grasp",
            "parameters": {{"force": 10.0}}
        }},
        {{
            "action": "lift",
            "parameters": {{"height": 0.1}}
        }}
    ],
    "safety_constraints": ["force_limiting", "collision_avoidance"]
}}

Ensure the manipulation plan is safe and appropriate for the object.
""",
            variables=["object_name", "object_properties", "target_action", "robot_state"],
            description="Template for generating manipulation plans",
            output_format="json",
            temperature=0.2
        )

        # Safety Check Template
        templates['safety_check'] = PromptTemplate(
            name="safety_check",
            template_type=PromptTemplateType.SAFETY_CHECK,
            template="""
Perform a safety analysis for the following action sequence:

Action Sequence: {action_sequence}
Environment Context: {environment_context}
Robot Capabilities: {robot_capabilities}

Analyze the following safety aspects:
1. Collision risks
2. Force limits
3. Environmental hazards
4. Human safety
5. Equipment safety

Provide the safety analysis in JSON format:
{{
    "collision_risk": "low|medium|high",
    "force_risk": "low|medium|high",
    "environmental_hazards": ["hazard1", "hazard2"],
    "safety_recommendations": ["recommendation1", "recommendation2"],
    "risk_level": "low|medium|high",
    "can_proceed": true|false,
    "required_safety_measures": ["measure1", "measure2"]
}}

Ensure the analysis is thorough and safety-focused.
""",
            variables=["action_sequence", "environment_context", "robot_capabilities"],
            description="Template for safety analysis of action sequences",
            output_format="json",
            temperature=0.1
        )

        return templates

    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name"""
        return self.templates.get(template_name)

    def render_template(self, template_name: str, **kwargs) -> Optional[str]:
        """
        Render a prompt template with provided variables

        Args:
            template_name: Name of the template to use
            **kwargs: Variables to substitute in the template

        Returns:
            Rendered prompt string or None if template not found
        """
        template = self.get_template(template_name)
        if not template:
            self.logger.error("PromptEngineeringUtilities", "render_template",
                            f"Template not found: {template_name}")
            return None

        try:
            # Check that all required variables are provided
            missing_vars = [var for var in template.variables if var not in kwargs]
            if missing_vars:
                self.logger.error("PromptEngineeringUtilities", "render_template",
                                f"Missing required variables: {missing_vars}")
                return None

            # Render the template
            rendered_prompt = template.template.format(**kwargs)

            self.logger.debug("PromptEngineeringUtilities", "render_template",
                            f"Rendered template '{template_name}' with variables: {list(kwargs.keys())}")

            return rendered_prompt

        except KeyError as e:
            self.logger.error("PromptEngineeringUtilities", "render_template",
                            f"Missing variable in template '{template_name}': {str(e)}")
            return None
        except Exception as e:
            self.logger.error("PromptEngineeringUtilities", "render_template",
                            f"Error rendering template '{template_name}': {str(e)}")
            return None

    def create_robotics_prompt(self, command: str, context: Dict[str, Any] = None,
                             prompt_type: str = "action_generation") -> Optional[str]:
        """
        Create a specialized robotics prompt based on command and context

        Args:
            command: Natural language command
            context: Additional context information
            prompt_type: Type of prompt to create

        Returns:
            Formatted prompt string or None if error occurs
        """
        context = context or {}

        # Prepare context variables
        variables = {
            "command": command,
            "context": json.dumps(context, indent=2),
            "available_actions": context.get("available_actions", ["navigate", "grasp", "speak", "detect"]),
            "capabilities": context.get("robot_capabilities", "Standard humanoid robot with arms and navigation"),
            "environment_context": json.dumps(context.get("environment", {}), indent=2),
            "robot_state": json.dumps(context.get("robot_state", {}), indent=2)
        }

        # Add specific variables based on prompt type
        if prompt_type == "navigation":
            variables.update({
                "current_position": context.get("current_position", {"x": 0.0, "y": 0.0, "z": 0.0}),
                "target_position": context.get("target_position", {"x": 1.0, "y": 1.0, "z": 0.0}),
                "environment_map": json.dumps(context.get("map", {}), indent=2),
                "obstacles": json.dumps(context.get("obstacles", []), indent=2)
            })

        elif prompt_type == "manipulation":
            variables.update({
                "object_name": context.get("object_name", "unknown_object"),
                "object_properties": json.dumps(context.get("object_properties", {}), indent=2),
                "target_action": context.get("target_action", "grasp"),
                "robot_state": json.dumps(context.get("robot_state", {}), indent=2)
            })

        elif prompt_type == "safety_check":
            variables.update({
                "action_sequence": json.dumps(context.get("action_sequence", {}), indent=2),
                "robot_capabilities": json.dumps(context.get("robot_capabilities", {}), indent=2)
            })

        # Render the appropriate template
        prompt = self.render_template(prompt_type, **variables)

        if prompt:
            self.logger.info("PromptEngineeringUtilities", "create_robotics_prompt",
                           f"Created {prompt_type} prompt for command: {command[:50]}...")
        else:
            self.logger.error("PromptEngineeringUtilities", "create_robotics_prompt",
                            f"Failed to create {prompt_type} prompt for command: {command}")

        return prompt

    def validate_response_format(self, response: str, template_name: str) -> Dict[str, Any]:
        """
        Validate that the LLM response matches the expected format for the template

        Args:
            response: LLM response to validate
            template_name: Name of the template used

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'parsed_data': None
        }

        try:
            # Try to extract JSON from response if it's wrapped in text
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                validation_results['is_valid'] = False
                validation_results['errors'].append("No JSON found in response")
                return validation_results

            json_str = json_match.group(0)
            parsed_data = json.loads(json_str)

            # Validate based on template type
            template = self.get_template(template_name)
            if not template:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Template not found: {template_name}")
                return validation_results

            if template.template_type == PromptTemplateType.TASK_PLANNING:
                # Validate task planning response
                required_fields = ['plan']
                for field in required_fields:
                    if field not in parsed_data:
                        validation_results['is_valid'] = False
                        validation_results['errors'].append(f"Missing required field: {field}")

                if 'plan' in parsed_data:
                    for i, step in enumerate(parsed_data['plan']):
                        if not isinstance(step, dict):
                            validation_results['is_valid'] = False
                            validation_results['errors'].append(f"Plan step {i} is not a dictionary")
                            continue

                        required_step_fields = ['action_type', 'parameters', 'description']
                        for field in required_step_fields:
                            if field not in step:
                                validation_results['is_valid'] = False
                                validation_results['errors'].append(f"Missing field in plan step {i}: {field}")

            elif template.template_type == PromptTemplateType.ACTION_GENERATION:
                # Validate action generation response
                required_fields = ['steps', 'original_command']
                for field in required_fields:
                    if field not in parsed_data:
                        validation_results['is_valid'] = False
                        validation_results['errors'].append(f"Missing required field: {field}")

                if 'steps' in parsed_data:
                    for i, step in enumerate(parsed_data['steps']):
                        if not isinstance(step, dict):
                            validation_results['is_valid'] = False
                            validation_results['errors'].append(f"Action step {i} is not a dictionary")
                            continue

                        required_step_fields = ['action_type', 'parameters', 'description']
                        for field in required_step_fields:
                            if field not in step:
                                validation_results['is_valid'] = False
                                validation_results['errors'].append(f"Missing field in action step {i}: {field}")

            validation_results['parsed_data'] = parsed_data

        except json.JSONDecodeError as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"JSON decode error: {str(e)}")
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")

        return validation_results

    def optimize_prompt(self, prompt: str, target_audience: str = "robotics_expert") -> str:
        """
        Optimize a prompt for better LLM performance in robotics context

        Args:
            prompt: Original prompt to optimize
            target_audience: Target audience for the prompt

        Returns:
            Optimized prompt string
        """
        # Add role specification based on target audience
        if target_audience == "robotics_expert":
            optimized_prompt = f"""
You are an expert robotics task planner with deep knowledge of autonomous systems,
motion planning, and safety protocols. Your responses should be precise, technically
accurate, and focused on executable robotic actions.

{prompt}
"""
        else:
            optimized_prompt = prompt

        # Add clarity and structure
        optimized_prompt += "\n\nBe specific, use precise terminology, and provide structured output."

        return optimized_prompt

    def create_context_aware_prompt(self, base_command: str, context_history: List[Dict[str, Any]],
                                  current_context: Dict[str, Any]) -> str:
        """
        Create a context-aware prompt that considers conversation history and current state

        Args:
            base_command: The current command to process
            context_history: List of previous commands and responses
            current_context: Current environmental and robot state

        Returns:
            Context-aware prompt string
        """
        # Format context history
        history_str = ""
        if context_history:
            history_str = "Previous interactions:\n"
            for i, interaction in enumerate(context_history[-3:]):  # Use last 3 interactions
                history_str += f"Interaction {i+1}: Command: {interaction.get('command', '')}\n"
                history_str += f"Response: {interaction.get('response', '')[:100]}...\n\n"

        # Format current context
        context_str = json.dumps(current_context, indent=2)

        # Create context-aware prompt
        context_aware_prompt = f"""
You are an expert robotics assistant. Consider the following context when processing the current command:

{history_str}

Current Context:
{context_str}

Current Command: {base_command}

Based on the previous interactions and current context, provide an appropriate response for the robot to execute.
"""

        return context_aware_prompt

    def get_template_variables(self, template_name: str) -> Optional[List[str]]:
        """Get the list of variables required by a template"""
        template = self.get_template(template_name)
        return template.variables if template else None


def main():
    """Main function to demonstrate the prompt engineering utilities"""
    print("Prompt Engineering Utilities Demo")
    print("="*50)

    # Initialize utilities
    prompt_utils = PromptEngineeringUtilities()

    # Show available templates
    print("Available Templates:")
    for name, template in prompt_utils.templates.items():
        print(f"  - {name}: {template.description}")

    # Create a sample command
    command = "Navigate to the kitchen and pick up the red cup from the table"
    context = {
        "current_position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "target_position": {"x": 5.0, "y": 3.0, "z": 0.0},
        "robot_capabilities": ["navigation", "manipulation", "object_detection"],
        "environment": {
            "room_layout": "kitchen is 5m east and 3m north of current position",
            "obstacles": ["sofa", "coffee_table"]
        }
    }

    # Create a robotics prompt
    prompt = prompt_utils.create_robotics_prompt(command, context, "action_generation")
    if prompt:
        print(f"\nGenerated prompt for command: {command}")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"First 200 chars: {prompt[:200]}...")

    # Validate a sample response
    sample_response = '''
    {
        "steps": [
            {
                "action_type": "navigate",
                "parameters": {"target_location": {"x": 5.0, "y": 3.0, "z": 0.0}},
                "description": "Navigate to kitchen",
                "priority": 3,
                "timeout": 60.0
            }
        ],
        "original_command": "Navigate to the kitchen and pick up the red cup from the table"
    }
    '''

    validation = prompt_utils.validate_response_format(sample_response, "action_generation")
    print(f"\nResponse validation: {'Valid' if validation['is_valid'] else 'Invalid'}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")


if __name__ == "__main__":
    main()