#!/usr/bin/env python3
"""
Action Sequence Data Model for VLA System
Defines the structure and validation for action sequences
Part of the Vision-Language-Action (VLA) pipeline for autonomous humanoid systems
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import uuid
import json
from .observability_utils import EducationalLogger


class ExecutionStatus(Enum):
    """Enumeration of action sequence execution statuses"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class ActionStatus(Enum):
    """Enumeration of individual action execution statuses"""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class ActionStep:
    """Represents a single action step within an action sequence"""
    id: str
    action_type: str  # Using string for flexibility (e.g., "navigation", "manipulation")
    parameters: Dict[str, Any]
    description: str
    priority: int = 3  # 1 (low) to 5 (high)
    timeout: float = 30.0  # seconds
    retry_attempts: int = 3
    dependencies: List[str] = field(default_factory=list)  # IDs of actions this depends on
    preconditions: List[Dict[str, Any]] = field(default_factory=list)  # Conditions that must be met
    postconditions: List[Dict[str, Any]] = field(default_factory=list)  # Expected outcomes
    safety_constraints: List[str] = field(default_factory=list)
    execution_status: ActionStatus = ActionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize the action step with default values if not provided"""
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert action step to dictionary representation"""
        return {
            'id': self.id,
            'action_type': self.action_type,
            'parameters': self.parameters,
            'description': self.description,
            'priority': self.priority,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts,
            'dependencies': self.dependencies,
            'preconditions': self.preconditions,
            'postconditions': self.postconditions,
            'safety_constraints': self.safety_constraints,
            'execution_status': self.execution_status.value,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'result': self.result
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionStep':
        """Create action step from dictionary representation"""
        data = data.copy()
        data['execution_status'] = ActionStatus(data['execution_status'])
        if data['started_at']:
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data['completed_at']:
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


@dataclass
class ActionSequenceExecution:
    """Represents the execution state of an action sequence"""
    sequence_id: str
    current_step_index: int = 0
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    progress: float = 0.0  # 0.0 to 1.0
    current_step_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert execution state to dictionary representation"""
        return {
            'sequence_id': self.sequence_id,
            'current_step_index': self.current_step_index,
            'execution_status': self.execution_status.value,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'execution_log': self.execution_log,
            'progress': self.progress,
            'current_step_id': self.current_step_id,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionSequenceExecution':
        """Create execution state from dictionary representation"""
        data = data.copy()
        data['execution_status'] = ExecutionStatus(data['execution_status'])
        if data['started_at']:
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data['completed_at']:
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


@dataclass
class ActionSequence:
    """
    Represents a complete action sequence with all steps and metadata
    This is the primary data structure for action sequence execution in the VLA system
    """
    id: str
    name: str
    description: str
    steps: List[ActionStep]
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    command_id: Optional[str] = None  # ID of the command that generated this sequence
    estimated_duration: float = 0.0  # seconds
    success_threshold: float = 0.95  # 0.0 to 1.0
    error_recovery_enabled: bool = True
    execution_constraints: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    dependencies: List[str] = field(default_factory=list)  # Other sequences this depends on
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_state: Optional[ActionSequenceExecution] = None

    def __post_init__(self):
        """Initialize the action sequence with default values if not provided"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if self.execution_state is None:
            self.execution_state = ActionSequenceExecution(sequence_id=self.id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert action sequence to dictionary representation"""
        result = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'steps': [step.to_dict() for step in self.steps],
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'command_id': self.command_id,
            'estimated_duration': self.estimated_duration,
            'success_threshold': self.success_threshold,
            'error_recovery_enabled': self.error_recovery_enabled,
            'execution_constraints': self.execution_constraints,
            'tags': self.tags,
            'version': self.version,
            'dependencies': self.dependencies,
            'metadata': self.metadata,
            'execution_state': self.execution_state.to_dict() if self.execution_state else None
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionSequence':
        """Create action sequence from dictionary representation"""
        data = data.copy()

        # Convert steps
        data['steps'] = [ActionStep.from_dict(step) for step in data['steps']]

        # Convert timestamps
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])

        # Convert execution state if present
        if data['execution_state']:
            data['execution_state'] = ActionSequenceExecution.from_dict(data['execution_state'])

        return cls(**data)

    def to_json(self) -> str:
        """Convert action sequence to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'ActionSequence':
        """Create action sequence from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> Dict[str, List[str]]:
        """
        Validate the action sequence and return validation results

        Returns:
            Dictionary with validation results containing errors and warnings
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        # Validate required fields
        if not self.id:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Action sequence ID is required")

        if not self.name.strip():
            validation_results['is_valid'] = False
            validation_results['errors'].append("Action sequence name cannot be empty")

        if not self.steps:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Action sequence must have at least one step")

        if self.success_threshold < 0 or self.success_threshold > 1:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Success threshold must be between 0 and 1")

        # Validate each step
        step_ids = set()
        for i, step in enumerate(self.steps):
            if step.id in step_ids:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Duplicate step ID at index {i}: {step.id}")
            step_ids.add(step.id)

            if step.priority < 1 or step.priority > 5:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Invalid priority in step {i}")

            if step.timeout <= 0:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Non-positive timeout in step {i}")

            if step.retry_attempts < 0:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Negative retry attempts in step {i}")

        # Check for circular dependencies
        try:
            self._check_circular_dependencies()
        except ValueError as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(str(e))

        return validation_results

    def _check_circular_dependencies(self):
        """Check for circular dependencies in action steps"""
        # Build dependency graph
        graph = {}
        step_id_to_index = {step.id: i for i, step in enumerate(self.steps)}

        for i, step in enumerate(self.steps):
            for dep_id in step.dependencies:
                if dep_id not in step_id_to_index:
                    raise ValueError(f"Dependency {dep_id} not found in sequence steps")
                if i not in graph:
                    graph[i] = []
                graph[i].append(step_id_to_index[dep_id])

        # Check for cycles using DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * len(self.steps)

        def dfs(node):
            if color[node] == GRAY:
                raise ValueError("Circular dependency detected in action steps")
            if color[node] == BLACK:
                return

            color[node] = GRAY
            if node in graph:
                for neighbor in graph[node]:
                    dfs(neighbor)
            color[node] = BLACK

        for i in range(len(self.steps)):
            if color[i] == WHITE:
                dfs(i)

    def get_step_by_id(self, step_id: str) -> Optional[ActionStep]:
        """Get an action step by its ID"""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_step_index(self, step_id: str) -> Optional[int]:
        """Get the index of an action step by its ID"""
        for i, step in enumerate(self.steps):
            if step.id == step_id:
                return i
        return None

    def get_next_step(self, current_step_id: str) -> Optional[ActionStep]:
        """Get the next step after the current one, considering dependencies"""
        current_index = self.get_step_index(current_step_id)
        if current_index is None:
            return None

        # Find next step that doesn't depend on the current step
        for i in range(current_index + 1, len(self.steps)):
            step = self.steps[i]
            # Check if this step can be executed (dependencies are satisfied)
            if self._are_dependencies_satisfied(step, current_index):
                return step

        return None

    def _are_dependencies_satisfied(self, step: ActionStep, current_index: int) -> bool:
        """Check if all dependencies of a step are satisfied"""
        for dep_id in step.dependencies:
            dep_index = self.get_step_index(dep_id)
            if dep_index is not None and dep_index > current_index:
                # Dependency comes after current step, so not satisfied
                return False
        return True

    def calculate_progress(self) -> float:
        """Calculate the execution progress of the sequence (0.0 to 1.0)"""
        if not self.steps:
            return 1.0

        completed_steps = sum(1 for step in self.steps
                            if step.execution_status in [ActionStatus.SUCCESS, ActionStatus.SKIPPED])
        return completed_steps / len(self.steps)

    def get_executable_steps(self, completed_step_ids: List[str]) -> List[ActionStep]:
        """Get steps that are ready to be executed based on completed steps"""
        executable_steps = []
        completed_set = set(completed_step_ids)

        for step in self.steps:
            # Check if all dependencies are completed
            if all(dep_id in completed_set for dep_id in step.dependencies):
                # Check if step hasn't been executed yet
                if step.execution_status == ActionStatus.PENDING:
                    executable_steps.append(step)

        return executable_steps

    def update_execution_state(self, step_id: str, status: ActionStatus,
                             result: Optional[Dict[str, Any]] = None,
                             error_message: Optional[str] = None):
        """Update the execution state of a specific step"""
        step = self.get_step_by_id(step_id)
        if step:
            step.execution_status = status
            step.updated_at = datetime.now()

            if status == ActionStatus.EXECUTING:
                step.started_at = datetime.now()
            elif status in [ActionStatus.SUCCESS, ActionStatus.FAILED, ActionStatus.SKIPPED, ActionStatus.TIMEOUT]:
                step.completed_at = datetime.now()

            if error_message:
                step.error_message = error_message
            if result:
                step.result = result

            # Update sequence progress
            if self.execution_state:
                self.execution_state.progress = self.calculate_progress()
                self.execution_state.current_step_id = step_id
                if status in [ActionStatus.SUCCESS, ActionStatus.FAILED, ActionStatus.SKIPPED, ActionStatus.TIMEOUT]:
                    # Move to next step if current step is completed
                    next_step = self.get_next_step(step_id)
                    if next_step:
                        self.execution_state.current_step_index = self.get_step_index(next_step.id) or 0

    def mark_as_executing(self):
        """Mark the sequence as currently executing"""
        if self.execution_state:
            self.execution_state.execution_status = ExecutionStatus.RUNNING
            self.execution_state.started_at = datetime.now()

    def mark_as_completed(self):
        """Mark the sequence as completed"""
        if self.execution_state:
            self.execution_state.execution_status = ExecutionStatus.COMPLETED
            self.execution_state.completed_at = datetime.now()

    def mark_as_failed(self, error_message: str):
        """Mark the sequence as failed"""
        if self.execution_state:
            self.execution_state.execution_status = ExecutionStatus.FAILED
            self.execution_state.error_message = error_message
            self.execution_state.completed_at = datetime.now()


class ActionSequenceFactory:
    """Factory class for creating action sequences with common patterns"""

    def __init__(self):
        self.logger = EducationalLogger("action_sequence_factory")

    def create_navigation_sequence(self, target_location: Dict[str, float],
                                 command_id: Optional[str] = None) -> ActionSequence:
        """Create a navigation action sequence"""
        steps = [
            ActionStep(
                id=str(uuid.uuid4()),
                action_type="detect_obstacles",
                parameters={"scan_radius": 2.0},
                description="Scan for obstacles in the environment"
            ),
            ActionStep(
                id=str(uuid.uuid4()),
                action_type="plan_path",
                parameters={"target_location": target_location},
                description="Plan a safe path to the target location"
            ),
            ActionStep(
                id=str(uuid.uuid4()),
                action_type="navigate",
                parameters={"target_location": target_location, "speed": 0.5},
                description="Navigate to the target location",
                dependencies=[f"step_{i}" for i in range(2)]  # Depends on previous steps
            )
        ]

        sequence = ActionSequence(
            id=str(uuid.uuid4()),
            name=f"Navigate to {target_location}",
            description=f"Navigation sequence to reach coordinates {target_location}",
            steps=steps,
            created_by="NavigationSequenceFactory",
            command_id=command_id,
            estimated_duration=120.0,
            tags=["navigation", "movement"]
        )

        self.logger.info("ActionSequenceFactory", "create_navigation_sequence",
                        f"Created navigation sequence: {sequence.name}")

        return sequence

    def create_manipulation_sequence(self, object_name: str, target_position: Dict[str, float],
                                   command_id: Optional[str] = None) -> ActionSequence:
        """Create a manipulation action sequence"""
        steps = [
            ActionStep(
                id=str(uuid.uuid4()),
                action_type="detect_object",
                parameters={"object_name": object_name},
                description=f"Detect the {object_name} object"
            ),
            ActionStep(
                id=str(uuid.uuid4()),
                action_type="approach_object",
                parameters={"object_name": object_name},
                description=f"Approach the {object_name} object",
                dependencies=["detect_object"]  # This would be the actual step ID
            ),
            ActionStep(
                id=str(uuid.uuid4()),
                action_type="grasp_object",
                parameters={"object_name": object_name},
                description=f"Grasp the {object_name} object",
                dependencies=["approach_object"]
            ),
            ActionStep(
                id=str(uuid.uuid4()),
                action_type="move_object",
                parameters={"target_position": target_position},
                description=f"Move the {object_name} to target position",
                dependencies=["grasp_object"]
            ),
            ActionStep(
                id=str(uuid.uuid4()),
                action_type="release_object",
                parameters={"target_position": target_position},
                description=f"Release the {object_name} at target position",
                dependencies=["move_object"]
            )
        ]

        sequence = ActionSequence(
            id=str(uuid.uuid4()),
            name=f"Manipulate {object_name}",
            description=f"Manipulation sequence to pick up {object_name} and place it at {target_position}",
            steps=steps,
            created_by="ManipulationSequenceFactory",
            command_id=command_id,
            estimated_duration=180.0,
            tags=["manipulation", "grasping"]
        )

        self.logger.info("ActionSequenceFactory", "create_manipulation_sequence",
                        f"Created manipulation sequence: {sequence.name}")

        return sequence

    def create_speech_sequence(self, text: str, command_id: Optional[str] = None) -> ActionSequence:
        """Create a speech action sequence"""
        steps = [
            ActionStep(
                id=str(uuid.uuid4()),
                action_type="text_to_speech",
                parameters={"text": text, "voice_type": "natural"},
                description=f"Convert text to speech: {text[:50]}..."
            )
        ]

        sequence = ActionSequence(
            id=str(uuid.uuid4()),
            name=f"Speech: {text[:30]}...",
            description=f"Speech sequence to speak the text: {text}",
            steps=steps,
            created_by="SpeechSequenceFactory",
            command_id=command_id,
            estimated_duration=10.0,
            tags=["speech", "communication"]
        )

        self.logger.info("ActionSequenceFactory", "create_speech_sequence",
                        f"Created speech sequence: {sequence.name}")

        return sequence


def main():
    """Main function to demonstrate the action sequence data model"""
    print("Action Sequence Data Model Demo")
    print("="*50)

    # Create an action sequence factory
    factory = ActionSequenceFactory()

    # Create example action sequences
    nav_sequence = factory.create_navigation_sequence(
        target_location={"x": 1.5, "y": 2.0, "z": 0.0},
        command_id="cmd_123"
    )

    manipulation_sequence = factory.create_manipulation_sequence(
        object_name="red cup",
        target_position={"x": 3.0, "y": 1.5, "z": 0.8},
        command_id="cmd_124"
    )

    print(f"Navigation Sequence: {nav_sequence.name}")
    print(f"Steps: {len(nav_sequence.steps)}")
    print(f"Estimated Duration: {nav_sequence.estimated_duration}s")

    print(f"\nManipulation Sequence: {manipulation_sequence.name}")
    print(f"Steps: {len(manipulation_sequence.steps)}")
    print(f"Tags: {manipulation_sequence.tags}")

    # Validate sequences
    nav_validation = nav_sequence.validate()
    print(f"\nNavigation Sequence Validation: {'Valid' if nav_validation['is_valid'] else 'Invalid'}")
    if nav_validation['errors']:
        print(f"Errors: {nav_validation['errors']}")
    if nav_validation['warnings']:
        print(f"Warnings: {nav_validation['warnings']}")

    # Update execution state
    if nav_sequence.steps:
        nav_sequence.update_execution_state(
            nav_sequence.steps[0].id,
            ActionStatus.SUCCESS,
            result={"location_reached": True, "coordinates": {"x": 1.5, "y": 2.0, "z": 0.0}}
        )
        print(f"\nUpdated execution progress: {nav_sequence.calculate_progress():.2%}")


if __name__ == "__main__":
    main()