#!/usr/bin/env python3
"""
Command Data Model for VLA System
Defines the structure and validation for natural language commands
Part of the Vision-Language-Action (VLA) pipeline for autonomous humanoid systems
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid
import json
from .observability_utils import EducationalLogger


class CommandSource(Enum):
    """Enumeration of command sources"""
    VOICE = "voice"
    TEXT = "text"
    GUI = "gui"
    API = "api"
    DEMO = "demo"
    SCRIPT = "script"


class CommandStatus(Enum):
    """Enumeration of command processing statuses"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class CommandType(Enum):
    """Enumeration of command types"""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    SPEECH = "speech"
    DETECTION = "detection"
    MOTION = "motion"
    COMPOSITE = "composite"
    QUERY = "query"
    CONFIGURATION = "configuration"


@dataclass
class CommandMetadata:
    """Metadata associated with a command"""
    timestamp: datetime
    source: CommandSource
    confidence: float = 1.0
    language: str = "en"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    location: Optional[Dict[str, float]] = None  # x, y, z coordinates
    environment_context: Optional[Dict[str, Any]] = None
    security_level: int = 1  # 1 (low) to 5 (high)


@dataclass
class Command:
    """
    Represents a natural language command with all necessary metadata
    This is the primary data structure for command processing in the VLA system
    """
    id: str
    text: str
    command_type: CommandType
    metadata: CommandMetadata
    original_text: str = ""
    processed_text: str = ""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)
    priority: int = 3  # 1 (low) to 5 (high)
    timeout: float = 300.0  # seconds
    retry_attempts: int = 3
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    status: CommandStatus = CommandStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize the command with default values if not provided"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.original_text:
            self.original_text = self.text
        if self.metadata.timestamp is None:
            self.metadata.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert command to dictionary representation"""
        result = {
            'id': self.id,
            'text': self.text,
            'command_type': self.command_type.value,
            'metadata': {
                'timestamp': self.metadata.timestamp.isoformat(),
                'source': self.metadata.source.value,
                'confidence': self.metadata.confidence,
                'language': self.metadata.language,
                'user_id': self.metadata.user_id,
                'session_id': self.metadata.session_id,
                'device_id': self.metadata.device_id,
                'location': self.metadata.location,
                'environment_context': self.metadata.environment_context,
                'security_level': self.metadata.security_level
            },
            'original_text': self.original_text,
            'processed_text': self.processed_text,
            'entities': self.entities,
            'intents': self.intents,
            'priority': self.priority,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts,
            'dependencies': self.dependencies,
            'context': self.context,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'result': self.result
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Command':
        """Create command from dictionary representation"""
        data = data.copy()

        # Convert string values back to enums
        data['command_type'] = CommandType(data['command_type'])

        # Create metadata
        metadata_data = data['metadata']
        metadata = CommandMetadata(
            timestamp=datetime.fromisoformat(metadata_data['timestamp']),
            source=CommandSource(metadata_data['source']),
            confidence=metadata_data['confidence'],
            language=metadata_data['language'],
            user_id=metadata_data.get('user_id'),
            session_id=metadata_data.get('session_id'),
            device_id=metadata_data.get('device_id'),
            location=metadata_data.get('location'),
            environment_context=metadata_data.get('environment_context'),
            security_level=metadata_data.get('security_level', 1)
        )
        data['metadata'] = metadata

        # Convert timestamps
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data['completed_at']:
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])

        # Convert status
        data['status'] = CommandStatus(data['status'])

        return cls(**data)

    def to_json(self) -> str:
        """Convert command to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Command':
        """Create command from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> Dict[str, List[str]]:
        """
        Validate the command and return validation results

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
            validation_results['errors'].append("Command ID is required")

        if not self.text.strip():
            validation_results['is_valid'] = False
            validation_results['errors'].append("Command text cannot be empty")

        if self.priority < 1 or self.priority > 5:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Priority must be between 1 and 5")

        if self.timeout <= 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Timeout must be positive")

        if self.retry_attempts < 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Retry attempts cannot be negative")

        # Validate metadata
        if self.metadata.confidence < 0 or self.metadata.confidence > 1:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Confidence must be between 0 and 1")

        if self.metadata.security_level < 1 or self.metadata.security_level > 5:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Security level must be between 1 and 5")

        # Check for potentially unsafe commands
        unsafe_patterns = [
            'shutdown', 'power off', 'turn off', 'disable safety',
            'ignore constraints', 'bypass safety', 'emergency stop override'
        ]
        text_lower = self.text.lower()
        for pattern in unsafe_patterns:
            if pattern in text_lower:
                validation_results['warnings'].append(f"Potentially unsafe command detected: {pattern}")

        return validation_results

    def update_status(self, new_status: CommandStatus, error_message: Optional[str] = None):
        """Update the command status and optionally set an error message"""
        self.status = new_status
        self.updated_at = datetime.now()

        if new_status == CommandStatus.COMPLETED:
            self.completed_at = datetime.now()

        if error_message:
            self.error_message = error_message

    def add_entity(self, entity_type: str, entity_value: str, confidence: float = 1.0):
        """Add an entity to the command"""
        entity = {
            'type': entity_type,
            'value': entity_value,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        self.entities.append(entity)

    def add_intent(self, intent: str):
        """Add an intent to the command"""
        if intent not in self.intents:
            self.intents.append(intent)

    def has_intent(self, intent: str) -> bool:
        """Check if the command has a specific intent"""
        return intent in self.intents

    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all entities of a specific type"""
        return [entity for entity in self.entities if entity['type'] == entity_type]

    def get_entity_value(self, entity_type: str, default: Optional[str] = None) -> Optional[str]:
        """Get the value of the first entity of a specific type"""
        entities = self.get_entities_by_type(entity_type)
        return entities[0]['value'] if entities else default

    def requires_action_planning(self) -> bool:
        """Check if the command requires complex action planning"""
        complex_types = [CommandType.NAVIGATION, CommandType.MANIPULATION, CommandType.COMPOSITE]
        return self.command_type in complex_types

    def is_safe_to_execute(self) -> bool:
        """Check if the command is safe to execute based on security level and content"""
        # Check security level
        if self.metadata.security_level > 3:
            # High security commands require additional validation
            if self.metadata.confidence < 0.8:
                return False

        # Check for unsafe patterns
        validation_results = self.validate()
        return len(validation_results['errors']) == 0


class CommandFactory:
    """Factory class for creating commands with common patterns"""

    def __init__(self):
        self.logger = EducationalLogger("command_factory")

    def create_voice_command(self, text: str, confidence: float = 0.9) -> Command:
        """Create a command from voice input"""
        metadata = CommandMetadata(
            timestamp=datetime.now(),
            source=CommandSource.VOICE,
            confidence=confidence,
            language="en"
        )

        command_type = self._infer_command_type(text)

        command = Command(
            id=str(uuid.uuid4()),
            text=text,
            command_type=command_type,
            metadata=metadata,
            original_text=text
        )

        self.logger.info("CommandFactory", "create_voice_command",
                        f"Created voice command: {text[:50]}...")

        return command

    def create_text_command(self, text: str, user_id: Optional[str] = None) -> Command:
        """Create a command from text input"""
        metadata = CommandMetadata(
            timestamp=datetime.now(),
            source=CommandSource.TEXT,
            confidence=1.0,
            language="en",
            user_id=user_id
        )

        command_type = self._infer_command_type(text)

        command = Command(
            id=str(uuid.uuid4()),
            text=text,
            command_type=command_type,
            metadata=metadata,
            original_text=text
        )

        self.logger.info("CommandFactory", "create_text_command",
                        f"Created text command: {text[:50]}...")

        return command

    def _infer_command_type(self, text: str) -> CommandType:
        """Infer command type from text content"""
        text_lower = text.lower()

        navigation_keywords = ['go to', 'move to', 'navigate to', 'walk to', 'drive to', 'go', 'move']
        manipulation_keywords = ['pick up', 'grasp', 'grab', 'place', 'put', 'move object', 'lift']
        speech_keywords = ['say', 'speak', 'tell', 'announce', 'repeat']
        detection_keywords = ['find', 'detect', 'locate', 'search for', 'look for', 'see']
        motion_keywords = ['stand up', 'sit down', 'turn', 'rotate', 'wave', 'dance', 'move arm']

        if any(keyword in text_lower for keyword in navigation_keywords):
            return CommandType.NAVIGATION
        elif any(keyword in text_lower for keyword in manipulation_keywords):
            return CommandType.MANIPULATION
        elif any(keyword in text_lower for keyword in speech_keywords):
            return CommandType.SPEECH
        elif any(keyword in text_lower for keyword in detection_keywords):
            return CommandType.DETECTION
        elif any(keyword in text_lower for keyword in motion_keywords):
            return CommandType.MOTION
        elif 'and' in text_lower or 'then' in text_lower:
            return CommandType.COMPOSITE
        else:
            return CommandType.QUERY


def main():
    """Main function to demonstrate the command data model"""
    print("Command Data Model Demo")
    print("="*40)

    # Create a command factory
    factory = CommandFactory()

    # Create example commands
    voice_command = factory.create_voice_command("Please navigate to the kitchen and pick up the red cup", confidence=0.85)
    text_command = factory.create_text_command("Move the robot to position x=1.5, y=2.0", user_id="user_123")

    print(f"Voice Command: {voice_command.text}")
    print(f"Command Type: {voice_command.command_type.value}")
    print(f"Source: {voice_command.metadata.source.value}")
    print(f"Confidence: {voice_command.metadata.confidence}")

    print(f"\nText Command: {text_command.text}")
    print(f"Command Type: {text_command.command_type.value}")
    print(f"User ID: {text_command.metadata.user_id}")

    # Validate commands
    voice_validation = voice_command.validate()
    print(f"\nVoice Command Validation: {'Valid' if voice_validation['is_valid'] else 'Invalid'}")
    if voice_validation['errors']:
        print(f"Errors: {voice_validation['errors']}")
    if voice_validation['warnings']:
        print(f"Warnings: {voice_validation['warnings']}")


if __name__ == "__main__":
    main()