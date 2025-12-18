"""
Voice Command Data Model for VLA System

This module defines the VoiceCommand data model based on the entity specification.
It represents a spoken instruction that needs to be processed through Whisper and converted to text.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class CommandTypes(Enum):
    """Enumeration of possible command types"""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMMUNICATION = "communication"


class VoiceCommandStatus(Enum):
    """Enumeration of voice command statuses"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Location:
    """Represents a 3D location in space"""
    x: float
    y: float
    z: float


@dataclass
class Orientation:
    """Represents orientation using roll, pitch, yaw"""
    roll: float
    pitch: float
    yaw: float


@dataclass
class Obstacle:
    """Represents an obstacle in the environment"""
    id: str
    position: Location
    size: Optional[Dict[str, float]] = None
    type: Optional[str] = None


@dataclass
class Environment:
    """Represents environmental context"""
    room_layout: str
    obstacles: List[Obstacle] = field(default_factory=list)


@dataclass
class Context:
    """Represents environmental context for the command"""
    location: Location
    orientation: Orientation
    environment: Environment


@dataclass
class VoiceCommand:
    """
    Represents a spoken instruction that needs to be processed through Whisper and converted to text.

    Attributes:
        id: Unique identifier for the command
        text: The transcribed text from voice input
        command_type: Type of command (navigation, manipulation, perception, communication)
        timestamp: When the command was received
        confidence: Confidence score of voice recognition (0.0-1.0)
        source: Source of the command (voice, text, API)
        user_id: Identifier of the user who issued the command (optional)
        status: Current status (pending, processing, completed, failed)
        error_message: Error message if processing failed (optional)
        context: Environmental context for the command
    """

    id: str
    text: str
    command_type: CommandTypes
    timestamp: datetime
    confidence: float
    source: str
    status: VoiceCommandStatus
    context: Context
    user_id: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate the VoiceCommand after initialization"""
        self._validate()

    def _validate(self):
        """Validate the voice command according to the data model rules"""
        # Validate text is not empty
        if not self.text or not self.text.strip():
            raise ValueError("Text must not be empty")

        # Validate confidence is between 0.0 and 1.0
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        # Validate timestamp is in the past or present
        if self.timestamp > datetime.now():
            raise ValueError("Timestamp must be in the past or present")

    def update_status(self, new_status: VoiceCommandStatus):
        """Update the status of the voice command"""
        # Define valid state transitions
        valid_transitions = {
            VoiceCommandStatus.PENDING: [VoiceCommandStatus.PROCESSING, VoiceCommandStatus.FAILED],
            VoiceCommandStatus.PROCESSING: [VoiceCommandStatus.COMPLETED, VoiceCommandStatus.FAILED],
            VoiceCommandStatus.COMPLETED: [],
            VoiceCommandStatus.FAILED: []
        }

        current_transitions = valid_transitions.get(self.status, [])
        if new_status not in current_transitions:
            raise ValueError(f"Invalid state transition from {self.status} to {new_status}")

        self.status = new_status

    def is_valid_for_processing(self) -> bool:
        """Check if the voice command is valid for processing"""
        try:
            self._validate()
            return self.status in [VoiceCommandStatus.PENDING, VoiceCommandStatus.PROCESSING]
        except ValueError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the VoiceCommand to a dictionary representation"""
        return {
            'id': self.id,
            'text': self.text,
            'command_type': self.command_type.value,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'source': self.source,
            'user_id': self.user_id,
            'status': self.status.value,
            'error_message': self.error_message,
            'context': {
                'location': {
                    'x': self.context.location.x,
                    'y': self.context.location.y,
                    'z': self.context.location.z
                },
                'orientation': {
                    'roll': self.context.orientation.roll,
                    'pitch': self.context.orientation.pitch,
                    'yaw': self.context.orientation.yaw
                },
                'environment': {
                    'room_layout': self.context.environment.room_layout,
                    'obstacles': [
                        {
                            'id': obs.id,
                            'position': {
                                'x': obs.position.x,
                                'y': obs.position.y,
                                'z': obs.position.z
                            },
                            'size': obs.size,
                            'type': obs.type
                        }
                        for obs in self.context.environment.obstacles
                    ]
                }
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceCommand':
        """Create a VoiceCommand from a dictionary representation"""
        # Parse location
        location_data = data['context']['location']
        location = Location(
            x=location_data['x'],
            y=location_data['y'],
            z=location_data['z']
        )

        # Parse orientation
        orientation_data = data['context']['orientation']
        orientation = Orientation(
            roll=orientation_data['roll'],
            pitch=orientation_data['pitch'],
            yaw=orientation_data['yaw']
        )

        # Parse obstacles
        obstacles = []
        for obs_data in data['context']['environment']['obstacles']:
            obs_position = Location(**obs_data['position'])
            obstacle = Obstacle(
                id=obs_data['id'],
                position=obs_position,
                size=obs_data.get('size'),
                type=obs_data.get('type')
            )
            obstacles.append(obstacle)

        # Create environment
        environment = Environment(
            room_layout=data['context']['environment']['room_layout'],
            obstacles=obstacles
        )

        # Create context
        context = Context(
            location=location,
            orientation=orientation,
            environment=environment
        )

        # Create and return VoiceCommand
        return cls(
            id=data['id'],
            text=data['text'],
            command_type=CommandTypes(data['command_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            confidence=data['confidence'],
            source=data['source'],
            user_id=data.get('user_id'),
            status=VoiceCommandStatus(data['status']),
            error_message=data.get('error_message'),
            context=context
        )


# Example usage and validation
if __name__ == "__main__":
    # Create a sample voice command
    sample_location = Location(x=1.0, y=2.0, z=0.0)
    sample_orientation = Orientation(roll=0.0, pitch=0.0, yaw=1.57)
    sample_environment = Environment(
        room_layout="living_room",
        obstacles=[
            Obstacle(
                id="obstacle_1",
                position=Location(x=1.5, y=2.5, z=0.0),
                size={"width": 0.5, "height": 0.5, "depth": 0.5},
                type="chair"
            )
        ]
    )
    sample_context = Context(
        location=sample_location,
        orientation=sample_orientation,
        environment=sample_environment
    )

    # Create a voice command
    voice_cmd = VoiceCommand(
        id="cmd_001",
        text="Go to the kitchen",
        command_type=CommandTypes.NAVIGATION,
        timestamp=datetime.now(),
        confidence=0.95,
        source="voice",
        status=VoiceCommandStatus.PENDING,
        context=sample_context
    )

    print(f"Created voice command: {voice_cmd.text}")
    print(f"Command ID: {voice_cmd.id}")
    print(f"Confidence: {voice_cmd.confidence}")
    print(f"Status: {voice_cmd.status.value}")

    # Convert to dict and back
    voice_cmd_dict = voice_cmd.to_dict()
    print(f"\nDictionary representation: {voice_cmd_dict['text']}")

    reconstructed_cmd = VoiceCommand.from_dict(voice_cmd_dict)
    print(f"\nReconstructed command: {reconstructed_cmd.text}")

    # Update status
    voice_cmd.update_status(VoiceCommandStatus.PROCESSING)
    print(f"Updated status: {voice_cmd.status.value}")