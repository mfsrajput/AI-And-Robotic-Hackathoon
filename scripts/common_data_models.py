"""
Common data models for VLA (Vision-Language-Action) system
Based on data-model.md specification
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class VoiceCommand:
    """
    Represents a voice command processed by the system
    """
    audio_data: bytes  # Raw audio data
    transcript: str    # Transcribed text
    timestamp: datetime  # When command was received
    confidence: float   # Confidence score (0.0-1.0)
    status: str        # Processing status: pending, processing, completed, failed
    user_id: Optional[str] = None  # Optional user identifier
    language: str = "en"  # Language code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "transcript": self.transcript,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "status": self.status,
            "user_id": self.user_id,
            "language": self.language
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceCommand':
        """Create instance from dictionary"""
        return cls(
            audio_data=b'',  # Audio data would be handled separately
            transcript=data["transcript"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            confidence=data["confidence"],
            status=data["status"],
            user_id=data.get("user_id"),
            language=data.get("language", "en")
        )


@dataclass
class ActionSequence:
    """
    Represents a sequence of actions to be executed by the robot
    """
    commands: List[Dict[str, Any]]  # List of action commands
    context: Dict[str, Any]        # Context for the action sequence
    status: str                    # Execution status: pending, executing, completed, failed
    created_at: datetime           # When sequence was created
    executed_at: Optional[datetime] = None  # When execution started
    completed_at: Optional[datetime] = None  # When execution completed
    robot_id: Optional[str] = None  # Target robot identifier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "commands": self.commands,
            "context": self.context,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }
        if self.executed_at:
            result["executed_at"] = self.executed_at.isoformat()
        if self.completed_at:
            result["completed_at"] = self.completed_at.isoformat()
        if self.robot_id:
            result["robot_id"] = self.robot_id
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionSequence':
        """Create instance from dictionary"""
        return cls(
            commands=data["commands"],
            context=data["context"],
            status=data["status"],
            created_at=datetime.fromisoformat(data["created_at"]),
            executed_at=datetime.fromisoformat(data["executed_at"]) if data.get("executed_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            robot_id=data.get("robot_id")
        )


@dataclass
class Command:
    """
    Represents a single command to be processed by the LLM planner
    """
    text: str                    # Natural language command text
    source: str                 # Source of command: voice, text, api
    timestamp: datetime         # When command was received
    processed: bool            # Whether command has been processed
    action_sequence: Optional[ActionSequence] = None  # Generated action sequence
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "text": self.text,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "processed": self.processed,
        }
        if self.action_sequence:
            result["action_sequence"] = self.action_sequence.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Command':
        """Create instance from dictionary"""
        action_sequence = None
        if data.get("action_sequence"):
            action_sequence = ActionSequence.from_dict(data["action_sequence"])
        
        return cls(
            text=data["text"],
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            processed=data["processed"],
            action_sequence=action_sequence,
            metadata=data.get("metadata")
        )


@dataclass
class MultiModalInput:
    """
    Represents multi-modal input combining voice, vision, and context
    """
    voice_input: Optional[VoiceCommand] = None  # Voice command component
    visual_input: Optional[Dict[str, Any]] = None  # Visual data (image paths, objects, etc.)
    context: Optional[Dict[str, Any]] = None  # Contextual information
    fusion_result: Optional[Dict[str, Any]] = None  # Result of multi-modal fusion
    timestamp: Optional[datetime] = None  # When input was received
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "context": self.context or {},
            "fusion_result": self.fusion_result or {},
        }
        if self.voice_input:
            result["voice_input"] = self.voice_input.to_dict()
        if self.visual_input:
            result["visual_input"] = self.visual_input
        if self.timestamp:
            result["timestamp"] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiModalInput':
        """Create instance from dictionary"""
        voice_input = None
        if data.get("voice_input"):
            voice_input = VoiceCommand.from_dict(data["voice_input"])
        
        return cls(
            voice_input=voice_input,
            visual_input=data.get("visual_input"),
            context=data.get("context"),
            fusion_result=data.get("fusion_result"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
        )


@dataclass
class AutonomousHumanoidSystem:
    """
    Represents the complete autonomous humanoid system state
    """
    system_id: str              # Unique system identifier
    current_state: str          # Current system state: idle, processing, executing, error
    active_tasks: List[str]     # Currently active task IDs
    capabilities: List[str]     # System capabilities
    sensors: List[str]          # Available sensors
    actuators: List[str]        # Available actuators
    last_update: datetime       # Last state update timestamp
    health_status: Dict[str, Any]  # Health status of system components
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "system_id": self.system_id,
            "current_state": self.current_state,
            "active_tasks": self.active_tasks,
            "capabilities": self.capabilities,
            "sensors": self.sensors,
            "actuators": self.actuators,
            "last_update": self.last_update.isoformat(),
            "health_status": self.health_status,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutonomousHumanoidSystem':
        """Create instance from dictionary"""
        return cls(
            system_id=data["system_id"],
            current_state=data["current_state"],
            active_tasks=data["active_tasks"],
            capabilities=data["capabilities"],
            sensors=data["sensors"],
            actuators=data["actuators"],
            last_update=datetime.fromisoformat(data["last_update"]),
            health_status=data["health_status"],
        )


@dataclass
class CapstoneProject:
    """
    Represents the capstone project configuration and state
    """
    project_id: str             # Unique project identifier
    title: str                 # Project title
    description: str           # Project description
    objectives: List[str]      # Learning objectives
    requirements: List[str]    # Technical requirements
    evaluation_criteria: List[Dict[str, Any]]  # Evaluation criteria
    current_phase: str         # Current project phase
    participants: List[str]    # Project participants
    start_date: datetime       # Project start date
    end_date: Optional[datetime] = None  # Project end date
    status: str = "not_started"  # Project status: not_started, in_progress, completed, failed
    artifacts: Optional[Dict[str, Any]] = None  # Project artifacts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            "project_id": self.project_id,
            "title": self.title,
            "description": self.description,
            "objectives": self.objectives,
            "requirements": self.requirements,
            "evaluation_criteria": self.evaluation_criteria,
            "current_phase": self.current_phase,
            "participants": self.participants,
            "start_date": self.start_date.isoformat(),
            "status": self.status,
        }
        if self.end_date:
            result["end_date"] = self.end_date.isoformat()
        if self.artifacts:
            result["artifacts"] = self.artifacts
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CapstoneProject':
        """Create instance from dictionary"""
        return cls(
            project_id=data["project_id"],
            title=data["title"],
            description=data["description"],
            objectives=data["objectives"],
            requirements=data["requirements"],
            evaluation_criteria=data["evaluation_criteria"],
            current_phase=data["current_phase"],
            participants=data["participants"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            status=data.get("status", "not_started"),
            artifacts=data.get("artifacts")
        )


# Utility functions for serialization
def serialize_model(obj) -> str:
    """Serialize a data model to JSON string"""
    if hasattr(obj, 'to_dict'):
        return json.dumps(obj.to_dict(), default=str)
    else:
        raise TypeError(f"Object {type(obj)} is not serializable")


def deserialize_model(json_str: str, model_class) -> Any:
    """Deserialize a JSON string to a data model"""
    if hasattr(model_class, 'from_dict'):
        data = json.loads(json_str)
        return model_class.from_dict(data)
    else:
        raise TypeError(f"Model class {model_class} does not support deserialization")
