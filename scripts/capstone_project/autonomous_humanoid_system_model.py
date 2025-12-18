"""
Autonomous Humanoid System Data Model for VLA System

This module defines the AutonomousHumanoidSystem data model based on the entity specification.
It represents the complete integrated system combining all textbook modules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class SystemStatus(Enum):
    """Enumeration of system statuses"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    EXECUTING = "executing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class MotorStatus(Enum):
    """Enumeration of motor statuses"""
    IDLE = "idle"
    MOVING = "moving"
    ERROR = "error"
    CALIBRATING = "calibrating"


class GripperStatus(Enum):
    """Enumeration of gripper statuses"""
    OPEN = "open"
    CLOSED = "closed"
    MOVING = "moving"
    ERROR = "error"


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
class AudioStatus:
    """Represents audio input status"""
    is_active: bool = True
    sensitivity: float = 0.5
    noise_level: float = 0.1


@dataclass
class VisionStatus:
    """Represents vision system status"""
    is_active: bool = True
    resolution: str = "1080p"
    frame_rate: int = 30


@dataclass
class IMUData:
    """Represents Inertial Measurement Unit data"""
    acceleration: Location
    angular_velocity: Orientation
    magnetic_field: Location


@dataclass
class EncoderData:
    """Represents joint encoder readings"""
    joint_positions: Dict[str, float] = field(default_factory=dict)
    joint_velocities: Dict[str, float] = field(default_factory=dict)
    joint_efforts: Dict[str, float] = field(default_factory=dict)


@dataclass
class Motor:
    """Represents a motor in the system"""
    id: str
    status: MotorStatus
    position: float = 0.0
    velocity: float = 0.0
    effort: float = 0.0
    temperature: float = 25.0


@dataclass
class Gripper:
    """Represents a gripper in the system"""
    id: str
    status: GripperStatus
    position: float = 0.0
    force: float = 0.0
    max_force: float = 100.0


@dataclass
class PerformanceMetrics:
    """Represents system performance metrics"""
    response_time: float = 0.0
    success_rate: float = 0.0
    uptime: float = 100.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ROSInterfaces:
    """Represents ROS interface connections to previous modules"""
    ros2_nodes: List[str] = field(default_factory=list)
    digital_twin_connection: Optional[str] = None
    isaac_sim_connection: Optional[str] = None


@dataclass
class ErrorLogEntry:
    """Represents an entry in the error log"""
    timestamp: datetime
    error_code: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class AutonomousHumanoidSystem:
    """
    Represents the complete integrated system combining all textbook modules.

    Attributes:
        id: Unique identifier for the system
        name: Name of the system
        status: Current system status
        current_behavior: Current behavior being executed
        sensors: Current sensor readings
        actuators: Current actuator status
        state: Current system state
        active_sequences: Currently active action sequences
        last_command: Last command processed
        command_history: History of recent commands
        performance_metrics: System performance metrics
        error_log: Recent error log entries
        ros_interfaces: ROS interface connections to previous modules
    """

    id: str
    name: str
    status: SystemStatus
    current_behavior: str
    sensors: Dict[str, Any]
    actuators: Dict[str, Any]
    state: Dict[str, Any]
    active_sequences: List[str]
    last_command: str
    command_history: List[str]
    performance_metrics: PerformanceMetrics
    error_log: List[ErrorLogEntry]
    ros_interfaces: ROSInterfaces

    def __post_init__(self):
        """Validate the AutonomousHumanoidSystem after initialization"""
        self._validate()

    def _validate(self):
        """Validate the system according to the data model rules"""
        # Validate battery level is between 0.0 and 100.0
        battery_level = self.state.get('battery_level', 100.0)
        if not 0.0 <= battery_level <= 100.0:
            raise ValueError("Battery level must be between 0.0 and 100.0")

        # Validate location coordinates are valid floats
        location = self.state.get('location', {})
        if 'x' in location and not isinstance(location['x'], (int, float)):
            raise ValueError("Location x coordinate must be a valid number")
        if 'y' in location and not isinstance(location['y'], (int, float)):
            raise ValueError("Location y coordinate must be a valid number")
        if 'z' in location and not isinstance(location['z'], (int, float)):
            raise ValueError("Location z coordinate must be a valid number")

    def update_status(self, new_status: SystemStatus):
        """Update the system status with valid state transitions"""
        # Define valid state transitions
        valid_transitions = {
            SystemStatus.IDLE: [SystemStatus.LISTENING, SystemStatus.ERROR],
            SystemStatus.LISTENING: [SystemStatus.PROCESSING, SystemStatus.IDLE, SystemStatus.ERROR],
            SystemStatus.PROCESSING: [SystemStatus.EXECUTING, SystemStatus.IDLE, SystemStatus.ERROR],
            SystemStatus.EXECUTING: [SystemStatus.IDLE, SystemStatus.ERROR],
            SystemStatus.ERROR: [SystemStatus.IDLE, SystemStatus.SHUTDOWN],
            SystemStatus.SHUTDOWN: []
        }

        current_transitions = valid_transitions.get(self.status, [])
        if new_status not in current_transitions:
            raise ValueError(f"Invalid state transition from {self.status} to {new_status}")

        self.status = new_status

    def add_error_log(self, error_code: str, description: str, severity: str):
        """Add an error log entry to the system"""
        entry = ErrorLogEntry(
            timestamp=datetime.now(),
            error_code=error_code,
            description=description,
            severity=severity
        )
        self.error_log.append(entry)

        # Keep only the last 100 error entries
        if len(self.error_log) > 100:
            self.error_log = self.error_log[-100:]

    def update_performance_metrics(self, response_time: Optional[float] = None,
                                   success_rate: Optional[float] = None,
                                   uptime: Optional[float] = None):
        """Update performance metrics"""
        if response_time is not None:
            self.performance_metrics.response_time = response_time
        if success_rate is not None:
            self.performance_metrics.success_rate = success_rate
        if uptime is not None:
            self.performance_metrics.uptime = uptime
        self.performance_metrics.last_updated = datetime.now()

    def add_command_to_history(self, command: str):
        """Add a command to the command history"""
        self.command_history.append(command)

        # Keep only the last 50 commands in history
        if len(self.command_history) > 50:
            self.command_history = self.command_history[-50:]

    def is_operational(self) -> bool:
        """Check if the system is operational (not in error or shutdown state)"""
        return self.status not in [SystemStatus.ERROR, SystemStatus.SHUTDOWN]

    def get_system_health(self) -> Dict[str, Any]:
        """Get a summary of system health"""
        battery_level = self.state.get('battery_level', 100.0)
        temperature = self.state.get('temperature', 25.0)

        health_status = "good"
        if battery_level < 20.0:
            health_status = "critical"
        elif battery_level < 30.0:
            health_status = "warning"

        if temperature > 70.0:
            health_status = "critical"
        elif temperature > 60.0 and health_status != "critical":
            health_status = "warning"

        return {
            "status": self.status.value,
            "battery_level": battery_level,
            "temperature": temperature,
            "health_status": health_status,
            "active_sequences_count": len(self.active_sequences),
            "last_error": self.error_log[-1].description if self.error_log else None,
            "performance": {
                "response_time": self.performance_metrics.response_time,
                "success_rate": self.performance_metrics.success_rate,
                "uptime": self.performance_metrics.uptime
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the AutonomousHumanoidSystem to a dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status.value,
            'current_behavior': self.current_behavior,
            'sensors': self.sensors,
            'actuators': self.actuators,
            'state': self.state,
            'active_sequences': self.active_sequences,
            'last_command': self.last_command,
            'command_history': self.command_history,
            'performance_metrics': {
                'response_time': self.performance_metrics.response_time,
                'success_rate': self.performance_metrics.success_rate,
                'uptime': self.performance_metrics.uptime,
                'last_updated': self.performance_metrics.last_updated.isoformat()
            },
            'error_log': [
                {
                    'timestamp': entry.timestamp.isoformat(),
                    'error_code': entry.error_code,
                    'description': entry.description,
                    'severity': entry.severity
                }
                for entry in self.error_log
            ],
            'ros_interfaces': {
                'ros2_nodes': self.ros_interfaces.ros2_nodes,
                'digital_twin_connection': self.ros_interfaces.digital_twin_connection,
                'isaac_sim_connection': self.ros_interfaces.isaac_sim_connection
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutonomousHumanoidSystem':
        """Create an AutonomousHumanoidSystem from a dictionary representation"""
        # Create performance metrics
        perf_data = data['performance_metrics']
        performance_metrics = PerformanceMetrics(
            response_time=perf_data['response_time'],
            success_rate=perf_data['success_rate'],
            uptime=perf_data['uptime'],
            last_updated=datetime.fromisoformat(perf_data['last_updated'])
        )

        # Create error log entries
        error_log = []
        for entry_data in data['error_log']:
            entry = ErrorLogEntry(
                timestamp=datetime.fromisoformat(entry_data['timestamp']),
                error_code=entry_data['error_code'],
                description=entry_data['description'],
                severity=entry_data['severity']
            )
            error_log.append(entry)

        # Create ROS interfaces
        ros_data = data['ros_interfaces']
        ros_interfaces = ROSInterfaces(
            ros2_nodes=ros_data['ros2_nodes'],
            digital_twin_connection=ros_data.get('digital_twin_connection'),
            isaac_sim_connection=ros_data.get('isaac_sim_connection')
        )

        # Create and return AutonomousHumanoidSystem
        return cls(
            id=data['id'],
            name=data['name'],
            status=SystemStatus(data['status']),
            current_behavior=data['current_behavior'],
            sensors=data['sensors'],
            actuators=data['actuators'],
            state=data['state'],
            active_sequences=data['active_sequences'],
            last_command=data['last_command'],
            command_history=data['command_history'],
            performance_metrics=performance_metrics,
            error_log=error_log,
            ros_interfaces=ros_interfaces
        )


# Example usage and validation
if __name__ == "__main__":
    # Create sample data for the autonomous humanoid system
    sample_sensors = {
        'audio': AudioStatus().__dict__,
        'vision': VisionStatus().__dict__,
        'imu': IMUData(acceleration=Location(0.0, 0.0, 9.81),
                       angular_velocity=Orientation(0.0, 0.0, 0.0),
                       magnetic_field=Location(25.0, 5.0, 40.0)).__dict__,
        'encoders': EncoderData().__dict__
    }

    sample_actuators = {
        'motors': [
            Motor(id="left_hip", status=MotorStatus.IDLE, position=0.0).__dict__,
            Motor(id="right_hip", status=MotorStatus.IDLE, position=0.0).__dict__
        ],
        'grippers': [
            Gripper(id="left_gripper", status=GripperStatus.OPEN).__dict__
        ]
    }

    sample_state = {
        'location': Location(x=0.0, y=0.0, z=0.0).__dict__,
        'orientation': Orientation(roll=0.0, pitch=0.0, yaw=0.0).__dict__,
        'battery_level': 85.0,
        'temperature': 30.0
    }

    # Create the autonomous humanoid system
    system = AutonomousHumanoidSystem(
        id="humanoid_001",
        name="VLA Humanoid Robot",
        status=SystemStatus.IDLE,
        current_behavior="idle",
        sensors=sample_sensors,
        actuators=sample_actuators,
        state=sample_state,
        active_sequences=[],
        last_command="",
        command_history=[],
        performance_metrics=PerformanceMetrics(),
        error_log=[],
        ros_interfaces=ROSInterfaces(
            ros2_nodes=["/navigation_node", "/manipulation_node"],
            digital_twin_connection="dt_connection_1",
            isaac_sim_connection="isaac_sim_1"
        )
    )

    print(f"Created autonomous humanoid system: {system.name}")
    print(f"System ID: {system.id}")
    print(f"Status: {system.status.value}")
    print(f"Health: {system.get_system_health()}")

    # Add a command to history
    system.add_command_to_history("Go to kitchen")
    print(f"Command history length: {len(system.command_history)}")

    # Update system status
    system.update_status(SystemStatus.LISTENING)
    print(f"Updated status: {system.status.value}")

    # Add an error log entry
    system.add_error_log("E001", "Low battery warning", "warning")
    print(f"Error log entries: {len(system.error_log)}")

    # Update performance metrics
    system.update_performance_metrics(response_time=0.5, success_rate=0.95)
    print(f"Updated response time: {system.performance_metrics.response_time}")

    # Convert to dict and back
    system_dict = system.to_dict()
    print(f"Dictionary keys: {list(system_dict.keys())}")

    reconstructed_system = AutonomousHumanoidSystem.from_dict(system_dict)
    print(f"Reconstructed system name: {reconstructed_system.name}")