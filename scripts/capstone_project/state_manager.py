"""
State Management System for Capstone Autonomous Humanoid
Manages complex system state and transitions for the complete autonomous system
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable, Union
import threading
import time
from datetime import datetime
import json
import copy

from common_data_models import AutonomousHumanoidSystem
from observability_utils import EducationalLogger, MetricsCollector, DebuggingHelper
from error_handling_utils import EducationalErrorHandler, InputValidator
from privacy_compliance import PrivacyComplianceMiddleware


class SystemState(Enum):
    """Top-level system states for the autonomous humanoid"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    SAFETY_STOP = "safety_stop"
    SHUTDOWN = "shutdown"


class TaskState(Enum):
    """States for individual tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"


class SafetyState(Enum):
    """Safety-related system states"""
    NORMAL = "normal"
    WARNING = "warning"
    EMERGENCY = "emergency"
    SAFETY_LOCKOUT = "safety_lockout"


@dataclass
class SystemContext:
    """Current context for the autonomous humanoid system"""
    current_state: SystemState
    previous_state: Optional[SystemState]
    active_tasks: Dict[str, TaskState]
    environment_state: Dict[str, Any]
    last_command: Optional[str]
    command_timestamp: Optional[datetime]
    error_count: int
    safety_status: Dict[str, Union[bool, str, float]]
    performance_metrics: Dict[str, float]
    robot_pose: Optional[Dict[str, float]]  # x, y, theta for 2D pose
    battery_level: float
    system_uptime: float
    active_sensors: List[str]
    active_actuators: List[str]


class StateTransition:
    """Represents a state transition with validation and callbacks"""
    def __init__(self, 
                 from_state: SystemState, 
                 to_state: SystemState, 
                 condition: Optional[Callable] = None,
                 callback: Optional[Callable] = None,
                 description: str = ""):
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition or (lambda context: True)  # Default: always allowed
        self.callback = callback
        self.description = description


class StateManager:
    """
    Advanced state management system for the autonomous humanoid
    Handles state transitions, persistence, and complex state logic
    """
    
    def __init__(self):
        self.logger = EducationalLogger("state_manager")
        self.metrics = MetricsCollector(self.logger)
        self.debug_helper = DebuggingHelper(self.logger)
        self.error_handler = EducationalErrorHandler()
        self.validator = InputValidator(self.error_handler)
        
        # Initialize privacy compliance
        self.privacy_middleware = PrivacyComplianceMiddleware()
        
        # Initialize state
        self.context = SystemContext(
            current_state=SystemState.IDLE,
            previous_state=None,
            active_tasks={},
            environment_state={},
            last_command=None,
            command_timestamp=None,
            error_count=0,
            safety_status={
                "emergency_stop": False,
                "collision_risk": False,
                "safety_state": SafetyState.NORMAL.value,
                "safety_violation_count": 0
            },
            performance_metrics={},
            robot_pose=None,
            battery_level=1.0,  # Full battery
            system_uptime=0.0,
            active_sensors=[],
            active_actuators=[]
        )
        
        # State management
        self.state_history = []
        self.transition_callbacks = {}
        self.state_lock = threading.Lock()
        self.start_time = datetime.now()
        
        # Define allowed state transitions
        self.allowed_transitions = self._define_allowed_transitions()
        
        # State persistence
        self.state_persistence_enabled = True
        self.state_persistence_file = "state_backup.json"
        
        # Safety constraints
        self.safety_constraints = {
            SystemState.SAFETY_STOP: [SystemState.IDLE, SystemState.SHUTDOWN],
            SystemState.ERROR: [SystemState.IDLE, SystemState.SAFETY_STOP, SystemState.SHUTDOWN]
        }
        
        self.logger.info("StateManager", "__init__", "State manager initialized", {
            'initial_state': self.context.current_state.value
        })
    
    def _define_allowed_transitions(self) -> List[StateTransition]:
        """Define allowed state transitions for the system"""
        return [
            # Normal operation transitions
            StateTransition(SystemState.IDLE, SystemState.LISTENING, description="Ready for voice command"),
            StateTransition(SystemState.LISTENING, SystemState.PROCESSING, description="Processing voice command"),
            StateTransition(SystemState.PROCESSING, SystemState.PLANNING, description="Planning action sequence"),
            StateTransition(SystemState.PLANNING, SystemState.EXECUTING, description="Executing action sequence"),
            StateTransition(SystemState.EXECUTING, SystemState.IDLE, description="Execution completed"),
            StateTransition(SystemState.EXECUTING, SystemState.WAITING, description="Waiting for condition"),
            StateTransition(SystemState.WAITING, SystemState.EXECUTING, description="Condition met, resuming execution"),
            
            # Error and safety transitions
            StateTransition(SystemState.LISTENING, SystemState.ERROR, description="Processing error during listening"),
            StateTransition(SystemState.PROCESSING, SystemState.ERROR, description="Error during processing"),
            StateTransition(SystemState.PLANNING, SystemState.ERROR, description="Error during planning"),
            StateTransition(SystemState.EXECUTING, SystemState.ERROR, description="Error during execution"),
            StateTransition(SystemState.WAITING, SystemState.ERROR, description="Error while waiting"),
            
            # Safety transitions
            StateTransition(SystemState.LISTENING, SystemState.SAFETY_STOP, description="Safety trigger during listening"),
            StateTransition(SystemState.PROCESSING, SystemState.SAFETY_STOP, description="Safety trigger during processing"),
            StateTransition(SystemState.PLANNING, SystemState.SAFETY_STOP, description="Safety trigger during planning"),
            StateTransition(SystemState.EXECUTING, SystemState.SAFETY_STOP, description="Safety trigger during execution"),
            StateTransition(SystemState.WAITING, SystemState.SAFETY_STOP, description="Safety trigger while waiting"),
            
            # Recovery transitions
            StateTransition(SystemState.ERROR, SystemState.IDLE, description="Recovery from error to idle"),
            StateTransition(SystemState.SAFETY_STOP, SystemState.IDLE, description="Recovery from safety stop"),
            StateTransition(SystemState.SAFETY_STOP, SystemState.SHUTDOWN, description="Safety shutdown"),
            
            # Shutdown transitions
            StateTransition(SystemState.IDLE, SystemState.SHUTDOWN, description="Normal shutdown"),
            StateTransition(SystemState.LISTENING, SystemState.SHUTDOWN, description="Shutdown during listening"),
            StateTransition(SystemState.PROCESSING, SystemState.SHUTDOWN, description="Shutdown during processing"),
            StateTransition(SystemState.PLANNING, SystemState.SHUTDOWN, description="Shutdown during planning"),
            StateTransition(SystemState.EXECUTING, SystemState.SHUTDOWN, description="Shutdown during execution"),
        ]
    
    def transition_to(self, new_state: SystemState, reason: str = "", 
                     validate_safety: bool = True) -> bool:
        """Transition to a new system state with safety validation"""
        with self.state_lock:
            old_state = self.context.current_state
            
            # Validate state transition
            if not self._is_valid_transition(old_state, new_state):
                self.logger.warning("StateManager", "transition_to", 
                                  f"Invalid state transition: {old_state.value} -> {new_state.value}", {
                                      'reason': reason
                                  })
                return False
            
            # Check safety constraints if enabled
            if validate_safety and not self._check_safety_constraints(old_state, new_state):
                self.logger.error("StateManager", "transition_to", 
                                f"Safety constraint violation: {old_state.value} -> {new_state.value}", {
                                    'reason': reason
                                })
                return False
            
            # Log state transition
            transition_info = {
                'from': old_state.value,
                'to': new_state.value,
                'reason': reason,
                'timestamp': datetime.now(),
                'context_snapshot': self._get_context_snapshot()
            }
            self.state_history.append(transition_info)
            
            # Update context
            self.context.previous_state = old_state
            self.context.current_state = new_state
            
            # Update system uptime
            self.context.system_uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Execute transition callback if registered
            if new_state in self.transition_callbacks:
                try:
                    self.transition_callbacks[new_state](self.context)
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context=f"transition_callback: {new_state.value}",
                        suggested_fix="Check transition callback implementation",
                        learning_objective="Handle transition callback errors"
                    )
            
            # Log the transition
            self.logger.info("StateManager", "transition_to", 
                           f"State transition: {old_state.value} -> {new_state.value}", {
                               'reason': reason,
                               'transition_count': len(self.state_history)
                           })
            
            # Update metrics
            self.metrics.record_metric('state_transitions', 1)
            self.metrics.record_metric(f'state_{new_state.value}_enter', 1)
            
            return True
    
    def _is_valid_transition(self, from_state: SystemState, to_state: SystemState) -> bool:
        """Check if a state transition is valid"""
        # Check against allowed transitions
        for transition in self.allowed_transitions:
            if transition.from_state == from_state and transition.to_state == to_state:
                # Check transition condition
                try:
                    return transition.condition(self.context)
                except Exception as e:
                    self.error_handler.handle_error(
                        e,
                        context=f"transition_condition: {from_state.value} -> {to_state.value}",
                        suggested_fix="Check transition condition logic",
                        learning_objective="Handle transition condition errors"
                    )
                    return False
        
        return False  # Transition not found in allowed list
    
    def _check_safety_constraints(self, from_state: SystemState, to_state: SystemState) -> bool:
        """Check safety constraints for state transition"""
        # Check if there are specific constraints for the target state
        if to_state in self.safety_constraints:
            allowed_from_states = self.safety_constraints[to_state]
            if from_state not in allowed_from_states:
                # Check if safety override is enabled (for emergency situations)
                if self.context.safety_status.get("emergency_override", False):
                    self.logger.warning("StateManager", "_check_safety_constraints", 
                                      f"Safety constraint overridden: {from_state.value} -> {to_state.value}")
                    return True
                else:
                    return False
        
        # Additional safety checks
        if to_state == SystemState.SAFETY_STOP and not self._is_safety_condition_met():
            return False  # Only transition to safety stop if safety condition is met
        
        return True
    
    def _is_safety_condition_met(self) -> bool:
        """Check if safety conditions are met for safety stop"""
        return (self.context.safety_status.get("emergency_stop", False) or
                self.context.safety_status.get("collision_risk", False) or
                self.context.safety_status.get("safety_state") == SafetyState.EMERGENCY.value)
    
    def _get_context_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current context for logging"""
        return {
            'active_tasks_count': len(self.context.active_tasks),
            'error_count': self.context.error_count,
            'safety_status': self.context.safety_status.copy(),
            'environment_objects': len(self.context.environment_state.get('objects', [])),
            'battery_level': self.context.battery_level,
            'system_uptime': self.context.system_uptime
        }
    
    def register_transition_callback(self, state: SystemState, callback: Callable):
        """Register a callback for when entering a specific state"""
        self.transition_callbacks[state] = callback
    
    def update_environment_state(self, new_state: Dict[str, Any]):
        """Update the environment state"""
        with self.state_lock:
            self.context.environment_state.update(new_state)
            
            # Update metrics
            self.metrics.record_metric('environment_updates', 1)
    
    def add_active_task(self, task_id: str, initial_state: TaskState = TaskState.PENDING):
        """Add a new active task"""
        with self.state_lock:
            self.context.active_tasks[task_id] = initial_state
            
            # Update metrics
            self.metrics.record_metric('active_tasks', len(self.context.active_tasks))
    
    def update_task_state(self, task_id: str, new_state: TaskState):
        """Update the state of an active task"""
        with self.state_lock:
            if task_id in self.context.active_tasks:
                old_state = self.context.active_tasks[task_id]
                self.context.active_tasks[task_id] = new_state
                
                # Log task state change
                self.logger.debug("StateManager", "update_task_state", 
                                f"Task {task_id}: {old_state.value} -> {new_state.value}")
                
                # Update metrics
                self.metrics.record_metric(f'task_{new_state.value}', 1)
    
    def remove_task(self, task_id: str):
        """Remove a task from active tasks"""
        with self.state_lock:
            if task_id in self.context.active_tasks:
                del self.context.active_tasks[task_id]
                
                # Update metrics
                self.metrics.record_metric('active_tasks', len(self.context.active_tasks))
    
    def update_safety_status(self, **kwargs):
        """Update safety status parameters"""
        with self.state_lock:
            for key, value in kwargs.items():
                self.context.safety_status[key] = value
            
            # Check if safety state needs to be updated
            if self._should_update_safety_state():
                self._update_safety_state()
    
    def _should_update_safety_state(self) -> bool:
        """Check if safety state should be updated based on current status"""
        current_safety = self.context.safety_status.get("safety_state", SafetyState.NORMAL.value)
        
        if self.context.safety_status.get("emergency_stop", False):
            return current_safety != SafetyState.EMERGENCY.value
        elif self.context.safety_status.get("collision_risk", False):
            return current_safety != SafetyState.WARNING.value
        else:
            return current_safety != SafetyState.NORMAL.value
    
    def _update_safety_state(self):
        """Update the safety state based on current status"""
        if self.context.safety_status.get("emergency_stop", False):
            self.context.safety_status["safety_state"] = SafetyState.EMERGENCY.value
        elif self.context.safety_status.get("collision_risk", False):
            self.context.safety_status["safety_state"] = SafetyState.WARNING.value
        else:
            self.context.safety_status["safety_state"] = SafetyState.NORMAL.value
    
    def update_robot_pose(self, x: float, y: float, theta: float):
        """Update robot pose in the context"""
        with self.state_lock:
            self.context.robot_pose = {
                'x': x,
                'y': y,
                'theta': theta,
                'timestamp': datetime.now().isoformat()
            }
    
    def update_battery_level(self, level: float):
        """Update battery level"""
        with self.state_lock:
            self.context.battery_level = max(0.0, min(1.0, level))  # Clamp between 0 and 1
    
    def increment_error_count(self):
        """Increment the error counter"""
        with self.state_lock:
            self.context.error_count += 1
            
            # Update metrics
            self.metrics.record_metric('errors', self.context.error_count)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self.state_lock:
            return {
                'current_state': self.context.current_state.value,
                'previous_state': self.context.previous_state.value if self.context.previous_state else None,
                'active_tasks': {k: v.value for k, v in self.context.active_tasks.items()},
                'error_count': self.context.error_count,
                'safety_status': self.context.safety_status,
                'state_history_length': len(self.state_history),
                'last_command': self.context.last_command,
                'system_uptime': self.context.system_uptime,
                'battery_level': self.context.battery_level,
                'robot_pose': self.context.robot_pose,
                'metrics': self.metrics.get_statistics()
            }
    
    def get_state_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get state transition history"""
        with self.state_lock:
            if limit:
                return self.state_history[-limit:]
            return self.state_history.copy()
    
    def reset_state_history(self):
        """Reset the state transition history"""
        with self.state_lock:
            self.state_history.clear()
    
    def save_state(self, filename: Optional[str] = None):
        """Save current state to persistent storage"""
        if not self.state_persistence_enabled:
            return
        
        save_file = filename or self.state_persistence_file
        try:
            # Create a serializable version of the context
            state_data = {
                'context': {
                    'current_state': self.context.current_state.value,
                    'previous_state': self.context.previous_state.value if self.context.previous_state else None,
                    'active_tasks': {k: v.value for k, v in self.context.active_tasks.items()},
                    'environment_state': self.context.environment_state,
                    'last_command': self.context.last_command,
                    'command_timestamp': self.context.command_timestamp.isoformat() if self.context.command_timestamp else None,
                    'error_count': self.context.error_count,
                    'safety_status': self.context.safety_status,
                    'performance_metrics': self.context.performance_metrics,
                    'robot_pose': self.context.robot_pose,
                    'battery_level': self.context.battery_level,
                    'system_uptime': self.context.system_uptime,
                    'active_sensors': self.context.active_sensors,
                    'active_actuators': self.context.active_actuators
                },
                'state_history': [
                    {
                        'from': t['from'],
                        'to': t['to'],
                        'reason': t['reason'],
                        'timestamp': t['timestamp'].isoformat(),
                        'context_snapshot': t['context_snapshot']
                    } for t in self.state_history
                ],
                'saved_at': datetime.now().isoformat()
            }
            
            with open(save_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.info("StateManager", "save_state", f"State saved to {save_file}")
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context=f"save_state: {save_file}",
                suggested_fix="Check file permissions and disk space",
                learning_objective="Handle state persistence errors"
            )
    
    def load_state(self, filename: Optional[str] = None):
        """Load state from persistent storage"""
        load_file = filename or self.state_persistence_file
        try:
            with open(load_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore context
            context_data = state_data['context']
            self.context.current_state = SystemState(context_data['current_state'])
            self.context.previous_state = SystemState(context_data['previous_state']) if context_data['previous_state'] else None
            self.context.active_tasks = {k: TaskState(v) for k, v in context_data['active_tasks'].items()}
            self.context.environment_state = context_data['environment_state']
            self.context.last_command = context_data['last_command']
            self.context.command_timestamp = datetime.fromisoformat(context_data['command_timestamp']) if context_data['command_timestamp'] else None
            self.context.error_count = context_data['error_count']
            self.context.safety_status = context_data['safety_status']
            self.context.performance_metrics = context_data['performance_metrics']
            self.context.robot_pose = context_data['robot_pose']
            self.context.battery_level = context_data['battery_level']
            self.context.system_uptime = context_data['system_uptime']
            self.context.active_sensors = context_data['active_sensors']
            self.context.active_actuators = context_data['active_actuators']
            
            # Restore state history
            self.state_history = []
            for t in state_data['state_history']:
                self.state_history.append({
                    'from': t['from'],
                    'to': t['to'],
                    'reason': t['reason'],
                    'timestamp': datetime.fromisoformat(t['timestamp']),
                    'context_snapshot': t['context_snapshot']
                })
            
            self.logger.info("StateManager", "load_state", f"State loaded from {load_file}")
            
        except FileNotFoundError:
            self.logger.info("StateManager", "load_state", f"State file not found: {load_file} (this is normal on first run)")
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context=f"load_state: {load_file}",
                suggested_fix="Check state file format and permissions",
                learning_objective="Handle state loading errors"
            )
    
    def emergency_stop(self, reason: str = "Emergency stop requested"):
        """Execute emergency stop procedure"""
        with self.state_lock:
            self.logger.warning("StateManager", "emergency_stop", f"Emergency stop initiated: {reason}")
            
            # Update safety status
            self.context.safety_status["emergency_stop"] = True
            self.context.safety_status["safety_state"] = SafetyState.EMERGENCY.value
            
            # Stop all active tasks
            for task_id in list(self.context.active_tasks.keys()):
                self.update_task_state(task_id, TaskState.CANCELLED)
            
            # Transition to safety stop state
            self.transition_to(SystemState.SAFETY_STOP, reason, validate_safety=False)
    
    def clear_emergency_stop(self):
        """Clear emergency stop condition"""
        with self.state_lock:
            self.logger.info("StateManager", "clear_emergency_stop", "Clearing emergency stop condition")
            
            # Update safety status
            self.context.safety_status["emergency_stop"] = False
            self.context.safety_status["safety_state"] = SafetyState.NORMAL.value
            
            # Transition back to idle
            self.transition_to(SystemState.IDLE, "Emergency stop cleared")


class HierarchicalStateManager:
    """
    Hierarchical state management for complex autonomous systems
    Manages states at multiple levels: system, subsystem, and task
    """
    
    def __init__(self):
        self.logger = EducationalLogger("hierarchical_state_manager")
        self.error_handler = EducationalErrorHandler()
        
        # Initialize the base state manager
        self.base_manager = StateManager()
        
        # Hierarchical state managers for subsystems
        self.subsystem_managers = {
            'voice': StateManager(),
            'planning': StateManager(), 
            'execution': StateManager(),
            'navigation': StateManager(),
            'manipulation': StateManager(),
            'perception': StateManager()
        }
        
        # Task-level state managers
        self.task_managers = {}
        self.task_manager_lock = threading.Lock()
        
        self.logger.info("HierarchicalStateManager", "__init__", "Hierarchical state manager initialized")
    
    def get_system_state(self) -> SystemState:
        """Get the overall system state"""
        return self.base_manager.context.current_state
    
    def get_subsystem_state(self, subsystem: str) -> Optional[SystemState]:
        """Get state of a specific subsystem"""
        if subsystem in self.subsystem_managers:
            return self.subsystem_managers[subsystem].context.current_state
        return None
    
    def get_task_state(self, task_id: str) -> Optional[TaskState]:
        """Get state of a specific task"""
        with self.task_manager_lock:
            if task_id in self.task_managers:
                return self.task_managers[task_id].context.current_state
        return None
    
    def create_task_manager(self, task_id: str) -> StateManager:
        """Create a state manager for a specific task"""
        with self.task_manager_lock:
            if task_id not in self.task_managers:
                self.task_managers[task_id] = StateManager()
            return self.task_managers[task_id]
    
    def update_system_state(self, new_state: SystemState, reason: str = ""):
        """Update the overall system state"""
        return self.base_manager.transition_to(new_state, reason)
    
    def update_subsystem_state(self, subsystem: str, new_state: SystemState, reason: str = ""):
        """Update the state of a specific subsystem"""
        if subsystem in self.subsystem_managers:
            return self.subsystem_managers[subsystem].transition_to(new_state, reason)
        return False
    
    def synchronize_states(self):
        """Synchronize states across hierarchy based on defined rules"""
        system_state = self.base_manager.context.current_state
        
        # Propagate system state changes to subsystems when appropriate
        if system_state in [SystemState.SAFETY_STOP, SystemState.ERROR, SystemState.SHUTDOWN]:
            for subsystem_name, manager in self.subsystem_managers.items():
                if manager.context.current_state not in [SystemState.SAFETY_STOP, SystemState.ERROR, SystemState.SHUTDOWN]:
                    manager.transition_to(SystemState.ERROR, f"Propagated from system state: {system_state.value}")
        
        # Update task managers based on system state
        with self.task_manager_lock:
            for task_manager in self.task_managers.values():
                if system_state in [SystemState.SAFETY_STOP, SystemState.ERROR, SystemState.SHUTDOWN]:
                    if task_manager.context.current_state not in [SystemState.SAFETY_STOP, SystemState.ERROR, SystemState.SHUTDOWN]:
                        task_manager.transition_to(SystemState.ERROR, f"Propagated from system state: {system_state.value}")


def main():
    """Main function to demonstrate the state management system"""
    print("Capstone State Management System - Autonomous Humanoid")
    print("="*58)
    
    state_manager = StateManager()
    
    print("State manager initialized.")
    print(f"Initial state: {state_manager.context.current_state.value}")
    
    # Show initial status
    status = state_manager.get_system_status()
    print(f"\nInitial system status: {status['current_state']}")
    
    try:
        # Demonstrate state transitions
        print("\nDemonstrating state transitions...")
        
        transitions = [
            (SystemState.LISTENING, "Ready for voice command"),
            (SystemState.PROCESSING, "Processing voice command"),
            (SystemState.PLANNING, "Planning action sequence"),
            (SystemState.EXECUTING, "Executing action sequence"),
            (SystemState.IDLE, "Execution completed"),
        ]
        
        for new_state, reason in transitions:
            success = state_manager.transition_to(new_state, reason)
            print(f"  Transition to {new_state.value}: {'✅' if success else '❌'}")
        
        # Add and update tasks
        print(f"\nManaging tasks...")
        state_manager.add_active_task("voice_processing_001", TaskState.IN_PROGRESS)
        state_manager.add_active_task("navigation_001", TaskState.PENDING)
        
        print(f"  Active tasks: {len(state_manager.context.active_tasks)}")
        
        state_manager.update_task_state("voice_processing_001", TaskState.COMPLETED)
        state_manager.update_task_state("navigation_001", TaskState.IN_PROGRESS)
        
        # Update environment and safety status
        print(f"\nUpdating environment and safety status...")
        state_manager.update_environment_state({
            "objects": ["person", "table", "cup"],
            "room": "kitchen",
            "lighting": "bright"
        })
        
        state_manager.update_safety_status(collision_risk=False, emergency_stop=False)
        
        # Show updated status
        updated_status = state_manager.get_system_status()
        print(f"\nUpdated system status: {updated_status['current_state']}")
        print(f"  Active tasks: {len(updated_status['active_tasks'])}")
        print(f"  Environment objects: {len(updated_status['environment_state'].get('objects', []))}")
        print(f"  Safety status: {updated_status['safety_status']}")
        
        # Show state history
        history = state_manager.get_state_history(limit=5)
        print(f"\nRecent state transitions:")
        for i, transition in enumerate(history[-5:], 1):
            print(f"  {i}. {transition['from']} -> {transition['to']} ({transition['reason']})")
        
        # Demonstrate emergency stop
        print(f"\nTesting emergency stop...")
        state_manager.emergency_stop("Test emergency stop")
        
        emergency_status = state_manager.get_system_status()
        print(f"  State after emergency stop: {emergency_status['current_state']}")
        print(f"  Safety status: {emergency_status['safety_status']}")
        
        # Clear emergency stop
        state_manager.clear_emergency_stop()
        cleared_status = state_manager.get_system_status()
        print(f"  State after clearing emergency: {cleared_status['current_state']}")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
    
    # Show final status
    final_status = state_manager.get_system_status()
    print(f"\nFinal system status:")
    print(f"  State: {final_status['current_state']}")
    print(f"  Total transitions: {len(state_manager.get_state_history())}")
    print(f"  Error count: {final_status['error_count']}")
    
    print(f"\nState management system demonstration completed.")


if __name__ == "__main__":
    main()
