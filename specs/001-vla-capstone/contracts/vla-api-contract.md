# API Contract: Vision-Language-Action (VLA) System

**Date**: 2025-12-10
**Feature**: 001-vla-capstone
**Version**: 1.0

## Overview

This document defines the API contracts for the Vision-Language-Action (VLA) system, focusing on the ROS 2 interfaces that enable voice-controlled humanoid robot behaviors. The contracts ensure consistent interfaces between voice processing, LLM task planning, and action execution components, with explicit support for ROS 2 action servers as required by FR-005.

## Action Server Contracts

### 1. Task Planning Action Server

**Action Name**: `/plan_task`

**Action Type**: `task_planning_msgs/action/PlanTask`

#### Goal Definition
```yaml
command_text: string  # Natural language command from user
context: string       # JSON string with environmental context
```

#### Result Definition
```yaml
success: bool                           # Whether planning was successful
action_sequence: string                # JSON string representing action sequence
error_message: string                  # Error message if success is false
```

#### Feedback Definition
```yaml
status: string     # Current planning status (e.g., "parsing", "generating", "validating")
progress: float    # Progress percentage (0.0 to 1.0)
```

#### Usage Example
```
Goal: {"command_text": "Go to the kitchen and pick up the red cup", "context": "{\"location\": \"living room\", \"objects\": [\"red cup\"]}"}
Result: {"success": true, "action_sequence": "[{\"action\": \"navigate\", \"params\": {\"target\": \"kitchen\"}}, {\"action\": \"detect_object\", \"params\": {\"object_type\": \"red cup\"}}]", "error_message": ""}
```

### 2. Navigation Action Server

**Action Name**: `/navigate_to_pose`

**Action Type**: `nav2_msgs/action/NavigateToPose` (Standard ROS 2 navigation action)

#### Goal Definition
```yaml
pose: geometry_msgs/PoseStamped  # Target pose for navigation
behavior_tree: string           # Behavior tree to use for navigation (optional)
```

#### Result Definition
```yaml
result_code: int8               # Result code (SUCCESS=1, FAILURE=0, CANCELED=-1)
final_pose: geometry_msgs/PoseStamped  # Final pose achieved
distance_remaining: float       # Distance remaining to goal
error_code: int8               # Error code if navigation failed
error_string: string           # Human-readable error description
```

#### Feedback Definition
```yaml
distance_remaining: float       # Distance remaining to goal
navigation_time: builtin_interfaces/Duration  # Time since navigation started
current_pose: geometry_msgs/PoseStamped       # Current robot pose
speed: float                   # Current speed
```

#### Usage Example
```
Goal: {"pose": {"header": {"frame_id": "map"}, "pose": {"position": {"x": 1.0, "y": 2.0, "z": 0.0}, "orientation": {"w": 1.0}}}}
Result: {"result_code": 1, "final_pose": {"header": {"frame_id": "map"}, "pose": {"position": {"x": 0.98, "y": 2.01, "z": 0.0}, "orientation": {"w": 0.99}}}, "distance_remaining": 0.05, "error_code": 0, "error_string": ""}
```

### 3. Manipulation Action Server

**Action Name**: `/execute_manipulation`

**Action Type**: `manipulation_msgs/action/ExecuteManipulation`

#### Goal Definition
```yaml
manipulation_type: string       # Type of manipulation (grasp, place, move, etc.)
object_id: string              # ID of the object to manipulate
target_pose: geometry_msgs/PoseStamped  # Target pose for manipulation
gripper_parameters: object     # Parameters for gripper control
  - force: float               # Gripper force to apply
  - width: float               # Gripper width
  - approach_angle: float      # Approach angle in degrees
```

#### Result Definition
```yaml
success: bool                  # Whether manipulation was successful
result_code: int8             # Result code (SUCCESS=1, FAILURE=0, CANCELED=-1)
error_message: string         # Error message if manipulation failed
object_state: string          # Final state of the object (grasped, placed, moved, etc.)
```

#### Feedback Definition
```yaml
current_stage: string         # Current stage of manipulation (approaching, grasping, lifting, etc.)
progress: float              # Progress percentage (0.0 to 1.0)
gripper_state: string        # Current gripper state (open, closed, force_applied, etc.)
error_code: int8             # Error code if any issues occurred
```

#### Usage Example
```
Goal: {"manipulation_type": "grasp", "object_id": "red_cup", "target_pose": {"header": {"frame_id": "base_link"}, "pose": {"position": {"x": 0.5, "y": 0.0, "z": 0.1}, "orientation": {"w": 1.0}}}, "gripper_parameters": {"force": 10.0, "width": 0.05, "approach_angle": 45.0}}
Result: {"success": true, "result_code": 1, "error_message": "", "object_state": "grasped"}
```

### 4. Action Execution Action Server

**Action Name**: `/execute_action_sequence`

**Action Type**: `task_planning_msgs/action/ExecuteActionSequence`

#### Goal Definition
```yaml
action_sequence: string  # JSON string representing the sequence of actions to execute
timeout: float           # Maximum time to wait for execution in seconds
```

#### Result Definition
```yaml
success: bool                    # Whether execution was successful
execution_log: string           # JSON string with detailed execution log
error_message: string           # Error message if success is false
```

#### Feedback Definition
```yaml
current_action: string  # Description of the currently executing action
progress: float         # Progress percentage (0.0 to 1.0)
status: string          # Current execution status (e.g., "executing", "waiting", "error")
```

#### Usage Example
```
Goal: {"action_sequence": "[{\"action\": \"navigate\", \"params\": {\"target\": \"kitchen\"}, \"ros_action_server\": \"/navigate_to_pose\"}]", "timeout": 300.0}
Result: {"success": true, "execution_log": "[{\"action\": \"navigate\", \"status\": \"completed\", \"timestamp\": \"...\"}]", "error_message": ""}
```

## Service Contracts

### 1. Voice Processing Service

**Service Name**: `/process_voice_command`

**Service Type**: `voice_processing_msgs/srv/ProcessVoiceCommand`

#### Request Definition
```yaml
audio_data: string    # Base64 encoded audio data
language: string      # Language code (e.g., "en", "es")
```

#### Response Definition
```yaml
success: bool              # Whether processing was successful
transcript: string         # Transcribed text from audio
confidence: float          # Confidence score (0.0 to 1.0)
error_message: string      # Error message if success is false
```

#### Usage Example
```
Request: {"audio_data": "base64_encoded_audio_data_here", "language": "en"}
Response: {"success": true, "transcript": "Please navigate to the kitchen", "confidence": 0.92, "error_message": ""}
```

### 2. LLM Query Service with Prompt Templates

**Service Name**: `/query_llm_with_template`

**Service Type**: `llm_msgs/srv/QueryLLMWithTemplate`

#### Request Definition
```yaml
template_name: string        # Name of the prompt template to use
template_parameters: object  # Parameters to substitute in the template
context: string              # JSON string with contextual information
model_params: string         # JSON string with model parameters (temperature, max_tokens, etc.)
```

#### Response Definition
```yaml
success: bool              # Whether query was successful
response: string           # LLM response
confidence: float          # Confidence score (0.0 to 1.0) - estimated by analyzing response coherence
error_message: string      # Error message if success is false
used_template: string      # Name of the template that was used
```

#### Usage Example
```
Request: {"template_name": "navigation_task_planning", "template_parameters": {"target_location": "kitchen", "start_location": "living room"}, "context": "{\"robot_capabilities\": [\"navigation\", \"manipulation\"]}", "model_params": "{\"temperature\": 0.3, \"max_tokens\": 512}"}
Response: {"success": true, "response": "{\"action\": \"navigate\", \"params\": {\"target\": \"kitchen\"}}", "confidence": 0.87, "error_message": "", "used_template": "navigation_task_planning"}
```

## Topic Contracts

### 1. Voice Command Topic

**Topic Name**: `/voice_commands`

**Message Type**: `std_msgs/msg/String`

**Description**: Topic for publishing voice commands after processing

**Message Format**:
- `data`: string containing the transcribed voice command

**Usage Example**:
```
Data: "Please navigate to the kitchen and pick up the red cup"
```

### 2. Action Sequence Topic

**Topic Name**: `/action_sequences`

**Message Type**: `std_msgs/msg/String`

**Description**: Topic for publishing generated action sequences

**Message Format**:
- `data`: JSON string containing the action sequence

**Usage Example**:
```
Data: "[{\"action\": \"navigate\", \"params\": {\"target\": \"kitchen\"}, \"timeout\": 120.0, \"ros_action_server\": \"/navigate_to_pose\"}, {\"action\": \"detect_object\", \"params\": {\"object_type\": \"red cup\", \"timeout\": 30.0}}]"
```

### 3. System Status Topic

**Topic Name**: `/vla_system_status`

**Message Type**: `std_msgs/msg/String`

**Description**: Topic for broadcasting system status updates

**Message Format**:
- `data`: JSON string containing system status information

**Usage Example**:
```
Data: "{\"status\": \"processing\", \"current_component\": \"llm\", \"command_in_flight\": true, \"timestamp\": \"2025-12-10T10:00:00Z\"}"
```

## Error Handling

### Standard Error Response Format
All services and actions follow this error format:

```yaml
success: false
error_message: string      # Human-readable error message
error_code: string         # Standardized error code
details: string            # Additional technical details (optional)
```

### Common Error Codes
- `VOICE_PROCESSING_FAILED`: Voice-to-text conversion failed
- `LLM_QUERY_FAILED`: LLM request failed
- `INVALID_COMMAND`: Command could not be understood or processed
- `ACTION_EXECUTION_FAILED`: Action execution failed
- `RESOURCE_UNAVAILABLE`: Required resource (e.g., model) not available
- `TIMEOUT`: Operation exceeded timeout
- `PRIVACY_VIOLATION`: Attempt to access external resources when prohibited
- `INVALID_ROS_ACTION_SERVER`: Invalid ROS action server name specified
- `CONNECTION_FAILED`: Failed to connect to required service/module

## Validation Requirements

### Request Validation
- All string inputs must be properly formatted
- Numerical values must be within valid ranges
- Required fields must be present
- Size limits must be respected (e.g., max 10KB for audio data)
- ROS action server names must be valid if specified

### Response Guarantees
- Responses will be provided within specified timeouts
- Error responses will include actionable information
- Success responses will include valid data structures
- All JSON strings will be properly formatted

## Performance Guarantees

### Response Times
- Voice processing: <2 seconds (achievable with Whisper.cpp on modern hardware)
- LLM response: <5 seconds (achievable with local Ollama setup)
- Action sequence generation: <3 seconds (achievable with optimized prompts)

### Availability
- System availability: >95% during testing hours
- Service uptime: >98% for non-LLM dependent services
- Failover capabilities for LLM services when model is unavailable

## Security & Privacy

### Data Handling
- All voice data processed locally
- No external API calls for privacy-compliant operation
- Data retention: <7 days for temporary processing artifacts
- All data transmission within system follows ROS 2 security protocols

### Access Control
- Services require proper ROS 2 authentication
- Sensitive operations require elevated permissions
- Audit logging for all system interactions

## Integration with Previous Modules

### ROS 2 Module (Module 1) Integration
- Uses standard ROS 2 action server patterns
- Compatible with ROS 2 Humble/Jazzy distributions
- Follows ROS 2 best practices for communication

### Digital Twin Module (Module 2) Integration
- Provides interfaces for simulating VLA capabilities in digital twin environment
- Use simulation for safe testing of complex action sequences
- Maintain consistency with Module 2's sensor simulation patterns

### Isaac Sim Module (Module 3) Integration
- Integrate Isaac Sim for realistic physics simulation of humanoid actions
- Use Isaac Sim's vision sensors for multi-modal training
- Ensure consistency with Module 3's simulation workflows