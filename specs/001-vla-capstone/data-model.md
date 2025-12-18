# Data Model: Module 4 - Vision-Language-Action (VLA) + Capstone

**Date**: 2025-12-10
**Feature**: 001-vla-capstone

## Entity Models

### VoiceCommand
**Description**: Represents a spoken instruction that needs to be processed through Whisper and converted to text

**Fields**:
- `id` (string): Unique identifier for the command
- `text` (string): The transcribed text from voice input
- `command_type` (enum): Type of command (navigation, manipulation, perception, communication)
- `timestamp` (datetime): When the command was received
- `confidence` (float): Confidence score of voice recognition (0.0-1.0)
- `source` (string): Source of the command (voice, text, API)
- `user_id` (string, optional): Identifier of the user who issued the command
- `status` (enum): Current status (pending, processing, completed, failed)
- `error_message` (string, optional): Error message if processing failed
- `context` (object): Environmental context for the command
  - `location` (object): Current robot location
    - `x` (float)
    - `y` (float)
    - `z` (float)
  - `orientation` (object): Current robot orientation
    - `roll` (float)
    - `pitch` (float)
    - `yaw` (float)
  - `environment` (object): Environmental data
    - `room_layout` (string)
    - `obstacles` (array of objects)

**Validation Rules**:
- `text` must not be empty
- `confidence` must be between 0.0 and 1.0
- `timestamp` must be in the past or present

**State Transitions**:
- pending → processing (when processing starts)
- processing → completed (when processing succeeds)
- processing → failed (when processing fails)
- pending → failed (when validation fails)

### ActionSequence
**Description**: Represents a series of ROS 2 actions generated from natural language commands

**Fields**:
- `id` (string): Unique identifier for the sequence
- `name` (string): Descriptive name of the sequence
- `description` (string): Detailed description of the sequence
- `steps` (array of ActionStep): Ordered list of actions to execute
- `created_by` (string): Entity that created this sequence
- `created_at` (datetime): When the sequence was created
- `updated_at` (datetime): When the sequence was last updated
- `command_id` (string): ID of the original command that generated this
- `estimated_duration` (float): Estimated time to complete the sequence in seconds
- `success_threshold` (float): Minimum success probability (0.0-1.0)
- `error_recovery_enabled` (boolean): Whether to attempt error recovery
- `execution_constraints` (object): Constraints for execution
  - `max_execution_time` (float)
  - `safety_limits` (object)
    - `max_force` (float)
    - `max_speed` (float)
- `tags` (array of string): Tags for categorizing the sequence
- `version` (string): Version of the sequence format
- `dependencies` (array of string): Other sequences this depends on
- `metadata` (object): Additional metadata
- `execution_state` (ActionSequenceExecution, optional): Current execution state

**Validation Rules**:
- `steps` must contain at least one step
- `success_threshold` must be between 0.0 and 1.0
- `estimated_duration` must be positive

### ActionStep
**Description**: Represents a single action within an action sequence

**Fields**:
- `id` (string): Unique identifier for the step
- `action_type` (string): Type of action (navigate_to, grasp_object, speak, etc.)
- `parameters` (object): Parameters for the action
- `description` (string): Human-readable description of what the action does
- `priority` (integer): Priority level (1-5, where 5 is highest)
- `timeout` (float): Maximum time to wait for action completion in seconds
- `retry_attempts` (integer): Number of retry attempts if action fails
- `dependencies` (array of string): IDs of steps this depends on
- `preconditions` (array of object): Conditions that must be met before execution
- `postconditions` (array of object): Expected outcomes after execution
- `safety_constraints` (array of string): Safety constraints to apply
- `execution_status` (enum): Current execution status
- `started_at` (datetime, optional): When the step started executing
- `completed_at` (datetime, optional): When the step completed
- `error_message` (string, optional): Error message if step failed
- `result` (object, optional): Result of the step execution
- `ros_action_server` (string, optional): Name of the ROS action server to use for this step

**Validation Rules**:
- `priority` must be between 1 and 5
- `timeout` must be positive
- `retry_attempts` must be non-negative
- If `ros_action_server` is specified, it must be a valid ROS action server name

**State Transitions**:
- pending → executing (when execution starts)
- executing → success (when execution completes successfully)
- executing → failed (when execution fails)
- executing → timeout (when execution times out)
- executing → skipped (when step is skipped)

### ActionSequenceExecution
**Description**: Represents the execution state of an action sequence

**Fields**:
- `sequence_id` (string): ID of the sequence being executed
- `current_step_index` (integer): Index of the currently executing step
- `execution_status` (enum): Current execution status
- `started_at` (datetime, optional): When execution started
- `completed_at` (datetime, optional): When execution completed
- `error_message` (string, optional): Error message if execution failed
- `execution_log` (array of object): Log of execution events
- `progress` (float): Progress percentage (0.0-1.0)
- `current_step_id` (string, optional): ID of the currently executing step
- `retry_count` (integer): Number of retry attempts for the sequence

**State Transitions**:
- pending → running (when execution starts)
- running → completed (when all steps complete successfully)
- running → failed (when execution fails)
- running → cancelled (when execution is cancelled)

### MultiModalInput
**Description**: Represents combined visual and linguistic inputs that guide robot behavior

**Fields**:
- `id` (string): Unique identifier for the multi-modal input
- `voice_input` (VoiceCommand): Voice command component
- `visual_input` (object): Visual perception data
  - `image_path` (string, optional): Path to image file
  - `object_detections` (array of object): Detected objects
    - `class` (string): Object class
    - `confidence` (float): Detection confidence
    - `bbox` (object): Bounding box coordinates
      - `x_min` (float)
      - `y_min` (float)
      - `x_max` (float)
      - `y_max` (float)
  - `scene_description` (string): Natural language description of the scene
  - `depth_data` (object, optional): Depth information
- `fusion_confidence` (float): Confidence in the multi-modal fusion (0.0-1.0)
- `timestamp` (datetime): When the input was captured
- `processing_status` (enum): Current processing status
- `fused_command` (string, optional): The fused command after processing both modalities

**Validation Rules**:
- `fusion_confidence` must be between 0.0 and 1.0
- Must have at least one of `voice_input` or `visual_input`

**State Transitions**:
- received → processing (when fusion starts)
- processing → fused (when fusion completes)
- processing → failed (when fusion fails)

### AutonomousHumanoidSystem
**Description**: Represents the complete integrated system combining all textbook modules

**Fields**:
- `id` (string): Unique identifier for the system
- `name` (string): Name of the system
- `status` (enum): Current system status
- `current_behavior` (string): Current behavior being executed
- `sensors` (object): Current sensor readings
  - `audio` (object): Audio input status
  - `vision` (object): Vision system status
  - `imu` (object): Inertial measurement unit data
  - `encoders` (object): Joint encoder readings
- `actuators` (object): Current actuator status
  - `motors` (array of object): Motor status
  - `grippers` (array of object): Gripper status
- `state` (object): Current system state
  - `location` (object): Current location
    - `x` (float)
    - `y` (float)
    - `z` (float)
  - `orientation` (object): Current orientation
    - `roll` (float)
    - `pitch` (float)
    - `yaw` (float)
  - `battery_level` (float): Battery level percentage
  - `temperature` (float): System temperature
- `active_sequences` (array of string): Currently active action sequences
- `last_command` (string): Last command processed
- `command_history` (array of string): History of recent commands
- `performance_metrics` (object): System performance metrics
  - `response_time` (float): Average response time
  - `success_rate` (float): Action success rate
  - `uptime` (float): System uptime percentage
- `error_log` (array of object): Recent error log entries
- `ros_interfaces` (object): ROS interface connections to previous modules
  - `ros2_nodes` (array of string): Connected ROS 2 nodes from Module 1
  - `digital_twin_connection` (string): Connection to Digital Twin from Module 2
  - `isaac_sim_connection` (string): Connection to Isaac Sim from Module 3

**Validation Rules**:
- `battery_level` must be between 0.0 and 100.0
- `temperature` must be within safe operating range
- `location` coordinates must be valid floats

### CapstoneProject
**Description**: Represents the final project that integrates all learning modules with comprehensive rubrics

**Fields**:
- `id` (string): Unique identifier for the project
- `title` (string): Title of the project
- `description` (string): Detailed description
- `learning_objectives` (array of string): Learning objectives to be demonstrated
- `requirements` (array of object): Technical requirements
  - `requirement_id` (string): Unique identifier for the requirement
  - `description` (string): Requirement description
  - `weight` (float): Weight in final evaluation (0.0-1.0)
  - `status` (enum): Implementation status
- `rubric` (object): Evaluation rubric
  - `criteria` (array of object): Evaluation criteria
    - `criterion_id` (string): Unique identifier for the criterion
    - `description` (string): Criterion description
    - `weight` (float): Weight in evaluation
    - `levels` (array of object): Evaluation levels
  - `total_points` (integer): Maximum possible points
- `submission` (object, optional): Student submission
  - `files` (array of string): Submitted files
  - `demonstration_video` (string, optional): Link to demonstration video
  - `report` (string): Written report
- `evaluation` (object, optional): Evaluation results
  - `scores` (array of object): Scores for each criterion
  - `feedback` (string): Instructor feedback
  - `grade` (string): Final letter grade
  - `evaluator` (string): Evaluator identifier
  - `evaluation_date` (datetime): Date of evaluation
- `status` (enum): Project status (draft, submitted, evaluated)
- `deadline` (datetime): Submission deadline
- `created_at` (datetime): Project creation date
- `updated_at` (datetime): Last update date

**Validation Rules**:
- Sum of requirement weights must equal 1.0
- Sum of rubric criterion weights must equal 1.0
- `weight` values must be between 0.0 and 1.0

### LLMPromptTemplate
**Description**: Represents a reusable prompt template for LLM-based task planning

**Fields**:
- `id` (string): Unique identifier for the template
- `name` (string): Name of the template
- `description` (string): Description of what the template does
- `template` (string): The prompt template with placeholders
- `parameters` (array of string): List of parameters that can be substituted
- `category` (string): Category of the prompt (navigation, manipulation, perception, etc.)
- `created_at` (datetime): When the template was created
- `updated_at` (datetime): When the template was last updated
- `version` (string): Version of the template
- `examples` (array of string): Example instantiations of the template

**Validation Rules**:
- `template` must contain at least one parameter placeholder
- `parameters` must match the placeholders in the template

## Relationships

### VoiceCommand → ActionSequence
- One VoiceCommand can generate one ActionSequence
- One ActionSequence is created from one VoiceCommand
- Relationship: 1:1 (VoiceCommand has optional ActionSequence)

### ActionSequence → ActionStep
- One ActionSequence contains multiple ActionSteps
- One ActionStep belongs to one ActionSequence
- Relationship: 1:many (ActionSequence has many ActionSteps)

### ActionSequence → ActionSequenceExecution
- One ActionSequence can have one execution state
- One ActionSequenceExecution belongs to one ActionSequence
- Relationship: 1:1 (ActionSequence has optional ActionSequenceExecution)

### MultiModalInput → VoiceCommand
- MultiModalInput optionally includes one VoiceCommand
- One VoiceCommand can be part of multiple MultiModalInputs
- Relationship: many:1 (MultiModalInput has optional VoiceCommand)

### AutonomousHumanoidSystem → ActionSequence
- One AutonomousHumanoidSystem can execute multiple ActionSequences
- One ActionSequence can be executed by one AutonomousHumanoidSystem
- Relationship: 1:many (AutonomousHumanoidSystem has many active ActionSequences)

### LLMPromptTemplate → ActionSequence
- One LLMPromptTemplate can be used to generate multiple ActionSequences
- One ActionSequence can use multiple LLMPromptTemplates
- Relationship: many:many (ActionSequences reference templates)

## State Diagrams

### ActionSequence Execution States
```
PENDING → PROCESSING → COMPLETED
    ↓         ↓           ↓
  FAILED ← ERROR ← TIMEOUT
```

### VoiceCommand Processing States
```
PENDING → PROCESSING → COMPLETED
    ↓         ↓           ↓
  FAILED ← VALIDATION ← REJECTED
```

### AutonomousHumanoidSystem States
```
IDLE → LISTENING → PROCESSING → EXECUTING → IDLE
  ↓      ↓           ↓           ↓         ↓
ERROR ← ERROR ← ERROR ← ERROR ← ERROR
```

## Data Validation Rules

### Cross-Entity Validation
1. An ActionSequence's command_id must reference an existing VoiceCommand
2. An ActionStep's dependencies must reference other ActionSteps in the same sequence
3. A MultiModalInput's voice_input must reference an existing VoiceCommand
4. CapstoneProject requirements must align with the module's functional requirements
5. LLMPromptTemplate parameters must match placeholders in the template
6. ActionStep ros_action_server must be a valid ROS action server name

### Business Logic Validation
1. Action sequences must not exceed maximum allowed steps (50)
2. Action steps must have appropriate timeouts based on action type
3. Safety constraints must be validated before execution
4. Privacy compliance must be maintained throughout the system
5. Battery levels must be within safe operating ranges
6. All coordinates must be valid for navigation systems

### ROS Integration Validation
1. Action steps with ros_action_server must have valid ROS action server names
2. AutonomousHumanoidSystem must maintain connections to previous modules as specified
3. Action sequences must be compatible with ROS action server interfaces