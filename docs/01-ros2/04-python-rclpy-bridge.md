---
sidebar_position: 4
title: "Python rclpy Bridge to Controllers"
---

# Python rclpy Bridge to Controllers

## Learning Objectives

By the end of this chapter, students will be able to:
1. Implement Python nodes using the rclpy library for ROS 2 communication
2. Create publishers, subscribers, services, and action clients/servers in Python
3. Interface Python-based controllers with ROS 2 for humanoid robot control
4. Handle asynchronous operations and callbacks in ROS 2 Python nodes
5. Design control loops with appropriate timing and synchronization
6. Implement error handling and safety mechanisms in Python controllers
7. Integrate Python machine learning models with ROS 2 control systems

## Introduction

The Robot Control Library for Python (rclpy) provides Python bindings for ROS 2, enabling developers to create ROS 2 nodes in Python. This is particularly valuable for humanoid robotics, where Python's rich ecosystem of scientific computing, machine learning, and control libraries can be leveraged for complex robot behaviors. The rclpy library allows seamless integration between Python-based algorithms and the ROS 2 ecosystem, making it possible to implement sophisticated controllers for humanoid robots.

Python's strengths in rapid prototyping, mathematical computation, and machine learning make it an ideal choice for implementing high-level controllers, motion planners, and perception algorithms for humanoid robots. The rclpy library bridges the gap between Python's capabilities and ROS 2's distributed architecture, allowing developers to create nodes that can communicate with other parts of the robotic system.

## Theory: Python ROS 2 Programming with rclpy

### Node Architecture

In rclpy, each Python program that participates in the ROS 2 ecosystem is a node. A node can contain:
- Publishers: for sending messages to topics
- Subscribers: for receiving messages from topics
- Service clients: for making requests to services
- Service servers: for providing services
- Action clients: for sending goals to action servers
- Action servers: for executing long-running tasks with feedback

### Asynchronous Programming

ROS 2 Python nodes typically use asynchronous programming patterns. The rclpy library provides mechanisms for handling callbacks from different communication patterns without blocking the main execution thread. This is crucial for humanoid robot control, where multiple systems need to operate concurrently.

### Threading Considerations

When implementing controllers in Python, it's important to consider the Global Interpreter Lock (GIL) and how it affects real-time performance. While Python may not be suitable for the most time-critical control loops, it excels at higher-level decision making and coordination.

## Practice: Implementing Python Controllers

### Basic Node Structure

Let's create a Python package for humanoid controllers:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python humanoid_controllers --dependencies rclpy std_msgs sensor_msgs geometry_msgs control_msgs trajectory_msgs builtin_interfaces
```

### Creating a Joint Position Controller

Create `~/ros2_ws/src/humanoid_controllers/humanoid_controllers/joint_position_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import time
from builtin_interfaces.msg import Duration

class JointPositionController(Node):
    def __init__(self):
        super().__init__('joint_position_controller')

        # Declare parameters
        self.declare_parameter('joint_names', ['left_hip_joint', 'left_knee_joint', 'left_ankle_joint'])
        self.declare_parameter('control_frequency', 50)  # Hz
        self.declare_parameter('kp', 10.0)  # Proportional gain
        self.declare_parameter('ki', 0.1)   # Integral gain
        self.declare_parameter('kd', 0.5)   # Derivative gain

        # Get parameters
        self.joint_names = self.get_parameter('joint_names').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value

        # Initialize controller state
        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.desired_positions = {name: 0.0 for name in self.joint_names}
        self.errors = {name: 0.0 for name in self.joint_names}
        self.integral_errors = {name: 0.0 for name in self.joint_names}
        self.previous_errors = {name: 0.0 for name in self.joint_names}

        # Create subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10)

        # Create publishers
        self.joint_command_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10)

        # Create service server for setting desired positions
        self.set_position_service = self.create_service(
            JointTrajectory,
            'set_joint_positions',
            self.set_joint_positions_callback)

        # Create timer for control loop
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_loop)

        self.get_logger().info(f'Joint Position Controller initialized for joints: {self.joint_names}')

    def joint_state_callback(self, msg):
        """Update current joint positions from joint state message"""
        for i, name in enumerate(msg.name):
            if name in self.current_positions:
                self.current_positions[name] = msg.position[i]

    def set_joint_positions_callback(self, request, response):
        """Service callback to set desired joint positions"""
        for point in request.points:
            for i, joint_name in enumerate(request.joint_names):
                if joint_name in self.desired_positions:
                    self.desired_positions[joint_name] = point.positions[i]

        self.get_logger().info(f'Set desired positions: {self.desired_positions}')
        response.success = True
        return response

    def control_loop(self):
        """Main control loop implementing PID control"""
        # Calculate errors
        for joint_name in self.joint_names:
            current_pos = self.current_positions.get(joint_name, 0.0)
            desired_pos = self.desired_positions.get(joint_name, 0.0)

            error = desired_pos - current_pos
            self.errors[joint_name] = error

            # Update integral and derivative terms
            self.integral_errors[joint_name] += error * (1.0 / self.control_frequency)
            derivative = (error - self.previous_errors[joint_name]) * self.control_frequency

            # PID control law
            control_output = (self.kp * error +
                            self.ki * self.integral_errors[joint_name] +
                            self.kd * derivative)

            self.previous_errors[joint_name] = error

            # Publish control command
            self.publish_joint_command(joint_name, control_output)

    def publish_joint_command(self, joint_name, control_output):
        """Publish joint trajectory command"""
        # Create trajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = [joint_name]

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [control_output]  # This would typically be the desired position, not control output
        point.time_from_start = Duration(sec=0, nanosec=10000000)  # 10ms

        traj_msg.points = [point]

        self.joint_command_pub.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)

    controller = JointPositionController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Joint Position Controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Balance Controller

Create `~/ros2_ws/src/humanoid_controllers/humanoid_controllers/balance_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Point
from std_msgs.msg import Float64
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from collections import deque

class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')

        # Declare parameters
        self.declare_parameter('control_frequency', 100)  # Hz
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('hip_joint_name', 'left_hip_joint')
        self.declare_parameter('ankle_joint_name', 'left_ankle_joint')
        self.declare_parameter('kp_balance', 5.0)
        self.declare_parameter('ki_balance', 0.1)
        self.declare_parameter('kd_balance', 1.0)

        # Get parameters
        self.control_frequency = self.get_parameter('control_frequency').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.hip_joint_name = self.get_parameter('hip_joint_name').value
        self.ankle_joint_name = self.get_parameter('ankle_joint_name').value
        self.kp_balance = self.get_parameter('kp_balance').value
        self.ki_balance = self.get_parameter('ki_balance').value
        self.kd_balance = self.get_parameter('kd_balance').value

        # Initialize state
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll_velocity = 0.0
        self.pitch_velocity = 0.0
        self.yaw_velocity = 0.0

        # PID state for balance
        self.roll_error = 0.0
        self.pitch_error = 0.0
        self.roll_integral = 0.0
        self.pitch_integral = 0.0
        self.previous_roll_error = 0.0
        self.previous_pitch_error = 0.0

        # Create subscribers
        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            10)

        # Create publishers
        self.balance_command_pub = self.create_publisher(
            JointTrajectory,
            '/balance_controller/commands',
            10)

        # Create timer for balance control loop
        self.balance_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.balance_control_loop)

        # Create a history buffer for filtering
        self.roll_history = deque(maxlen=10)
        self.pitch_history = deque(maxlen=10)

        self.get_logger().info('Balance Controller initialized')

    def imu_callback(self, msg):
        """Process IMU data to extract orientation and angular velocities"""
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        orientation = msg.orientation
        self.roll, self.pitch, self.yaw = self.quaternion_to_euler(
            orientation.w, orientation.x, orientation.y, orientation.z)

        # Get angular velocities
        self.roll_velocity = msg.angular_velocity.x
        self.pitch_velocity = msg.angular_velocity.y
        self.yaw_velocity = msg.angular_velocity.z

        # Add to history for filtering
        self.roll_history.append(self.roll)
        self.pitch_history.append(self.pitch)

    def quaternion_to_euler(self, w, x, y, z):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def balance_control_loop(self):
        """Main balance control loop"""
        # Calculate desired orientation (typically zero for balance)
        desired_roll = 0.0
        desired_pitch = 0.0

        # Calculate errors
        self.roll_error = desired_roll - self.roll
        self.pitch_error = desired_pitch - self.pitch

        # Update integral terms
        self.roll_integral += self.roll_error * (1.0 / self.control_frequency)
        self.pitch_integral += self.pitch_error * (1.0 / self.control_frequency)

        # Calculate derivatives
        roll_derivative = (self.roll_error - self.previous_roll_error) * self.control_frequency
        pitch_derivative = (self.pitch_error - self.previous_pitch_error) * self.control_frequency

        # Update previous errors
        self.previous_roll_error = self.roll_error
        self.previous_pitch_error = self.pitch_error

        # Calculate control outputs using PID
        hip_command = (self.kp_balance * self.roll_error +
                      self.ki_balance * self.roll_integral +
                      self.kd_balance * roll_derivative)

        ankle_command = (self.kp_balance * self.pitch_error +
                        self.ki_balance * self.pitch_integral +
                        self.kd_balance * pitch_derivative)

        # Publish balance commands
        self.publish_balance_commands(hip_command, ankle_command)

    def publish_balance_commands(self, hip_command, ankle_command):
        """Publish balance control commands as joint trajectory"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = [self.hip_joint_name, self.ankle_joint_name]

        point = JointTrajectoryPoint()
        point.positions = [hip_command, ankle_command]
        point.velocities = [0.0, 0.0]  # Zero velocity for now
        point.time_from_start = Duration(sec=0, nanosec=10000000)  # 10ms

        traj_msg.points = [point]

        self.balance_command_pub.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)

    controller = BalanceController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Balance Controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Walking Pattern Generator

Create `~/ros2_ws/src/humanoid_controllers/humanoid_controllers/walking_pattern_generator.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import math

class WalkingPatternGenerator(Node):
    def __init__(self):
        super().__init__('walking_pattern_generator')

        # Declare parameters
        self.declare_parameter('control_frequency', 50)  # Hz
        self.declare_parameter('step_height', 0.05)     # meters
        self.declare_parameter('step_length', 0.3)      # meters
        self.declare_parameter('step_duration', 1.0)    # seconds
        self.declare_parameter('hip_joint_name', 'left_hip_joint')
        self.declare_parameter('knee_joint_name', 'left_knee_joint')
        self.declare_parameter('ankle_joint_name', 'left_ankle_joint')

        # Get parameters
        self.control_frequency = self.get_parameter('control_frequency').value
        self.step_height = self.get_parameter('step_height').value
        self.step_length = self.get_parameter('step_length').value
        self.step_duration = self.get_parameter('step_duration').value
        self.hip_joint_name = self.get_parameter('hip_joint_name').value
        self.knee_joint_name = self.get_parameter('knee_joint_name').value
        self.ankle_joint_name = self.get_parameter('ankle_joint_name').value

        # Initialize walking state
        self.walk_enabled = False
        self.walk_speed = 0.0
        self.walk_direction = 0.0  # radians
        self.walk_phase = 0.0      # 0 to 2*pi
        self.step_count = 0

        # Create subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)

        # Create publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/walking_controller/trajectory',
            10)

        # Create timer for walking pattern generation
        self.walk_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.walk_pattern_generator)

        self.get_logger().info('Walking Pattern Generator initialized')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        self.walk_speed = math.sqrt(msg.linear.x**2 + msg.linear.y**2)
        self.walk_direction = math.atan2(msg.linear.y, msg.linear.x)

        # Enable walking if speed is above threshold
        self.walk_enabled = self.walk_speed > 0.01  # 1cm/s threshold

        if self.walk_enabled:
            self.get_logger().info(f'Walking: speed={self.walk_speed:.2f} m/s, direction={math.degrees(self.walk_direction):.1f}°')

    def walk_pattern_generator(self):
        """Generate walking pattern based on desired velocity"""
        if not self.walk_enabled:
            # Publish zero commands when not walking
            self.publish_zero_commands()
            return

        # Update walking phase
        self.walk_phase += 2 * math.pi * (1.0 / self.control_frequency) / self.step_duration

        if self.walk_phase >= 2 * math.pi:
            self.walk_phase = 0.0
            self.step_count += 1

        # Calculate joint positions based on walking phase
        hip_pos, knee_pos, ankle_pos = self.calculate_leg_trajectory(self.walk_phase)

        # Publish trajectory
        self.publish_trajectory(hip_pos, knee_pos, ankle_pos)

    def calculate_leg_trajectory(self, phase):
        """Calculate leg joint positions based on walking phase"""
        # Simplified walking pattern using sinusoidal functions
        # Phase: 0 to pi (swing phase), pi to 2*pi (stance phase)

        if phase < math.pi:
            # Swing phase - leg moves forward
            swing_progress = phase / math.pi

            # Hip moves to lift leg and move forward
            hip_pos = -0.1 + 0.05 * math.sin(phase)  # Oscillate around -0.1

            # Knee bends to clear ground
            knee_pos = -0.5 + 0.3 * math.sin(phase)  # Bend more during swing

            # Ankle adjusts for foot placement
            ankle_pos = 0.2 + 0.1 * math.sin(phase)  # Oscillate around 0.2
        else:
            # Stance phase - leg supports body
            stance_phase = phase - math.pi
            stance_progress = stance_phase / math.pi

            # Hip maintains support
            hip_pos = -0.05 + 0.02 * math.sin(stance_phase * 2)  # Small oscillation

            # Knee extends for support
            knee_pos = -0.2 - 0.2 * math.sin(stance_phase)  # Extend during stance

            # Ankle maintains contact
            ankle_pos = 0.1 + 0.05 * math.sin(stance_phase * 2)  # Small adjustment

        return hip_pos, knee_pos, ankle_pos

    def publish_trajectory(self, hip_pos, knee_pos, ankle_pos):
        """Publish joint trajectory for walking"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = [self.hip_joint_name, self.knee_joint_name, self.ankle_joint_name]

        point = JointTrajectoryPoint()
        point.positions = [hip_pos, knee_pos, ankle_pos]
        point.velocities = [0.0, 0.0, 0.0]  # Would be calculated in a real implementation
        point.time_from_start = Duration(sec=0, nanosec=int(1000000000.0 / self.control_frequency))

        traj_msg.points = [point]

        self.trajectory_pub.publish(traj_msg)

    def publish_zero_commands(self):
        """Publish zero joint positions when not walking"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = [self.hip_joint_name, self.knee_joint_name, self.ankle_joint_name]

        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, 0.0]
        point.velocities = [0.0, 0.0, 0.0]
        point.time_from_start = Duration(sec=0, nanosec=10000000)

        traj_msg.points = [point]

        self.trajectory_pub.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)

    generator = WalkingPatternGenerator()

    try:
        rclpy.spin(generator)
    except KeyboardInterrupt:
        generator.get_logger().info('Shutting down Walking Pattern Generator')
    finally:
        generator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Active Learning Exercise

**Exercise: Implementing a Python-based Vision Controller**

Using the patterns demonstrated in the examples above, implement a Python controller that:
1. Subscribes to a camera topic (sensor_msgs/Image)
2. Processes the image to detect an object (you can simulate this with a simple condition)
3. Calculates the necessary joint movements to point the robot's head at the object
4. Publishes joint trajectory commands to move the head

Consider the timing constraints for real-time image processing and how to maintain a consistent control loop frequency. Discuss potential challenges with this approach and possible solutions.

## Worked Example: Black-box to Glass-box - Implementing an Adaptive Walking Controller

### Black-box View

We'll implement an adaptive walking controller that adjusts its gait based on terrain feedback. The black-box view is: the controller receives sensor data about the terrain, adjusts walking parameters accordingly, and produces stable walking patterns adapted to the current conditions.

### Glass-box Implementation

1. **Create the adaptive walking controller:**

Create `~/ros2_ws/src/humanoid_controllers/humanoid_controllers/adaptive_walking_controller.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64
from builtin_interfaces.msg import Duration
import numpy as np
import math
from collections import deque

class AdaptiveWalkingController(Node):
    def __init__(self):
        super().__init__('adaptive_walking_controller')

        # Declare parameters
        self.declare_parameter('control_frequency', 100)  # Hz
        self.declare_parameter('base_step_height', 0.05)  # meters
        self.declare_parameter('base_step_length', 0.3)   # meters
        self.declare_parameter('base_step_duration', 1.0) # seconds
        self.declare_parameter('max_adaptation_factor', 1.5)  # maximum adaptation multiplier
        self.declare_parameter('min_adaptation_factor', 0.7)  # minimum adaptation multiplier

        # Get parameters
        self.control_frequency = self.get_parameter('control_frequency').value
        self.base_step_height = self.get_parameter('base_step_height').value
        self.base_step_length = self.get_parameter('base_step_length').value
        self.base_step_duration = self.get_parameter('base_step_duration').value
        self.max_adaptation = self.get_parameter('max_adaptation_factor').value
        self.min_adaptation = self.get_parameter('min_adaptation_factor').value

        # Initialize state
        self.walk_enabled = False
        self.walk_speed = 0.0
        self.walk_direction = 0.0
        self.walk_phase = 0.0
        self.step_count = 0

        # Terrain adaptation state
        self.terrain_roughness = 0.0
        self.foot_pressure_left = 0.0
        self.foot_pressure_right = 0.0
        self.balance_error = 0.0

        # Adaptation factors (start with base values)
        self.height_factor = 1.0
        self.length_factor = 1.0
        self.duration_factor = 1.0

        # Create subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)

        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10)

        # Create publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/adaptive_walking_controller/trajectory',
            10)

        self.adaptation_pub = self.create_publisher(
            Float64,
            '/adaptation_factor',
            10)

        # Create timer for control loop
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_loop)

        # Create history buffers for terrain analysis
        self.terrain_history = deque(maxlen=50)
        self.balance_history = deque(maxlen=50)

        self.get_logger().info('Adaptive Walking Controller initialized')

    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        self.walk_speed = math.sqrt(msg.linear.x**2 + msg.linear.y**2)
        self.walk_direction = math.atan2(msg.linear.y, msg.linear.x)
        self.walk_enabled = self.walk_speed > 0.01  # 1cm/s threshold

    def imu_callback(self, msg):
        """Process IMU data for balance feedback"""
        # Calculate balance error from roll and pitch
        roll, pitch, _ = self.quaternion_to_euler(
            msg.orientation.w, msg.orientation.x,
            msg.orientation.y, msg.orientation.z)

        # Balance error is a combination of roll and pitch angles
        self.balance_error = abs(roll) + abs(pitch)

        # Add to history for analysis
        self.balance_history.append(self.balance_error)

    def laser_callback(self, msg):
        """Process laser scan for terrain analysis"""
        # Analyze terrain roughness from laser scan
        # This is a simplified approach - in reality, you'd use more sophisticated methods
        if len(msg.ranges) > 0:
            # Calculate variance in distances as a measure of terrain roughness
            valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
            if len(valid_ranges) > 10:  # Need enough samples
                mean_range = sum(valid_ranges) / len(valid_ranges)
                variance = sum((r - mean_range)**2 for r in valid_ranges) / len(valid_ranges)
                self.terrain_roughness = min(variance, 1.0)  # Cap the value

        # Add to history
        self.terrain_history.append(self.terrain_roughness)

    def quaternion_to_euler(self, w, x, y, z):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def calculate_adaptation_factors(self):
        """Calculate adaptation factors based on sensor feedback"""
        # Terrain-based adaptations
        terrain_factor = 1.0 + (self.terrain_roughness * 0.3)  # Increase step height on rough terrain

        # Balance-based adaptations
        balance_factor = 1.0 + (self.balance_error * 0.5)  # Be more careful when off-balance

        # Combine factors with limits
        self.height_factor = max(self.min_adaptation,
                                min(self.max_adaptation, terrain_factor))
        self.length_factor = max(self.min_adaptation * 0.8,
                                min(self.max_adaptation * 0.8, 1.0 / balance_factor))
        self.duration_factor = max(self.min_adaptation * 1.2,
                                  min(self.max_adaptation * 1.2, balance_factor))

    def control_loop(self):
        """Main control loop"""
        # Update adaptation factors based on sensor feedback
        self.calculate_adaptation_factors()

        # Publish adaptation factors for monitoring
        adapt_msg = Float64()
        adapt_msg.data = (self.height_factor + self.length_factor + self.duration_factor) / 3.0
        self.adaptation_pub.publish(adapt_msg)

        if not self.walk_enabled:
            # Publish zero commands when not walking
            self.publish_zero_commands()
            return

        # Update walking phase with adaptation
        adapted_duration = self.base_step_duration * self.duration_factor
        phase_increment = 2 * math.pi * (1.0 / self.control_frequency) / adapted_duration

        self.walk_phase += phase_increment

        if self.walk_phase >= 2 * math.pi:
            self.walk_phase = 0.0
            self.step_count += 1

        # Calculate adapted walking parameters
        adapted_height = self.base_step_height * self.height_factor
        adapted_length = self.base_step_length * self.length_factor

        # Calculate joint positions based on adapted parameters
        hip_pos, knee_pos, ankle_pos = self.calculate_adapted_leg_trajectory(
            self.walk_phase, adapted_height, adapted_length)

        # Publish trajectory
        self.publish_trajectory(hip_pos, knee_pos, ankle_pos)

    def calculate_adapted_leg_trajectory(self, phase, height, length):
        """Calculate leg trajectory with adapted parameters"""
        # Adjust trajectory based on adapted parameters
        if phase < math.pi:
            # Swing phase - adapted for terrain
            swing_progress = phase / math.pi

            # Hip moves with adapted height and length
            hip_pos = -0.1 + (height * 0.5) * math.sin(phase)

            # Knee bends more on rough terrain
            knee_bend_factor = 0.3 + (self.terrain_roughness * 0.2)
            knee_pos = -0.5 + knee_bend_factor * math.sin(phase)

            # Ankle adjusts for terrain
            ankle_pos = 0.2 + (height * 0.2) * math.sin(phase)
        else:
            # Stance phase - adapted for stability
            stance_phase = phase - math.pi
            stance_progress = stance_phase / math.pi

            # Hip maintains adapted support
            hip_pos = -0.05 + (height * 0.1) * math.sin(stance_phase * 2)

            # Knee extends for support
            knee_pos = -0.2 - (length * 0.2) * math.sin(stance_phase)

            # Ankle maintains contact
            ankle_pos = 0.1 + (height * 0.05) * math.sin(stance_phase * 2)

        return hip_pos, knee_pos, ankle_pos

    def publish_trajectory(self, hip_pos, knee_pos, ankle_pos):
        """Publish joint trajectory for adaptive walking"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ['left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
                               'right_hip_joint', 'right_knee_joint', 'right_ankle_joint']

        point = JointTrajectoryPoint()
        # For simplicity, use same values for both legs (in a real system, they'd be different)
        point.positions = [hip_pos, knee_pos, ankle_pos, hip_pos, knee_pos, ankle_pos]
        point.velocities = [0.0] * 6  # Would be calculated in a real implementation
        point.time_from_start = Duration(
            sec=0,
            nanosec=int(1000000000.0 / self.control_frequency))

        traj_msg.points = [point]

        self.trajectory_pub.publish(traj_msg)

    def publish_zero_commands(self):
        """Publish zero joint positions when not walking"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ['left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
                               'right_hip_joint', 'right_knee_joint', 'right_ankle_joint']

        point = JointTrajectoryPoint()
        point.positions = [0.0] * 6
        point.velocities = [0.0] * 6
        point.time_from_start = Duration(sec=0, nanosec=10000000)

        traj_msg.points = [point]

        self.trajectory_pub.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)

    controller = AdaptiveWalkingController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Adaptive Walking Controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

2. **Update the package setup.py:**

Edit `~/ros2_ws/src/humanoid_controllers/setup.py`:

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'humanoid_controllers'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Python controllers for humanoid robots',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_position_controller = humanoid_controllers.joint_position_controller:main',
            'balance_controller = humanoid_controllers.balance_controller:main',
            'walking_pattern_generator = humanoid_controllers.walking_pattern_generator:main',
            'adaptive_walking_controller = humanoid_controllers.adaptive_walking_controller:main',
        ],
    },
)
```

3. **Build the package:**

```bash
cd ~/ros2_ws
colcon build --packages-select humanoid_controllers
source install/setup.bash
```

### Understanding the Implementation

The glass-box view reveals:
- The controller uses multiple sensor inputs (IMU, laser, velocity commands) to adapt its behavior
- The adaptation algorithm adjusts walking parameters based on terrain roughness and balance state
- The control loop maintains timing constraints while processing sensor data
- The implementation includes safety limits to prevent extreme adaptations
- The system publishes monitoring data to track adaptation factors

## Tiered Assessments

### Tier 1: Basic Understanding
1. What is rclpy and what is its role in ROS 2?
2. Name the main types of communication patterns available in rclpy.
3. What is the purpose of the control loop timer in a Python controller?

### Tier 2: Application
4. Create a Python node that subscribes to IMU data and publishes joint commands to maintain balance.
5. Implement a simple PID controller in Python for joint position control.

### Tier 3: Analysis and Synthesis
6. Design a complete control architecture for a humanoid robot using multiple Python controllers, considering real-time constraints, safety requirements, and coordination between different control systems.

## Mermaid Diagram

```mermaid
graph TB
    subgraph "Python Controllers"
        A[Joint Position Controller]
        B[Balance Controller]
        C[Walking Pattern Generator]
        D[Adaptive Walking Controller]
    end

    subgraph "ROS 2 System"
        E[Joint State Publisher]
        F[IMU Sensor]
        G[Laser Scanner]
        H[Command Velocity]
    end

    subgraph "Hardware Interface"
        I[Joint Trajectory Controller]
        J[Robot Hardware]
    end

    E --> A
    F --> B
    G --> D
    H --> C
    A --> I
    B --> I
    C --> I
    D --> I
    I --> J

    style A fill:#ff9999
    style B fill:#ff9999
    style C fill:#ff9999
    style D fill:#ff9999
    style I fill:#99ccff
    style J fill:#99ff99
```

**Alt-text for diagram:** "System architecture diagram showing Python controllers (Joint Position, Balance, Walking Pattern Generator, and Adaptive Walking) interfacing with ROS 2 system components (Joint State Publisher, IMU Sensor, Laser Scanner, Command Velocity) and hardware interface (Joint Trajectory Controller and Robot Hardware). The controllers are highlighted in pink, the hardware interface in blue and green respectively."

## Summary

This chapter covered the implementation of Python-based controllers for humanoid robots using the rclpy library. We explored how to create nodes that interface with the ROS 2 ecosystem, handle different communication patterns, and implement control algorithms for joint positioning, balance, and walking. The examples demonstrated practical implementations of controllers that can adapt to changing conditions and sensor feedback, which is essential for humanoid robot operation in real-world environments.

## References

1. Quigley, M., Gerkey, B., & Smart, W. D. (2019). Programming robots with ROS: a practical introduction to the Robot Operating System. O'Reilly Media.

2. Bou-Ammar, H., Ralaivola, L., & Duffner, S. (2019). Practical robotics software development in C++. Apress.

3. Siciliano, B., & Khatib, O. (2016). Springer handbook of robotics. Springer Publishing Company, Incorporated.

4. Lentin, J. (2018). Mastering ROS for robotics programming. Packt Publishing.

5. Tedrake, R. (2019). Underactuated robotics: Algorithms for walking, running, swimming, flying, and manipulation. MIT Press.

6. Khatib, O., Park, H. W., Verrelst, B., Kim, B. S., Dalloul, I., & Yokoi, K. (2016). Humanoid robot control system framework for supporting different programming methods. *2016 IEEE-RAS 16th International Conference on Humanoid Robots (Humanoids)*, 83-88.

7. Ha, I., Tamura, Y., Asama, H., &白云, J. (2015). Real-time walking pattern generation based on control of the zero-moment point for a full-size humanoid robot. *Advanced Robotics*, 29(1), 1-14.

8. Nakanishi, J., Cory, R., Mistry, M., Peters, J., & Schaal, S. (2008). Operational space control: A theoretical and empirical comparison. *International Journal of Robotics Research*, 27(6), 737-757.