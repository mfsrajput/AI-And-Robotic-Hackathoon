---
sidebar_position: 3
title: "URDF + Xacro for Humanoids"
---

# URDF + Xacro for Humanoids

## Learning Objectives

By the end of this chapter, students will be able to:
1. Construct complete URDF models for complex humanoid robots with multiple degrees of freedom
2. Utilize Xacro macros to create reusable and maintainable robot descriptions
3. Define joint limits, safety controllers, and physical properties for humanoid robots
4. Incorporate sensor models and mounting points in URDF descriptions
5. Generate and visualize robot models using ROS 2 tools
6. Implement parametric robot designs that can be customized for different configurations
7. Validate URDF models for kinematic and dynamic correctness

## Introduction

Unified Robot Description Format (URDF) is the standard XML-based format for representing robot models in ROS. For humanoid robots, which typically have 20+ degrees of freedom and complex kinematic chains, URDF becomes essential for defining the robot's physical structure, joint relationships, and inertial properties. However, raw URDF files for humanoid robots can become extremely verbose and difficult to maintain, which is where Xacro (XML Macros) becomes invaluable.

Xacro allows the use of macros, constants, and expressions to create more readable and maintainable robot descriptions. For humanoid robots with symmetrical structures like arms and legs, Xacro macros can significantly reduce redundancy while improving consistency across the model.

## Theory: URDF Fundamentals for Humanoid Robots

### Robot Structure

A humanoid robot typically consists of:
- A base/torso segment
- Two symmetrical arms (shoulder, elbow, wrist joints)
- Two symmetrical legs (hip, knee, ankle joints)
- A head/neck assembly
- Various sensors mounted on different links

### URDF Elements for Humanoids

Key URDF elements for humanoid robots include:
- `<link>`: Represents rigid bodies (limbs, torso, etc.)
- `<joint>`: Defines connections between links with specific joint types
- `<material>`: Defines visual appearance
- `<visual>`: Specifies how links appear in simulation
- `<collision>`: Defines collision geometry
- `<inertial>`: Specifies mass properties for physics simulation

### Xacro Advantages

Xacro provides several benefits for humanoid robot descriptions:
- Macros for repetitive structures (left/right arms/legs)
- Constants for consistent measurements
- Expressions for calculated values
- Include statements for modular design

## Practice: Creating a Humanoid Robot Model

### Basic URDF Structure

Let's create a simple humanoid model using Xacro. First, create the package:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake humanoid_description --dependencies
mkdir -p ~/ros2_ws/src/humanoid_description/urdf
mkdir -p ~/ros2_ws/src/humanoid_description/meshes
```

### Creating the Main Xacro File

Create `~/ros2_ws/src/humanoid_description/urdf/humanoid.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_humanoid">

  <!-- Constants for robot dimensions -->
  <xacro:property name="PI" value="3.1415926535897931"/>
  <xacro:property name="mass_torso" value="10.0"/>
  <xacro:property name="mass_arm" value="2.0"/>
  <xacro:property name="mass_leg" value="5.0"/>
  <xacro:property name="mass_head" value="1.5"/>

  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0.0 0.0 0.15" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="orange"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_torso}"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-PI/4}" upper="${PI/4}" effort="100" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${mass_head}"/>
      <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <!-- Include macros for arms and legs -->
  <xacro:include filename="$(find humanoid_description)/urdf/humanoid_macros.xacro"/>

  <!-- Left Arm -->
  <xacro:arm side="left"
            parent="torso"
            xyz="0.2 0.1 0.1"
            rpy="0 0 0"
            color="blue"/>

  <!-- Right Arm -->
  <xacro:arm side="right"
            parent="torso"
            xyz="0.2 -0.1 0.1"
            rpy="0 0 0"
            color="blue"/>

  <!-- Left Leg -->
  <xacro:leg side="left"
            parent="base_link"
            xyz="-0.1 0.05 -0.1"
            rpy="0 0 0"
            color="green"/>

  <!-- Right Leg -->
  <xacro:leg side="right"
            parent="base_link"
            xyz="-0.1 -0.05 -0.1"
            rpy="0 0 0"
            color="green"/>

  <!-- ROS Control interface -->
  <ros2_control name="GazeboSystem" type="system">
    <hardware>
      <plugin>gazebo_ros2_control/GazeboSystem</plugin>
    </hardware>
    <joint name="neck_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="left_shoulder_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="left_elbow_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="right_shoulder_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="right_elbow_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>

  <!-- Gazebo plugins -->
  <gazebo reference="base_link">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="torso">
    <material>Gazebo/Orange</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

</robot>
```

### Creating the Xacro Macros File

Create `~/ros2_ws/src/humanoid_description/urdf/humanoid_macros.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Arm Macro -->
  <xacro:macro name="arm" params="side parent xyz rpy color">
    <!-- Shoulder joint and link -->
    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-PI/2}" upper="${PI/2}" effort="100" velocity="1"/>
    </joint>

    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
        <material name="${color}"/>
        <origin xyz="0 0 0.15" rpy="0 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
        <origin xyz="0 0 0.15" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="${mass_arm}"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
    </link>

    <!-- Elbow joint and link -->
    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-PI/2}" upper="${PI/2}" effort="100" velocity="1"/>
    </joint>

    <link name="${side}_lower_arm">
      <visual>
        <geometry>
          <cylinder length="0.25" radius="0.04"/>
        </geometry>
        <material name="${color}"/>
        <origin xyz="0 0 0.125" rpy="0 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.25" radius="0.04"/>
        </geometry>
        <origin xyz="0 0 0.125" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="${mass_arm * 0.7}"/>
        <inertia ixx="0.007" ixy="0.0" ixz="0.0" iyy="0.007" iyz="0.0" izz="0.007"/>
      </inertial>
    </link>

    <!-- Hand joint and link -->
    <joint name="${side}_wrist_joint" type="revolute">
      <parent link="${side}_lower_arm"/>
      <child link="${side}_hand"/>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-PI/4}" upper="${PI/4}" effort="50" velocity="1"/>
    </joint>

    <link name="${side}_hand">
      <visual>
        <geometry>
          <box size="0.1 0.08 0.05"/>
        </geometry>
        <material name="${color}"/>
      </visual>
      <collision>
        <geometry>
          <box size="0.1 0.08 0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Leg Macro -->
  <xacro:macro name="leg" params="side parent xyz rpy color">
    <!-- Hip joint and link -->
    <joint name="${side}_hip_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_upper_leg"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-PI/2}" upper="${PI/2}" effort="200" velocity="1"/>
    </joint>

    <link name="${side}_upper_leg">
      <visual>
        <geometry>
          <cylinder length="0.4" radius="0.06"/>
        </geometry>
        <material name="${color}"/>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.4" radius="0.06"/>
        </geometry>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="${mass_leg}"/>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
      </inertial>
    </link>

    <!-- Knee joint and link -->
    <joint name="${side}_knee_joint" type="revolute">
      <parent link="${side}_upper_leg"/>
      <child link="${side}_lower_leg"/>
      <origin xyz="0 0 -0.4" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="0" upper="${PI/2}" effort="200" velocity="1"/>
    </joint>

    <link name="${side}_lower_leg">
      <visual>
        <geometry>
          <cylinder length="0.4" radius="0.05"/>
        </geometry>
        <material name="${color}"/>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="0.4" radius="0.05"/>
        </geometry>
        <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="${mass_leg * 0.8}"/>
        <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.015"/>
      </inertial>
    </link>

    <!-- Ankle joint and foot -->
    <joint name="${side}_ankle_joint" type="revolute">
      <parent link="${side}_lower_leg"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 -0.4" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-PI/6}" upper="${PI/6}" effort="100" velocity="1"/>
    </joint>

    <link name="${side}_foot">
      <visual>
        <geometry>
          <box size="0.2 0.1 0.05"/>
        </geometry>
        <material name="${color}"/>
        <origin xyz="0 0 0.025" rpy="0 0 0"/>
      </visual>
      <collision>
        <geometry>
          <box size="0.2 0.1 0.05"/>
        </geometry>
        <origin xyz="0 0 0.025" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.003"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Sensor Macro -->
  <xacro:macro name="camera_sensor" params="name parent xyz rpy">
    <joint name="${name}_joint" type="fixed">
      <parent link="${parent}"/>
      <child link="${name}_link"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
    </joint>

    <link name="${name}_link">
      <visual>
        <geometry>
          <box size="0.02 0.04 0.02"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <geometry>
          <box size="0.02 0.04 0.02"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
      </inertial>
    </link>

    <gazebo reference="${name}_link">
      <sensor name="${name}_camera" type="camera">
        <always_on>true</always_on>
        <update_rate>30.0</update_rate>
        <camera name="head">
          <horizontal_fov>1.3962634</horizontal_fov>
          <image>
            <format>R8G8B8</format>
            <width>640</width>
            <height>480</height>
          </image>
          <clip>
            <near>0.1</near>
            <far>100</far>
          </clip>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
          <frame_name>${name}_optical_frame</frame_name>
        </plugin>
      </sensor>
    </gazebo>
  </xacro:macro>

</robot>
```

## Active Learning Exercise

**Exercise: Customizing a Humanoid Robot**

Using the Xacro macros we created, modify the humanoid model to:
1. Create a version with longer arms (10% longer)
2. Add a camera sensor to the head
3. Adjust joint limits to be more restrictive for safety
4. Change the physical properties to make the robot lighter

Implement these changes by creating a new Xacro file that includes the original one but with your modifications. Discuss how these changes might affect the robot's capabilities and performance.

## Worked Example: Black-box to Glass-box - Creating a Parametric Humanoid Model

### Black-box View

We'll create a parametric humanoid model that can be customized for different configurations. The black-box view is: we provide parameters (like height, weight, number of fingers), and the system generates a complete URDF model that's ready for simulation and control.

### Glass-box Implementation

1. **Create a parametric Xacro file:**

Create `~/ros2_ws/src/humanoid_description/urdf/parametric_humanoid.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="parametric_humanoid">

  <!-- Parameters for customization -->
  <xacro:arg name="robot_height" default="1.7"/>
  <xacro:arg name="robot_weight" default="70.0"/>
  <xacro:arg name="has_lidar" default="false"/>
  <xacro:arg name="has_camera" default="true"/>
  <xacro:arg name="arm_dof" default="7"/>
  <xacro:arg name="leg_dof" default="6"/>

  <!-- Calculate derived parameters -->
  <xacro:property name="height" value="$(arg robot_height)"/>
  <xacro:property name="weight" value="$(arg robot_weight)"/>
  <xacro:property name="torso_height" value="${height * 0.4}"/>
  <xacro:property name="leg_length" value="${height * 0.45}"/>
  <xacro:property name="arm_length" value="${height * 0.35}"/>
  <xacro:property name="head_radius" value="${height * 0.08}"/>

  <!-- Mass distribution -->
  <xacro:property name="torso_mass" value="${weight * 0.4}"/>
  <xacro:property name="head_mass" value="${weight * 0.08}"/>
  <xacro:property name="arm_mass" value="${weight * 0.1}"/>
  <xacro:property name="leg_mass" value="${weight * 0.16}"/>

  <!-- Constants -->
  <xacro:property name="PI" value="3.1415926535897931"/>

  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0.0 0.0 0.15" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 ${torso_height}"/>
      </geometry>
      <material name="orange"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 ${torso_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${torso_mass}"/>
      <inertia ixx="${torso_mass * torso_height * torso_height / 12}"
               ixy="0.0" ixz="0.0"
               iyy="${torso_mass * 0.3 * 0.3 / 12}"
               iyz="0.0"
               izz="${torso_mass * 0.2 * 0.2 / 12}"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 ${torso_height/2 + head_radius}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-PI/4}" upper="${PI/4}" effort="100" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="${head_radius}"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="${head_radius}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${head_mass}"/>
      <inertia ixx="${0.4 * head_mass * head_radius * head_radius}"
               ixy="0.0" ixz="0.0"
               iyy="${0.4 * head_mass * head_radius * head_radius}"
               iyz="0.0"
               izz="${0.4 * head_mass * head_radius * head_radius}"/>
    </inertial>
  </link>

  <!-- Include macros -->
  <xacro:include filename="$(find humanoid_description)/urdf/humanoid_macros.xacro"/>

  <!-- Arms based on DOF parameter -->
  <xacro:if value="${arm_dof > 5}">
    <!-- More complex arm with additional joints -->
    <xacro:macro name="complex_arm" params="side parent xyz rpy color">
      <!-- Shoulder joint and link -->
      <joint name="${side}_shoulder_pitch_joint" type="revolute">
        <parent link="${parent}"/>
        <child link="${side}_shoulder"/>
        <origin xyz="${xyz}" rpy="${rpy}"/>
        <axis xyz="0 1 0"/>
        <limit lower="${-PI/2}" upper="${PI/2}" effort="100" velocity="1"/>
      </joint>

      <link name="${side}_shoulder">
        <visual>
          <geometry>
            <cylinder length="0.1" radius="0.05"/>
          </geometry>
          <material name="${color}"/>
        </visual>
        <collision>
          <geometry>
            <cylinder length="0.1" radius="0.05"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="${arm_mass * 0.1}"/>
          <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
        </inertial>
      </link>

      <joint name="${side}_shoulder_roll_joint" type="revolute">
        <parent link="${side}_shoulder"/>
        <child link="${side}_upper_arm"/>
        <origin xyz="0 0 0.05" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="${-PI/2}" upper="${PI/2}" effort="100" velocity="1"/>
      </joint>

      <link name="${side}_upper_arm">
        <visual>
          <geometry>
            <cylinder length="${arm_length * 0.4}" radius="0.05"/>
          </geometry>
          <material name="${color}"/>
          <origin xyz="0 0 ${arm_length * 0.2}" rpy="0 0 0"/>
        </visual>
        <collision>
          <geometry>
            <cylinder length="${arm_length * 0.4}" radius="0.05"/>
          </geometry>
          <origin xyz="0 0 ${arm_length * 0.2}" rpy="0 0 0"/>
        </collision>
        <inertial>
          <mass value="${arm_mass * 0.3}"/>
          <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
      </link>

      <joint name="${side}_elbow_joint" type="revolute">
        <parent link="${side}_upper_arm"/>
        <child link="${side}_forearm"/>
        <origin xyz="0 0 ${arm_length * 0.4}" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="${-PI/2}" upper="${PI/2}" effort="100" velocity="1"/>
      </joint>

      <link name="${side}_forearm">
        <visual>
          <geometry>
            <cylinder length="${arm_length * 0.35}" radius="0.04"/>
          </geometry>
          <material name="${color}"/>
          <origin xyz="0 0 ${arm_length * 0.175}" rpy="0 0 0"/>
        </visual>
        <collision>
          <geometry>
            <cylinder length="${arm_length * 0.35}" radius="0.04"/>
          </geometry>
          <origin xyz="0 0 ${arm_length * 0.175}" rpy="0 0 0"/>
        </collision>
        <inertial>
          <mass value="${arm_mass * 0.25}"/>
          <inertia ixx="0.007" ixy="0.0" ixz="0.0" iyy="0.007" iyz="0.0" izz="0.007"/>
        </inertial>
      </link>

      <joint name="${side}_wrist_pitch_joint" type="revolute">
        <parent link="${side}_forearm"/>
        <child link="${side}_wrist"/>
        <origin xyz="0 0 ${arm_length * 0.35}" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="${-PI/4}" upper="${PI/4}" effort="50" velocity="1"/>
      </joint>

      <link name="${side}_wrist">
        <visual>
          <geometry>
            <cylinder length="0.05" radius="0.03"/>
          </geometry>
          <material name="${color}"/>
        </visual>
        <collision>
          <geometry>
            <cylinder length="0.05" radius="0.03"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="0.2"/>
          <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
        </inertial>
      </link>

      <joint name="${side}_wrist_yaw_joint" type="revolute">
        <parent link="${side}_wrist"/>
        <child link="${side}_hand"/>
        <origin xyz="0 0 0.025" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="${-PI/3}" upper="${PI/3}" effort="30" velocity="1"/>
      </joint>

      <link name="${side}_hand">
        <visual>
          <geometry>
            <box size="0.1 0.08 0.05"/>
          </geometry>
          <material name="${color}"/>
        </visual>
        <collision>
          <geometry>
            <box size="0.1 0.08 0.05"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="0.3"/>
          <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
        </inertial>
      </link>
    </xacro:macro>

    <!-- Use complex arms -->
    <xacro:complex_arm side="left"
                      parent="torso"
                      xyz="${0.15} ${0.1} ${torso_height/4}"
                      rpy="0 0 0"
                      color="blue"/>
    <xacro:complex_arm side="right"
                      parent="torso"
                      xyz="${0.15} ${-0.1} ${torso_height/4}"
                      rpy="0 0 0"
                      color="blue"/>
  </xacro:if>

  <xacro:unless value="${arm_dof > 5}">
    <!-- Simple arms -->
    <xacro:arm side="left"
              parent="torso"
              xyz="${0.15} ${0.1} ${torso_height/4}"
              rpy="0 0 0"
              color="blue"/>
    <xacro:arm side="right"
              parent="torso"
              xyz="${0.15} ${-0.1} ${torso_height/4}"
              rpy="0 0 0"
              color="blue"/>
  </xacro:unless>

  <!-- Legs based on DOF parameter -->
  <xacro:if value="${leg_dof > 5}">
    <!-- More complex leg with additional joints -->
    <xacro:macro name="complex_leg" params="side parent xyz rpy color">
      <!-- Hip joint and link -->
      <joint name="${side}_hip_yaw_joint" type="revolute">
        <parent link="${parent}"/>
        <child link="${side}_hip"/>
        <origin xyz="${xyz}" rpy="${rpy}"/>
        <axis xyz="0 0 1"/>
        <limit lower="${-PI/6}" upper="${PI/6}" effort="200" velocity="1"/>
      </joint>

      <link name="${side}_hip">
        <visual>
          <geometry>
            <cylinder length="0.08" radius="0.06"/>
          </geometry>
          <material name="${color}"/>
        </visual>
        <collision>
          <geometry>
            <cylinder length="0.08" radius="0.06"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="${leg_mass * 0.1}"/>
          <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
        </inertial>
      </link>

      <joint name="${side}_hip_pitch_joint" type="revolute">
        <parent link="${side}_hip"/>
        <child link="${side}_upper_leg"/>
        <origin xyz="0 0 -0.04" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="${-PI/2}" upper="${PI/2}" effort="200" velocity="1"/>
      </joint>

      <link name="${side}_upper_leg">
        <visual>
          <geometry>
            <cylinder length="${leg_length * 0.45}" radius="0.06"/>
          </geometry>
          <material name="${color}"/>
          <origin xyz="0 0 ${-leg_length * 0.225}" rpy="0 0 0"/>
        </visual>
        <collision>
          <geometry>
            <cylinder length="${leg_length * 0.45}" radius="0.06"/>
          </geometry>
          <origin xyz="0 0 ${-leg_length * 0.225}" rpy="0 0 0"/>
        </collision>
        <inertial>
          <mass value="${leg_mass * 0.4}"/>
          <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
        </inertial>
      </link>

      <joint name="${side}_knee_joint" type="revolute">
        <parent link="${side}_upper_leg"/>
        <child link="${side}_lower_leg"/>
        <origin xyz="0 0 ${-leg_length * 0.45}" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="0" upper="${PI/2}" effort="200" velocity="1"/>
      </joint>

      <link name="${side}_lower_leg">
        <visual>
          <geometry>
            <cylinder length="${leg_length * 0.45}" radius="0.05"/>
          </geometry>
          <material name="${color}"/>
          <origin xyz="0 0 ${-leg_length * 0.225}" rpy="0 0 0"/>
        </visual>
        <collision>
          <geometry>
            <cylinder length="${leg_length * 0.45}" radius="0.05"/>
          </geometry>
          <origin xyz="0 0 ${-leg_length * 0.225}" rpy="0 0 0"/>
        </collision>
        <inertial>
          <mass value="${leg_mass * 0.35}"/>
          <inertia ixx="0.015" ixy="0.0" ixz="0.0" iyy="0.015" iyz="0.0" izz="0.015"/>
        </inertial>
      </link>

      <joint name="${side}_ankle_pitch_joint" type="revolute">
        <parent link="${side}_lower_leg"/>
        <child link="${side}_ankle"/>
        <origin xyz="0 0 ${-leg_length * 0.45}" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="${-PI/6}" upper="${PI/6}" effort="100" velocity="1"/>
      </joint>

      <link name="${side}_ankle">
        <visual>
          <geometry>
            <cylinder length="0.04" radius="0.05"/>
          </geometry>
          <material name="${color}"/>
        </visual>
        <collision>
          <geometry>
            <cylinder length="0.04" radius="0.05"/>
          </geometry>
        </collision>
        <inertial>
          <mass value="0.5"/>
          <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
        </inertial>
      </link>

      <joint name="${side}_ankle_roll_joint" type="revolute">
        <parent link="${side}_ankle"/>
        <child link="${side}_foot"/>
        <origin xyz="0 0 -0.02" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="${-PI/6}" upper="${PI/6}" effort="100" velocity="1"/>
      </joint>

      <link name="${side}_foot">
        <visual>
          <geometry>
            <box size="0.2 0.1 0.05"/>
          </geometry>
          <material name="${color}"/>
          <origin xyz="0 0 0.025" rpy="0 0 0"/>
        </visual>
        <collision>
          <geometry>
            <box size="0.2 0.1 0.05"/>
          </geometry>
          <origin xyz="0 0 0.025" rpy="0 0 0"/>
        </collision>
        <inertial>
          <mass value="1.0"/>
          <inertia ixx="0.003" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.003"/>
        </inertial>
      </link>
    </xacro:macro>

    <!-- Use complex legs -->
    <xacro:complex_leg side="left"
                      parent="base_link"
                      xyz="${-0.1} ${0.05} -0.1"
                      rpy="0 0 0"
                      color="green"/>
    <xacro:complex_leg side="right"
                      parent="base_link"
                      xyz="${-0.1} ${-0.05} -0.1"
                      rpy="0 0 0"
                      color="green"/>
  </xacro:if>

  <xacro:unless value="${leg_dof > 5}">
    <!-- Simple legs -->
    <xacro:leg side="left"
              parent="base_link"
              xyz="${-0.1} ${0.05} -0.1"
              rpy="0 0 0"
              color="green"/>
    <xacro:leg side="right"
              parent="base_link"
              xyz="${-0.1} ${-0.05} -0.1"
              rpy="0 0 0"
              color="green"/>
  </xacro:unless>

  <!-- Add sensors based on parameters -->
  <xacro:if value="$(arg has_camera)">
    <xacro:camera_sensor name="head_camera"
                         parent="head"
                         xyz="0.05 0 0"
                         rpy="0 0 0"/>
  </xacro:if>

  <xacro:if value="$(arg has_lidar)">
    <joint name="lidar_joint" type="fixed">
      <parent link="head"/>
      <child link="lidar_link"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
    </joint>

    <link name="lidar_link">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.04"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.04"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <gazebo reference="lidar_link">
      <sensor name="head_lidar" type="ray">
        <always_on>true</always_on>
        <update_rate>10</update_rate>
        <ray>
          <scan>
            <horizontal>
              <samples>360</samples>
              <resolution>1</resolution>
              <min_angle>-3.14159</min_angle>
              <max_angle>3.14159</max_angle>
            </horizontal>
          </scan>
          <range>
            <min>0.1</min>
            <max>30.0</max>
            <resolution>0.01</resolution>
          </range>
        </ray>
        <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
          <ros>
            <remapping>~/out:=scan</remapping>
          </ros>
          <output_type>sensor_msgs/LaserScan</output_type>
        </plugin>
      </sensor>
    </gazebo>
  </xacro:if>

  <!-- ROS Control interface -->
  <ros2_control name="GazeboSystem" type="system">
    <hardware>
      <plugin>gazebo_ros2_control/GazeboSystem</plugin>
    </hardware>
    <joint name="neck_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="left_shoulder_pitch_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="left_shoulder_roll_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="left_elbow_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="right_shoulder_pitch_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="right_shoulder_roll_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="right_elbow_joint">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>

  <!-- Gazebo plugins -->
  <gazebo reference="base_link">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="torso">
    <material>Gazebo/Orange</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

</robot>
```

2. **Create a launch file to visualize the robot:**

Create `~/ros2_ws/src/humanoid_description/launch/view_robot.launch.py`:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    height_arg = DeclareLaunchArgument(
        'robot_height',
        default_value='1.7',
        description='Height of the robot in meters'
    )

    weight_arg = DeclareLaunchArgument(
        'robot_weight',
        default_value='70.0',
        description='Weight of the robot in kg'
    )

    has_camera_arg = DeclareLaunchArgument(
        'has_camera',
        default_value='true',
        description='Whether the robot has a camera'
    )

    has_lidar_arg = DeclareLaunchArgument(
        'has_lidar',
        default_value='false',
        description='Whether the robot has a lidar'
    )

    # Get launch configuration
    robot_height = LaunchConfiguration('robot_height')
    robot_weight = LaunchConfiguration('robot_weight')
    has_camera = LaunchConfiguration('has_camera')
    has_lidar = LaunchConfiguration('has_lidar')

    # Get URDF file path
    urdf_path = os.path.join(
        get_package_share_directory('humanoid_description'),
        'urdf',
        'parametric_humanoid.xacro'
    )

    # Define robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': f'$(find xacro)/xacro {urdf_path} robot_height:={robot_height} robot_weight:={robot_weight} has_camera:={has_camera} has_lidar:={has_lidar}'
        }]
    )

    # Define joint state publisher GUI node
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )

    # Define RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory('humanoid_description'), 'rviz', 'urdf.rviz')]
    )

    return LaunchDescription([
        height_arg,
        weight_arg,
        has_camera_arg,
        has_lidar_arg,
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz_node
    ])
```

3. **Create an RViz configuration file:**

Create the RViz directory and config:

```bash
mkdir -p ~/ros2_ws/src/humanoid_description/rviz
```

Create `~/ros2_ws/src/humanoid_description/rviz/urdf.rviz`:

```yaml
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
        - /RobotModel1/Links1
      Splitter Ratio: 0.5
    Tree Height: 606
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description File: ""
      Description Source: Topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
      Line color: 128; 128; 0
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Name: Current View
      Target Frame: base_link
      Value: Orbit (rviz)
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 846
  Hide Left Dock: false
  Hide Right Dock: true
  QMainWindow State: 000000ff00000000fd000000040000000000000156000002f4fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d000002f4000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000002f4fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073000000003d000002f4000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004420000003efc0100000002fb0000000800540069006d00650100000000000004420000000000000000fb0000000800540069006d00650100000000000004500000000000000000000003a0000002f400000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Width: 1200
  X: 60
  Y: 60