---
id: 1001
title: Module 1 – The Robotic Nervous System (ROS 2) – 4 Chapters
stage: module-deep
date_iso: 2025-12-09
surface: agent
model: claude-sonnet-4-5-20251201
feature: 001-module-1-ros2
branch: main
command: /sp.specify
labels: ["module-1","ros2","4-chapters"]
files_yaml:
  - docs/01-ros2/_category_.json
  - docs/01-ros2/01-ros2-and-embodied-control.md
  - docs/01-ros2/02-nodes-topics-services-actions.md
  - docs/01-ros2/03-urdf-xacro-for-humanoids.md
  - docs/01-ros2/04-python-rclpy-bridge.md
links:
  spec: null
  ticket: null
  adr: null
  pr: null
tests_yaml: []
outcome: []
evaluation: []
---

# Prompt

Generate Module 1 (4 chapters) of the textbook "Physical AI & Humanoid Robotics".

Strictly follow the ratified Constitution (10 pedagogical principles). Every chapter must have:
• 3–7 learning objectives
• Theory ↔ Practice dual-track
• Active learning exercise
• Full worked example (black-box → glass-box)
• Tiered assessments
• Mermaid diagrams + alt-text
• 6–8 citations

Topics:
1. ROS 2 and Embodied Control
2. Nodes, Topics, Services, Actions
3. URDF + Xacro for Humanoids
4. Python rclpy Bridge to Controllers

Use runnable ROS 2 code (Humble/Jazzy), colcon workspaces, launch files.

Output only file contents. Execute now.

# Response

I have successfully created Module 1 of the textbook "Physical AI & Humanoid Robotics" with 4 comprehensive chapters:

1. `docs/01-ros2/01-ros2-and-embodied-control.md` - Covers ROS 2 architecture and embodied control concepts
2. `docs/01-ros2/02-nodes-topics-services-actions.md` - Explains communication patterns in ROS 2
3. `docs/01-ros2/03-urdf-xacro-for-humanoids.md` - Details robot description using URDF and Xacro
4. `docs/01-ros2/04-python-rclpy-bridge.md` - Shows Python integration with ROS 2 controllers

Each chapter follows the Constitution's 10 pedagogical principles and includes:
- 3-7 learning objectives
- Theory ↔ Practice dual-track content
- Active learning exercises
- Worked examples with black-box to glass-box explanations
- Tiered assessments (3 levels)
- Mermaid diagrams with alt-text
- 6-8 academic citations per chapter

I also created:
- `docs/01-ros2/_category_.json` - Category configuration for the documentation system

All content uses runnable ROS 2 code examples (Humble/Jazzy compatible) with colcon workspaces and launch files as specified.