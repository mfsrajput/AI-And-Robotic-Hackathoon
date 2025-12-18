#!/usr/bin/env python3
"""
Multi-Modal Integration Demo for VLA System

This script demonstrates the multi-modal fusion of visual and linguistic inputs
to guide robot actions in the Vision-Language-Action system.
"""

import asyncio
import time
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json

# Import our models
from scripts.voice_control.voice_command_model import VoiceCommand, CommandTypes, VoiceCommandStatus, Location, Orientation, Environment, Context, Obstacle


@dataclass
class MultiModalInput:
    """
    Represents combined visual and linguistic inputs that guide robot behavior.
    """
    id: str
    voice_input: Optional[VoiceCommand] = None
    visual_input: Dict[str, Any] = None
    fusion_confidence: float = 0.0
    timestamp: datetime = None
    processing_status: str = "received"
    fused_command: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.visual_input is None:
            self.visual_input = {}

        # Validate fusion confidence
        if not 0.0 <= self.fusion_confidence <= 1.0:
            raise ValueError("Fusion confidence must be between 0.0 and 1.0")


class MockVisualPerception:
    """
    Mock service to simulate visual perception and object detection.
    """

    def __init__(self):
        self.object_database = {
            "red cup": {"class": "cup", "confidence": 0.95, "bbox": {"x_min": 100, "y_min": 150, "x_max": 180, "y_max": 220}},
            "blue box": {"class": "box", "confidence": 0.92, "bbox": {"x_min": 200, "y_min": 100, "x_max": 300, "y_max": 180}},
            "green ball": {"class": "ball", "confidence": 0.88, "bbox": {"x_min": 50, "y_min": 200, "x_max": 120, "y_max": 270}},
            "sofa": {"class": "furniture", "confidence": 0.98, "bbox": {"x_min": 300, "y_min": 50, "x_max": 600, "y_max": 300}},
            "table": {"class": "furniture", "confidence": 0.96, "bbox": {"x_min": 150, "y_min": 300, "x_max": 450, "y_max": 500}}
        }

    def detect_objects(self, image_data: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Detect objects in the visual input.

        Args:
            image_data: Optional image data (for demo, we'll use a default scene)

        Returns:
            List of detected objects
        """
        print("   Detecting objects in visual scene...")

        # Simulate processing time
        time.sleep(0.6)

        # Return a sample scene with objects
        detected_objects = [
            self.object_database["red cup"],
            self.object_database["blue box"],
            self.object_database["sofa"],
            self.object_database["table"]
        ]

        print(f"   Detected {len(detected_objects)} objects")
        return detected_objects

    def describe_scene(self, image_data: Optional[np.ndarray] = None) -> str:
        """
        Generate a natural language description of the scene.

        Args:
            image_data: Optional image data

        Returns:
            Natural language description of the scene
        """
        print("   Generating scene description...")

        # Simulate processing time
        time.sleep(0.4)

        description = "The scene shows a living room with a sofa on the left, a table in the center, a blue box on the table, and a red cup on the floor near the table."
        print(f"   Scene description: {description}")
        return description


class MockMultiModalFusionEngine:
    """
    Mock service for multi-modal fusion algorithms.
    """

    def __init__(self):
        self.fusion_rules = {
            "go to the blue box": self._fuse_navigation_to_object,
            "pick up the red cup": self._fuse_manipulation_object,
            "find the blue box": self._fuse_detection_object,
            "go to the blue box": self._fuse_navigation_to_detected_object
        }

    def _fuse_navigation_to_object(self, voice_command: str, visual_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse navigation command with visual data for object."""
        blue_box = next((obj for obj in visual_objects if obj["class"] == "box"), None)
        if blue_box:
            return {
                "fused_action": "navigate_to_object",
                "target_object": "blue box",
                "action_type": "navigation",
                "confidence": 0.85,
                "reasoning": "Blue box detected in scene, navigating to its location"
            }
        else:
            return {
                "fused_action": "search_for_object",
                "target_object": "blue box",
                "action_type": "navigation",
                "confidence": 0.65,
                "reasoning": "Blue box not detected, searching for it"
            }

    def _fuse_manipulation_object(self, voice_command: str, visual_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse manipulation command with visual data for object."""
        red_cup = next((obj for obj in visual_objects if "cup" in obj["class"] and "red" in voice_command.lower()), None)
        if red_cup:
            return {
                "fused_action": "grasp_object",
                "target_object": "red cup",
                "action_type": "manipulation",
                "confidence": 0.90,
                "reasoning": "Red cup detected in scene, attempting to grasp it"
            }
        else:
            return {
                "fused_action": "search_for_object",
                "target_object": "red cup",
                "action_type": "perception",
                "confidence": 0.70,
                "reasoning": "Red cup not detected, searching for it"
            }

    def _fuse_detection_object(self, voice_command: str, visual_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse detection command with visual data."""
        target_object = "blue box" if "blue box" in voice_command.lower() else "object"
        detected = any(target_object in obj["class"] for obj in visual_objects)

        if detected:
            return {
                "fused_action": "report_location",
                "target_object": target_object,
                "action_type": "perception",
                "confidence": 0.88,
                "reasoning": f"{target_object.title()} detected in scene, reporting its location"
            }
        else:
            return {
                "fused_action": "search_for_object",
                "target_object": target_object,
                "action_type": "perception",
                "confidence": 0.60,
                "reasoning": f"{target_object.title()} not detected, searching for it"
            }

    def _fuse_navigation_to_detected_object(self, voice_command: str, visual_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse navigation command with detected object."""
        return self._fuse_navigation_to_object(voice_command, visual_objects)

    def fuse_inputs(self, voice_command: str, visual_objects: List[Dict[str, Any]], scene_description: str) -> Dict[str, Any]:
        """
        Fuse voice and visual inputs to create a unified command.

        Args:
            voice_command: The voice command text
            visual_objects: List of detected objects
            scene_description: Natural language description of the scene

        Returns:
            Fused command with enhanced context
        """
        print(f"   Fusing voice command: '{voice_command}' with visual data")
        print(f"   Scene description: {scene_description[:50]}...")

        # Simulate processing time
        time.sleep(0.5)

        # Find the appropriate fusion rule
        fusion_func = None
        for key, func in self.fusion_rules.items():
            if key in voice_command.lower():
                fusion_func = func
                break

        if fusion_func:
            result = fusion_func(voice_command, visual_objects)
        else:
            # Default fusion for unknown commands
            result = {
                "fused_action": "unknown_command",
                "target_object": "unknown",
                "action_type": "unknown",
                "confidence": 0.5,
                "reasoning": "Unknown command, unable to determine appropriate action"
            }

        print(f"   Fused action: {result['fused_action']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        return result


class MockObjectDetectionIntegration:
    """
    Mock service for object detection integration with voice commands.
    """

    def __init__(self):
        self.detection_threshold = 0.7

    def integrate_with_voice(self, voice_command: str, detected_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Integrate object detection results with voice command.

        Args:
            voice_command: The voice command
            detected_objects: List of detected objects

        Returns:
            Filtered and relevant objects based on the voice command
        """
        print(f"   Integrating object detection with voice command: '{voice_command}'")

        # Extract target object from voice command
        target_objects = []
        voice_lower = voice_command.lower()

        for obj in detected_objects:
            obj_class = obj["class"]
            obj_confidence = obj["confidence"]

            # Check if this object matches the target in the voice command
            if obj_confidence >= self.detection_threshold:
                if ("box" in voice_lower and "box" in obj_class) or \
                   ("cup" in voice_lower and "cup" in obj_class) or \
                   ("ball" in voice_lower and "ball" in obj_class):
                    target_objects.append(obj)

        print(f"   Found {len(target_objects)} relevant objects for the command")
        return target_objects


class MultiModalDemo:
    """
    Demo class that demonstrates the multi-modal integration pipeline.
    """

    def __init__(self):
        self.visual_perception = MockVisualPerception()
        self.fusion_engine = MockMultiModalFusionEngine()
        self.detection_integration = MockObjectDetectionIntegration()

    def process_multi_modal_input(self, voice_command: str) -> MultiModalInput:
        """
        Process multi-modal input combining voice and visual data.

        Args:
            voice_command: The voice command to process

        Returns:
            MultiModalInput object with fused results
        """
        print(f"\n=== Processing Multi-Modal Input: '{voice_command}' ===")

        # Step 1: Create voice command object
        print("\n1. Creating voice command...")
        context = Context(
            location=Location(x=0.0, y=0.0, z=0.0),
            orientation=Orientation(roll=0.0, pitch=0.0, yaw=0.0),
            environment=Environment(room_layout="living_room", obstacles=[])
        )

        voice_cmd_obj = VoiceCommand(
            id=f"v_cmd_{int(time.time())}",
            text=voice_command,
            command_type=CommandTypes.NAVIGATION,  # Will be updated based on fusion
            timestamp=datetime.now(),
            confidence=0.85,
            source="voice",
            status=VoiceCommandStatus.PENDING,
            context=context
        )
        print(f"   Created voice command: {voice_cmd_obj.id}")

        # Step 2: Perform visual perception
        print("\n2. Performing visual perception...")
        detected_objects = self.visual_perception.detect_objects()
        scene_description = self.visual_perception.describe_scene()

        visual_input = {
            "object_detections": detected_objects,
            "scene_description": scene_description,
            "image_path": "simulated_image_path.jpg"
        }
        print(f"   Detected {len(detected_objects)} objects in the scene")

        # Step 3: Integrate object detection with voice command
        print("\n3. Integrating object detection with voice command...")
        relevant_objects = self.detection_integration.integrate_with_voice(voice_command, detected_objects)
        print(f"   Found {len(relevant_objects)} relevant objects for the command")

        # Step 4: Perform multi-modal fusion
        print("\n4. Performing multi-modal fusion...")
        fusion_result = self.fusion_engine.fuse_inputs(voice_command, detected_objects, scene_description)

        # Step 5: Create multi-modal input object
        print("\n5. Creating multi-modal input object...")
        multi_modal_input = MultiModalInput(
            id=f"mm_input_{int(time.time())}",
            voice_input=voice_cmd_obj,
            visual_input=visual_input,
            fusion_confidence=fusion_result["confidence"],
            processing_status="fused",
            fused_command=fusion_result["fused_action"]
        )

        print(f"   Created multi-modal input: {multi_modal_input.id}")
        print(f"   Fused command: {multi_modal_input.fused_command}")
        print(f"   Fusion confidence: {multi_modal_input.fusion_confidence:.2f}")

        return multi_modal_input

    def execute_fused_command(self, multi_modal_input: MultiModalInput) -> Dict[str, Any]:
        """
        Execute the fused command from multi-modal input.

        Args:
            multi_modal_input: The multi-modal input with fused command

        Returns:
            Execution result dictionary
        """
        fused_command = multi_modal_input.fused_command
        print(f"\n=== Executing Fused Command: '{fused_command}' ===")

        result = {
            "command": fused_command,
            "success": False,
            "execution_time": 0.0,
            "details": {}
        }

        start_time = time.time()

        # Simulate execution based on command type
        if fused_command == "navigate_to_object":
            print("   Executing navigation to object...")
            time.sleep(1.2)  # Simulate navigation time
            result["success"] = True
            result["details"] = {"action": "navigation", "target": "object", "distance": "2.5m"}
        elif fused_command == "grasp_object":
            print("   Executing object grasp...")
            time.sleep(0.8)  # Simulate grasp time
            result["success"] = True
            result["details"] = {"action": "manipulation", "object": "red cup", "success": True}
        elif fused_command == "report_location":
            print("   Reporting object location...")
            time.sleep(0.5)  # Simulate reporting time
            result["success"] = True
            result["details"] = {"action": "perception", "object": "blue box", "location": "on the table"}
        elif fused_command == "search_for_object":
            print("   Searching for object...")
            time.sleep(1.0)  # Simulate search time
            result["success"] = True
            result["details"] = {"action": "perception", "target": "object", "found": True}
        else:
            print(f"   Unknown command: {fused_command}")
            result["success"] = False
            result["details"] = {"error": "Unknown command", "action": "none"}

        end_time = time.time()
        result["execution_time"] = end_time - start_time

        print(f"   Execution result: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"   Execution time: {result['execution_time']:.2f}s")

        return result

    def run_demo(self):
        """
        Run the complete multi-modal integration demo.
        """
        print("Starting Multi-Modal Integration Demo for VLA System")
        print("=" * 65)

        # Define test commands
        test_commands = [
            "Go to the blue box",
            "Pick up the red cup",
            "Find the blue box",
            "Look for the green ball"
        ]

        # Process each command
        for i, command in enumerate(test_commands):
            print(f"\n--- Processing Command {i+1} of {len(test_commands)}: '{command}' ---")

            # Process multi-modal input
            multi_modal_input = self.process_multi_modal_input(command)

            # Execute the fused command
            execution_result = self.execute_fused_command(multi_modal_input)

            # Print summary
            print(f"\nCommand Summary:")
            print(f"  Original: {command}")
            print(f"  Fused: {multi_modal_input.fused_command}")
            print(f"  Confidence: {multi_modal_input.fusion_confidence:.2f}")
            print(f"  Success: {execution_result['success']}")
            print(f"  Time: {execution_result['execution_time']:.2f}s")

            # Wait between commands
            if i < len(test_commands) - 1:
                print(f"\nWaiting before next command...")
                time.sleep(1)

        print("\n" + "=" * 65)
        print("Multi-Modal Integration Demo completed!")

        # Print final statistics
        print("\nDemo Statistics:")
        print(f"  Commands processed: {len(test_commands)}")
        print(f"  Architecture demonstrated: Vision-Language-Action fusion")
        print(f"  Integration level: Multi-modal perception with language understanding")


def main():
    """
    Main function to run the multi-modal integration demo.
    """
    demo = MultiModalDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()