#!/usr/bin/env python3
"""
Voice Processing Demo for VLA System

This script demonstrates the voice processing pipeline for the Vision-Language-Action system.
It shows how voice commands are captured, processed through Whisper, and converted to actions.
"""

import asyncio
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import sounddevice as sd
import queue

# Import our voice command model
from scripts.voice_control.voice_command_model import VoiceCommand, CommandTypes, VoiceCommandStatus, Location, Orientation, Environment, Context, Obstacle


class MockWhisperService:
    """
    Mock service to simulate Whisper model functionality.
    In a real implementation, this would interface with Whisper.cpp
    """

    def __init__(self):
        self.command_mapping = {
            "go to the kitchen": ("Navigate to the kitchen", CommandTypes.NAVIGATION),
            "pick up the red cup": ("Grasp the red cup", CommandTypes.MANIPULATION),
            "find the blue box": ("Detect the blue box", CommandTypes.PERCEPTION),
            "hello robot": ("Greet the user", CommandTypes.COMMUNICATION),
            "move forward": ("Move forward", CommandTypes.NAVIGATION),
            "turn left": ("Turn left", CommandTypes.NAVIGATION),
            "grasp the object": ("Grasp the object", CommandTypes.MANIPULATION),
            "detect obstacles": ("Detect obstacles", CommandTypes.PERCEPTION),
        }

    def transcribe(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Simulate Whisper transcription of audio data.

        Args:
            audio_data: Audio data to transcribe

        Returns:
            Dictionary with transcription results
        """
        # In a real implementation, this would call Whisper.cpp
        # For this demo, we'll simulate based on predefined commands

        # Simulate processing time
        time.sleep(0.5)

        # Simulate some audio analysis to determine which command was said
        # In a real implementation, this would be the actual Whisper output
        command_text = "go to the kitchen"  # Default simulation
        confidence = 0.92  # Simulated confidence

        # Return mock transcription result
        return {
            "text": command_text,
            "confidence": confidence,
            "command_type": self.command_mapping.get(command_text.lower(), (command_text, CommandTypes.NAVIGATION))[1]
        }


class MockAudioInputHandler:
    """
    Mock audio input handler to simulate capturing audio from microphone.
    """

    def __init__(self):
        self.sample_rate = 16000  # Standard sample rate for Whisper
        self.duration = 3  # Record for 3 seconds
        self.audio_queue = queue.Queue()

    def record_audio(self) -> np.ndarray:
        """
        Record audio from the microphone.

        Returns:
            numpy array containing audio data
        """
        print("Recording audio for 3 seconds...")

        # Record audio using sounddevice
        audio_data = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to complete

        print("Audio recording complete.")
        return audio_data.flatten()


class VoiceProcessingDemo:
    """
    Demo class that demonstrates the complete voice processing pipeline.
    """

    def __init__(self):
        self.whisper_service = MockWhisperService()
        self.audio_handler = MockAudioInputHandler()
        self.command_id_counter = 1

    def create_sample_context(self) -> Context:
        """
        Create a sample context for the voice command.

        Returns:
            Context object with sample environment data
        """
        location = Location(x=0.0, y=0.0, z=0.0)
        orientation = Orientation(roll=0.0, pitch=0.0, yaw=0.0)
        environment = Environment(
            room_layout="living_room",
            obstacles=[
                Obstacle(
                    id="chair_1",
                    position=Location(x=1.0, y=1.0, z=0.0),
                    size={"width": 0.5, "height": 0.8, "depth": 0.5},
                    type="chair"
                ),
                Obstacle(
                    id="table_1",
                    position=Location(x=2.0, y=0.0, z=0.0),
                    size={"width": 1.0, "height": 0.7, "depth": 0.6},
                    type="table"
                )
            ]
        )
        return Context(
            location=location,
            orientation=orientation,
            environment=environment
        )

    def process_voice_command(self) -> Optional[VoiceCommand]:
        """
        Process a voice command through the complete pipeline.

        Returns:
            VoiceCommand object if successful, None otherwise
        """
        print("\n=== Voice Command Processing Demo ===")

        # Step 1: Capture audio
        print("\n1. Capturing audio input...")
        audio_data = self.audio_handler.record_audio()
        print(f"   Captured {len(audio_data)} audio samples")

        # Step 2: Transcribe using Whisper
        print("\n2. Processing with Whisper model...")
        transcription_result = self.whisper_service.transcribe(audio_data)
        print(f"   Transcribed text: '{transcription_result['text']}'")
        print(f"   Confidence: {transcription_result['confidence']:.2f}")
        print(f"   Command type: {transcription_result['command_type'].value}")

        # Step 3: Create VoiceCommand object
        print("\n3. Creating VoiceCommand object...")
        command_id = f"cmd_{self.command_id_counter:03d}"
        self.command_id_counter += 1

        context = self.create_sample_context()

        try:
            voice_command = VoiceCommand(
                id=command_id,
                text=transcription_result['text'],
                command_type=transcription_result['command_type'],
                timestamp=datetime.now(),
                confidence=transcription_result['confidence'],
                source="voice",
                status=VoiceCommandStatus.PENDING,
                context=context
            )
            print(f"   Created VoiceCommand with ID: {voice_command.id}")

            # Step 4: Update status to processing
            print("\n4. Updating command status...")
            voice_command.update_status(VoiceCommandStatus.PROCESSING)
            print(f"   Status updated to: {voice_command.status.value}")

            # Step 5: Simulate command execution
            print("\n5. Simulating command execution...")
            self.execute_command(voice_command)

            # Step 6: Update status to completed
            voice_command.update_status(VoiceCommandStatus.COMPLETED)
            print(f"   Status updated to: {voice_command.status.value}")

            return voice_command

        except ValueError as e:
            print(f"   Error creating VoiceCommand: {e}")
            if 'voice_command' in locals():
                voice_command.update_status(VoiceCommandStatus.FAILED)
                voice_command.error_message = str(e)
            return None

    def execute_command(self, voice_command: VoiceCommand):
        """
        Simulate command execution based on command type.

        Args:
            voice_command: The voice command to execute
        """
        print(f"   Executing {voice_command.command_type.value} command: '{voice_command.text}'")

        # Simulate different execution based on command type
        if voice_command.command_type == CommandTypes.NAVIGATION:
            print("   - Planning navigation path...")
            print("   - Moving to target location...")
            time.sleep(1)  # Simulate movement time
            print("   - Navigation completed")

        elif voice_command.command_type == CommandTypes.MANIPULATION:
            print("   - Identifying target object...")
            print("   - Planning grasp trajectory...")
            print("   - Executing manipulation...")
            time.sleep(1)  # Simulate manipulation time
            print("   - Manipulation completed")

        elif voice_command.command_type == CommandTypes.PERCEPTION:
            print("   - Activating perception sensors...")
            print("   - Processing visual data...")
            print("   - Detecting objects...")
            time.sleep(1)  # Simulate perception time
            print("   - Perception completed")

        elif voice_command.command_type == CommandTypes.COMMUNICATION:
            print("   - Processing communication request...")
            print("   - Generating response...")
            time.sleep(0.5)  # Simulate response time
            print("   - Communication completed")

    def run_demo(self):
        """
        Run the complete voice processing demo.
        """
        print("Starting Voice Processing Demo for VLA System")
        print("=" * 50)

        # Process multiple commands to demonstrate the pipeline
        for i in range(2):  # Process 2 commands for the demo
            print(f"\n--- Processing Command {i+1} ---")
            voice_command = self.process_voice_command()

            if voice_command:
                print(f"\nVoice Command Summary:")
                print(f"  ID: {voice_command.id}")
                print(f"  Text: {voice_command.text}")
                print(f"  Type: {voice_command.command_type.value}")
                print(f"  Confidence: {voice_command.confidence:.2f}")
                print(f"  Status: {voice_command.status.value}")
                print(f"  Timestamp: {voice_command.timestamp}")
            else:
                print("Command processing failed.")

            if i < 1:  # Don't wait after the last command
                print(f"\nWaiting before next command...")
                time.sleep(2)

        print("\n" + "=" * 50)
        print("Voice Processing Demo completed!")


def main():
    """
    Main function to run the voice processing demo.
    """
    demo = VoiceProcessingDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()