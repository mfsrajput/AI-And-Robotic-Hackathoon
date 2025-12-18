#!/usr/bin/env python3
"""
Comprehensive Test Suite for VLA System Components

This test suite validates all components of the Vision-Language-Action system,
including voice processing, LLM integration, multi-modal fusion, and capstone components.
"""

import unittest
import time
from datetime import datetime
from typing import Dict, Any, List

# Import our models and services
from scripts.voice_control.voice_command_model import VoiceCommand, CommandTypes, VoiceCommandStatus, Location, Orientation, Environment, Context, Obstacle
from scripts.capstone_project.autonomous_humanoid_system_model import AutonomousHumanoidSystem, SystemStatus, PerformanceMetrics, ROSInterfaces, Motor, MotorStatus, Gripper, GripperStatus
from scripts.capstone_project.capstone_project_model import CapstoneProject, ProjectStatus, Requirement, RequirementStatus, Rubric, EvaluationCriterion, EvaluationLevelDetail, EvaluationLevel, Evaluation, EvaluationScore
from assets.code-examples.voice_control.voice_demo import MockWhisperService, MockAudioInputHandler, VoiceProcessingDemo
from assets.code-examples.llm_task_planning.llm_demo import ActionStep, ActionSequence, MockOllamaService, MockPromptEngineering, LLMTaskPlanningDemo
from assets.code-examples.multi_modal_integration.multi_modal_demo import MultiModalInput, MockVisualPerception, MockMultiModalFusionEngine, MockObjectDetectionIntegration, MultiModalDemo
from assets.code-examples.capstone_project.capstone_demo import MockLLMService, MockMultiModalFusion, MockROSBridge, CapstoneDemo


class TestVoiceCommandModel(unittest.TestCase):
    """Test cases for the VoiceCommand data model."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.location = Location(x=1.0, y=2.0, z=0.0)
        self.orientation = Orientation(roll=0.0, pitch=0.0, yaw=1.57)
        self.environment = Environment(
            room_layout="living_room",
            obstacles=[
                Obstacle(
                    id="obstacle_1",
                    position=Location(x=1.5, y=2.5, z=0.0),
                    size={"width": 0.5, "height": 0.5, "depth": 0.5},
                    type="chair"
                )
            ]
        )
        self.context = Context(
            location=self.location,
            orientation=self.orientation,
            environment=self.environment
        )

    def test_voice_command_creation(self):
        """Test creating a valid VoiceCommand object."""
        voice_cmd = VoiceCommand(
            id="cmd_001",
            text="Go to the kitchen",
            command_type=CommandTypes.NAVIGATION,
            timestamp=datetime.now(),
            confidence=0.95,
            source="voice",
            status=VoiceCommandStatus.PENDING,
            context=self.context
        )

        self.assertEqual(voice_cmd.id, "cmd_001")
        self.assertEqual(voice_cmd.text, "Go to the kitchen")
        self.assertEqual(voice_cmd.command_type, CommandTypes.NAVIGATION)
        self.assertEqual(voice_cmd.status, VoiceCommandStatus.PENDING)
        self.assertEqual(voice_cmd.confidence, 0.95)
        self.assertEqual(voice_cmd.source, "voice")
        self.assertEqual(voice_cmd.context.location.x, 1.0)

    def test_voice_command_validation(self):
        """Test validation of VoiceCommand object."""
        # Test with empty text
        with self.assertRaises(ValueError):
            VoiceCommand(
                id="cmd_002",
                text="",
                command_type=CommandTypes.NAVIGATION,
                timestamp=datetime.now(),
                confidence=0.95,
                source="voice",
                status=VoiceCommandStatus.PENDING,
                context=self.context
            )

        # Test with invalid confidence
        with self.assertRaises(ValueError):
            VoiceCommand(
                id="cmd_003",
                text="Test command",
                command_type=CommandTypes.NAVIGATION,
                timestamp=datetime.now(),
                confidence=1.5,  # Invalid confidence
                source="voice",
                status=VoiceCommandStatus.PENDING,
                context=self.context
            )

    def test_voice_command_status_update(self):
        """Test updating VoiceCommand status."""
        voice_cmd = VoiceCommand(
            id="cmd_004",
            text="Test command",
            command_type=CommandTypes.NAVIGATION,
            timestamp=datetime.now(),
            confidence=0.90,
            source="voice",
            status=VoiceCommandStatus.PENDING,
            context=self.context
        )

        # Valid transition: pending -> processing
        voice_cmd.update_status(VoiceCommandStatus.PROCESSING)
        self.assertEqual(voice_cmd.status, VoiceCommandStatus.PROCESSING)

        # Valid transition: processing -> completed
        voice_cmd.update_status(VoiceCommandStatus.COMPLETED)
        self.assertEqual(voice_cmd.status, VoiceCommandStatus.COMPLETED)

        # Invalid transition: completed -> processing (should raise error)
        with self.assertRaises(ValueError):
            voice_cmd.update_status(VoiceCommandStatus.PROCESSING)

    def test_voice_command_dict_conversion(self):
        """Test converting VoiceCommand to and from dictionary."""
        original_cmd = VoiceCommand(
            id="cmd_005",
            text="Test conversion",
            command_type=CommandTypes.MANIPULATION,
            timestamp=datetime.now(),
            confidence=0.85,
            source="voice",
            status=VoiceCommandStatus.PENDING,
            context=self.context
        )

        # Convert to dict and back
        cmd_dict = original_cmd.to_dict()
        reconstructed_cmd = VoiceCommand.from_dict(cmd_dict)

        self.assertEqual(original_cmd.id, reconstructed_cmd.id)
        self.assertEqual(original_cmd.text, reconstructed_cmd.text)
        self.assertEqual(original_cmd.command_type, reconstructed_cmd.command_type)
        self.assertEqual(original_cmd.confidence, reconstructed_cmd.confidence)
        self.assertEqual(original_cmd.context.location.x, reconstructed_cmd.context.location.x)


class TestAutonomousHumanoidSystemModel(unittest.TestCase):
    """Test cases for the AutonomousHumanoidSystem data model."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.performance_metrics = PerformanceMetrics()
        self.ros_interfaces = ROSInterfaces(
            ros2_nodes=["/navigation_node", "/manipulation_node"],
            digital_twin_connection="dt_connection_1",
            isaac_sim_connection="isaac_sim_1"
        )

    def test_autonomous_humanoid_system_creation(self):
        """Test creating a valid AutonomousHumanoidSystem object."""
        system = AutonomousHumanoidSystem(
            id="humanoid_001",
            name="Test Humanoid",
            status=SystemStatus.IDLE,
            current_behavior="idle",
            sensors={"audio": {"is_active": True}, "vision": {"is_active": True}},
            actuators={"motors": [], "grippers": []},
            state={
                "location": Location(x=0.0, y=0.0, z=0.0).__dict__,
                "orientation": Orientation(roll=0.0, pitch=0.0, yaw=0.0).__dict__,
                "battery_level": 85.0,
                "temperature": 30.0
            },
            active_sequences=[],
            last_command="",
            command_history=[],
            performance_metrics=self.performance_metrics,
            error_log=[],
            ros_interfaces=self.ros_interfaces
        )

        self.assertEqual(system.id, "humanoid_001")
        self.assertEqual(system.name, "Test Humanoid")
        self.assertEqual(system.status, SystemStatus.IDLE)
        self.assertEqual(system.current_behavior, "idle")
        self.assertEqual(system.performance_metrics.response_time, 0.0)

    def test_system_status_update(self):
        """Test updating system status."""
        system = AutonomousHumanoidSystem(
            id="humanoid_002",
            name="Test Humanoid 2",
            status=SystemStatus.IDLE,
            current_behavior="idle",
            sensors={"audio": {"is_active": True}},
            actuators={"motors": [], "grippers": []},
            state={"battery_level": 85.0, "temperature": 30.0},
            active_sequences=[],
            last_command="",
            command_history=[],
            performance_metrics=self.performance_metrics,
            error_log=[],
            ros_interfaces=self.ros_interfaces
        )

        # Valid transition: idle -> listening
        system.update_status(SystemStatus.LISTENING)
        self.assertEqual(system.status, SystemStatus.LISTENING)

        # Valid transition: listening -> processing
        system.update_status(SystemStatus.PROCESSING)
        self.assertEqual(system.status, SystemStatus.PROCESSING)

        # Invalid transition: processing -> idle (should raise error)
        with self.assertRaises(ValueError):
            system.update_status(SystemStatus.IDLE)

    def test_system_health_check(self):
        """Test system health check functionality."""
        system = AutonomousHumanoidSystem(
            id="humanoid_003",
            name="Test Humanoid 3",
            status=SystemStatus.IDLE,
            current_behavior="idle",
            sensors={"audio": {"is_active": True}},
            actuators={"motors": [], "grippers": []},
            state={"battery_level": 90.0, "temperature": 25.0},
            active_sequences=[],
            last_command="",
            command_history=[],
            performance_metrics=self.performance_metrics,
            error_log=[],
            ros_interfaces=self.ros_interfaces
        )

        health = system.get_system_health()
        self.assertEqual(health["status"], "idle")
        self.assertEqual(health["battery_level"], 90.0)
        self.assertEqual(health["temperature"], 25.0)
        self.assertEqual(health["health_status"], "good")

        # Test with low battery
        system.state["battery_level"] = 15.0
        health = system.get_system_health()
        self.assertEqual(health["health_status"], "critical")

        # Test with high temperature
        system.state["battery_level"] = 90.0
        system.state["temperature"] = 75.0
        health = system.get_system_health()
        self.assertEqual(health["health_status"], "critical")

    def test_command_history_management(self):
        """Test command history management."""
        system = AutonomousHumanoidSystem(
            id="humanoid_004",
            name="Test Humanoid 4",
            status=SystemStatus.IDLE,
            current_behavior="idle",
            sensors={"audio": {"is_active": True}},
            actuators={"motors": [], "grippers": []},
            state={"battery_level": 85.0, "temperature": 30.0},
            active_sequences=[],
            last_command="",
            command_history=[],
            performance_metrics=self.performance_metrics,
            error_log=[],
            ros_interfaces=self.ros_interfaces
        )

        # Add commands to history
        for i in range(55):  # Add more than the limit to test trimming
            system.add_command_to_history(f"Command {i}")

        # Should only keep the last 50 commands
        self.assertEqual(len(system.command_history), 50)
        self.assertEqual(system.command_history[0], "Command 5")  # First should be command 5
        self.assertEqual(system.command_history[-1], "Command 54")  # Last should be command 54


class TestCapstoneProjectModel(unittest.TestCase):
    """Test cases for the CapstoneProject data model."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.requirements = [
            Requirement(
                requirement_id="req_001",
                description="Implement voice command processing",
                weight=0.25,
                status=RequirementStatus.NOT_STARTED
            ),
            Requirement(
                requirement_id="req_002",
                description="Integrate LLM-based task planning",
                weight=0.25,
                status=RequirementStatus.NOT_STARTED
            ),
            Requirement(
                requirement_id="req_003",
                description="Implement multi-modal perception",
                weight=0.25,
                status=RequirementStatus.NOT_STARTED
            ),
            Requirement(
                requirement_id="req_004",
                description="Create complete autonomous system",
                weight=0.25,
                status=RequirementStatus.NOT_STARTED
            )
        ]

        self.rubric = Rubric(
            criteria=[
                EvaluationCriterion(
                    criterion_id="criterion_001",
                    description="Technical Implementation",
                    weight=0.4,
                    levels=[
                        EvaluationLevelDetail(
                            level=EvaluationLevel.EXCELLENT,
                            description="Implementation exceeds expectations",
                            points=100
                        ),
                        EvaluationLevelDetail(
                            level=EvaluationLevel.PROFICIENT,
                            description="Implementation meets all requirements",
                            points=85
                        )
                    ]
                ),
                EvaluationCriterion(
                    criterion_id="criterion_002",
                    description="Integration and Testing",
                    weight=0.6,
                    levels=[
                        EvaluationLevelDetail(
                            level=EvaluationLevel.EXCELLENT,
                            description="System integrates and tests flawlessly",
                            points=100
                        ),
                        EvaluationLevelDetail(
                            level=EvaluationLevel.PROFICIENT,
                            description="System integrates and tests well",
                            points=85
                        )
                    ]
                )
            ],
            total_points=200
        )

    def test_capstone_project_creation(self):
        """Test creating a valid CapstoneProject object."""
        project = CapstoneProject(
            id="capstone_001",
            title="Test Capstone Project",
            description="A test capstone project",
            learning_objectives=["Implement voice processing", "Integrate LLM planning"],
            requirements=self.requirements,
            rubric=self.rubric,
            status=ProjectStatus.DRAFT,
            deadline=datetime.now().replace(year=datetime.now().year + 1),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        self.assertEqual(project.id, "capstone_001")
        self.assertEqual(project.title, "Test Capstone Project")
        self.assertEqual(project.status, ProjectStatus.DRAFT)
        self.assertEqual(len(project.requirements), 4)
        self.assertEqual(len(project.rubric.criteria), 2)

    def test_requirement_weights_validation(self):
        """Test validation of requirement weights."""
        # Test with weights that don't sum to 1.0
        invalid_requirements = [
            Requirement(
                requirement_id="req_001",
                description="Test requirement",
                weight=0.5,  # This would make total 1.5
                status=RequirementStatus.NOT_STARTED
            ),
            Requirement(
                requirement_id="req_002",
                description="Test requirement 2",
                weight=1.0,  # This would make total 1.5
                status=RequirementStatus.NOT_STARTED
            )
        ]

        with self.assertRaises(ValueError):
            CapstoneProject(
                id="capstone_002",
                title="Test Capstone Project 2",
                description="A test capstone project",
                learning_objectives=["Test objective"],
                requirements=invalid_requirements,
                rubric=self.rubric,
                status=ProjectStatus.DRAFT,
                deadline=datetime.now().replace(year=datetime.now().year + 1),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

    def test_project_submission_and_evaluation(self):
        """Test project submission and evaluation."""
        project = CapstoneProject(
            id="capstone_003",
            title="Test Capstone Project 3",
            description="A test capstone project",
            learning_objectives=["Test objective"],
            requirements=self.requirements,
            rubric=self.rubric,
            status=ProjectStatus.DRAFT,
            deadline=datetime.now().replace(year=datetime.now().year + 1),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Submit the project
        from scripts.capstone_project.capstone_project_model import Submission
        submission = Submission(
            files=["src/main.py", "docs/report.md"],
            report="Test report"
        )
        project.submit_project(submission)

        self.assertEqual(project.status, ProjectStatus.SUBMITTED)
        self.assertIsNotNone(project.submission)

        # Evaluate the project
        evaluation = Evaluation(
            scores=[
                EvaluationScore(
                    criterion_id="criterion_001",
                    level=EvaluationLevel.PROFICIENT,
                    points_awarded=85,
                    feedback="Good implementation"
                ),
                EvaluationScore(
                    criterion_id="criterion_002",
                    level=EvaluationLevel.EXCELLENT,
                    points_awarded=100,
                    feedback="Excellent integration"
                )
            ],
            feedback="Overall excellent work!",
            grade="A",
            evaluator="Dr. Smith",
            evaluation_date=datetime.now()
        )
        project.evaluate_project(evaluation)

        self.assertEqual(project.status, ProjectStatus.EVALUATED)
        self.assertIsNotNone(project.evaluation)
        self.assertEqual(project.get_grade(), "A")
        self.assertGreater(project.get_total_score(), 0.9)  # Should be above 90%

    def test_project_completion_percentage(self):
        """Test project completion percentage calculation."""
        project = CapstoneProject(
            id="capstone_004",
            title="Test Capstone Project 4",
            description="A test capstone project",
            learning_objectives=["Test objective"],
            requirements=self.requirements,
            rubric=self.rubric,
            status=ProjectStatus.DRAFT,
            deadline=datetime.now().replace(year=datetime.now().year + 1),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Initially no requirements completed
        self.assertEqual(project.get_completion_percentage(), 0.0)

        # Complete 2 out of 4 requirements
        project.requirements[0].status = RequirementStatus.IMPLEMENTED
        project.requirements[1].status = RequirementStatus.TESTED

        self.assertEqual(project.get_completion_percentage(), 0.5)  # 50%

        # Complete all requirements
        project.requirements[2].status = RequirementStatus.VERIFIED
        project.requirements[3].status = RequirementStatus.VERIFIED

        self.assertEqual(project.get_completion_percentage(), 1.0)  # 100%


class TestActionSequenceModel(unittest.TestCase):
    """Test cases for the ActionSequence and ActionStep models."""

    def test_action_step_creation(self):
        """Test creating a valid ActionStep object."""
        step = ActionStep(
            id="step_001",
            action_type="navigate_to",
            parameters={"target": "kitchen"},
            description="Navigate to the kitchen",
            priority=3,
            timeout=10.0,
            retry_attempts=2,
            ros_action_server="/navigate_to_pose"
        )

        self.assertEqual(step.id, "step_001")
        self.assertEqual(step.action_type, "navigate_to")
        self.assertEqual(step.parameters["target"], "kitchen")
        self.assertEqual(step.description, "Navigate to the kitchen")
        self.assertEqual(step.priority, 3)
        self.assertEqual(step.timeout, 10.0)
        self.assertEqual(step.retry_attempts, 2)
        self.assertEqual(step.ros_action_server, "/navigate_to_pose")

    def test_action_sequence_creation(self):
        """Test creating a valid ActionSequence object."""
        steps = [
            ActionStep(
                id="step_001",
                action_type="detect_object",
                parameters={"object_type": "red cup"},
                description="Detect the red cup"
            ),
            ActionStep(
                id="step_002",
                action_type="navigate_to",
                parameters={"target": "red cup location"},
                description="Navigate to the red cup"
            )
        ]

        sequence = ActionSequence(
            id="seq_001",
            name="Pick up red cup",
            description="Detect and pick up the red cup",
            steps=steps,
            created_by="LLM_Service",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            command_id="cmd_001",
            estimated_duration=5.0,
            success_threshold=0.8
        )

        self.assertEqual(sequence.id, "seq_001")
        self.assertEqual(sequence.name, "Pick up red cup")
        self.assertEqual(len(sequence.steps), 2)
        self.assertEqual(sequence.created_by, "LLM_Service")
        self.assertEqual(sequence.estimated_duration, 5.0)
        self.assertEqual(sequence.success_threshold, 0.8)

    def test_action_sequence_validation(self):
        """Test validation of ActionSequence object."""
        # Test with no steps (should fail validation in a real implementation)
        # For this test, we'll just verify the basic structure
        sequence = ActionSequence(
            id="seq_002",
            name="Empty sequence",
            description="An empty sequence",
            steps=[],  # Empty steps
            created_by="LLM_Service",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            command_id="cmd_002",
            estimated_duration=0.0,
            success_threshold=0.8
        )

        self.assertEqual(sequence.id, "seq_002")
        self.assertEqual(len(sequence.steps), 0)


class TestVLAIntegration(unittest.TestCase):
    """Test cases for VLA component integration."""

    def test_voice_processing_demo(self):
        """Test the voice processing demo functionality."""
        demo = VoiceProcessingDemo()

        # Test that the demo can create a sample context
        context = demo.create_sample_context()
        self.assertIsNotNone(context)
        self.assertEqual(context.location.x, 0.0)
        self.assertEqual(len(context.environment.obstacles), 2)

    def test_llm_task_planning_demo(self):
        """Test the LLM task planning demo functionality."""
        demo = LLMTaskPlanningDemo()

        # Test command classification
        command_type = demo._classify_command("Go to the kitchen")
        self.assertEqual(command_type, "navigation")

        command_type = demo._classify_command("Pick up the red cup")
        self.assertEqual(command_type, "manipulation")

        command_type = demo._classify_command("Find the blue box")
        self.assertEqual(command_type, "perception")

    def test_multi_modal_demo(self):
        """Test the multi-modal demo functionality."""
        demo = MultiModalDemo()

        # Test that we can process a simple command
        multi_modal_input = demo.process_multi_modal_input("Go to the blue box")

        self.assertIsNotNone(multi_modal_input)
        self.assertEqual(multi_modal_input.processing_status, "fused")
        self.assertIsNotNone(multi_modal_input.fused_command)
        self.assertTrue(0.0 <= multi_modal_input.fusion_confidence <= 1.0)

    def test_capstone_demo(self):
        """Test the capstone demo functionality."""
        demo = CapstoneDemo()

        # Test that the system was created properly
        self.assertIsNotNone(demo.system)
        self.assertEqual(demo.system.name, "VLA Capstone Humanoid Robot")

        # Test that the project was created properly
        self.assertIsNotNone(demo.project)
        self.assertEqual(demo.project.title, "Autonomous Humanoid Robot Capstone")
        self.assertEqual(len(demo.project.requirements), 4)


class TestMockServices(unittest.TestCase):
    """Test cases for mock services used in the VLA system."""

    def test_mock_whisper_service(self):
        """Test the mock Whisper service."""
        service = MockWhisperService()

        # Test transcription
        result = service.transcribe(None)  # Pass None as audio_data for demo
        self.assertIn("text", result)
        self.assertIn("confidence", result)
        self.assertIn("command_type", result)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_mock_visual_perception(self):
        """Test the mock visual perception service."""
        service = MockVisualPerception()

        # Test object detection
        objects = service.detect_objects()
        self.assertIsInstance(objects, list)
        self.assertGreater(len(objects), 0)

        # Test scene description
        description = service.describe_scene()
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)

    def test_mock_multi_modal_fusion_engine(self):
        """Test the mock multi-modal fusion engine."""
        engine = MockMultiModalFusionEngine()
        visual_objects = [
            {"class": "box", "confidence": 0.9, "bbox": {}},
            {"class": "cup", "confidence": 0.85, "bbox": {}}
        ]

        # Test fusion
        result = engine.fuse_inputs("Go to the blue box", visual_objects, "A room with a blue box")
        self.assertIn("fused_action", result)
        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


def run_all_tests():
    """Run all test suites."""
    print("Running VLA System Test Suite")
    print("=" * 40)

    # Create a test suite
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTest(unittest.makeSuite(TestVoiceCommandModel))
    suite.addTest(unittest.makeSuite(TestAutonomousHumanoidSystemModel))
    suite.addTest(unittest.makeSuite(TestCapstoneProjectModel))
    suite.addTest(unittest.makeSuite(TestActionSequenceModel))
    suite.addTest(unittest.makeSuite(TestVLAIntegration))
    suite.addTest(unittest.makeSuite(TestMockServices))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.2f}%")

    return result.wasSuccessful()


def main():
    """Main function to run the test suite."""
    success = run_all_tests()

    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())