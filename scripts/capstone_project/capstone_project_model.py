"""
Capstone Project Data Model for VLA System

This module defines the CapstoneProject data model based on the entity specification.
It represents the final project that integrates all learning modules with comprehensive rubrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class ProjectStatus(Enum):
    """Enumeration of project statuses"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    EVALUATED = "evaluated"


class RequirementStatus(Enum):
    """Enumeration of requirement statuses"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    TESTED = "tested"
    VERIFIED = "verified"


class EvaluationLevel(Enum):
    """Enumeration of evaluation levels"""
    EXCELLENT = "excellent"
    PROFICIENT = "proficient"
    ADEQUATE = "adequate"
    NEEDS_IMPROVEMENT = "needs_improvement"
    UNSATISFACTORY = "unsatisfactory"


@dataclass
class Requirement:
    """Represents a technical requirement for the capstone project"""
    requirement_id: str
    description: str
    weight: float  # Weight in final evaluation (0.0-1.0)
    status: RequirementStatus


@dataclass
class EvaluationLevelDetail:
    """Represents an evaluation level within a criterion"""
    level: EvaluationLevel
    description: str
    points: int


@dataclass
class EvaluationCriterion:
    """Represents an evaluation criterion in the rubric"""
    criterion_id: str
    description: str
    weight: float  # Weight in evaluation (0.0-1.0)
    levels: List[EvaluationLevelDetail]


@dataclass
class Rubric:
    """Represents the evaluation rubric for the project"""
    criteria: List[EvaluationCriterion]
    total_points: int

    def __post_init__(self):
        """Validate the rubric after initialization"""
        self._validate()

    def _validate(self):
        """Validate the rubric according to the data model rules"""
        # Validate that sum of criterion weights equals 1.0
        total_weight = sum(criterion.weight for criterion in self.criteria)
        if abs(total_weight - 1.0) > 0.001:  # Allow for floating point precision
            raise ValueError(f"Sum of criterion weights must equal 1.0, got {total_weight}")


@dataclass
class Submission:
    """Represents a student submission for the project"""
    files: List[str]
    demonstration_video: Optional[str] = None
    report: str = ""


@dataclass
class EvaluationScore:
    """Represents a score for a specific criterion"""
    criterion_id: str
    level: EvaluationLevel
    points_awarded: int
    feedback: str = ""


@dataclass
class Evaluation:
    """Represents the evaluation results for the project"""
    scores: List[EvaluationScore]
    feedback: str
    grade: str  # Final letter grade
    evaluator: str
    evaluation_date: datetime


@dataclass
class CapstoneProject:
    """
    Represents the final project that integrates all learning modules with comprehensive rubrics.

    Attributes:
        id: Unique identifier for the project
        title: Title of the project
        description: Detailed description
        learning_objectives: Learning objectives to be demonstrated
        requirements: Technical requirements
        rubric: Evaluation rubric
        submission: Student submission
        evaluation: Evaluation results
        status: Project status (draft, submitted, evaluated)
        deadline: Submission deadline
        created_at: Project creation date
        updated_at: Last update date
    """

    id: str
    title: str
    description: str
    learning_objectives: List[str]
    requirements: List[Requirement]
    rubric: Rubric
    status: ProjectStatus
    deadline: datetime
    created_at: datetime
    updated_at: datetime

    submission: Optional[Submission] = None
    evaluation: Optional[Evaluation] = None

    def __post_init__(self):
        """Validate the CapstoneProject after initialization"""
        self._validate()

    def _validate(self):
        """Validate the project according to the data model rules"""
        # Validate sum of requirement weights equals 1.0
        total_req_weight = sum(req.weight for req in self.requirements)
        if abs(total_req_weight - 1.0) > 0.001:  # Allow for floating point precision
            raise ValueError(f"Sum of requirement weights must equal 1.0, got {total_req_weight}")

        # Validate weight values are between 0.0 and 1.0
        for req in self.requirements:
            if not 0.0 <= req.weight <= 1.0:
                raise ValueError(f"Requirement weight must be between 0.0 and 1.0, got {req.weight}")

    def add_requirement(self, requirement: Requirement):
        """Add a requirement to the project"""
        self.requirements.append(requirement)
        self.updated_at = datetime.now()

        # Revalidate requirement weights sum
        total_weight = sum(req.weight for req in self.requirements)
        if abs(total_weight - 1.0) > 0.001:
            # Remove the requirement if it violates the constraint
            self.requirements.pop()
            raise ValueError(f"Adding this requirement would make total weight {total_weight}, which is not 1.0")

    def submit_project(self, submission: Submission):
        """Submit the project"""
        if self.status != ProjectStatus.DRAFT:
            raise ValueError(f"Cannot submit project with status {self.status.value}")

        self.submission = submission
        self.status = ProjectStatus.SUBMITTED
        self.updated_at = datetime.now()

    def evaluate_project(self, evaluation: Evaluation):
        """Evaluate the submitted project"""
        if self.status != ProjectStatus.SUBMITTED:
            raise ValueError(f"Cannot evaluate project with status {self.status.value}")

        self.evaluation = evaluation
        self.status = ProjectStatus.EVALUATED
        self.updated_at = datetime.now()

    def get_total_score(self) -> float:
        """Calculate the total score based on evaluation"""
        if not self.evaluation:
            return 0.0

        total_points = sum(score.points_awarded for score in self.evaluation.scores)
        max_possible_points = sum(
            max(level.points for level in criterion.levels)
            for criterion in self.rubric.criteria
        )

        return total_points / max_possible_points if max_possible_points > 0 else 0.0

    def get_grade(self) -> str:
        """Get the letter grade based on evaluation"""
        if not self.evaluation:
            return "N/A"

        return self.evaluation.grade

    def is_overdue(self) -> bool:
        """Check if the project is overdue"""
        return datetime.now() > self.deadline and self.status == ProjectStatus.DRAFT

    def is_complete(self) -> bool:
        """Check if the project is complete (evaluated)"""
        return self.status == ProjectStatus.EVALUATED

    def get_completion_percentage(self) -> float:
        """Get the completion percentage based on requirements status"""
        if not self.requirements:
            return 0.0

        completed_requirements = sum(
            1 for req in self.requirements
            if req.status in [RequirementStatus.IMPLEMENTED, RequirementStatus.TESTED, RequirementStatus.VERIFIED]
        )

        return completed_requirements / len(self.requirements)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the CapstoneProject to a dictionary representation"""
        result = {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'learning_objectives': self.learning_objectives,
            'requirements': [
                {
                    'requirement_id': req.requirement_id,
                    'description': req.description,
                    'weight': req.weight,
                    'status': req.status.value
                }
                for req in self.requirements
            ],
            'rubric': {
                'criteria': [
                    {
                        'criterion_id': crit.criterion_id,
                        'description': crit.description,
                        'weight': crit.weight,
                        'levels': [
                            {
                                'level': level.level.value,
                                'description': level.description,
                                'points': level.points
                            }
                            for level in crit.levels
                        ]
                    }
                    for crit in self.rubric.criteria
                ],
                'total_points': self.rubric.total_points
            },
            'status': self.status.value,
            'deadline': self.deadline.isoformat(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

        # Add optional fields if they exist
        if self.submission:
            result['submission'] = {
                'files': self.submission.files,
                'demonstration_video': self.submission.demonstration_video,
                'report': self.submission.report
            }

        if self.evaluation:
            result['evaluation'] = {
                'scores': [
                    {
                        'criterion_id': score.criterion_id,
                        'level': score.level.value,
                        'points_awarded': score.points_awarded,
                        'feedback': score.feedback
                    }
                    for score in self.evaluation.scores
                ],
                'feedback': self.evaluation.feedback,
                'grade': self.evaluation.grade,
                'evaluator': self.evaluation.evaluator,
                'evaluation_date': self.evaluation.evaluation_date.isoformat()
            }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CapstoneProject':
        """Create a CapstoneProject from a dictionary representation"""
        # Create requirements
        requirements = []
        for req_data in data['requirements']:
            req = Requirement(
                requirement_id=req_data['requirement_id'],
                description=req_data['description'],
                weight=req_data['weight'],
                status=RequirementStatus(req_data['status'])
            )
            requirements.append(req)

        # Create rubric
        criteria = []
        for crit_data in data['rubric']['criteria']:
            levels = []
            for level_data in crit_data['levels']:
                level = EvaluationLevelDetail(
                    level=EvaluationLevel(level_data['level']),
                    description=level_data['description'],
                    points=level_data['points']
                )
                levels.append(level)

            criterion = EvaluationCriterion(
                criterion_id=crit_data['criterion_id'],
                description=crit_data['description'],
                weight=crit_data['weight'],
                levels=levels
            )
            criteria.append(criterion)

        rubric = Rubric(
            criteria=criteria,
            total_points=data['rubric']['total_points']
        )

        # Create submission if present
        submission = None
        if 'submission' in data:
            sub_data = data['submission']
            submission = Submission(
                files=sub_data['files'],
                demonstration_video=sub_data.get('demonstration_video'),
                report=sub_data.get('report', "")
            )

        # Create evaluation if present
        evaluation = None
        if 'evaluation' in data:
            eval_data = data['evaluation']
            scores = []
            for score_data in eval_data['scores']:
                score = EvaluationScore(
                    criterion_id=score_data['criterion_id'],
                    level=EvaluationLevel(score_data['level']),
                    points_awarded=score_data['points_awarded'],
                    feedback=score_data.get('feedback', "")
                )
                scores.append(score)

            evaluation = Evaluation(
                scores=scores,
                feedback=eval_data['feedback'],
                grade=eval_data['grade'],
                evaluator=eval_data['evaluator'],
                evaluation_date=datetime.fromisoformat(eval_data['evaluation_date'])
            )

        # Create and return CapstoneProject
        return cls(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            learning_objectives=data['learning_objectives'],
            requirements=requirements,
            rubric=rubric,
            status=ProjectStatus(data['status']),
            deadline=datetime.fromisoformat(data['deadline']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            submission=submission,
            evaluation=evaluation
        )


# Example usage and validation
if __name__ == "__main__":
    # Create sample requirements
    requirements = [
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

    # Create sample rubric
    rubric = Rubric(
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
                    ),
                    EvaluationLevelDetail(
                        level=EvaluationLevel.ADEQUATE,
                        description="Implementation meets basic requirements",
                        points=70
                    ),
                    EvaluationLevelDetail(
                        level=EvaluationLevel.NEEDS_IMPROVEMENT,
                        description="Implementation has significant issues",
                        points=55
                    ),
                    EvaluationLevelDetail(
                        level=EvaluationLevel.UNSATISFACTORY,
                        description="Implementation does not meet requirements",
                        points=40
                    )
                ]
            ),
            EvaluationCriterion(
                criterion_id="criterion_002",
                description="Documentation and Presentation",
                weight=0.3,
                levels=[
                    EvaluationLevelDetail(
                        level=EvaluationLevel.EXCELLENT,
                        description="Documentation is comprehensive and clear",
                        points=100
                    ),
                    EvaluationLevelDetail(
                        level=EvaluationLevel.PROFICIENT,
                        description="Documentation is good with minor issues",
                        points=85
                    ),
                    EvaluationLevelDetail(
                        level=EvaluationLevel.ADEQUATE,
                        description="Documentation meets basic requirements",
                        points=70
                    ),
                    EvaluationLevelDetail(
                        level=EvaluationLevel.NEEDS_IMPROVEMENT,
                        description="Documentation has significant issues",
                        points=55
                    ),
                    EvaluationLevelDetail(
                        level=EvaluationLevel.UNSATISFACTORY,
                        description="Documentation is inadequate",
                        points=40
                    )
                ]
            ),
            EvaluationCriterion(
                criterion_id="criterion_003",
                description="Integration and Testing",
                weight=0.3,
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
                    ),
                    EvaluationLevelDetail(
                        level=EvaluationLevel.ADEQUATE,
                        description="System integrates with basic functionality",
                        points=70
                    ),
                    EvaluationLevelDetail(
                        level=EvaluationLevel.NEEDS_IMPROVEMENT,
                        description="System has integration issues",
                        points=55
                    ),
                    EvaluationLevelDetail(
                        level=EvaluationLevel.UNSATISFACTORY,
                        description="System fails to integrate properly",
                        points=40
                    )
                ]
            )
        ],
        total_points=300
    )

    # Create the capstone project
    project = CapstoneProject(
        id="capstone_001",
        title="Autonomous Humanoid Robot Capstone",
        description="Develop a complete autonomous humanoid robot that responds to voice commands, navigates environments, and performs tasks",
        learning_objectives=[
            "Implement voice command processing with local Whisper models",
            "Integrate LLM-based task and motion planning",
            "Combine visual perception and language understanding",
            "Create a complete autonomous system"
        ],
        requirements=requirements,
        rubric=rubric,
        status=ProjectStatus.DRAFT,
        deadline=datetime.now().replace(year=datetime.now().year + 1),
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    print(f"Created capstone project: {project.title}")
    print(f"Project ID: {project.id}")
    print(f"Status: {project.status.value}")
    print(f"Requirements: {len(project.requirements)}")
    print(f"Criteria: {len(project.rubric.criteria)}")
    print(f"Completion: {project.get_completion_percentage():.2%}")

    # Add a new requirement
    try:
        new_requirement = Requirement(
            requirement_id="req_005",
            description="Ensure WCAG 2.1 AA compliance",
            weight=0.0,  # This would cause validation error when summing to 1.0
            status=RequirementStatus.NOT_STARTED
        )
        project.add_requirement(new_requirement)
        print("Successfully added requirement")
    except ValueError as e:
        print(f"Could not add requirement: {e}")

    # Create and submit a sample submission
    submission = Submission(
        files=["src/main.py", "docs/report.md", "assets/demo.mp4"],
        demonstration_video="https://example.com/demo_video",
        report="This is the project report..."
    )

    # Create a sample evaluation
    evaluation = Evaluation(
        scores=[
            EvaluationScore(
                criterion_id="criterion_001",
                level=EvaluationLevel.PROFICIENT,
                points_awarded=85,
                feedback="Good implementation with minor issues"
            ),
            EvaluationScore(
                criterion_id="criterion_002",
                level=EvaluationLevel.EXCELLENT,
                points_awarded=100,
                feedback="Excellent documentation"
            ),
            EvaluationScore(
                criterion_id="criterion_003",
                level=EvaluationLevel.PROFICIENT,
                points_awarded=85,
                feedback="Good integration with some minor bugs"
            )
        ],
        feedback="Overall excellent work on the capstone project!",
        grade="A",
        evaluator="Dr. Smith",
        evaluation_date=datetime.now()
    )

    # Submit and evaluate the project
    project.submit_project(submission)
    print(f"Project status after submission: {project.status.value}")

    project.evaluate_project(evaluation)
    print(f"Project status after evaluation: {project.status.value}")
    print(f"Final grade: {project.get_grade()}")
    print(f"Total score: {project.get_total_score():.2%}")

    # Convert to dict and back
    project_dict = project.to_dict()
    print(f"Dictionary keys: {list(project_dict.keys())}")

    reconstructed_project = CapstoneProject.from_dict(project_dict)
    print(f"Reconstructed project title: {reconstructed_project.title}")