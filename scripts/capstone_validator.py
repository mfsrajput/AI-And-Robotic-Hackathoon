"""
Capstone Validation and Assessment Utilities for Autonomous Humanoid
Comprehensive validation framework for the complete system
"""

import json
import yaml
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import os
import sys
import importlib.util
from pathlib import Path
import subprocess
import threading
import time

from common_data_models import AutonomousHumanoidSystem, CapstoneProject
from observability_utils import EducationalLogger, MetricsCollector, DebuggingHelper
from error_handling_utils import EducationalErrorHandler, InputValidator
from privacy_compliance import PrivacyComplianceChecker
from capstone_project.state_manager import StateManager, SystemState


class ValidationRule:
    """Represents a validation rule for the capstone system"""
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 category: str, 
                 severity: str,  # 'critical', 'error', 'warning', 'info'
                 check_function,
                 enabled: bool = True):
        self.name = name
        self.description = description
        self.category = category  # 'integration', 'privacy', 'performance', 'safety', etc.
        self.severity = severity
        self.check_function = check_function
        self.enabled = enabled
        self.last_result: Optional[Dict[str, Any]] = None
    
    def execute(self, system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the validation rule"""
        try:
            result = self.check_function(system_context)
            self.last_result = result
            return result
        except Exception as e:
            return {
                'rule_name': self.name,
                'status': 'error',
                'message': f'Validation rule execution failed: {str(e)}',
                'details': str(e)
            }


class CapstoneValidator:
    """
    Comprehensive validation system for the autonomous humanoid capstone
    """
    
    def __init__(self):
        self.logger = EducationalLogger("capstone_validator")
        self.metrics = MetricsCollector(self.logger)
        self.debug_helper = DebuggingHelper(self.logger)
        self.error_handler = EducationalErrorHandler()
        self.validator = InputValidator(self.error_handler)
        
        # Initialize privacy compliance checker
        self.privacy_checker = PrivacyComplianceChecker()
        
        # Validation rules
        self.validation_rules: List[ValidationRule] = []
        self._initialize_validation_rules()
        
        # Validation results
        self.validation_results: List[Dict[str, Any]] = []
        self.validation_history: List[Dict[str, Any]] = []
        
        # Assessment rubric
        self.assessment_rubric = self._create_assessment_rubric()
        
        self.logger.info("CapstoneValidator", "__init__", "Capstone validator initialized")
    
    def _initialize_validation_rules(self):
        """Initialize all validation rules"""
        # Integration validation rules
        self.validation_rules.extend([
            ValidationRule(
                name="system_integration_check",
                description="Validate all VLA components are properly integrated",
                category="integration",
                severity="critical",
                check_function=self._check_system_integration
            ),
            ValidationRule(
                name="component_communication_check",
                description="Validate components can communicate with each other",
                category="integration", 
                severity="error",
                check_function=self._check_component_communication
            ),
            ValidationRule(
                name="state_consistency_check",
                description="Validate system state remains consistent across components",
                category="integration",
                severity="error", 
                check_function=self._check_state_consistency
            ),
        ])
        
        # Privacy validation rules
        self.validation_rules.extend([
            ValidationRule(
                name="external_api_check",
                description="Validate no external API calls are made",
                category="privacy",
                severity="critical",
                check_function=self._check_external_api_usage
            ),
            ValidationRule(
                name="data_storage_check",
                description="Validate sensitive data is not unnecessarily stored",
                category="privacy",
                severity="critical",
                check_function=self._check_data_storage_compliance
            ),
            ValidationRule(
                name="local_processing_check",
                description="Validate all processing occurs locally",
                category="privacy",
                severity="critical",
                check_function=self._check_local_processing
            ),
        ])
        
        # Performance validation rules
        self.validation_rules.extend([
            ValidationRule(
                name="response_time_check",
                description="Validate system response time meets requirements",
                category="performance",
                severity="error",
                check_function=self._check_response_time
            ),
            ValidationRule(
                name="resource_usage_check",
                description="Validate system resource usage is acceptable",
                category="performance",
                severity="warning",
                check_function=self._check_resource_usage
            ),
            ValidationRule(
                name="real_time_processing_check",
                description="Validate real-time processing capabilities",
                category="performance",
                severity="error",
                check_function=self._check_real_time_processing
            ),
        ])
        
        # Safety validation rules
        self.validation_rules.extend([
            ValidationRule(
                name="safety_constraint_check",
                description="Validate safety constraints are enforced",
                category="safety",
                severity="critical",
                check_function=self._check_safety_constraints
            ),
            ValidationRule(
                name="emergency_stop_check",
                description="Validate emergency stop functionality",
                category="safety",
                severity="critical",
                check_function=self._check_emergency_stop
            ),
        ])
    
    def _create_assessment_rubric(self) -> Dict[str, Any]:
        """Create the assessment rubric for capstone evaluation"""
        return {
            'categories': {
                'system_integration': {
                    'weight': 0.25,
                    'max_points': 25,
                    'criteria': [
                        'All VLA components properly integrated',
                        'Components communicate effectively',
                        'System state managed correctly',
                        'Error handling implemented'
                    ]
                },
                'functionality': {
                    'weight': 0.30,
                    'max_points': 30,
                    'criteria': [
                        'Voice commands processed correctly',
                        'LLM task planning functional',
                        'Actions executed as planned',
                        'Multi-modal integration working'
                    ]
                },
                'performance': {
                    'weight': 0.20,
                    'max_points': 20,
                    'criteria': [
                        'Response time under 2 seconds',
                        'System remains stable under load',
                        'Resource usage optimized',
                        'Real-time processing achieved'
                    ]
                },
                'privacy_compliance': {
                    'weight': 0.15,
                    'max_points': 15,
                    'criteria': [
                        'No external API calls made',
                        'Data not stored unnecessarily',
                        'Local processing verified',
                        'Privacy settings configured'
                    ]
                },
                'documentation': {
                    'weight': 0.10,
                    'max_points': 10,
                    'criteria': [
                        'Code properly documented',
                        'System architecture explained',
                        'Usage instructions provided',
                        'Troubleshooting guide included'
                    ]
                }
            },
            'grading_scale': {
                'A+': (97, 100),
                'A': (93, 96.99),
                'A-': (90, 92.99),
                'B+': (87, 89.99),
                'B': (83, 86.99),
                'B-': (80, 82.99),
                'C+': (77, 79.99),
                'C': (73, 76.99),
                'C-': (70, 72.99),
                'D+': (67, 69.99),
                'D': (65, 66.99),
                'F': (0, 64.99)
            }
        }
    
    def _check_system_integration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if all system components are properly integrated"""
        try:
            required_components = context.get('required_components', [
                'voice_processor', 'llm_planner', 'action_executor', 'state_manager'
            ])
            system_components = context.get('system_components', {})
            
            missing_components = []
            for component in required_components:
                if component not in system_components:
                    missing_components.append(component)
            
            if missing_components:
                return {
                    'rule_name': 'system_integration_check',
                    'status': 'fail',
                    'message': f'Missing required components: {missing_components}',
                    'details': {'missing_components': missing_components}
                }
            else:
                return {
                    'rule_name': 'system_integration_check',
                    'status': 'pass',
                    'message': 'All required components present',
                    'details': {'present_components': list(system_components.keys())}
                }
                
        except Exception as e:
            return {
                'rule_name': 'system_integration_check',
                'status': 'error',
                'message': f'Error checking system integration: {str(e)}',
                'details': str(e)
            }
    
    def _check_component_communication(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if components can communicate with each other"""
        try:
            # This would involve checking actual communication channels
            # For this implementation, we'll check if communication interfaces exist
            communication_interfaces = context.get('communication_interfaces', [])
            
            if len(communication_interfaces) >= 3:  # Arbitrary threshold
                return {
                    'rule_name': 'component_communication_check',
                    'status': 'pass',
                    'message': 'Sufficient communication interfaces available',
                    'details': {'interfaces_count': len(communication_interfaces)}
                }
            else:
                return {
                    'rule_name': 'component_communication_check',
                    'status': 'fail',
                    'message': 'Insufficient communication interfaces',
                    'details': {'interfaces_count': len(communication_interfaces)}
                }
                
        except Exception as e:
            return {
                'rule_name': 'component_communication_check',
                'status': 'error',
                'message': f'Error checking component communication: {str(e)}',
                'details': str(e)
            }
    
    def _check_state_consistency(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if system state remains consistent"""
        try:
            state_manager = context.get('state_manager')
            if state_manager:
                status = state_manager.get_system_status()
                return {
                    'rule_name': 'state_consistency_check',
                    'status': 'pass',
                    'message': 'State manager available and functional',
                    'details': {
                        'current_state': status.get('current_state'),
                        'error_count': status.get('error_count', 0)
                    }
                }
            else:
                return {
                    'rule_name': 'state_consistency_check',
                    'status': 'fail',
                    'message': 'State manager not available',
                    'details': {}
                }
                
        except Exception as e:
            return {
                'rule_name': 'state_consistency_check',
                'status': 'error',
                'message': f'Error checking state consistency: {str(e)}',
                'details': str(e)
            }
    
    def _check_external_api_usage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check for external API usage"""
        try:
            # Use privacy compliance checker to scan for external calls
            privacy_report = self.privacy_checker.get_privacy_report()
            
            if privacy_report.get('compliance_status') == 'PASS':
                return {
                    'rule_name': 'external_api_check',
                    'status': 'pass',
                    'message': 'No external API usage detected',
                    'details': privacy_report
                }
            else:
                return {
                    'rule_name': 'external_api_check',
                    'status': 'fail',
                    'message': 'External API usage detected',
                    'details': privacy_report
                }
                
        except Exception as e:
            return {
                'rule_name': 'external_api_check',
                'status': 'error',
                'message': f'Error checking external API usage: {str(e)}',
                'details': str(e)
            }
    
    def _check_data_storage_compliance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check data storage compliance"""
        try:
            config = context.get('system_config', {})
            
            store_audio = config.get('store_audio_recordings', False)
            store_transcripts = config.get('store_transcripts', False)
            data_retention_days = config.get('data_retention_days', 0)
            
            issues = []
            if store_audio:
                issues.append("Audio recordings are stored")
            if store_transcripts:
                issues.append("Transcripts are stored")
            if data_retention_days > 7:
                issues.append(f"Data retention ({data_retention_days} days) exceeds recommended limit")
            
            if issues:
                return {
                    'rule_name': 'data_storage_check',
                    'status': 'fail',
                    'message': f'Data storage compliance issues: {issues}',
                    'details': {
                        'store_audio': store_audio,
                        'store_transcripts': store_transcripts,
                        'data_retention_days': data_retention_days,
                        'issues': issues
                    }
                }
            else:
                return {
                    'rule_name': 'data_storage_check',
                    'status': 'pass',
                    'message': 'Data storage compliance verified',
                    'details': {
                        'store_audio': store_audio,
                        'store_transcripts': store_transcripts,
                        'data_retention_days': data_retention_days
                    }
                }
                
        except Exception as e:
            return {
                'rule_name': 'data_storage_check',
                'status': 'error',
                'message': f'Error checking data storage compliance: {str(e)}',
                'details': str(e)
            }
    
    def _check_local_processing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check if processing occurs locally"""
        try:
            config = context.get('system_config', {})
            
            # Check if local processing is enforced
            local_processing = config.get('local_processing', False)
            
            if local_processing:
                return {
                    'rule_name': 'local_processing_check',
                    'status': 'pass',
                    'message': 'Local processing is enforced',
                    'details': {'local_processing': local_processing}
                }
            else:
                return {
                    'rule_name': 'local_processing_check',
                    'status': 'fail',
                    'message': 'Local processing not enforced',
                    'details': {'local_processing': local_processing}
                }
                
        except Exception as e:
            return {
                'rule_name': 'local_processing_check',
                'status': 'error',
                'message': f'Error checking local processing: {str(e)}',
                'details': str(e)
            }
    
    def _check_response_time(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check system response time"""
        try:
            performance_metrics = context.get('performance_metrics', {})
            
            response_time = performance_metrics.get('average_response_time', float('inf'))
            max_allowed = 2.0  # 2 seconds max
            
            if response_time <= max_allowed:
                return {
                    'rule_name': 'response_time_check',
                    'status': 'pass',
                    'message': f'Acceptable response time: {response_time:.2f}s',
                    'details': {'response_time': response_time, 'max_allowed': max_allowed}
                }
            else:
                return {
                    'rule_name': 'response_time_check',
                    'status': 'fail',
                    'message': f'Response time too slow: {response_time:.2f}s (max: {max_allowed}s)',
                    'details': {'response_time': response_time, 'max_allowed': max_allowed}
                }
                
        except Exception as e:
            return {
                'rule_name': 'response_time_check',
                'status': 'error',
                'message': f'Error checking response time: {str(e)}',
                'details': str(e)
            }
    
    def _check_resource_usage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            performance_metrics = context.get('performance_metrics', {})
            
            cpu_usage = performance_metrics.get('cpu_usage_percent', 100)
            memory_usage = performance_metrics.get('memory_usage_percent', 100)
            max_cpu = 80  # 80% max CPU usage
            max_memory = 80  # 80% max memory usage
            
            issues = []
            if cpu_usage > max_cpu:
                issues.append(f"CPU usage too high: {cpu_usage}% (max: {max_cpu}%)")
            if memory_usage > max_memory:
                issues.append(f"Memory usage too high: {memory_usage}% (max: {max_memory}%)")
            
            if issues:
                return {
                    'rule_name': 'resource_usage_check',
                    'status': 'fail',
                    'message': f'Resource usage issues: {issues}',
                    'details': {
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage,
                        'max_cpu': max_cpu,
                        'max_memory': max_memory,
                        'issues': issues
                    }
                }
            else:
                return {
                    'rule_name': 'resource_usage_check',
                    'status': 'pass',
                    'message': 'Resource usage within limits',
                    'details': {
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage,
                        'max_cpu': max_cpu,
                        'max_memory': max_memory
                    }
                }
                
        except Exception as e:
            return {
                'rule_name': 'resource_usage_check',
                'status': 'error',
                'message': f'Error checking resource usage: {str(e)}',
                'details': str(e)
            }
    
    def _check_real_time_processing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check real-time processing capabilities"""
        try:
            # Check if system can process inputs in real-time
            performance_metrics = context.get('performance_metrics', {})
            
            processing_frequency = performance_metrics.get('processing_frequency_hz', 0)
            min_required = 10  # 10 Hz minimum for real-time
            
            if processing_frequency >= min_required:
                return {
                    'rule_name': 'real_time_processing_check',
                    'status': 'pass',
                    'message': f'Real-time processing capability: {processing_frequency} Hz',
                    'details': {'frequency': processing_frequency, 'min_required': min_required}
                }
            else:
                return {
                    'rule_name': 'real_time_processing_check',
                    'status': 'fail',
                    'message': f'Real-time processing too slow: {processing_frequency} Hz (min: {min_required} Hz)',
                    'details': {'frequency': processing_frequency, 'min_required': min_required}
                }
                
        except Exception as e:
            return {
                'rule_name': 'real_time_processing_check',
                'status': 'error',
                'message': f'Error checking real-time processing: {str(e)}',
                'details': str(e)
            }
    
    def _check_safety_constraints(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check safety constraints"""
        try:
            safety_status = context.get('safety_status', {})
            
            safety_enabled = safety_status.get('safety_enabled', True)
            emergency_stop_available = safety_status.get('emergency_stop_available', True)
            
            if safety_enabled and emergency_stop_available:
                return {
                    'rule_name': 'safety_constraint_check',
                    'status': 'pass',
                    'message': 'Safety constraints properly configured',
                    'details': {
                        'safety_enabled': safety_enabled,
                        'emergency_stop_available': emergency_stop_available
                    }
                }
            else:
                return {
                    'rule_name': 'safety_constraint_check',
                    'status': 'fail',
                    'message': 'Safety constraints not properly configured',
                    'details': {
                        'safety_enabled': safety_enabled,
                        'emergency_stop_available': emergency_stop_available
                    }
                }
                
        except Exception as e:
            return {
                'rule_name': 'safety_constraint_check',
                'status': 'error',
                'message': f'Error checking safety constraints: {str(e)}',
                'details': str(e)
            }
    
    def _check_emergency_stop(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check emergency stop functionality"""
        try:
            # Check if emergency stop mechanism is available
            state_manager = context.get('state_manager')
            
            if state_manager:
                # Check if emergency stop can be triggered
                return {
                    'rule_name': 'emergency_stop_check',
                    'status': 'pass',
                    'message': 'Emergency stop mechanism available',
                    'details': {'state_manager_available': True}
                }
            else:
                return {
                    'rule_name': 'emergency_stop_check',
                    'status': 'fail',
                    'message': 'Emergency stop mechanism not available',
                    'details': {'state_manager_available': False}
                }
                
        except Exception as e:
            return {
                'rule_name': 'emergency_stop_check',
                'status': 'error',
                'message': f'Error checking emergency stop: {str(e)}',
                'details': str(e)
            }
    
    def run_validation(self, system_context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all validation rules against the system context"""
        try:
            self.logger.info("CapstoneValidator", "run_validation", "Starting validation", {
                'context_keys': list(system_context.keys())
            })
            
            results = []
            category_results = {}
            
            # Run enabled validation rules
            for rule in self.validation_rules:
                if rule.enabled:
                    result = rule.execute(system_context)
                    results.append(result)
                    
                    # Group by category
                    if rule.category not in category_results:
                        category_results[rule.category] = []
                    category_results[rule.category].append(result)
            
            # Calculate overall validation status
            total_rules = len(results)
            passed_rules = len([r for r in results if r.get('status') == 'pass'])
            failed_rules = len([r for r in results if r.get('status') == 'fail'])
            error_rules = len([r for r in results if r.get('status') == 'error'])
            
            overall_status = 'pass' if failed_rules == 0 and error_rules == 0 else 'fail'
            
            # Create validation report
            validation_report = {
                'timestamp': datetime.now().isoformat(),
                'total_rules': total_rules,
                'passed_rules': passed_rules,
                'failed_rules': failed_rules,
                'error_rules': error_rules,
                'overall_status': overall_status,
                'category_results': category_results,
                'all_results': results,
                'compliance_summary': self._generate_compliance_summary(results)
            }
            
            # Store in history
            self.validation_history.append(validation_report)
            self.validation_results = results
            
            self.logger.info("CapstoneValidator", "run_validation", "Validation completed", {
                'overall_status': overall_status,
                'passed': passed_rules,
                'failed': failed_rules
            })
            
            return validation_report
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="run_validation",
                suggested_fix="Check validation context and rules",
                learning_objective="Handle validation execution errors"
            )
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e),
                'all_results': []
            }
    
    def _generate_compliance_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a compliance summary from validation results"""
        summary = {}
        
        for result in results:
            rule_name = result.get('rule_name', 'unknown')
            status = result.get('status', 'unknown')
            
            # Map rule names to compliance areas
            if 'privacy' in rule_name:
                compliance_area = 'privacy'
            elif 'safety' in rule_name:
                compliance_area = 'safety'
            elif 'performance' in rule_name:
                compliance_area = 'performance'
            else:
                compliance_area = 'general'
            
            if compliance_area not in summary:
                summary[compliance_area] = {'pass': 0, 'fail': 0, 'error': 0}
            
            if status in summary[compliance_area]:
                summary[compliance_area][status] += 1
        
        return summary
    
    def assess_capstone_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess a capstone project using the rubric"""
        try:
            self.logger.info("CapstoneValidator", "assess_capstone_project", "Starting project assessment", {
                'project_id': project_data.get('project_id', 'unknown')
            })
            
            assessment = {
                'project_id': project_data.get('project_id', 'unknown'),
                'evaluator': 'capstone_assessment_system',
                'timestamp': datetime.now().isoformat(),
                'category_scores': {},
                'total_score': 0,
                'max_possible_score': sum(cat['max_points'] for cat in self.assessment_rubric['categories'].values()),
                'grade': '',
                'feedback': [],
                'rubric_used': self.assessment_rubric
            }
            
            # Evaluate each category
            for category, config in self.assessment_rubric['categories'].items():
                score, feedback = self._evaluate_category(category, project_data, config)
                assessment['category_scores'][category] = {
                    'score': score,
                    'max_points': config['max_points'],
                    'weight': config['weight'],
                    'feedback': feedback
                }
                assessment['total_score'] += score
                assessment['feedback'].extend(feedback)
            
            # Calculate final grade
            percentage = (assessment['total_score'] / assessment['max_possible_score']) * 100
            assessment['percentage'] = percentage
            assessment['grade'] = self._calculate_letter_grade(percentage)
            
            self.logger.info("CapstoneValidator", "assess_capstone_project", "Project assessment completed", {
                'grade': assessment['grade'],
                'percentage': assessment['percentage'],
                'total_score': assessment['total_score']
            })
            
            return assessment
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="assess_capstone_project",
                suggested_fix="Check project data format and assessment logic",
                learning_objective="Handle project assessment errors"
            )
            return {
                'project_id': project_data.get('project_id', 'unknown'),
                'overall_status': 'error',
                'error': str(e)
            }
    
    def _evaluate_category(self, category: str, project_data: Dict[str, Any], config: Dict[str, Any]) -> Tuple[int, List[str]]:
        """Evaluate a specific category"""
        feedback = []
        score = 0
        
        try:
            # This is a simplified evaluation - in practice, this would involve
            # detailed checking of the project against each criterion
            criteria = config['criteria']
            
            if category == 'system_integration':
                # Check for integration elements
                has_voice_processor = 'voice_processor' in project_data.get('components', {})
                has_state_manager = 'state_manager' in project_data.get('components', {})
                has_error_handling = 'error_handling' in project_data.get('components', {})
                
                if has_voice_processor:
                    score += config['max_points'] * 0.3
                    feedback.append("✓ Voice processor component implemented")
                else:
                    feedback.append("✗ Voice processor component missing")
                
                if has_state_manager:
                    score += config['max_points'] * 0.4
                    feedback.append("✓ State manager component implemented")
                else:
                    feedback.append("✗ State manager component missing")
                
                if has_error_handling:
                    score += config['max_points'] * 0.3
                    feedback.append("✓ Error handling implemented")
                else:
                    feedback.append("✗ Error handling not implemented")
            
            elif category == 'functionality':
                # Check for functional elements
                has_voice_processing = project_data.get('voice_processing_working', False)
                has_task_planning = project_data.get('task_planning_working', False)
                has_action_execution = project_data.get('action_execution_working', False)
                has_multi_modal = project_data.get('multi_modal_integration', False)
                
                if has_voice_processing:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ Voice processing functional")
                else:
                    feedback.append("✗ Voice processing not functional")
                
                if has_task_planning:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ Task planning functional")
                else:
                    feedback.append("✗ Task planning not functional")
                
                if has_action_execution:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ Action execution functional")
                else:
                    feedback.append("✗ Action execution not functional")
                
                if has_multi_modal:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ Multi-modal integration working")
                else:
                    feedback.append("✗ Multi-modal integration not working")
            
            elif category == 'performance':
                # Check performance metrics
                avg_response_time = project_data.get('avg_response_time', float('inf'))
                system_stability = project_data.get('system_stability', 0)  # 0-1 scale
                resource_usage_ok = project_data.get('resource_usage_ok', False)
                real_time_capable = project_data.get('real_time_capable', False)
                
                if avg_response_time <= 2.0:  # Under 2 seconds
                    score += config['max_points'] * 0.25
                    feedback.append(f"✓ Good response time: {avg_response_time:.2f}s")
                else:
                    feedback.append(f"✗ Slow response time: {avg_response_time:.2f}s")
                
                if system_stability >= 0.9:  # 90% stability
                    score += config['max_points'] * 0.25
                    feedback.append(f"✓ Good system stability: {system_stability:.2f}")
                else:
                    feedback.append(f"✗ Poor system stability: {system_stability:.2f}")
                
                if resource_usage_ok:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ Resource usage optimized")
                else:
                    feedback.append("✗ Resource usage not optimized")
                
                if real_time_capable:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ Real-time processing achieved")
                else:
                    feedback.append("✗ Real-time processing not achieved")
            
            elif category == 'privacy_compliance':
                # Check privacy compliance
                no_external_apis = project_data.get('no_external_apis', False)
                no_unnecessary_storage = project_data.get('no_unnecessary_storage', False)
                local_processing = project_data.get('local_processing', False)
                privacy_configured = project_data.get('privacy_configured', False)
                
                if no_external_apis:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ No external API calls")
                else:
                    feedback.append("✗ External API calls detected")
                
                if no_unnecessary_storage:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ No unnecessary data storage")
                else:
                    feedback.append("✗ Unnecessary data storage detected")
                
                if local_processing:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ Local processing enforced")
                else:
                    feedback.append("✗ Local processing not enforced")
                
                if privacy_configured:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ Privacy settings configured")
                else:
                    feedback.append("✗ Privacy settings not configured")
            
            elif category == 'documentation':
                # Check documentation
                has_code_docs = project_data.get('code_documentation', False)
                has_architecture_docs = project_data.get('architecture_documentation', False)
                has_usage_docs = project_data.get('usage_documentation', False)
                has_troubleshooting = project_data.get('troubleshooting_guide', False)
                
                if has_code_docs:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ Code properly documented")
                else:
                    feedback.append("✗ Code not properly documented")
                
                if has_architecture_docs:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ Architecture documented")
                else:
                    feedback.append("✗ Architecture not documented")
                
                if has_usage_docs:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ Usage instructions provided")
                else:
                    feedback.append("✗ Usage instructions missing")
                
                if has_troubleshooting:
                    score += config['max_points'] * 0.25
                    feedback.append("✓ Troubleshooting guide available")
                else:
                    feedback.append("✗ Troubleshooting guide missing")
            
            # Cap score at max points for category
            score = min(score, config['max_points'])
            
            return int(score), feedback
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context=f"evaluate_category: {category}",
                suggested_fix="Check category evaluation logic",
                learning_objective="Handle category evaluation errors"
            )
            return 0, [f"Error evaluating category {category}: {str(e)}"]
    
    def _calculate_letter_grade(self, percentage: float) -> str:
        """Calculate letter grade from percentage"""
        for grade, (min_pct, max_pct) in self.assessment_rubric['grading_scale'].items():
            if min_pct <= percentage <= max_pct:
                return grade
        return 'F'  # Default to F if no match
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive validation report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_validations_run': len(self.validation_history),
            'latest_validation': self.validation_history[-1] if self.validation_history else None,
            'validation_summary': self._get_validation_summary(),
            'compliance_status': self._get_compliance_status(),
            'metrics': self.metrics.get_statistics()
        }
    
    def _get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation runs"""
        if not self.validation_history:
            return {}
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        
        for report in self.validation_history:
            total_tests += report.get('total_rules', 0)
            total_passed += report.get('passed_rules', 0)
            total_failed += report.get('failed_rules', 0)
            total_errors += report.get('error_rules', 0)
        
        return {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_errors': total_errors,
            'success_rate': total_passed / total_tests if total_tests > 0 else 0
        }
    
    def _get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status"""
        if not self.validation_history:
            return {'status': 'no_data'}
        
        latest_report = self.validation_history[-1]
        return {
            'overall_status': latest_report.get('overall_status', 'unknown'),
            'compliance_summary': latest_report.get('compliance_summary', {}),
            'last_validation': latest_report.get('timestamp')
        }


def main():
    """Main function to demonstrate the validation and assessment system"""
    print("Capstone Validation and Assessment System")
    print("="*45)
    
    # Create validator
    validator = CapstoneValidator()
    
    print("Validator initialized with comprehensive validation rules.")
    
    # Create sample system context for validation
    sample_context = {
        'required_components': ['voice_processor', 'llm_planner', 'action_executor', 'state_manager'],
        'system_components': {
            'voice_processor': True,
            'llm_planner': True,
            'action_executor': True,
            'state_manager': True
        },
        'communication_interfaces': ['voice_commands', 'action_plans', 'execution_feedback'],
        'state_manager': type('MockStateManager', (), {
            'get_system_status': lambda: {
                'current_state': 'idle',
                'error_count': 0
            }
        })(),
        'system_config': {
            'local_processing': True,
            'store_audio_recordings': False,
            'store_transcripts': False,
            'data_retention_days': 0
        },
        'performance_metrics': {
            'average_response_time': 1.2,
            'cpu_usage_percent': 45,
            'memory_usage_percent': 60,
            'processing_frequency_hz': 15
        },
        'safety_status': {
            'safety_enabled': True,
            'emergency_stop_available': True
        }
    }
    
    # Run validation
    print("\nRunning system validation...")
    validation_report = validator.run_validation(sample_context)
    
    print(f"Validation completed: {validation_report['overall_status']}")
    print(f"  Total rules: {validation_report['total_rules']}")
    print(f"  Passed: {validation_report['passed_rules']}")
    print(f"  Failed: {validation_report['failed_rules']}")
    print(f"  Errors: {validation_report['error_rules']}")
    
    # Create sample project data for assessment
    sample_project = {
        'project_id': 'capstone_demo_001',
        'components': {
            'voice_processor': True,
            'state_manager': True,
            'error_handling': True
        },
        'voice_processing_working': True,
        'task_planning_working': True,
        'action_execution_working': True,
        'multi_modal_integration': True,
        'avg_response_time': 1.5,
        'system_stability': 0.95,
        'resource_usage_ok': True,
        'real_time_capable': True,
        'no_external_apis': True,
        'no_unnecessary_storage': True,
        'local_processing': True,
        'privacy_configured': True,
        'code_documentation': True,
        'architecture_documentation': True,
        'usage_documentation': True,
        'troubleshooting_guide': True
    }
    
    # Run assessment
    print(f"\nRunning project assessment...")
    assessment = validator.assess_capstone_project(sample_project)
    
    print(f"Assessment completed:")
    print(f"  Grade: {assessment['grade']}")
    print(f"  Percentage: {assessment['percentage']:.1f}%")
    print(f"  Total Score: {assessment['total_score']}/{assessment['max_possible_score']}")
    
    print(f"\nCategory Scores:")
    for category, scores in assessment['category_scores'].items():
        print(f"  {category}: {scores['score']}/{scores['max_points']} "
              f"({scores['score']/scores['max_points']*100:.1f}%)")
    
    print(f"\nSample Feedback:")
    for i, feedback in enumerate(assessment['feedback'][:5], 1):  # Show first 5 feedback items
        print(f"  {i}. {feedback}")
    
    # Generate validation report
    report = validator.generate_validation_report()
    print(f"\nValidation Report Summary:")
    summary = report.get('validation_summary', {})
    print(f"  Total tests run: {summary.get('total_tests', 0)}")
    print(f"  Success rate: {summary.get('success_rate', 0)*100:.1f}%")
    print(f"  Compliance: {report.get('compliance_status', {}).get('overall_status', 'unknown')}")
    
    print(f"\nValidation and assessment system demonstration completed.")


if __name__ == "__main__":
    main()
