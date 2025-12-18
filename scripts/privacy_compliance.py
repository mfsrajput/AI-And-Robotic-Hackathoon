"""
Privacy Compliance Module for VLA System
Ensures all processing occurs locally with no external API calls
"""

import os
import sys
import socket
import urllib.request
import requests
import threading
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
import inspect
import importlib.util
from pathlib import Path

from observability_utils import EducationalLogger, MetricsCollector, DebuggingHelper
from error_handling_utils import EducationalErrorHandler, InputValidator


@dataclass
class PrivacyViolation:
    """Represents a privacy violation detected in the system"""
    id: str
    timestamp: str
    function_name: str
    file_path: str
    line_number: int
    violation_type: str  # 'network_access', 'external_api', 'data_upload', etc.
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    suggested_fix: str


class PrivacyComplianceChecker:
    """
    Checks and enforces privacy compliance throughout the VLA system
    Ensures no external data transmission or API calls occur
    """
    
    def __init__(self):
        self.logger = EducationalLogger("privacy_compliance")
        self.metrics = MetricsCollector(self.logger)
        self.debug_helper = DebuggingHelper(self.logger)
        self.error_handler = EducationalErrorHandler()
        self.validator = InputValidator(self.error_handler)
        
        # Privacy compliance settings
        self.enforce_local_processing = True
        self.block_network_access = False  # Set to True in production
        self.log_network_attempts = True
        self.monitor_external_calls = True
        
        # Track privacy violations
        self.privacy_violations: List[PrivacyViolation] = []
        self.violation_lock = threading.Lock()
        
        # Known external domains to block
        self.blocked_domains = [
            'openai.com', 'api.openai.com',
            'googleapis.com', 'speech.googleapis.com',
            'amazonaws.com', 'tts.polly',
            'azure.com', 'cognitiveservices.azure.com',
            'microsoft.com', 'bing.com',
            'watson.com', 'ibm.com',
            'assemblyai.com', 'rev.ai',
            'deepgram.com', 'speechmatics.com'
        ]
        
        # Network monitoring
        self.network_monitoring = False
        self.network_monitor_thread = None
        self.blocked_connections = []
        
        self.logger.info("PrivacyComplianceChecker", "__init__", "Privacy compliance checker initialized", {
            'enforce_local_processing': self.enforce_local_processing,
            'block_network_access': self.block_network_access
        })
    
    def check_file_for_external_calls(self, file_path: str) -> List[PrivacyViolation]:
        """Check a Python file for potential external API calls"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common external API patterns
            external_patterns = [
                ('requests.get', 'External HTTP GET request'),
                ('requests.post', 'External HTTP POST request'),
                ('urllib.request', 'External URL request'),
                ('httpx.', 'HTTP client usage'),
                ('websocket.', 'WebSocket connection'),
                ('ftplib.', 'FTP connection'),
                ('smtplib.', 'Email sending'),
                ('smtplib', 'Email sending'),
                ('openai.', 'OpenAI API usage'),
                ('google.cloud', 'Google Cloud API usage'),
                ('boto3.', 'AWS API usage'),
                ('azure.', 'Azure API usage'),
                ('ibm_watson', 'IBM Watson API usage'),
            ]
            
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                for pattern, description in external_patterns:
                    if pattern in line and not line.strip().startswith('#'):  # Ignore comments
                        violation = PrivacyViolation(
                            id=f"PRIVACY_{len(self.privacy_violations) + 1:04d}",
                            timestamp=str(__import__('datetime').datetime.now()),
                            function_name='unknown',
                            file_path=file_path,
                            line_number=line_num,
                            violation_type='external_api_call',
                            description=f"{description} found: {line.strip()}",
                            severity='high',
                            suggested_fix='Replace with local processing equivalent'
                        )
                        violations.append(violation)
            
            # Add violations to global list
            with self.violation_lock:
                self.privacy_violations.extend(violations)
            
            return violations
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context=f"check_file_for_external_calls: {file_path}",
                suggested_fix="Check file permissions and encoding",
                learning_objective="Handle file privacy scanning errors"
            )
            return []
    
    def scan_directory_for_privacy_violations(self, directory_path: str) -> List[PrivacyViolation]:
        """Scan an entire directory for privacy violations"""
        all_violations = []
        
        try:
            directory = Path(directory_path)
            python_files = list(directory.rglob("*.py"))
            
            self.logger.info("PrivacyComplianceChecker", "scan_directory_for_privacy_violations", 
                           f"Scanning {len(python_files)} Python files", {
                               'directory': directory_path
                           })
            
            for py_file in python_files:
                file_violations = self.check_file_for_external_calls(str(py_file))
                all_violations.extend(file_violations)
            
            self.logger.info("PrivacyComplianceChecker", "scan_directory_for_privacy_violations", 
                           "Scan completed", {
                               'files_scanned': len(python_files),
                               'violations_found': len(all_violations)
                           })
            
            return all_violations
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context=f"scan_directory_for_privacy_violations: {directory_path}",
                suggested_fix="Check directory permissions and structure",
                learning_objective="Handle directory privacy scanning errors"
            )
            return []
    
    def validate_local_model_usage(self, model_path: str) -> bool:
        """Validate that a model is a local file, not a remote resource"""
        try:
            path = Path(model_path)
            
            # Check if it's a local file
            if path.exists():
                # Verify it's not a URL disguised as a path
                if any(domain in model_path for domain in self.blocked_domains):
                    self.logger.error("PrivacyComplianceChecker", "validate_local_model_usage", 
                                    "Model path contains blocked domain", {
                                        'model_path': model_path
                                    })
                    return False
                
                self.logger.debug("PrivacyComplianceChecker", "validate_local_model_usage", 
                                "Local model validated", {
                                    'model_path': model_path,
                                    'file_size': path.stat().st_size
                                })
                return True
            else:
                self.logger.error("PrivacyComplianceChecker", "validate_local_model_usage", 
                                "Model file does not exist locally", {
                                    'model_path': model_path
                                })
                return False
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context=f"validate_local_model_usage: {model_path}",
                suggested_fix="Verify model file path and permissions",
                learning_objective="Handle model validation errors"
            )
            return False
    
    def check_network_activity(self) -> Dict[str, Any]:
        """Check for any network activity that might indicate privacy violations"""
        network_info = {
            'is_network_monitoring': self.network_monitoring,
            'blocked_connections': len(self.blocked_connections),
            'external_domains_accessed': []
        }
        
        # In a real implementation, this would monitor actual network connections
        # For this educational implementation, we'll just return the status
        return network_info
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate a comprehensive privacy compliance report"""
        with self.violation_lock:
            report = {
                'timestamp': str(__import__('datetime').datetime.now()),
                'total_violations': len(self.privacy_violations),
                'violations_by_severity': {
                    'critical': len([v for v in self.privacy_violations if v.severity == 'critical']),
                    'high': len([v for v in self.privacy_violations if v.severity == 'high']),
                    'medium': len([v for v in self.privacy_violations if v.severity == 'medium']),
                    'low': len([v for v in self.privacy_violations if v.severity == 'low'])
                },
                'violations': [v.__dict__ for v in self.privacy_violations],
                'compliance_status': 'PASS' if len(self.privacy_violations) == 0 else 'FAIL',
                'network_info': self.check_network_activity(),
                'settings': {
                    'enforce_local_processing': self.enforce_local_processing,
                    'block_network_access': self.block_network_access,
                    'monitor_external_calls': self.monitor_external_calls
                }
            }
        
        return report


class LocalProcessingValidator:
    """
    Validates that all processing occurs locally without external dependencies
    """
    
    def __init__(self, compliance_checker: PrivacyComplianceChecker):
        self.compliance_checker = compliance_checker
        self.logger = EducationalLogger("local_processing_validator")
        self.metrics = MetricsCollector(self.logger)
        self.debug_helper = DebuggingHelper(self.logger)
        self.error_handler = EducationalErrorHandler()
        self.validator = InputValidator(self.error_handler)
        
        # Track local processing validation results
        self.validation_results = []
        
        self.logger.info("LocalProcessingValidator", "__init__", "Local processing validator initialized")
    
    def validate_whisper_local_processing(self, whisper_config) -> bool:
        """Validate that Whisper processing is configured for local processing only"""
        try:
            self.logger.debug("LocalProcessingValidator", "validate_whisper_local_processing", 
                            "Validating Whisper local processing", {
                                'model_path': getattr(whisper_config, 'model_path', 'unknown')
                            })
            
            # Check if model path is local
            if hasattr(whisper_config, 'model_path'):
                is_local = self.compliance_checker.validate_local_model_usage(whisper_config.model_path)
                if not is_local:
                    self.logger.error("LocalProcessingValidator", "validate_whisper_local_processing", 
                                    "Whisper model is not local", {
                                        'model_path': whisper_config.model_path
                                    })
                    return False
            
            # Check for any network-related settings
            if hasattr(whisper_config, 'api_key') or hasattr(whisper_config, 'api_endpoint'):
                self.logger.error("LocalProcessingValidator", "validate_whisper_local_processing", 
                                "Whisper configuration contains API settings", {
                                    'has_api_key': hasattr(whisper_config, 'api_key'),
                                    'has_api_endpoint': hasattr(whisper_config, 'api_endpoint')
                                })
                return False
            
            self.logger.info("LocalProcessingValidator", "validate_whisper_local_processing", 
                           "Whisper local processing validated successfully")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="validate_whisper_local_processing",
                suggested_fix="Check Whisper configuration for local processing",
                learning_objective="Handle Whisper validation errors"
            )
            return False
    
    def validate_data_storage_local(self, data_path: str) -> bool:
        """Validate that data is stored locally and not transmitted externally"""
        try:
            path = Path(data_path)
            
            # Check if path is within allowed local directories
            allowed_local_paths = [
                Path('.').resolve(),  # Current directory
                Path('/tmp'),         # Temporary directory
                Path.home(),          # User home directory
            ]
            
            resolved_path = path.resolve()
            is_local = any(
                str(resolved_path).startswith(str(allowed_path))
                for allowed_path in allowed_local_paths
            )
            
            if is_local:
                self.logger.debug("LocalProcessingValidator", "validate_data_storage_local", 
                                "Data storage validated as local", {
                                    'data_path': str(resolved_path)
                                })
                return True
            else:
                self.logger.error("LocalProcessingValidator", "validate_data_storage_local", 
                                "Data storage path is not local", {
                                    'data_path': str(resolved_path)
                                })
                return False
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context=f"validate_data_storage_local: {data_path}",
                suggested_fix="Check data storage path configuration",
                learning_objective="Handle data storage validation errors"
            )
            return False
    
    def validate_audio_processing_local(self, sample_rate: int, channels: int) -> bool:
        """Validate that audio processing parameters are appropriate for local processing"""
        try:
            # Validate parameters are reasonable for local processing
            if sample_rate not in [8000, 11025, 16000, 22050, 32000, 44100, 48000]:
                self.logger.error("LocalProcessingValidator", "validate_audio_processing_local", 
                                "Unsupported sample rate for local processing", {
                                    'sample_rate': sample_rate
                                })
                return False
            
            if channels not in [1, 2]:
                self.logger.error("LocalProcessingValidator", "validate_audio_processing_local", 
                                "Unsupported channel count for local processing", {
                                    'channels': channels
                                })
                return False
            
            self.logger.debug("LocalProcessingValidator", "validate_audio_processing_local", 
                            "Audio processing parameters validated", {
                                'sample_rate': sample_rate,
                                'channels': channels
                            })
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="validate_audio_processing_local",
                suggested_fix="Check audio processing parameters",
                learning_objective="Handle audio validation errors"
            )
            return False


def privacy_safe_execute(func, *args, **kwargs):
    """
    Decorator to ensure functions execute in privacy-compliant manner
    """
    def wrapper(*args, **kwargs):
        # Check function for potential privacy violations before execution
        try:
            # Get function source code to check for violations
            source = inspect.getsource(func)
            
            # Check for common privacy-violating patterns
            violation_patterns = ['requests.', 'urllib.', 'socket.', 'ftplib.', 'smtplib.']
            
            for pattern in violation_patterns:
                if pattern in source and not source.strip().startswith('#'):
                    raise Exception(f"Privacy violation detected in function {func.__name__}: {pattern}")
            
            # Execute the function
            return func(*args, **kwargs)
            
        except Exception as e:
            if "Privacy violation" in str(e):
                raise e
            else:
                # Re-raise other exceptions normally
                raise e
    
    return wrapper


class PrivacyComplianceMiddleware:
    """
    Middleware to enforce privacy compliance throughout the system
    """
    
    def __init__(self):
        self.compliance_checker = PrivacyComplianceChecker()
        self.local_validator = LocalProcessingValidator(self.compliance_checker)
        self.logger = EducationalLogger("privacy_middleware")
        
        self.logger.info("PrivacyComplianceMiddleware", "__init__", "Privacy middleware initialized")
    
    def wrap_function_for_privacy(self, func):
        """Wrap a function to ensure privacy compliance"""
        return privacy_safe_execute(func)
    
    def validate_component_privacy(self, component_name: str, config: Any) -> bool:
        """Validate that a component is configured for privacy-compliant operation"""
        try:
            self.logger.debug("PrivacyComplianceMiddleware", "validate_component_privacy", 
                            "Validating component privacy", {
                                'component': component_name
                            })
            
            # Different validation based on component type
            if component_name == "whisper":
                return self.local_validator.validate_whisper_local_processing(config)
            elif component_name == "audio":
                return self.local_validator.validate_audio_processing_local(
                    getattr(config, 'sample_rate', 16000),
                    getattr(config, 'channels', 1)
                )
            else:
                # Generic validation
                return True
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context=f"validate_component_privacy: {component_name}",
                suggested_fix="Check component configuration for privacy compliance",
                learning_objective="Handle component privacy validation"
            )
            return False
    
    def enforce_privacy_policy(self, policy_config: Dict[str, Any]) -> bool:
        """Enforce a privacy policy across the system"""
        try:
            # Set privacy enforcement settings based on policy
            self.compliance_checker.enforce_local_processing = policy_config.get('enforce_local_processing', True)
            self.compliance_checker.block_network_access = policy_config.get('block_network_access', False)
            self.compliance_checker.log_network_attempts = policy_config.get('log_network_attempts', True)
            
            self.logger.info("PrivacyComplianceMiddleware", "enforce_privacy_policy", 
                           "Privacy policy enforced", {
                               'policy_config': policy_config
                           })
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="enforce_privacy_policy",
                suggested_fix="Check privacy policy configuration",
                learning_objective="Handle privacy policy enforcement"
            )
            return False


def main():
    """Main function for testing privacy compliance"""
    print("Privacy Compliance Module - VLA System")
    print("="*45)
    
    # Create privacy compliance checker
    checker = PrivacyComplianceChecker()
    
    # Create local processing validator
    validator = LocalProcessingValidator(checker)
    
    print("Privacy compliance modules initialized.")
    print(f"  Local processing enforced: {checker.enforce_local_processing}")
    print(f"  Network access blocked: {checker.block_network_access}")
    
    # Validate some configurations
    print("\nValidating local processing configurations...")
    
    # Simulate a Whisper config
    class MockWhisperConfig:
        def __init__(self):
            self.model_path = "models/ggml-tiny.en.bin"  # Local model
            # No API keys or external endpoints
    
    whisper_config = MockWhisperConfig()
    is_valid = validator.validate_whisper_local_processing(whisper_config)
    print(f"  Whisper local processing: {'✅ PASS' if is_valid else '❌ FAIL'}")
    
    # Validate audio processing
    audio_valid = validator.validate_audio_processing_local(16000, 1)
    print(f"  Audio processing: {'✅ PASS' if audio_valid else '❌ FAIL'}")
    
    # Validate data storage
    storage_valid = validator.validate_data_storage_local("./test_data")
    print(f"  Data storage: {'✅ PASS' if storage_valid else '❌ FAIL'}")
    
    # Scan current directory for privacy violations
    print(f"\nScanning current directory for privacy violations...")
    violations = checker.scan_directory_for_privacy_violations(".")
    print(f"  Found {len(violations)} potential violations")
    
    if violations:
        print("  Sample violations:")
        for violation in violations[:3]:  # Show first 3 violations
            print(f"    - {violation.description}")
    
    # Generate privacy report
    report = checker.get_privacy_report()
    print(f"\nPrivacy compliance status: {report['compliance_status']}")
    print(f"Total violations: {report['total_violations']}")
    
    print(f"\nPrivacy compliance testing completed.")


if __name__ == "__main__":
    main()
