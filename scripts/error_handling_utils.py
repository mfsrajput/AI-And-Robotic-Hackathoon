"""
Error handling utilities for VLA system with educational focus
Provides comprehensive error handling with clear explanations for students
"""

import logging
import traceback
import sys
from typing import Type, Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
import functools


class ErrorCategory(Enum):
    """Categories of errors for educational classification"""
    INPUT_ERROR = "input_error"
    PROCESSING_ERROR = "processing_error"
    CONNECTION_ERROR = "connection_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"
    USER_ERROR = "user_error"
    CONFIGURATION_ERROR = "configuration_error"
    PRIVACY_ERROR = "privacy_error"


@dataclass
class EducationalError:
    """Structure for educational error information"""
    error_id: str
    category: ErrorCategory
    message: str
    details: Dict[str, Any]
    suggested_fix: str
    learning_objective: str
    timestamp: str
    severity: str  # debug, info, warning, error, critical


class EducationalErrorHandler:
    """
    Error handler designed specifically for educational purposes
    Provides detailed error information with learning context
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[EducationalError] = []
        self.error_counts: Dict[str, int] = {}
    
    def handle_error(
        self, 
        error: Exception, 
        context: str = "", 
        suggested_fix: str = "Check the code and error message for details",
        learning_objective: str = "Understand error handling and debugging techniques",
        severity: str = "error"
    ) -> EducationalError:
        """Handle an error with educational context"""
        
        error_id = f"ERR_{len(self.error_history) + 1:04d}"
        category = self._categorize_error(error)
        
        # Create educational error object
        educational_error = EducationalError(
            error_id=error_id,
            category=category,
            message=str(error),
            details={
                "error_type": type(error).__name__,
                "context": context,
                "traceback": traceback.format_exc(),
                "args": error.args if hasattr(error, 'args') else [],
            },
            suggested_fix=suggested_fix,
            learning_objective=learning_objective,
            timestamp=str(__import__('datetime').datetime.now()),
            severity=severity
        )
        
        # Store in history
        self.error_history.append(educational_error)
        
        # Update error counts
        error_key = f"{category.value}_{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log the error with educational context
        self._log_educational_error(educational_error)
        
        return educational_error
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error for educational purposes"""
        error_type = type(error).__name__.lower()
        
        if 'input' in error_type or 'value' in error_type or 'key' in error_type:
            return ErrorCategory.INPUT_ERROR
        elif 'connection' in error_type or 'timeout' in error_type or 'network' in error_type.lower():
            return ErrorCategory.CONNECTION_ERROR
        elif 'validation' in error_type or 'assert' in error_type:
            return ErrorCategory.VALIDATION_ERROR
        elif 'config' in error_type or 'file' in error_type or 'path' in error_type:
            return ErrorCategory.CONFIGURATION_ERROR
        elif 'privacy' in error_type.lower() or 'security' in error_type.lower():
            return ErrorCategory.PRIVACY_ERROR
        else:
            return ErrorCategory.SYSTEM_ERROR
    
    def _log_educational_error(self, educational_error: EducationalError):
        """Log error with educational context"""
        log_msg = f"[{educational_error.error_id}] {educational_error.message}"
        log_msg += f"\n  Category: {educational_error.category.value}"
        log_msg += f"\n  Context: {educational_error.details.get('context', 'N/A')}"
        log_msg += f"\n  Suggested Fix: {educational_error.suggested_fix}"
        log_msg += f"\n  Learning Objective: {educational_error.learning_objective}"
        
        # Log based on severity
        if educational_error.severity == "debug":
            self.logger.debug(log_msg)
        elif educational_error.severity == "info":
            self.logger.info(log_msg)
        elif educational_error.severity == "warning":
            self.logger.warning(log_msg)
        elif educational_error.severity == "error":
            self.logger.error(log_msg)
        elif educational_error.severity == "critical":
            self.logger.critical(log_msg)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors for educational analysis"""
        return {
            "total_errors": len(self.error_history),
            "error_counts": self.error_counts,
            "categories": {category.value: sum(1 for e in self.error_history if e.category == category) 
                          for category in ErrorCategory},
            "recent_errors": [e.error_id for e in self.error_history[-5:]]  # Last 5 errors
        }
    
    def print_error_analysis(self):
        """Print educational error analysis"""
        summary = self.get_error_summary()
        
        print("\n" + "="*60)
        print("EDUCATIONAL ERROR ANALYSIS")
        print("="*60)
        print(f"Total Errors Encountered: {summary['total_errors']}")
        print("\nError Categories:")
        for category, count in summary['categories'].items():
            if count > 0:
                print(f"  {category}: {count}")
        
        print(f"\nError Counts by Type:")
        for error_type, count in summary['error_counts'].items():
            print(f"  {error_type}: {count}")
        
        if summary['recent_errors']:
            print(f"\nRecent Errors (last 5): {', '.join(summary['recent_errors'])}")
        
        print("="*60)


def safe_execute(
    func: Callable, 
    error_handler: EducationalErrorHandler = None,
    context: str = "",
    suggested_fix: str = "Check function implementation and inputs",
    learning_objective: str = "Understand safe execution patterns"
) -> Any:
    """
    Safely execute a function with educational error handling
    """
    if error_handler is None:
        error_handler = EducationalErrorHandler()
    
    try:
        return func()
    except Exception as e:
        error = error_handler.handle_error(
            e,
            context=f"{context} - {func.__name__}",
            suggested_fix=suggested_fix,
            learning_objective=learning_objective
        )
        return None


def educational_try_except(func: Callable) -> Callable:
    """
    Decorator that adds educational error handling to functions
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        handler = EducationalErrorHandler()
        context = f"{func.__module__}.{func.__name__}"
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Determine suggested fix based on function name and error type
            func_name = func.__name__.lower()
            if 'validate' in func_name:
                suggested_fix = "Check input validation logic and constraints"
                learning_obj = "Understand input validation and error prevention"
            elif 'process' in func_name:
                suggested_fix = "Verify data processing steps and input formats"
                learning_obj = "Learn about data processing error handling"
            elif 'connect' in func_name:
                suggested_fix = "Check network connectivity and configuration"
                learning_obj = "Understand network error handling and resilience"
            else:
                suggested_fix = "Review function implementation and error conditions"
                learning_obj = "Develop systematic debugging skills"
            
            handler.handle_error(
                e,
                context=context,
                suggested_fix=suggested_fix,
                learning_objective=learning_obj
            )
            raise  # Re-raise the exception after logging
    
    return wrapper


class InputValidator:
    """
    Educational input validation with detailed error reporting
    """
    
    def __init__(self, error_handler: EducationalErrorHandler = None):
        self.error_handler = error_handler or EducationalErrorHandler()
    
    def validate_type(self, value: Any, expected_type: Type, param_name: str = "parameter") -> bool:
        """Validate that a value is of the expected type"""
        if not isinstance(value, expected_type):
            error_msg = f"{param_name} must be of type {expected_type.__name__}, got {type(value).__name__}"
            self.error_handler.handle_error(
                TypeError(error_msg),
                context=f"Input validation for {param_name}",
                suggested_fix=f"Ensure {param_name} is of type {expected_type.__name__}",
                learning_objective="Understand type validation and type safety"
            )
            return False
        return True
    
    def validate_range(self, value: Union[int, float], min_val: Union[int, float], 
                      max_val: Union[int, float], param_name: str = "parameter") -> bool:
        """Validate that a value is within the specified range"""
        if not (min_val <= value <= max_val):
            error_msg = f"{param_name} must be between {min_val} and {max_val}, got {value}"
            self.error_handler.handle_error(
                ValueError(error_msg),
                context=f"Range validation for {param_name}",
                suggested_fix=f"Ensure {param_name} is between {min_val} and {max_val}",
                learning_objective="Learn about input range validation"
            )
            return False
        return True
    
    def validate_not_empty(self, value: Union[str, list, dict], param_name: str = "parameter") -> bool:
        """Validate that a value is not empty"""
        if hasattr(value, '__len__') and len(value) == 0:
            error_msg = f"{param_name} cannot be empty"
            self.error_handler.handle_error(
                ValueError(error_msg),
                context=f"Empty validation for {param_name}",
                suggested_fix=f"Provide a non-empty value for {param_name}",
                learning_objective="Understand input validation best practices"
            )
            return False
        return True
    
    def validate_enum(self, value: Any, valid_values: List[Any], param_name: str = "parameter") -> bool:
        """Validate that a value is in a list of valid values"""
        if value not in valid_values:
            error_msg = f"{param_name} must be one of {valid_values}, got {value}"
            self.error_handler.handle_error(
                ValueError(error_msg),
                context=f"Enum validation for {param_name}",
                suggested_fix=f"Use one of the valid values: {valid_values}",
                learning_objective="Learn about constrained value validation"
            )
            return False
        return True


class EducationalRetryHandler:
    """
    Retry mechanism with educational logging for transient failures
    """
    
    def __init__(self, error_handler: EducationalErrorHandler = None):
        self.error_handler = error_handler or EducationalErrorHandler()
    
    def retry_with_backoff(
        self,
        func: Callable,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
        retryable_exceptions: List[Type[Exception]] = None
    ) -> Any:
        """
        Execute function with exponential backoff retry
        """
        if retryable_exceptions is None:
            retryable_exceptions = [ConnectionError, TimeoutError, OSError]
        
        for attempt in range(max_attempts):
            try:
                return func()
            except tuple(retryable_exceptions) as e:
                if attempt == max_attempts - 1:  # Last attempt
                    self.error_handler.handle_error(
                        e,
                        context=f"Final attempt failed after {max_attempts} tries",
                        suggested_fix="Check system connectivity and resource availability",
                        learning_objective="Understand retry mechanisms and transient failure handling"
                    )
                    raise e
                else:
                    delay = base_delay * (backoff_factor ** attempt)
                    self.error_handler.handle_error(
                        e,
                        context=f"Attempt {attempt + 1} failed, retrying in {delay}s",
                        suggested_fix=f"Wait {delay}s and retry automatically",
                        learning_objective="Learn about retry strategies with exponential backoff"
                    )
                    import time
                    time.sleep(delay)
            except Exception as e:
                # Non-retryable exception
                self.error_handler.handle_error(
                    e,
                    context="Non-retryable error occurred",
                    suggested_fix="Fix the underlying issue before retrying",
                    learning_objective="Distinguish between retryable and non-retryable errors"
                )
                raise e


class PrivacyComplianceError(Exception):
    """Custom exception for privacy compliance violations"""
    pass


class SafetyValidationError(Exception):
    """Custom exception for safety validation failures"""
    pass


def privacy_safe_execute(func: Callable) -> Callable:
    """
    Decorator that ensures privacy compliance in execution
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check for potential privacy violations
        for arg in args:
            if hasattr(arg, '__class__') and 'password' in str(arg).lower():
                raise PrivacyComplianceError("Potential password in arguments - privacy violation")
        
        for key, value in kwargs.items():
            if 'password' in key.lower() or 'secret' in key.lower():
                raise PrivacyComplianceError(f"Potential secret in parameter '{key}' - privacy violation")
        
        return func(*args, **kwargs)
    
    return wrapper


def safety_validated_execute(func: Callable) -> Callable:
    """
    Decorator that performs safety validation before execution
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Perform safety validation
        # This is a simplified example - real implementation would be more complex
        result = func(*args, **kwargs)
        
        # Verify result is safe (example: no negative values for coordinates)
        if isinstance(result, (int, float)) and result < 0 and 'distance' in func.__name__:
            raise SafetyValidationError(f"Negative distance result: {result}")
        
        return result
    
    return wrapper


# Global error handler for the VLA system
vla_error_handler = EducationalErrorHandler()


def setup_educational_error_handling(name: str = "vla_system") -> tuple:
    """
    Set up educational error handling for a module
    Returns (error_handler, validator, retry_handler)
    """
    handler = EducationalErrorHandler()
    validator = InputValidator(handler)
    retry_handler = EducationalRetryHandler(handler)
    
    return handler, validator, retry_handler


# Example usage for educational purposes
if __name__ == "__main__":
    print("Educational Error Handling Examples")
    print("="*50)
    
    # Create error handler
    handler = EducationalErrorHandler()
    
    # Example 1: Handling a simple error
    print("\n1. Handling a simple error:")
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        handler.handle_error(
            e,
            context="Mathematical operation",
            suggested_fix="Avoid division by zero by checking denominator first",
            learning_objective="Understand common mathematical errors and prevention"
        )
    
    # Example 2: Input validation
    print("\n2. Input validation example:")
    validator = InputValidator(handler)
    validator.validate_type("not_a_number", int, "user_age")
    validator.validate_range(150, 0, 120, "user_age")
    
    # Example 3: Safe execution
    print("\n3. Safe execution example:")
    def risky_function():
        raise ValueError("This is a test error")
    
    result = safe_execute(
        risky_function,
        handler,
        context="Test function execution",
        suggested_fix="Handle the ValueError in the function",
        learning_objective="Practice safe execution patterns"
    )
    
    # Example 4: Decorator usage
    print("\n4. Decorator usage example:")
    @educational_try_except
    def example_function(x, y):
        if y == 0:
            raise ValueError("Division by zero")
        return x / y
    
    try:
        result = example_function(10, 0)
    except ValueError:
        print("Caught and handled by decorator")
    
    # Print error analysis
    handler.print_error_analysis()
