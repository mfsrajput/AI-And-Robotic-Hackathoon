"""
Observability utilities for VLA system with educational focus
Provides comprehensive logging, metrics, and debugging tools for students
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
import functools
import traceback
import sys


@dataclass
class LogEntry:
    """Structure for educational log entries"""
    timestamp: datetime
    level: str
    module: str
    function: str
    message: str
    details: Dict[str, Any]
    student_relevant: bool = True  # Whether this log is relevant for students


class EducationalLogger:
    """
    Logger designed specifically for educational purposes in the VLA system
    Provides detailed, student-friendly logs with educational context
    """
    
    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.level = getattr(logging, level.upper())
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Create formatter that's educational-friendly
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _log_educational_entry(self, level: str, module: str, function: str, 
                             message: str, details: Dict[str, Any] = None):
        """Create and log an educational entry"""
        if details is None:
            details = {}
        
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            module=module,
            function=function,
            message=message,
            details=details
        )
        
        # Format educational message
        edu_msg = f"[{module}.{function}] {message}"
        if details:
            edu_msg += f" | Details: {json.dumps(details, default=str)}"
        
        # Log based on level
        if level == "DEBUG":
            self.logger.debug(edu_msg)
        elif level == "INFO":
            self.logger.info(edu_msg)
        elif level == "WARNING":
            self.logger.warning(edu_msg)
        elif level == "ERROR":
            self.logger.error(edu_msg)
        elif level == "CRITICAL":
            self.logger.critical(edu_msg)
    
    def debug(self, module: str, function: str, message: str, details: Dict[str, Any] = None):
        """Educational debug logging"""
        if self.level <= logging.DEBUG:
            self._log_educational_entry("DEBUG", module, function, message, details)
    
    def info(self, module: str, function: str, message: str, details: Dict[str, Any] = None):
        """Educational info logging"""
        if self.level <= logging.INFO:
            self._log_educational_entry("INFO", module, function, message, details)
    
    def warning(self, module: str, function: str, message: str, details: Dict[str, Any] = None):
        """Educational warning logging"""
        if self.level <= logging.WARNING:
            self._log_educational_entry("WARNING", module, function, message, details)
    
    def error(self, module: str, function: str, message: str, details: Dict[str, Any] = None):
        """Educational error logging"""
        if self.level <= logging.ERROR:
            self._log_educational_entry("ERROR", module, function, message, details)
    
    def critical(self, module: str, function: str, message: str, details: Dict[str, Any] = None):
        """Educational critical logging"""
        if self.level <= logging.CRITICAL:
            self._log_educational_entry("CRITICAL", module, function, message, details)


class MetricsCollector:
    """
    Collects and reports performance metrics for educational analysis
    """
    
    def __init__(self, logger: EducationalLogger):
        self.logger = logger
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, metric_name: str):
        """Start timing for a specific metric"""
        self.start_times[metric_name] = time.time()
        self.logger.debug("Metrics", "start_timer", f"Started timer for {metric_name}")
    
    def stop_timer(self, metric_name: str) -> float:
        """Stop timing and return elapsed time"""
        if metric_name in self.start_times:
            elapsed = time.time() - self.start_times[metric_name]
            self.logger.debug("Metrics", "stop_timer", f"Timer {metric_name} stopped: {elapsed:.3f}s")
            
            # Store metric
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(elapsed)
            
            del self.start_times[metric_name]
            return elapsed
        else:
            self.logger.warning("Metrics", "stop_timer", f"No timer found for {metric_name}")
            return 0.0
    
    def record_metric(self, metric_name: str, value: float):
        """Record a specific metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        self.logger.debug("Metrics", "record_metric", f"Recorded {metric_name}: {value}")
    
    def get_average(self, metric_name: str) -> Optional[float]:
        """Get average value for a metric"""
        if metric_name in self.metrics and self.metrics[metric_name]:
            avg = sum(self.metrics[metric_name]) / len(self.metrics[metric_name])
            return avg
        return None
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive statistics for all metrics"""
        stats = {}
        for name, values in self.metrics.items():
            if values:
                stats[name] = {
                    'count': len(values),
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'total': sum(values)
                }
        return stats
    
    def print_summary(self):
        """Print a summary of collected metrics"""
        stats = self.get_statistics()
        self.logger.info("Metrics", "print_summary", "Performance Metrics Summary", stats)


def educational_trace(func: Callable) -> Callable:
    """
    Decorator for educational tracing - logs function entry/exit with timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = EducationalLogger(f"trace.{func.__module__}")
        
        # Log function entry
        logger.info(
            func.__module__, 
            func.__name__, 
            f"Entering function", 
            {
                'args_count': len(args),
                'kwargs_count': len(kwargs),
                'function': func.__name__
            }
        )
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            logger.info(
                func.__module__,
                func.__name__,
                f"Exiting function successfully",
                {
                    'duration': f"{elapsed:.3f}s",
                    'function': func.__name__,
                    'result_type': type(result).__name__ if result is not None else 'None'
                }
            )
            
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                func.__module__,
                func.__name__,
                f"Function raised exception",
                {
                    'duration': f"{elapsed:.3f}s",
                    'function': func.__name__,
                    'exception': str(e),
                    'exception_type': type(e).__name__
                }
            )
            raise
    
    return wrapper


class DebuggingHelper:
    """
    Provides debugging utilities for educational purposes
    """
    
    def __init__(self, logger: EducationalLogger):
        self.logger = logger
    
    def print_variables(self, variables: Dict[str, Any], context: str = ""):
        """Print variables with their types and values for debugging"""
        var_info = {}
        for name, value in variables.items():
            var_info[name] = {
                'type': type(value).__name__,
                'value': str(value)[:100] + "..." if len(str(value)) > 100 else str(value),  # Limit length
                'length': len(value) if hasattr(value, '__len__') else None
            }
        
        self.logger.info("Debug", "print_variables", f"Variable inspection in {context}", var_info)
    
    def trace_execution_path(self, path_point: str, details: Dict[str, Any] = None):
        """Trace execution path for debugging"""
        self.logger.debug("Debug", "trace_execution", f"Execution path: {path_point}", details)
    
    def validate_state(self, state: Dict[str, Any], expected_keys: list):
        """Validate that state contains expected keys"""
        missing_keys = [key for key in expected_keys if key not in state]
        if missing_keys:
            self.logger.error("Debug", "validate_state", "State validation failed", {
                'missing_keys': missing_keys,
                'available_keys': list(state.keys())
            })
            return False
        else:
            self.logger.info("Debug", "validate_state", "State validation passed", {
                'validated_keys': expected_keys
            })
            return True
    
    def log_exception_context(self, e: Exception, context: str = ""):
        """Log exception with full context for educational debugging"""
        self.logger.error("Debug", "log_exception_context", f"Exception in {context}: {str(e)}", {
            'exception_type': type(e).__name__,
            'exception_args': str(e.args),
            'traceback': traceback.format_exc()
        })


class SystemMonitor:
    """
    Monitors system health and performance for educational observability
    """
    
    def __init__(self, logger: EducationalLogger):
        self.logger = logger
        self.healthy_components = set()
        self.unhealthy_components = set()
    
    def register_component(self, component_name: str):
        """Register a system component for monitoring"""
        self.healthy_components.add(component_name)
        self.logger.info("Monitor", "register_component", f"Registered component: {component_name}")
    
    def report_health(self, component_name: str, is_healthy: bool, details: Dict[str, Any] = None):
        """Report health status of a component"""
        if is_healthy:
            self.healthy_components.add(component_name)
            self.unhealthy_components.discard(component_name)
            status = "HEALTHY"
        else:
            self.unhealthy_components.add(component_name)
            self.healthy_components.discard(component_name)
            status = "UNHEALTHY"
        
        self.logger.info("Monitor", "report_health", f"Component {component_name} status: {status}", details)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health report"""
        total_components = len(self.healthy_components) + len(self.unhealthy_components)
        healthy_count = len(self.healthy_components)
        
        health_report = {
            'total_components': total_components,
            'healthy_components': list(self.healthy_components),
            'unhealthy_components': list(self.unhealthy_components),
            'health_percentage': (healthy_count / total_components * 100) if total_components > 0 else 100,
            'system_status': 'HEALTHY' if not self.unhealthy_components else 'DEGRADED'
        }
        
        return health_report
    
    def log_system_status(self):
        """Log current system status"""
        health_report = self.get_system_health()
        self.logger.info("Monitor", "log_system_status", "System health report", health_report)


# Global instances for easy access in educational code
default_logger = EducationalLogger("vla_system")
metrics_collector = MetricsCollector(default_logger)
debug_helper = DebuggingHelper(default_logger)
system_monitor = SystemMonitor(default_logger)


def setup_educational_logging(name: str = "vla_system", level: str = "INFO") -> tuple:
    """
    Set up educational logging utilities for a module
    Returns (logger, metrics_collector, debug_helper, system_monitor)
    """
    logger = EducationalLogger(name, level)
    metrics = MetricsCollector(logger)
    debug = DebuggingHelper(logger)
    monitor = SystemMonitor(logger)
    
    return logger, metrics, debug, monitor


# Example usage for educational purposes
if __name__ == "__main__":
    # Example of using educational logging
    logger = EducationalLogger("example_module")
    
    logger.info("Example", "main", "Starting example", {"example_type": "logging"})
    
    # Example of using metrics
    metrics = MetricsCollector(logger)
    metrics.start_timer("example_operation")
    
    # Simulate some work
    time.sleep(0.1)
    
    elapsed = metrics.stop_timer("example_operation")
    logger.info("Example", "main", f"Operation completed in {elapsed:.3f}s")
    
    # Example of using debugging helper
    debug = DebuggingHelper(logger)
    test_vars = {"x": 42, "y": [1, 2, 3, 4, 5], "name": "test"}
    debug.print_variables(test_vars, "example_function")
    
    # Example of using system monitor
    monitor = SystemMonitor(logger)
    monitor.register_component("voice_processor")
    monitor.register_component("llm_planner")
    monitor.report_health("voice_processor", True, {"latency": 0.2})
    monitor.report_health("llm_planner", False, {"error": "timeout"})
    
    health = monitor.get_system_health()
    logger.info("Example", "main", "Final system health", health)
    
    # Print metrics summary
    metrics.print_summary()
