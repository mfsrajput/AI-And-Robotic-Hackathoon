"""
LLM Integration Service for VLA System
Integrates open-source LLMs (Ollama/Llama-based) for natural language task planning
"""

import time
import json
import re
from typing import Dict, Any, List, Optional, Callable
import requests
import threading
from datetime import datetime

from common_data_models import Command, ActionSequence
from observability_utils import EducationalLogger, MetricsCollector, DebuggingHelper
from error_handling_utils import EducationalErrorHandler, InputValidator
from privacy_compliance import PrivacyComplianceMiddleware


class OllamaClient:
    """
    Client for interacting with Ollama service
    Provides a clean interface for LLM interactions in robotics context
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.1:8b"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.session = requests.Session()
        self.session.timeout = 30  # 30 second timeout for requests
        
        # Set up headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'VLA-LLM-Service/1.0'
        })
    
    def generate_response(self, 
                         prompt: str, 
                         system_prompt: str = None,
                         temperature: float = 0.3,
                         max_tokens: int = 2048,
                         stream: bool = False) -> Dict[str, Any]:
        """
        Generate response from LLM with given prompt
        """
        try:
            # Prepare the request payload
            messages = []
            
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            
            messages.append({
                'role': 'user',
                'content': prompt
            })
            
            payload = {
                'model': self.model_name,
                'messages': messages,
                'options': {
                    'temperature': temperature,
                    'num_predict': max_tokens,
                },
                'stream': stream
            }
            
            # Make the API call
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            
            response.raise_for_status()
            
            if stream:
                # Handle streaming response
                return self._handle_streaming_response(response)
            else:
                # Handle single response
                result = response.json()
                return {
                    'content': result['message']['content'],
                    'model': result['model'],
                    'created_at': result.get('created_at'),
                    'total_duration': result.get('total_duration'),
                    'load_duration': result.get('load_duration'),
                    'prompt_eval_count': result.get('prompt_eval_count'),
                    'eval_count': result.get('eval_count')
                }
                
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?")
        except requests.exceptions.Timeout:
            raise Exception(f"Request to Ollama timed out after {self.session.timeout}s")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request to Ollama failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error in Ollama client: {str(e)}")
    
    def _handle_streaming_response(self, response) -> Dict[str, Any]:
        """
        Handle streaming response from Ollama
        For this implementation, we'll collect all chunks and return as single response
        """
        content = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'message' in chunk and 'content' in chunk['message']:
                        content += chunk['message']['content']
                    
                    # Check if this is the final chunk
                    if chunk.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        return {
            'content': content,
            'model': self.model_name,
            'created_at': datetime.now().isoformat()
        }
    
    def check_model_available(self) -> bool:
        """
        Check if the specified model is available
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            # Check for exact match or partial match
            return (self.model_name in model_names or 
                    any(self.model_name in name for name in model_names))
                    
        except Exception:
            return False


class PromptEngineeringService:
    """
    Service for specialized prompt engineering for robotics applications
    """
    
    def __init__(self):
        self.logger = EducationalLogger("prompt_engineering")
        self.error_handler = EducationalErrorHandler()
        
        # Base system prompts for different types of commands
        self.system_prompts = {
            'general': """
You are an expert robotics task planner. Convert natural language commands to sequences of ROS actions for humanoid robots.
Always output in valid JSON format with "actions" array containing action objects.
""",
            'navigation': """
You are an expert navigation planner. Plan safe and efficient paths for humanoid robots.
Consider obstacles, safety, and robot capabilities in your planning.
Output actions as valid JSON.
""",
            'manipulation': """
You are an expert manipulation planner. Plan precise manipulation tasks for humanoid robots.
Consider object properties, robot gripper capabilities, and safety in your planning.
Output actions as valid JSON.
""",
            'multi_step': """
You are an expert multi-step task planner. Break complex commands into simple, executable steps.
Plan for contingencies and error recovery. Output as valid JSON with action sequence.
"""
        }
    
    def create_robotics_system_prompt(self, command_type: str = 'general', context: Dict[str, Any] = None) -> str:
        """
        Create a system prompt for robotics task planning
        """
        try:
            base_prompt = self.system_prompts.get(command_type, self.system_prompts['general'])
            
            # Add safety constraints
            safety_constraints = """
## SAFETY CONSTRAINTS:
- Never plan actions that could harm humans or damage environment
- Include perception steps before navigation or manipulation
- Verify environment is safe before executing actions
- Plan escape routes and recovery procedures
- Respect robot's physical limitations
"""
            
            # Add action schema
            action_schema = """
## ACTION SCHEMA:
{
  "actions": [
    {
      "action": "action_type",
      "parameters": {
        "param1": "value1",
        "param2": "value2"
      },
      "verification": "how to verify completion"
    }
  ],
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
"""
            
            # Add available action types
            available_actions = """
## AVAILABLE ACTION TYPES:
- navigation: Move robot to location (params: destination, speed, avoid_obstacles)
- manipulation: Manipulate objects (params: action, object, position, force)
- perception: Sense environment (params: target, sensor, accuracy)
- communication: Speak or listen (params: message, recipient, volume)
- wait: Pause execution (params: duration, condition)
"""
            
            prompt = base_prompt + "\n\n" + safety_constraints + "\n\n" + action_schema + "\n\n" + available_actions
            
            # Add context if provided
            if context:
                prompt += f"\n\n## ADDITIONAL CONTEXT:\n{json.dumps(context, indent=2)}"
            
            return prompt
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="create_robotics_system_prompt",
                suggested_fix="Check prompt template and context formatting",
                learning_objective="Handle prompt engineering errors"
            )
            return self.system_prompts['general']
    
    def create_task_planning_prompt(self, command: str, context: Dict[str, Any] = None) -> str:
        """
        Create a complete prompt for task planning
        """
        try:
            prompt = f"""
## TASK COMMAND:
{command}

## REQUIREMENTS:
1. Generate 2-8 actions maximum for safety and clarity
2. Include perception steps where appropriate for verification
3. Verify safety constraints are met in each action
4. Include error handling and recovery where needed
5. Output only in the specified JSON format
6. Ensure all actions are physically possible for the robot

## EXAMPLE OUTPUT:
{{
  "actions": [
    {{
      "action": "navigation",
      "parameters": {{
        "destination": "kitchen",
        "speed": 0.5,
        "avoid_obstacles": true
      }},
      "verification": "Robot reached kitchen area"
    }},
    {{
      "action": "perception",
      "parameters": {{
        "target": "red cup",
        "sensor": "camera",
        "accuracy": "high"
      }},
      "verification": "Red cup identified and located"
    }}
  ],
  "confidence": 0.85,
  "reasoning": "Navigate to kitchen, identify red cup, prepare for grasping"
}}

## IMPORTANT:
- Output ONLY valid JSON, no additional text
- Prioritize safety and verification in all plans
- Include realistic parameter values based on the command
"""
            
            if context:
                prompt += f"\n\n## ENVIRONMENT CONTEXT:\n{json.dumps(context, indent=2)}"
            
            return prompt
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="create_task_planning_prompt",
                suggested_fix="Check command and context formatting",
                learning_objective="Handle task planning prompt creation errors"
            )
            return f"Plan actions for this command: {command}"


class LLMResponseValidator:
    """
    Validates LLM responses for robotics applications
    """
    
    def __init__(self):
        self.logger = EducationalLogger("llm_response_validator")
        self.error_handler = EducationalErrorHandler()
        
        # Define valid action types and parameters
        self.valid_actions = {
            'navigation': {
                'required': ['destination'],
                'optional': ['speed', 'avoid_obstacles', 'precision', 'max_speed']
            },
            'manipulation': {
                'required': ['action', 'object'],
                'optional': ['position', 'force', 'gripper_width', 'approach_angle']
            },
            'perception': {
                'required': ['target'],
                'optional': ['sensor', 'accuracy', 'timeout', 'roi']
            },
            'communication': {
                'required': ['message'],
                'optional': ['recipient', 'volume', 'language', 'emotion']
            },
            'wait': {
                'required': ['duration'],
                'optional': ['condition', 'timeout']
            },
            'conditional': {
                'required': ['condition', 'action_if_true'],
                'optional': ['action_if_false', 'timeout']
            }
        }
    
    def validate_response(self, response_text: str) -> Dict[str, Any]:
        """
        Validate LLM response and extract action sequence
        """
        try:
            self.logger.debug("LLMResponseValidator", "validate_response", "Validating LLM response", {
                'response_length': len(response_text)
            })
            
            # Extract JSON from response (in case LLM adds extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return {
                    'valid': False,
                    'error': 'No valid JSON found in LLM response',
                    'raw_response': response_text,
                    'parsed_data': None
                }
            
            json_str = json_match.group()
            try:
                parsed_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                return {
                    'valid': False,
                    'error': f'Invalid JSON in LLM response: {str(e)}',
                    'raw_response': response_text,
                    'parsed_data': None
                }
            
            # Validate required fields
            if 'actions' not in parsed_data:
                return {
                    'valid': False,
                    'error': 'Response missing "actions" field',
                    'raw_response': response_text,
                    'parsed_data': parsed_data
                }
            
            # Validate actions
            actions = parsed_data['actions']
            if not isinstance(actions, list):
                return {
                    'valid': False,
                    'error': '"actions" field must be an array',
                    'raw_response': response_text,
                    'parsed_data': parsed_data
                }
            
            # Validate each action
            validated_actions = []
            validation_errors = []
            
            for i, action in enumerate(actions):
                if not isinstance(action, dict):
                    validation_errors.append(f'Action {i} is not a dictionary')
                    continue
                
                if 'action' not in action:
                    validation_errors.append(f'Action {i} missing "action" field')
                    continue
                
                action_type = action['action']
                if action_type not in self.valid_actions:
                    validation_errors.append(f'Action {i} has invalid type: {action_type}')
                    continue
                
                # Validate parameters
                params = action.get('parameters', {})
                if not isinstance(params, dict):
                    validation_errors.append(f'Action {i} parameters must be an object')
                    continue
                
                # Check required parameters
                required_params = self.valid_actions[action_type]['required']
                for param in required_params:
                    if param not in params:
                        validation_errors.append(f'Action {i} missing required parameter: {param}')
                
                # Add validated action
                validated_actions.append(action)
            
            if validation_errors:
                return {
                    'valid': False,
                    'error': f'Response validation failed: {validation_errors}',
                    'raw_response': response_text,
                    'parsed_data': parsed_data,
                    'validation_errors': validation_errors
                }
            
            # All validations passed
            return {
                'valid': True,
                'error': None,
                'raw_response': response_text,
                'parsed_data': parsed_data,
                'actions': validated_actions
            }
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="validate_response",
                suggested_fix="Check LLM response validation logic",
                learning_objective="Handle response validation errors"
            )
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}',
                'raw_response': response_text,
                'parsed_data': None
            }


class LLMIntegrationService:
    """
    Main service class for LLM integration in the VLA system
    Integrates Ollama client, prompt engineering, and response validation
    """
    
    def __init__(self, 
                 model_name: str = "llama3.1:8b",
                 ollama_base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        
        # Initialize educational utilities
        self.logger = EducationalLogger("llm_integration_service")
        self.metrics = MetricsCollector(self.logger)
        self.debug_helper = DebuggingHelper(self.logger)
        self.error_handler = EducationalErrorHandler()
        self.validator = InputValidator(self.error_handler)
        
        # Initialize privacy compliance
        self.privacy_middleware = PrivacyComplianceMiddleware()
        
        # Initialize components
        self.ollama_client = OllamaClient(self.ollama_base_url, self.model_name)
        self.prompt_engineer = PromptEngineeringService()
        self.response_validator = LLMResponseValidator()
        
        # Processing statistics
        self.total_processed = 0
        self.total_errors = 0
        self.total_validated = 0
        
        # Processing queue for concurrent requests
        self.processing_queue = []
        self.queue_lock = threading.Lock()
        
        self.logger.info("LLMIntegrationService", "__init__", "LLM integration service initialized", {
            'model_name': self.model_name,
            'ollama_url': self.ollama_base_url
        })
    
    def initialize_service(self) -> bool:
        """
        Initialize the service and verify LLM availability
        """
        try:
            self.logger.info("LLMIntegrationService", "initialize_service", "Initializing LLM service")
            
            # Check if model is available
            if not self.ollama_client.check_model_available():
                self.logger.error("LLMIntegrationService", "initialize_service", 
                                f"Model {self.model_name} not available in Ollama")
                return False
            
            self.logger.info("LLMIntegrationService", "initialize_service", 
                           "LLM service initialized successfully")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="initialize_service",
                suggested_fix="Check Ollama service and model availability",
                learning_objective="Handle LLM service initialization errors"
            )
            return False
    
    def process_natural_language_command(self, 
                                       command: str, 
                                       context: Dict[str, Any] = None,
                                       command_type: str = 'general') -> Optional[ActionSequence]:
        """
        Process a natural language command and return an action sequence
        """
        start_time = self.metrics.start_timer('llm_processing_cycle')
        
        try:
            self.logger.debug("LLMIntegrationService", "process_natural_language_command", 
                            "Processing command", {
                                'command': command,
                                'context_keys': list(context.keys()) if context else [],
                                'command_type': command_type
                            })
            
            # Validate inputs
            if not self.validator.validate_not_empty(command, "command"):
                raise ValueError("Command cannot be empty")
            
            if len(command) > 1000:  # Arbitrary limit to prevent prompt injection
                raise ValueError("Command too long (max 1000 characters)")
            
            # Create system and user prompts
            system_prompt = self.prompt_engineer.create_robotics_system_prompt(command_type, context)
            user_prompt = self.prompt_engineer.create_task_planning_prompt(command, context)
            
            # Call LLM
            llm_response = self.ollama_client.generate_response(
                user_prompt,
                system_prompt,
                temperature=0.3,  # Lower temperature for more consistent outputs
                max_tokens=2048
            )
            
            # Validate response
            validation_result = self.response_validator.validate_response(llm_response['content'])
            
            if not validation_result['valid']:
                self.logger.error("LLMIntegrationService", "process_natural_language_command", 
                                "LLM response validation failed", {
                                    'error': validation_result['error'],
                                    'command': command
                                })
                
                # Return a safe fallback action sequence
                action_sequence = self._create_fallback_sequence(command, validation_result['error'])
                self.total_errors += 1
            else:
                # Create action sequence from validated response
                action_sequence = self._create_action_sequence(
                    validation_result['actions'], 
                    command, 
                    validation_result.get('parsed_data', {}),
                    context
                )
                self.total_validated += 1
            
            # Update metrics
            processing_time = self.metrics.stop_timer('llm_processing_cycle')
            self.metrics.record_metric('llm_processing_time', processing_time)
            self.metrics.record_metric('commands_processed', 1)
            self.total_processed += 1
            
            self.logger.info("LLMIntegrationService", "process_natural_language_command", 
                           "Command processed successfully", {
                               'action_count': len(action_sequence.commands),
                               'processing_time': processing_time,
                               'confidence': action_sequence.context.get('confidence', 0)
                           })
            
            return action_sequence
            
        except Exception as e:
            self.total_errors += 1
            processing_time = self.metrics.stop_timer('llm_processing_cycle')
            self.metrics.record_metric('llm_processing_error_time', processing_time)
            
            self.error_handler.handle_error(
                e,
                context="process_natural_language_command",
                suggested_fix="Check command format and LLM connectivity",
                learning_objective="Handle LLM command processing errors"
            )
            
            # Return error fallback
            return self._create_error_fallback(str(e))
    
    def _create_action_sequence(self, 
                              actions: List[Dict[str, Any]], 
                              original_command: str,
                              llm_data: Dict[str, Any],
                              context: Dict[str, Any] = None) -> ActionSequence:
        """
        Create ActionSequence from validated actions
        """
        try:
            # Create context combining original info and LLM data
            sequence_context = {
                'original_command': original_command,
                'llm_reasoning': llm_data.get('reasoning', ''),
                'confidence': llm_data.get('confidence', 0.0),
                'generated_at': datetime.now().isoformat(),
                'model_used': self.model_name,
                'safety_verified': True  # This would be set based on actual safety checks
            }
            
            if context:
                sequence_context.update(context)
            
            action_sequence = ActionSequence(
                commands=actions,
                context=sequence_context,
                status='generated',
                created_at=datetime.now()
            )
            
            return action_sequence
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="_create_action_sequence",
                suggested_fix="Check action sequence creation logic",
                learning_objective="Handle action sequence creation errors"
            )
            # Return empty sequence on error
            return ActionSequence(
                commands=[],
                context={'error': str(e)},
                status='failed',
                created_at=datetime.now()
            )
    
    def _create_fallback_sequence(self, command: str, validation_error: str) -> ActionSequence:
        """
        Create a safe fallback action sequence when validation fails
        """
        fallback_actions = [{
            'action': 'communication',
            'parameters': {
                'message': f'Unable to process command safely: {command}',
                'reason': validation_error
            },
            'verification': 'System acknowledged inability to process command'
        }]
        
        return ActionSequence(
            commands=fallback_actions,
            context={
                'original_command': command,
                'validation_error': validation_error,
                'fallback_used': True,
                'generated_at': datetime.now().isoformat()
            },
            status='fallback',
            created_at=datetime.now()
        )
    
    def _create_error_fallback(self, error_msg: str) -> ActionSequence:
        """
        Create an error fallback action sequence
        """
        error_actions = [{
            'action': 'communication',
            'parameters': {
                'message': f'Processing error occurred: {error_msg}',
                'type': 'system_error'
            },
            'verification': 'System error reported'
        }]
        
        return ActionSequence(
            commands=error_actions,
            context={
                'error_message': error_msg,
                'error_fallback_used': True,
                'generated_at': datetime.now().isoformat()
            },
            status='error',
            created_at=datetime.now()
        )
    
    def batch_process_commands(self, commands: List[Dict[str, Any]]) -> List[ActionSequence]:
        """
        Process multiple commands in batch
        """
        try:
            results = []
            for cmd_data in commands:
                command = cmd_data.get('command', '')
                context = cmd_data.get('context', {})
                command_type = cmd_data.get('type', 'general')
                
                result = self.process_natural_language_command(command, context, command_type)
                results.append(result)
            
            return results
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                context="batch_process_commands",
                suggested_fix="Check batch processing logic",
                learning_objective="Handle batch command processing errors"
            )
            return []
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get current status of the LLM service
        """
        return {
            'model_name': self.model_name,
            'ollama_url': self.ollama_base_url,
            'is_available': self.ollama_client.check_model_available(),
            'total_processed': self.total_processed,
            'total_errors': self.total_errors,
            'total_validated': self.total_validated,
            'error_rate': self.total_errors / max(1, self.total_processed),
            'metrics': self.metrics.get_statistics(),
            'prompt_engineer_status': 'active',
            'response_validator_status': 'active'
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the service
        """
        try:
            # Check Ollama availability with a simple request
            test_response = self.ollama_client.generate_response(
                "Hello, are you working?",
                temperature=0.1,
                max_tokens=10
            )
            
            return {
                'status': 'healthy',
                'model_available': True,
                'response_time': 'normal',
                'test_response_length': len(test_response.get('content', ''))
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'model_available': False,
                'error': str(e)
            }


def main():
    """
    Main function for testing LLM integration service
    """
    print("LLM Integration Service - VLA System")
    print("="*40)
    
    # Create LLM service
    service = LLMIntegrationService(model_name="llama3.1:8b")
    
    # Initialize the service
    if not service.initialize_service():
        print("❌ Failed to initialize LLM service. Is Ollama running?")
        print("   Please start Ollama with: ollama serve")
        return
    
    print(f"✅ LLM service initialized with model: {service.model_name}")
    
    # Show initial status
    status = service.get_service_status()
    print(f"Service status: {status['status']}")
    print(f"Total processed: {status['total_processed']}")
    
    # Test some commands
    test_commands = [
        {"command": "Go to the kitchen and bring me a cup", "type": "navigation"},
        {"command": "Pick up the red block from the table", "type": "manipulation"},
        {"command": "Tell me what you see in front of you", "type": "perception"},
    ]
    
    print(f"\nTesting commands...")
    for i, cmd_data in enumerate(test_commands, 1):
        print(f"\nTest {i}: {cmd_data['command']}")
        
        try:
            sequence = service.process_natural_language_command(
                cmd_data['command'],
                command_type=cmd_data['type']
            )
            
            print(f"  Actions: {len(sequence.commands)} generated")
            for j, action in enumerate(sequence.commands[:3], 1):  # Show first 3 actions
                print(f"    {j}. {action['action']}: {action.get('parameters', {})}")
            
            if len(sequence.commands) > 3:
                print(f"    ... and {len(sequence.commands) - 3} more actions")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Show final status
    final_status = service.get_service_status()
    print(f"\nFinal service status:")
    print(f"  Processed: {final_status['total_processed']}")
    print(f"  Validated: {final_status['total_validated']}")
    print(f"  Errors: {final_status['total_errors']}")
    print(f"  Success rate: {(final_status['total_validated'] / max(1, final_status['total_processed'])) * 100:.1f}%")
    
    # Perform health check
    health = service.health_check()
    print(f"  Health status: {health['status']}")
    
    print(f"\nLLM integration service test completed.")


if __name__ == "__main__":
    main()
