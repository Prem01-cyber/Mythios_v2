"""
Tool Executor - Command Execution Engine for Security Tools

This module provides a robust execution engine for security tools defined in YAML manifests.
It handles command execution, output capture, timeout management, and result structuring.
"""

import subprocess
import time
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json
import signal
from enum import Enum
import random


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Execution status codes"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"
    KILLED = "killed"


class FailureType(Enum):
    """Types of failures for retry logic"""
    TRANSIENT = "transient"  # Network issues, temporary unavailability
    PERMANENT = "permanent"  # Invalid command, permission denied
    UNKNOWN = "unknown"


@dataclass
class ExecutionContext:
    """
    Context information for tool execution tracking
    
    Tracks: what tool, when executed, why executed, and related metadata
    """
    tool_name: str
    command_type: str
    reason: str  # Why this tool is being executed
    target: Optional[str] = None  # Target being scanned/tested
    parent_execution_id: Optional[str] = None  # For chained executions
    execution_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    tags: List[str] = field(default_factory=list)  # Custom tags for categorization
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context
    initiated_by: Optional[str] = None  # User, system, or agent name
    priority: int = 5  # Priority level (1-10, higher = more important)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class RetryAttempt:
    """Information about a single retry attempt"""
    attempt_number: int
    timestamp: str
    exit_code: int
    error_message: Optional[str]
    execution_time: float
    backoff_delay: float  # Delay before this attempt


@dataclass
class ExecutionResult:
    """Structured result from tool execution"""
    tool_name: str
    command: str
    command_type: str
    status: ExecutionStatus
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    timestamp: str
    pid: Optional[int] = None
    timeout_occurred: bool = False
    error_message: Optional[str] = None
    success_indicators_found: List[str] = None
    failure_indicators_found: List[str] = None
    
    # Execution context tracking
    context: Optional[ExecutionContext] = None
    
    # Retry information
    retry_count: int = 0
    retry_attempts: List[RetryAttempt] = None
    failure_type: Optional[FailureType] = None
    is_retryable: bool = True
    
    def __post_init__(self):
        if self.success_indicators_found is None:
            self.success_indicators_found = []
        if self.failure_indicators_found is None:
            self.failure_indicators_found = []
        if self.retry_attempts is None:
            self.retry_attempts = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with enum handling"""
        data = asdict(self)
        data['status'] = self.status.value
        if self.failure_type:
            data['failure_type'] = self.failure_type.value
        if self.context:
            data['context'] = self.context.to_dict()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class RetryPolicy:
    """Configuration for retry behavior"""
    max_attempts: int = 3  # Maximum retry attempts
    initial_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd
    retry_on_timeout: bool = True  # Retry on timeout errors
    retry_on_connection_error: bool = True  # Retry on connection errors
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given retry attempt with exponential backoff"""
        delay = min(self.initial_delay * (self.backoff_multiplier ** attempt), self.max_delay)
        
        if self.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


class ToolManifest:
    """Represents a tool manifest loaded from YAML"""
    
    def __init__(self, manifest_path: Path):
        self.path = manifest_path
        self.data = self._load_manifest()
        
    def _load_manifest(self) -> Dict[str, Any]:
        """Load and parse YAML manifest"""
        try:
            with open(self.path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest {self.path}: {e}")
            raise
    
    @property
    def name(self) -> str:
        return self.data.get('name', '')
    
    @property
    def version(self) -> str:
        return self.data.get('version', '')
    
    @property
    def category(self) -> str:
        return self.data.get('category', '')
    
    @property
    def commands(self) -> Dict[str, Any]:
        return self.data.get('commands', {})
    
    @property
    def success_indicators(self) -> List[str]:
        return self.data.get('success_indicators', [])
    
    @property
    def failure_indicators(self) -> List[str]:
        return self.data.get('failure_indicators', [])
    
    @property
    def requires_root(self) -> bool:
        return self.data.get('requires_root', False)
    
    def get_command_template(self, command_type: str) -> Optional[str]:
        """Get command template for a specific command type"""
        cmd_data = self.commands.get(command_type)
        if cmd_data:
            return cmd_data.get('template')
        return None
    
    def get_command_info(self, command_type: str) -> Optional[Dict[str, Any]]:
        """Get full command information"""
        return self.commands.get(command_type)


class ToolExecutor:
    """
    Main tool execution engine
    
    Handles:
    - Loading tool manifests
    - Command template substitution
    - Process execution and monitoring
    - Output capture (stdout, stderr, exit code)
    - Timeout management
    - Result structuring
    """
    
    def __init__(
        self, 
        manifests_dir: str = "tools/manifests",
        default_retry_policy: Optional[RetryPolicy] = None
    ):
        self.manifests_dir = Path(manifests_dir)
        self.manifests: Dict[str, ToolManifest] = {}
        self.default_retry_policy = default_retry_policy or RetryPolicy()
        self.execution_history: List[ExecutionResult] = []  # Track all executions
        self._load_all_manifests()
    
    def _load_all_manifests(self):
        """Load all YAML manifests from the manifests directory"""
        if not self.manifests_dir.exists():
            logger.warning(f"Manifests directory not found: {self.manifests_dir}")
            return
        
        manifest_files = list(self.manifests_dir.glob("*.yaml")) + list(self.manifests_dir.glob("*.yml"))
        
        for manifest_file in manifest_files:
            try:
                manifest = ToolManifest(manifest_file)
                self.manifests[manifest.name] = manifest
                logger.info(f"Loaded manifest: {manifest.name} v{manifest.version}")
            except Exception as e:
                logger.error(f"Failed to load manifest {manifest_file}: {e}")
        
        logger.info(f"Loaded {len(self.manifests)} tool manifests")
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.manifests.keys())
    
    def get_tool_commands(self, tool_name: str) -> List[str]:
        """Get available commands for a specific tool"""
        if tool_name not in self.manifests:
            return []
        return list(self.manifests[tool_name].commands.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get full tool information from manifest"""
        if tool_name not in self.manifests:
            return None
        return self.manifests[tool_name].data
    
    def _classify_failure(
        self, 
        exit_code: int, 
        stderr: str, 
        error_message: Optional[str],
        timeout_occurred: bool
    ) -> FailureType:
        """
        Classify failure type to determine if retry is appropriate
        
        Args:
            exit_code: Process exit code
            stderr: Standard error output
            error_message: Error message if any
            timeout_occurred: Whether a timeout occurred
            
        Returns:
            FailureType indicating if failure is transient or permanent
        """
        # Combine all error information
        error_text = (stderr + " " + (error_message or "")).lower()
        
        # Transient failure indicators
        transient_indicators = [
            "connection refused",
            "connection timed out",
            "connection reset",
            "network is unreachable",
            "temporarily unavailable",
            "resource temporarily unavailable",
            "too many open files",
            "cannot allocate memory",
            "no route to host",
            "name or service not known",
            "host is unreachable",
            "broken pipe",
            "socket timeout",
            "recv failed",
        ]
        
        # Permanent failure indicators
        permanent_indicators = [
            "command not found",
            "permission denied",
            "no such file or directory",
            "invalid argument",
            "invalid option",
            "syntax error",
            "access denied",
            "authentication failed",
            "bad credentials",
            "unauthorized",
            "forbidden",
        ]
        
        # Check for transient failures
        for indicator in transient_indicators:
            if indicator in error_text:
                return FailureType.TRANSIENT
        
        # Timeout is usually transient
        if timeout_occurred:
            return FailureType.TRANSIENT
        
        # Check for permanent failures
        for indicator in permanent_indicators:
            if indicator in error_text:
                return FailureType.PERMANENT
        
        # Exit code analysis
        if exit_code == 127:  # Command not found
            return FailureType.PERMANENT
        elif exit_code == 126:  # Permission denied
            return FailureType.PERMANENT
        elif exit_code == 1:  # Generic error - could be either
            return FailureType.UNKNOWN
        
        return FailureType.UNKNOWN
    
    def _should_retry(
        self,
        failure_type: FailureType,
        retry_count: int,
        retry_policy: RetryPolicy,
        timeout_occurred: bool
    ) -> bool:
        """
        Determine if execution should be retried
        
        Args:
            failure_type: Type of failure that occurred
            retry_count: Number of retries already attempted
            retry_policy: Retry policy configuration
            timeout_occurred: Whether a timeout occurred
            
        Returns:
            True if retry should be attempted
        """
        # Don't retry if max attempts reached
        if retry_count >= retry_policy.max_attempts:
            return False
        
        # Don't retry permanent failures
        if failure_type == FailureType.PERMANENT:
            return False
        
        # Retry transient failures
        if failure_type == FailureType.TRANSIENT:
            return True
        
        # Retry timeouts if policy allows
        if timeout_occurred and retry_policy.retry_on_timeout:
            return True
        
        # For unknown failures, retry but with caution
        if failure_type == FailureType.UNKNOWN and retry_count < retry_policy.max_attempts - 1:
            return True
        
        return False
    
    def build_command(
        self, 
        tool_name: str, 
        command_type: str, 
        parameters: Dict[str, Any]
    ) -> Optional[str]:
        """
        Build executable command from template and parameters
        
        Args:
            tool_name: Name of the tool
            command_type: Type of command (e.g., 'basic_scan', 'full_scan')
            parameters: Dictionary of parameters to substitute in template
            
        Returns:
            Formatted command string or None if tool/command not found
        """
        if tool_name not in self.manifests:
            logger.error(f"Tool not found: {tool_name}")
            return None
        
        manifest = self.manifests[tool_name]
        template = manifest.get_command_template(command_type)
        
        if not template:
            logger.error(f"Command type not found: {command_type} for tool {tool_name}")
            return None
        
        try:
            # Substitute parameters in template
            command = template.format(**parameters)
            return command
        except KeyError as e:
            logger.error(f"Missing required parameter: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to build command: {e}")
            return None
    
    def execute_command(
        self,
        command: str,
        tool_name: str = "unknown",
        command_type: str = "unknown",
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True,
        shell: bool = True,
        context: Optional[ExecutionContext] = None,
        retry_policy: Optional[RetryPolicy] = None,
        enable_retry: bool = True
    ) -> ExecutionResult:
        """
        Execute a command and capture all output with retry logic
        
        Args:
            command: Command string to execute
            tool_name: Name of the tool being executed
            command_type: Type of command being executed
            timeout: Timeout in seconds (None = no timeout)
            cwd: Working directory for execution
            env: Environment variables
            capture_output: Whether to capture stdout/stderr
            shell: Whether to use shell execution
            context: Execution context for tracking
            retry_policy: Retry policy (uses default if None)
            enable_retry: Whether to enable retry logic
            
        Returns:
            ExecutionResult with all execution details
        """
        # Use provided retry policy or default
        retry_policy = retry_policy or self.default_retry_policy
        
        # Create context if not provided
        if context is None:
            context = ExecutionContext(
                tool_name=tool_name,
                command_type=command_type,
                reason="Manual execution"
            )
        
        logger.info(f"Executing [{context.execution_id}]: {command}")
        if context.reason:
            logger.info(f"  Reason: {context.reason}")
        if context.target:
            logger.info(f"  Target: {context.target}")
        
        # Prepare execution environment
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)
        
        # Initialize tracking variables
        retry_count = 0
        retry_attempts: List[RetryAttempt] = []
        overall_start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Final result variables
        final_stdout = ""
        final_stderr = ""
        final_exit_code = -1
        final_pid = None
        final_timeout_occurred = False
        final_error_message = None
        final_status = ExecutionStatus.ERROR
        failure_type = FailureType.UNKNOWN
        
        # Retry loop
        while True:
            attempt_start_time = time.time()
            stdout_data = ""
            stderr_data = ""
            exit_code = -1
            pid = None
            timeout_occurred = False
            error_message = None
            status = ExecutionStatus.ERROR
            
            # Calculate backoff delay for retries
            if retry_count > 0:
                backoff_delay = retry_policy.calculate_delay(retry_count - 1)
                logger.info(f"Retry attempt {retry_count}/{retry_policy.max_attempts} after {backoff_delay:.2f}s delay")
                time.sleep(backoff_delay)
            else:
                backoff_delay = 0.0
            
            try:
                # Execute command
                process = subprocess.Popen(
                    command,
                    shell=shell,
                    stdout=subprocess.PIPE if capture_output else None,
                    stderr=subprocess.PIPE if capture_output else None,
                    cwd=cwd,
                    env=exec_env,
                    text=True,
                    preexec_fn=os.setsid if os.name != 'nt' else None  # Process group for Unix
                )
                
                pid = process.pid
                logger.info(f"Process started with PID: {pid}")
            
                try:
                    # Wait for process with timeout
                    stdout_data, stderr_data = process.communicate(timeout=timeout)
                    exit_code = process.returncode
                    
                    # Determine status based on exit code
                    if exit_code == 0:
                        status = ExecutionStatus.SUCCESS
                    else:
                        status = ExecutionStatus.FAILURE
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"Command timeout after {timeout} seconds")
                    timeout_occurred = True
                    status = ExecutionStatus.TIMEOUT
                    
                    # Kill the process group
                    try:
                        if os.name != 'nt':
                            os.killpg(os.getpgid(pid), signal.SIGTERM)
                        else:
                            process.terminate()
                        
                        # Give it a chance to terminate gracefully
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            # Force kill if it doesn't terminate
                            if os.name != 'nt':
                                os.killpg(os.getpgid(pid), signal.SIGKILL)
                            else:
                                process.kill()
                            status = ExecutionStatus.KILLED
                            
                    except Exception as kill_error:
                        logger.error(f"Failed to kill process: {kill_error}")
                    
                    # Capture any output before timeout
                    try:
                        stdout_data, stderr_data = process.communicate(timeout=1)
                    except:
                        stdout_data = stdout_data or ""
                        stderr_data = stderr_data or ""
                    
                    exit_code = process.returncode if process.returncode is not None else -1
                    error_message = f"Command exceeded timeout of {timeout} seconds"
                    
            except FileNotFoundError:
                error_message = f"Command not found: {command.split()[0]}"
                logger.error(error_message)
                status = ExecutionStatus.ERROR
                
            except PermissionError:
                error_message = f"Permission denied: {command}"
                logger.error(error_message)
                status = ExecutionStatus.ERROR
                
            except Exception as e:
                error_message = f"Execution error: {str(e)}"
                logger.error(error_message)
                status = ExecutionStatus.ERROR
            
            attempt_execution_time = time.time() - attempt_start_time
            
            # Record this attempt
            retry_attempts.append(RetryAttempt(
                attempt_number=retry_count,
                timestamp=datetime.now().isoformat(),
                exit_code=exit_code,
                error_message=error_message,
                execution_time=attempt_execution_time,
                backoff_delay=backoff_delay
            ))
            
            # Update final values
            final_stdout = stdout_data
            final_stderr = stderr_data
            final_exit_code = exit_code
            final_pid = pid
            final_timeout_occurred = timeout_occurred
            final_error_message = error_message
            final_status = status
            
            # Check if we succeeded
            if status == ExecutionStatus.SUCCESS:
                logger.info(f"Execution successful on attempt {retry_count + 1}")
                break
            
            # Classify the failure
            failure_type = self._classify_failure(exit_code, stderr_data, error_message, timeout_occurred)
            logger.info(f"Failure classified as: {failure_type.value}")
            
            # Check if we should retry
            if not enable_retry or not self._should_retry(failure_type, retry_count, retry_policy, timeout_occurred):
                logger.info(f"Not retrying (attempts: {retry_count + 1}, failure_type: {failure_type.value})")
                break
            
            retry_count += 1
            logger.info(f"Will retry (attempt {retry_count + 1}/{retry_policy.max_attempts + 1})")
        
        # End of retry loop
        total_execution_time = time.time() - overall_start_time
        
        # Analyze output for success/failure indicators
        success_indicators_found = []
        failure_indicators_found = []
        
        if tool_name in self.manifests:
            manifest = self.manifests[tool_name]
            output_text = final_stdout + final_stderr
            
            for indicator in manifest.success_indicators:
                if indicator.lower() in output_text.lower():
                    success_indicators_found.append(indicator)
            
            for indicator in manifest.failure_indicators:
                if indicator.lower() in output_text.lower():
                    failure_indicators_found.append(indicator)
        
        # Create result object with context and retry info
        result = ExecutionResult(
            tool_name=tool_name,
            command=command,
            command_type=command_type,
            status=final_status,
            exit_code=final_exit_code,
            stdout=final_stdout,
            stderr=final_stderr,
            execution_time=total_execution_time,
            timestamp=timestamp,
            pid=final_pid,
            timeout_occurred=final_timeout_occurred,
            error_message=final_error_message,
            success_indicators_found=success_indicators_found,
            failure_indicators_found=failure_indicators_found,
            context=context,
            retry_count=retry_count,
            retry_attempts=retry_attempts,
            failure_type=failure_type if final_status != ExecutionStatus.SUCCESS else None,
            is_retryable=failure_type != FailureType.PERMANENT
        )
        
        # Add to execution history
        self.execution_history.append(result)
        
        logger.info(f"Execution completed [{context.execution_id}]: {final_status.value} " 
                   f"(exit_code={final_exit_code}, retries={retry_count}, time={total_execution_time:.2f}s)")
        
        return result
    
    def execute_tool(
        self,
        tool_name: str,
        command_type: str,
        parameters: Dict[str, Any],
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        context: Optional[ExecutionContext] = None,
        retry_policy: Optional[RetryPolicy] = None,
        enable_retry: bool = True
    ) -> ExecutionResult:
        """
        High-level method to execute a tool command with context tracking
        
        Args:
            tool_name: Name of the tool
            command_type: Type of command to execute
            parameters: Parameters for command template
            timeout: Execution timeout in seconds
            cwd: Working directory
            env: Environment variables
            context: Execution context for tracking
            retry_policy: Retry policy (uses default if None)
            enable_retry: Whether to enable retry logic
            
        Returns:
            ExecutionResult with all execution details
        """
        # Create context if not provided
        if context is None:
            target = parameters.get('target', parameters.get('url', 'unknown'))
            context = ExecutionContext(
                tool_name=tool_name,
                command_type=command_type,
                reason=f"Execute {tool_name} {command_type}",
                target=str(target) if target else None
            )
        
        # Build command from template
        command = self.build_command(tool_name, command_type, parameters)
        
        if not command:
            return ExecutionResult(
                tool_name=tool_name,
                command="",
                command_type=command_type,
                status=ExecutionStatus.ERROR,
                exit_code=-1,
                stdout="",
                stderr="",
                execution_time=0.0,
                timestamp=datetime.now().isoformat(),
                error_message=f"Failed to build command for {tool_name}:{command_type}",
                context=context
            )
        
        # Get estimated time if no timeout specified
        if timeout is None and tool_name in self.manifests:
            cmd_info = self.manifests[tool_name].get_command_info(command_type)
            if cmd_info and 'estimated_time' in cmd_info:
                # Add 50% buffer to estimated time
                timeout = int(cmd_info['estimated_time'] * 1.5)
        
        # Execute the command
        return self.execute_command(
            command=command,
            tool_name=tool_name,
            command_type=command_type,
            timeout=timeout,
            cwd=cwd,
            env=env,
            context=context,
            retry_policy=retry_policy,
            enable_retry=enable_retry
        )
    
    def get_execution_history(
        self,
        tool_name: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        limit: Optional[int] = None
    ) -> List[ExecutionResult]:
        """
        Get execution history with optional filtering
        
        Args:
            tool_name: Filter by tool name
            status: Filter by execution status
            limit: Maximum number of results
            
        Returns:
            List of ExecutionResult objects
        """
        results = self.execution_history
        
        if tool_name:
            results = [r for r in results if r.tool_name == tool_name]
        
        if status:
            results = [r for r in results if r.status == status]
        
        if limit:
            results = results[-limit:]
        
        return results
    
    def get_execution_by_id(self, execution_id: str) -> Optional[ExecutionResult]:
        """Get execution by context ID"""
        for result in self.execution_history:
            if result.context and result.context.execution_id == execution_id:
                return result
        return None
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about all executions"""
        total = len(self.execution_history)
        if total == 0:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "total_retries": 0,
                "by_status": {},
                "by_tool": {}
            }
        
        success_count = sum(1 for r in self.execution_history if r.status == ExecutionStatus.SUCCESS)
        total_retries = sum(r.retry_count for r in self.execution_history)
        avg_time = sum(r.execution_time for r in self.execution_history) / total
        
        # Count by status
        by_status = {}
        for result in self.execution_history:
            status_key = result.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1
        
        # Count by tool
        by_tool = {}
        for result in self.execution_history:
            tool = result.tool_name
            if tool not in by_tool:
                by_tool[tool] = {"count": 0, "success": 0, "failure": 0}
            by_tool[tool]["count"] += 1
            if result.status == ExecutionStatus.SUCCESS:
                by_tool[tool]["success"] += 1
            else:
                by_tool[tool]["failure"] += 1
        
        return {
            "total_executions": total,
            "success_count": success_count,
            "success_rate": success_count / total,
            "average_execution_time": avg_time,
            "total_retries": total_retries,
            "by_status": by_status,
            "by_tool": by_tool
        }
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        logger.info("Execution history cleared")


# Example usage and testing
if __name__ == "__main__":
    # Initialize executor with custom retry policy
    custom_retry_policy = RetryPolicy(
        max_attempts=2,
        initial_delay=0.5,
        backoff_multiplier=2.0
    )
    executor = ToolExecutor(default_retry_policy=custom_retry_policy)
    
    print(f"\n{'='*60}")
    print("Tool Executor - Command Execution Engine")
    print("With Execution Context Tracking & Retry Logic")
    print(f"{'='*60}\n")
    
    # List available tools
    tools = executor.get_available_tools()
    print(f"Available tools ({len(tools)}):")
    for tool in sorted(tools):
        print(f"  - {tool}")
    
    # Example: Get nmap info
    if 'nmap' in tools:
        print(f"\n{'='*60}")
        print("Example: Nmap Tool Information")
        print(f"{'='*60}\n")
        
        nmap_commands = executor.get_tool_commands('nmap')
        print(f"Available nmap commands: {', '.join(nmap_commands)}")
        
        # Build a test command
        command = executor.build_command(
            tool_name='nmap',
            command_type='basic_scan',
            parameters={'target': '127.0.0.1'}
        )
        print(f"\nBuilt command: {command}")
        
        # Note: Actual execution would require nmap to be installed
        # Uncomment below to test actual execution with context:
        # context = ExecutionContext(
        #     tool_name='nmap',
        #     command_type='basic_scan',
        #     reason='Initial network reconnaissance',
        #     target='127.0.0.1',
        #     tags=['test', 'localhost'],
        #     initiated_by='admin'
        # )
        # result = executor.execute_tool(
        #     tool_name='nmap',
        #     command_type='basic_scan',
        #     parameters={'target': '127.0.0.1'},
        #     timeout=30,
        #     context=context
        # )
        # print(f"\nExecution Result:")
        # print(result.to_json())
    
    # Test 1: Simple command with context tracking
    print(f"\n{'='*60}")
    print("Example 1: Simple Command with Context Tracking")
    print(f"{'='*60}\n")
    
    context = ExecutionContext(
        tool_name="test",
        command_type="echo_test",
        reason="Testing basic command execution",
        tags=["demo", "echo"],
        initiated_by="test_user"
    )
    
    result = executor.execute_command(
        command="echo 'Hello from Tool Executor'",
        tool_name="test",
        command_type="echo_test",
        timeout=5,
        context=context
    )
    
    print("Result:")
    print(f"  Execution ID: {result.context.execution_id}")
    print(f"  Status: {result.status.value}")
    print(f"  Exit Code: {result.exit_code}")
    print(f"  Stdout: {result.stdout.strip()}")
    print(f"  Execution Time: {result.execution_time:.3f}s")
    print(f"  Retry Count: {result.retry_count}")
    
    # Test 2: Timeout with retry
    print(f"\n{'='*60}")
    print("Example 2: Timeout with Retry Logic")
    print(f"{'='*60}\n")
    
    context = ExecutionContext(
        tool_name="test",
        command_type="sleep_test",
        reason="Testing timeout and retry mechanism",
        tags=["demo", "timeout", "retry"]
    )
    
    result = executor.execute_command(
        command="sleep 10",
        tool_name="test",
        command_type="sleep_test",
        timeout=1,
        context=context,
        enable_retry=True
    )
    
    print("Result:")
    print(f"  Execution ID: {result.context.execution_id}")
    print(f"  Status: {result.status.value}")
    print(f"  Timeout Occurred: {result.timeout_occurred}")
    print(f"  Error Message: {result.error_message}")
    print(f"  Execution Time: {result.execution_time:.3f}s")
    print(f"  Retry Count: {result.retry_count}")
    print(f"  Failure Type: {result.failure_type.value if result.failure_type else 'N/A'}")
    print(f"  Is Retryable: {result.is_retryable}")
    print(f"  Retry Attempts: {len(result.retry_attempts)}")
    
    # Test 3: Transient failure simulation (command not found)
    print(f"\n{'='*60}")
    print("Example 3: Permanent Failure (No Retry)")
    print(f"{'='*60}\n")
    
    context = ExecutionContext(
        tool_name="test",
        command_type="invalid_test",
        reason="Testing permanent failure detection",
        tags=["demo", "error"]
    )
    
    result = executor.execute_command(
        command="this_command_does_not_exist_xyz",
        tool_name="test",
        command_type="invalid_test",
        timeout=5,
        context=context,
        enable_retry=True
    )
    
    print("Result:")
    print(f"  Execution ID: {result.context.execution_id}")
    print(f"  Status: {result.status.value}")
    print(f"  Error Message: {result.error_message}")
    print(f"  Retry Count: {result.retry_count}")
    print(f"  Failure Type: {result.failure_type.value if result.failure_type else 'N/A'}")
    print(f"  Is Retryable: {result.is_retryable}")
    
    # Show execution statistics
    print(f"\n{'='*60}")
    print("Execution Statistics")
    print(f"{'='*60}\n")
    
    stats = executor.get_execution_statistics()
    print(f"Total Executions: {stats['total_executions']}")
    print(f"Success Count: {stats['success_count']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Execution Time: {stats['average_execution_time']:.3f}s")
    print(f"Total Retries: {stats['total_retries']}")
    print(f"\nBy Status:")
    for status, count in stats['by_status'].items():
        print(f"  {status}: {count}")
    
    print(f"\n{'='*60}\n")
