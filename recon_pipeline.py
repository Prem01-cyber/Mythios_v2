"""
Reconnaissance Pipeline - Complete Integration Layer

This module integrates the three core components:
1. Tool Executor - Executes security tools
2. Output Parser - Parses tool outputs
3. Information State Tracker - Tracks discovered information

This provides a complete end-to-end pipeline for the RL agent.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from tool_executor import ToolExecutor, ExecutionResult, ExecutionContext, RetryPolicy
from output_parser import UniversalOutputParser, ParseResult
from information_state import (
    InformationStateTracker, 
    InformationDelta,
    CompletenessScore
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReconResult:
    """
    Complete result of a reconnaissance action
    
    Combines execution, parsing, and state tracking results
    """
    # Execution details
    execution: ExecutionResult
    
    # Parsing details
    parse_result: ParseResult
    
    # State tracking details
    information_delta: InformationDelta
    completeness: CompletenessScore
    
    # Summary metrics
    points_gained: int
    new_information_count: int
    redundant_information_count: int
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "execution": {
                "tool": self.execution.tool_name,
                "command": self.execution.command,
                "status": self.execution.status.value,
                "exit_code": self.execution.exit_code,
                "execution_time": self.execution.execution_time,
                "retry_count": self.execution.retry_count
            },
            "parsing": {
                "format": self.parse_result.format_detected.value,
                "confidence": self.parse_result.confidence,
                "warnings": self.parse_result.warnings
            },
            "information": {
                "points_gained": self.points_gained,
                "new_items": self.new_information_count,
                "redundant_items": self.redundant_information_count,
                "completeness": f"{self.completeness.percentage:.1f}%"
            }
        }
    
    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict(), indent=2)


class ReconnaissancePipeline:
    """
    Complete reconnaissance pipeline
    
    Orchestrates tool execution, output parsing, and state tracking
    """
    
    def __init__(
        self,
        tools_manifest_dir: str = "tools/manifests",
        use_mock_llm: bool = True,
        retry_policy: Optional[RetryPolicy] = None
    ):
        """
        Initialize the reconnaissance pipeline
        
        Args:
            tools_manifest_dir: Directory containing tool manifests
            use_mock_llm: Whether to use mock LLM (True for testing)
            retry_policy: Optional retry policy for tool execution
        """
        self.executor = ToolExecutor(
            manifests_dir=tools_manifest_dir,
            default_retry_policy=retry_policy
        )
        self.parser = UniversalOutputParser(use_mock_llm=use_mock_llm)
        self.state_tracker = InformationStateTracker()
        
        logger.info("Reconnaissance Pipeline initialized")
        logger.info(f"Loaded {len(self.executor.get_available_tools())} tools")
    
    def execute_tool(
        self,
        target_id: str,
        tool_name: str,
        command_type: str,
        parameters: Dict[str, Any],
        reason: str = "Reconnaissance",
        timeout: Optional[int] = None,
        enable_retry: bool = True
    ) -> ReconResult:
        """
        Execute a single tool and process results through complete pipeline
        
        Args:
            target_id: Target identifier (IP, hostname, etc.)
            tool_name: Name of the tool to execute
            command_type: Type of command to run
            parameters: Parameters for the command template
            reason: Why this tool is being executed
            timeout: Optional timeout in seconds
            enable_retry: Whether to enable retry logic
            
        Returns:
            ReconResult with complete pipeline results
        """
        logger.info(f"Pipeline: Executing {tool_name} {command_type} against {target_id}")
        logger.info(f"Reason: {reason}")
        
        # Step 1: Execute tool
        context = ExecutionContext(
            tool_name=tool_name,
            command_type=command_type,
            reason=reason,
            target=target_id,
            tags=[target_id, tool_name]
        )
        
        execution_result = self.executor.execute_tool(
            tool_name=tool_name,
            command_type=command_type,
            parameters=parameters,
            timeout=timeout,
            context=context,
            enable_retry=enable_retry
        )
        
        logger.info(f"Execution: {execution_result.status.value} " +
                   f"(exit_code={execution_result.exit_code}, " +
                   f"time={execution_result.execution_time:.2f}s)")
        
        # Step 2: Parse output
        parse_result = self.parser.parse(
            tool_name=tool_name,
            stdout=execution_result.stdout,
            stderr=execution_result.stderr,
            exit_code=execution_result.exit_code,
            context={"command_type": command_type, "target": target_id}
        )
        
        logger.info(f"Parsing: {parse_result.format_detected.value} format " +
                   f"(confidence={parse_result.confidence:.2f})")
        
        # Step 3: Update state tracker
        information_delta = self.state_tracker.ingest_tool_output(
            target_id=target_id,
            tool_name=tool_name,
            stdout=execution_result.stdout,
            stderr=execution_result.stderr,
            timestamp=execution_result.timestamp
        )
        
        logger.info(f"Information: +{information_delta.points_gained} points " +
                   f"({len(information_delta.new_items)} new, " +
                   f"{len(information_delta.redundant_items)} redundant)")
        
        # Step 4: Calculate completeness
        completeness = self.state_tracker.calculate_completeness(target_id)
        
        logger.info(f"Completeness: {completeness.percentage:.1f}%")
        
        # Create comprehensive result
        recon_result = ReconResult(
            execution=execution_result,
            parse_result=parse_result,
            information_delta=information_delta,
            completeness=completeness,
            points_gained=information_delta.points_gained,
            new_information_count=len(information_delta.new_items),
            redundant_information_count=len(information_delta.redundant_items),
            execution_time=execution_result.execution_time
        )
        
        return recon_result
    
    def execute_sequence(
        self,
        target_id: str,
        tool_sequence: List[Dict[str, Any]],
        stop_on_error: bool = False
    ) -> List[ReconResult]:
        """
        Execute a sequence of tools
        
        Args:
            target_id: Target identifier
            tool_sequence: List of tool configurations:
                [
                    {
                        "tool_name": "nmap",
                        "command_type": "basic_scan",
                        "parameters": {"target": "192.168.1.100"},
                        "reason": "Initial port scan"
                    },
                    ...
                ]
            stop_on_error: Whether to stop sequence on first error
            
        Returns:
            List of ReconResult for each tool
        """
        logger.info(f"Executing sequence of {len(tool_sequence)} tools against {target_id}")
        
        results = []
        for i, tool_config in enumerate(tool_sequence, 1):
            logger.info(f"\n[{i}/{len(tool_sequence)}] Executing {tool_config['tool_name']}")
            
            try:
                result = self.execute_tool(
                    target_id=target_id,
                    tool_name=tool_config['tool_name'],
                    command_type=tool_config['command_type'],
                    parameters=tool_config['parameters'],
                    reason=tool_config.get('reason', 'Sequence execution')
                )
                results.append(result)
                
                # Check if we should stop
                if stop_on_error and result.execution.exit_code != 0:
                    logger.warning(f"Stopping sequence due to error in {tool_config['tool_name']}")
                    break
                    
            except Exception as e:
                logger.error(f"Error executing {tool_config['tool_name']}: {e}")
                if stop_on_error:
                    break
        
        return results
    
    def get_recommended_next_tools(self, target_id: str, max_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get recommended next tools based on current state
        
        Uses missing categories to suggest what to run next
        """
        missing = self.state_tracker.get_missing_categories(target_id)
        completeness = self.state_tracker.calculate_completeness(target_id)
        
        recommendations = []
        
        # Prioritize by missing critical items
        critical_missing = [m for m in missing if m['priority'] == 'CRITICAL']
        high_missing = [m for m in missing if m['priority'] == 'HIGH']
        
        for item in critical_missing + high_missing:
            suggested_tools = item.get('suggested_tools', [])
            for tool in suggested_tools[:1]:  # Take first suggestion per item
                # Parse tool name and command if specified
                if ' ' in tool:
                    tool_parts = tool.split(maxsplit=1)
                    tool_name = tool_parts[0]
                    command_hint = tool_parts[1] if len(tool_parts) > 1 else None
                else:
                    tool_name = tool
                    command_hint = None
                
                recommendations.append({
                    "tool_name": tool_name,
                    "reason": f"Gather {item['item']}",
                    "priority": item['priority'],
                    "category": item['category'],
                    "command_hint": command_hint
                })
            
            if len(recommendations) >= max_recommendations:
                break
        
        return recommendations[:max_recommendations]
    
    def get_target_summary(self, target_id: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of a target
        """
        if target_id not in self.state_tracker.dossiers:
            return {"error": "Target not found"}
        
        completeness = self.state_tracker.calculate_completeness(target_id)
        stats = self.state_tracker.query_state(target_id, 'statistics')
        missing = self.state_tracker.get_missing_categories(target_id)
        recommendations = self.get_recommended_next_tools(target_id)
        
        return {
            "target_id": target_id,
            "completeness": completeness.to_dict(),
            "statistics": stats,
            "missing_items": len(missing),
            "missing_critical": len([m for m in missing if m['priority'] == 'CRITICAL']),
            "recommendations": recommendations
        }
    
    def export_report(self, target_id: str, filepath: Optional[str] = None) -> str:
        """
        Export complete reconnaissance report
        """
        summary = self.get_target_summary(target_id)
        full_state = self.state_tracker.export_state(target_id)
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "target": target_id,
            "summary": summary,
            "full_state": json.loads(full_state)
        }
        
        report_json = json.dumps(report, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(report_json)
            logger.info(f"Report exported to {filepath}")
        
        return report_json


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPLETE RECONNAISSANCE PIPELINE - Demo")
    print("Tool Executor → Output Parser → Information State Tracker")
    print("="*70 + "\n")
    
    # Initialize pipeline
    pipeline = ReconnaissancePipeline(use_mock_llm=True)
    target = "192.168.1.100"
    
    # Demo: Execute sequence of tools
    print("Executing reconnaissance sequence...")
    print("="*70 + "\n")
    
    # Tool sequence (would be actual tools in production)
    sequence = [
        {
            "tool_name": "test",
            "command_type": "echo_test",
            "parameters": {},
            "reason": "Test execution (simulates nmap port scan)"
        }
    ]
    
    # For demo purposes, we'll simulate tool outputs manually
    print("Simulating nmap scan...")
    nmap_output = """
    Nmap scan report for 192.168.1.100
    PORT     STATE SERVICE      VERSION
    139/tcp  open  netbios-ssn  Microsoft Windows netbios-ssn
    445/tcp  open  microsoft-ds Microsoft Windows XP microsoft-ds
    3389/tcp open  ms-wbt-server
    OS details: Microsoft Windows XP SP2
    """
    
    # Manually process (in production, this would be automatic)
    parse_result = pipeline.parser.parse("nmap", nmap_output, "")
    delta = pipeline.state_tracker.ingest_tool_output(target, "nmap", nmap_output, "")
    completeness = pipeline.state_tracker.calculate_completeness(target)
    
    print(f"Results:")
    print(f"  Format detected: {parse_result.format_detected.value}")
    print(f"  Parsing confidence: {parse_result.confidence:.2f}")
    print(f"  Points gained: +{delta.points_gained}")
    print(f"  New information: {len(delta.new_items)} items")
    print(f"  Completeness: {completeness.percentage:.1f}%")
    
    print("\n" + "-"*70)
    print("Simulating enum4linux scan...")
    enum_output = """
    [+] Got domain/workgroup name: WORKGROUP
    [+] Administrator (RID: 500)
    [+] Guest (RID: 501)
    [+] john.doe (RID: 1001)
    
    Sharename       Type      Comment
    ---------       ----      -------
    C$              Disk      Default share
    ADMIN$          Disk      Remote Admin
    Backup          Disk      Backup files
    """
    
    parse_result2 = pipeline.parser.parse("enum4linux", enum_output, "")
    delta2 = pipeline.state_tracker.ingest_tool_output(target, "enum4linux", enum_output, "")
    completeness2 = pipeline.state_tracker.calculate_completeness(target)
    
    print(f"Results:")
    print(f"  Format detected: {parse_result2.format_detected.value}")
    print(f"  Parsing confidence: {parse_result2.confidence:.2f}")
    print(f"  Points gained: +{delta2.points_gained}")
    print(f"  New information: {len(delta2.new_items)} items")
    print(f"  Redundant information: {len(delta2.redundant_items)} items")
    print(f"  Completeness: {completeness2.percentage:.1f}%")
    
    # Show recommendations
    print("\n" + "="*70)
    print("Intelligent Recommendations")
    print("="*70 + "\n")
    
    recommendations = pipeline.get_recommended_next_tools(target, max_recommendations=3)
    print("Based on current state, recommended next actions:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. Run {rec['tool_name']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Priority: {rec['priority']}")
        print(f"   Category: {rec['category']}")
    
    # Show summary
    print("\n" + "="*70)
    print("Target Summary")
    print("="*70 + "\n")
    
    summary = pipeline.get_target_summary(target)
    print(json.dumps(summary, indent=2))
    
    print("\n" + "="*70)
    print("Pipeline Capabilities Demonstrated:")
    print("="*70)
    print("✓ End-to-end tool execution → parsing → state tracking")
    print("✓ Automatic output format detection")
    print("✓ Information extraction and normalization")
    print("✓ Deduplication and redundancy detection")
    print("✓ Completeness measurement")
    print("✓ Intelligent next-tool recommendations")
    print("✓ Comprehensive target reporting")
    print("\n" + "="*70 + "\n")
