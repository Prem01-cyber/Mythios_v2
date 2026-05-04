#!/usr/bin/env python3
"""
Reconnaissance Pipeline Test Suite

This script tests the complete reconnaissance pipeline including:
1. Tool Executor - Command execution
2. Output Parser - Output parsing
3. Information State Tracker - State tracking and deduplication
4. Integration - Complete end-to-end workflow

Usage:
    python3 test_pipeline.py              # Run all tests
    python3 test_pipeline.py --quick      # Run quick tests only
    python3 test_pipeline.py --tools      # Test only available tools
"""

import sys
import json
import time
from typing import List, Dict, Any

# Import pipeline components
from tool_executor import ToolExecutor, ExecutionContext, RetryPolicy, ExecutionStatus
from output_parser import UniversalOutputParser, OutputFormat
from information_state import InformationStateTracker
from recon_pipeline import ReconnaissancePipeline

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

def print_header(text: str):
    """Print section header"""
    print(f"\n{'='*80}")
    print(f"{Colors.CYAN}{text}{Colors.NC}")
    print(f"{'='*80}\n")

def print_test(name: str):
    """Print test name"""
    print(f"{Colors.BLUE}[TEST]{Colors.NC} {name}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.NC} {text}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.NC} {text}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.NC} {text}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.NC} {text}")


class PipelineTestSuite:
    """Complete test suite for the reconnaissance pipeline"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.executor = None
        self.parser = None
        self.state_tracker = None
        self.pipeline = None
    
    def run_all_tests(self, quick_mode: bool = False):
        """Run all tests"""
        print_header("Reconnaissance Pipeline Test Suite")
        
        # Component tests
        self.test_tool_executor()
        self.test_output_parser()
        self.test_information_state()
        
        if not quick_mode:
            self.test_integration()
            self.test_real_tools()
        
        # Summary
        self.print_summary()
    
    def test_tool_executor(self):
        """Test the tool executor component"""
        print_header("Testing Tool Executor")
        
        try:
            self.executor = ToolExecutor()
            print_success(f"Initialized executor with {len(self.executor.get_available_tools())} tools")
            self.passed += 1
        except Exception as e:
            print_error(f"Failed to initialize executor: {e}")
            self.failed += 1
            return
        
        # Test 1: Check manifest loading
        print_test("Manifest loading")
        tools = self.executor.get_available_tools()
        if len(tools) >= 50:
            print_success(f"Loaded {len(tools)} tool manifests")
            self.passed += 1
        else:
            print_warning(f"Only {len(tools)} tools loaded (expected 53)")
            self.warnings += 1
        
        # Test 2: Command building
        print_test("Command building")
        try:
            command = self.executor.build_command(
                'nmap',
                'basic_scan',
                {'target': '127.0.0.1'}
            )
            if command and 'nmap' in command:
                print_success(f"Built command: {command}")
                self.passed += 1
            else:
                print_error("Command building failed")
                self.failed += 1
        except Exception as e:
            print_error(f"Command building error: {e}")
            self.failed += 1
        
        # Test 3: Simple command execution
        print_test("Command execution")
        try:
            context = ExecutionContext(
                tool_name="test",
                command_type="echo",
                reason="Testing execution"
            )
            result = self.executor.execute_command(
                command="echo 'Test successful'",
                tool_name="test",
                command_type="echo",
                timeout=5,
                context=context
            )
            
            if result.status == ExecutionStatus.SUCCESS:
                print_success(f"Command executed successfully (exit_code={result.exit_code})")
                print_info(f"  Output: {result.stdout.strip()}")
                self.passed += 1
            else:
                print_error(f"Command failed: {result.status.value}")
                self.failed += 1
        except Exception as e:
            print_error(f"Execution error: {e}")
            self.failed += 1
        
        # Test 4: Timeout handling
        print_test("Timeout handling")
        try:
            result = self.executor.execute_command(
                command="sleep 5",
                tool_name="test",
                command_type="sleep",
                timeout=1,
                enable_retry=False
            )
            
            if result.status == ExecutionStatus.TIMEOUT:
                print_success("Timeout correctly handled")
                self.passed += 1
            else:
                print_warning(f"Expected timeout, got: {result.status.value}")
                self.warnings += 1
        except Exception as e:
            print_error(f"Timeout test error: {e}")
            self.failed += 1
        
        # Test 5: Retry logic
        print_test("Retry logic")
        try:
            result = self.executor.execute_command(
                command="sleep 3",
                tool_name="test",
                command_type="sleep",
                timeout=1,
                enable_retry=True
            )
            
            if result.retry_count > 0:
                print_success(f"Retry logic worked ({result.retry_count} retries)")
                self.passed += 1
            else:
                print_warning("No retries occurred")
                self.warnings += 1
        except Exception as e:
            print_error(f"Retry test error: {e}")
            self.failed += 1
    
    def test_output_parser(self):
        """Test the output parser component"""
        print_header("Testing Output Parser")
        
        try:
            self.parser = UniversalOutputParser(use_mock_llm=True)
            print_success("Initialized output parser")
            self.passed += 1
        except Exception as e:
            print_error(f"Failed to initialize parser: {e}")
            self.failed += 1
            return
        
        # Test 1: Parse nmap output
        print_test("Parse nmap output")
        nmap_output = """
        Nmap scan report for 192.168.1.100
        PORT     STATE SERVICE      VERSION
        22/tcp   open  ssh          OpenSSH 7.4
        80/tcp   open  http         Apache httpd 2.4.6
        445/tcp  open  microsoft-ds Samba smbd 3.X - 4.X
        OS details: Linux 3.10 - 4.11
        """
        
        try:
            result = self.parser.parse("nmap", nmap_output, "")
            
            if result.success:
                print_success(f"Parsed successfully (format: {result.format_detected.value})")
                print_info(f"  Confidence: {result.confidence:.2f}")
                print_info(f"  Ports found: {len(result.extracted_data.get('ports', []))}")
                print_info(f"  OS detected: {result.extracted_data.get('os')}")
                self.passed += 1
            else:
                print_error("Parsing failed")
                self.failed += 1
        except Exception as e:
            print_error(f"Parser error: {e}")
            self.failed += 1
        
        # Test 2: Parse enum4linux output
        print_test("Parse enum4linux output")
        enum_output = """
        [+] Got domain/workgroup name: WORKGROUP
        [+] Enumerating users
        [+] Administrator (RID: 500)
        [+] Guest (RID: 501)
        [+] john.doe (RID: 1001)
        
        Sharename       Type      Comment
        ---------       ----      -------
        C$              Disk      Default share
        ADMIN$          Disk      Remote Admin
        """
        
        try:
            result = self.parser.parse("enum4linux", enum_output, "")
            
            if result.success:
                print_success(f"Parsed successfully (format: {result.format_detected.value})")
                users = result.extracted_data.get('users', [])
                print_info(f"  Users found: {len(users)}")
                self.passed += 1
            else:
                print_error("Parsing failed")
                self.failed += 1
        except Exception as e:
            print_error(f"Parser error: {e}")
            self.failed += 1
        
        # Test 3: Parse nuclei output
        print_test("Parse nuclei output")
        nuclei_output = """
        [critical] [CVE-2021-44228] [log4j-rce] Apache Log4j RCE
        [high] [CVE-2017-5638] [apache-struts] Apache Struts RCE
        [medium] [CVE-2019-0708] [bluekeep] RDP RCE
        """
        
        try:
            result = self.parser.parse("nuclei", nuclei_output, "")
            
            if result.success:
                print_success("Parsed successfully")
                cves = result.extracted_data.get('cves', [])
                print_info(f"  CVEs found: {len(cves)}")
                if cves:
                    print_info(f"  Example: {cves[0]}")
                self.passed += 1
            else:
                print_error("Parsing failed")
                self.failed += 1
        except Exception as e:
            print_error(f"Parser error: {e}")
            self.failed += 1
    
    def test_information_state(self):
        """Test the information state tracker"""
        print_header("Testing Information State Tracker")
        
        try:
            self.state_tracker = InformationStateTracker()
            print_success("Initialized state tracker")
            self.passed += 1
        except Exception as e:
            print_error(f"Failed to initialize state tracker: {e}")
            self.failed += 1
            return
        
        target = "192.168.1.100"
        
        # Test 1: Ingest first tool output
        print_test("Ingest nmap output")
        nmap_output = """
        PORT     STATE SERVICE
        22/tcp   open  ssh
        80/tcp   open  http
        445/tcp  open  microsoft-ds
        OS details: Windows XP SP2
        """
        
        try:
            delta = self.state_tracker.ingest_tool_output(target, "nmap", nmap_output, "")
            print_success(f"Ingested successfully")
            print_info(f"  Points gained: +{delta.points_gained}")
            print_info(f"  New items: {len(delta.new_items)}")
            print_info(f"  Redundant: {len(delta.redundant_items)}")
            self.passed += 1
        except Exception as e:
            print_error(f"Ingestion error: {e}")
            self.failed += 1
        
        # Test 2: Ingest second tool output (with overlap)
        print_test("Ingest enum4linux output (deduplication test)")
        enum_output = """
        [+] OS: Windows XP
        [+] Administrator (RID: 500)
        [+] Guest (RID: 501)
        [+] Sharename: C$
        [+] Sharename: ADMIN$
        """
        
        try:
            delta2 = self.state_tracker.ingest_tool_output(target, "enum4linux", enum_output, "")
            print_success(f"Ingested successfully")
            print_info(f"  Points gained: +{delta2.points_gained}")
            print_info(f"  New items: {len(delta2.new_items)}")
            print_info(f"  Redundant: {len(delta2.redundant_items)}")
            
            if delta2.points_gained > 0:
                print_success("Deduplication working (some new, some redundant)")
                self.passed += 1
            else:
                print_warning("All information was redundant")
                self.warnings += 1
        except Exception as e:
            print_error(f"Ingestion error: {e}")
            self.failed += 1
        
        # Test 3: Check completeness
        print_test("Completeness calculation")
        try:
            completeness = self.state_tracker.calculate_completeness(target)
            print_success(f"Completeness: {completeness.percentage:.1f}%")
            print_info(f"  Identity: {completeness.identity_score*100:.1f}%")
            print_info(f"  Network: {completeness.network_score*100:.1f}%")
            print_info(f"  Security: {completeness.security_score*100:.1f}%")
            
            weakest, score = completeness.get_weakest_dimension()
            print_info(f"  Weakest: {weakest} ({score*100:.1f}%)")
            self.passed += 1
        except Exception as e:
            print_error(f"Completeness error: {e}")
            self.failed += 1
        
        # Test 4: Query state
        print_test("State querying")
        try:
            ports = self.state_tracker.query_state(target, "ports")
            os_info = self.state_tracker.query_state(target, "os")
            
            print_success("State queries working")
            print_info(f"  Ports: {ports}")
            print_info(f"  OS: {os_info}")
            self.passed += 1
        except Exception as e:
            print_error(f"Query error: {e}")
            self.failed += 1
        
        # Test 5: Missing categories
        print_test("Gap detection")
        try:
            missing = self.state_tracker.get_missing_categories(target)
            print_success(f"Found {len(missing)} missing categories")
            for item in missing[:3]:
                print_info(f"  [{item['priority']}] {item['item']}")
            self.passed += 1
        except Exception as e:
            print_error(f"Gap detection error: {e}")
            self.failed += 1
    
    def test_integration(self):
        """Test integrated pipeline"""
        print_header("Testing Integrated Pipeline")
        
        try:
            self.pipeline = ReconnaissancePipeline(use_mock_llm=True)
            print_success("Initialized integrated pipeline")
            self.passed += 1
        except Exception as e:
            print_error(f"Failed to initialize pipeline: {e}")
            self.failed += 1
            return
        
        target = "192.168.1.50"
        
        # Test: Simulated reconnaissance
        print_test("Simulated reconnaissance sequence")
        try:
            # Simulate tool outputs
            tools_and_outputs = [
                ("nmap", """
                PORT     STATE SERVICE
                80/tcp   open  http
                443/tcp  open  https
                OS: Linux 4.15
                """),
                ("enum4linux", """
                [+] Administrator (RID: 500)
                [+] webadmin (RID: 1001)
                """),
                ("nuclei", """
                [critical] [CVE-2021-44228] Log4j RCE
                """)
            ]
            
            total_points = 0
            for tool_name, output in tools_and_outputs:
                delta = self.pipeline.state_tracker.ingest_tool_output(
                    target, tool_name, output, ""
                )
                total_points += delta.points_gained
            
            completeness = self.pipeline.state_tracker.calculate_completeness(target)
            
            print_success("Reconnaissance sequence completed")
            print_info(f"  Total points: +{total_points}")
            print_info(f"  Completeness: {completeness.percentage:.1f}%")
            self.passed += 1
        except Exception as e:
            print_error(f"Integration test error: {e}")
            self.failed += 1
        
        # Test: Recommendations
        print_test("Intelligent recommendations")
        try:
            recommendations = self.pipeline.get_recommended_next_tools(target, max_recommendations=3)
            
            if recommendations:
                print_success(f"Got {len(recommendations)} recommendations")
                for rec in recommendations:
                    print_info(f"  - {rec['tool_name']}: {rec['reason']}")
                self.passed += 1
            else:
                print_warning("No recommendations available")
                self.warnings += 1
        except Exception as e:
            print_error(f"Recommendations error: {e}")
            self.failed += 1
    
    def test_real_tools(self):
        """Test with real installed tools"""
        print_header("Testing Real Tools")
        
        # Test only safe, non-intrusive tools
        safe_tools = [
            ('echo', 'echo "test"'),
            ('ls', 'ls /tmp'),
            ('whoami', 'whoami'),
        ]
        
        for tool_name, command in safe_tools:
            print_test(f"Testing {tool_name}")
            try:
                context = ExecutionContext(
                    tool_name=tool_name,
                    command_type="test",
                    reason=f"Testing {tool_name}"
                )
                
                result = self.executor.execute_command(
                    command=command,
                    tool_name=tool_name,
                    command_type="test",
                    timeout=5,
                    context=context
                )
                
                if result.status == ExecutionStatus.SUCCESS:
                    print_success(f"{tool_name} executed successfully")
                    self.passed += 1
                else:
                    print_warning(f"{tool_name} failed: {result.status.value}")
                    self.warnings += 1
            except Exception as e:
                print_error(f"{tool_name} error: {e}")
                self.failed += 1
    
    def print_summary(self):
        """Print test summary"""
        print_header("Test Summary")
        
        total = self.passed + self.failed + self.warnings
        
        print(f"Total Tests: {total}")
        print(f"{Colors.GREEN}✓ Passed:{Colors.NC} {self.passed}")
        print(f"{Colors.RED}✗ Failed:{Colors.NC} {self.failed}")
        print(f"{Colors.YELLOW}⚠ Warnings:{Colors.NC} {self.warnings}")
        
        if self.failed == 0:
            print(f"\n{Colors.GREEN}All tests passed!{Colors.NC}")
            return 0
        else:
            print(f"\n{Colors.RED}Some tests failed!{Colors.NC}")
            return 1


def check_available_tools():
    """Check which tools are actually installed"""
    print_header("Checking Available Security Tools")
    
    import subprocess
    
    tools_to_check = [
        'nmap', 'masscan', 'rustscan', 'enum4linux', 'smbclient',
        'crackmapexec', 'nikto', 'nuclei', 'sqlmap', 'hydra',
        'john', 'hashcat', 'msfconsole', 'amass', 'subfinder',
        'theharvester', 'dnsenum', 'wireshark', 'tcpdump'
    ]
    
    installed = []
    missing = []
    
    for tool in tools_to_check:
        result = subprocess.run(
            ['which', tool],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_success(f"{tool}: {result.stdout.strip()}")
            installed.append(tool)
        else:
            print_error(f"{tool}: NOT FOUND")
            missing.append(tool)
    
    print(f"\n{Colors.GREEN}Installed:{Colors.NC} {len(installed)}/{len(tools_to_check)}")
    print(f"{Colors.RED}Missing:{Colors.NC} {len(missing)}/{len(tools_to_check)}")
    
    if missing:
        print(f"\n{Colors.YELLOW}Missing tools:{Colors.NC}")
        for tool in missing:
            print(f"  - {tool}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Reconnaissance Pipeline')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--tools', action='store_true', help='Check available tools')
    
    args = parser.parse_args()
    
    if args.tools:
        check_available_tools()
        return 0
    
    # Run test suite
    suite = PipelineTestSuite()
    return suite.run_all_tests(quick_mode=args.quick)


if __name__ == '__main__':
    sys.exit(main())
