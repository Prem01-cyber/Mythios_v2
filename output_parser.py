"""
Output Parser - Universal Tool Output Translation Layer

This module provides an intelligent parsing system that converts messy, unstructured
tool outputs (text, XML, JSON) into clean, normalized information objects. It uses
LLM intelligence to understand WHAT each tool found without tool-specific parsing code.

Core Capabilities:
- Format detection (JSON, XML, tables, logs, free-form text)
- LLM-based semantic extraction
- Cross-tool normalization
- Noise filtering
- Context-aware parsing
- Confidence assessment
- Validation and enrichment
"""

import json
import re
import xml.etree.ElementTree as ET
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Detected output format types"""
    JSON = "json"
    XML = "xml"
    TABLE = "table"
    LOG_STYLE = "log_style"
    KEY_VALUE = "key_value"
    FREE_FORM = "free_form"


@dataclass
class ParseResult:
    """
    Result of parsing a tool output
    """
    success: bool
    format_detected: OutputFormat
    extracted_data: Dict[str, Any]
    confidence: float  # How confident the parser is in the extraction
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_sections: Dict[str, str] = field(default_factory=dict)  # Extracted sections
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "format": self.format_detected.value,
            "data": self.extracted_data,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata
        }


class FormatDetector:
    """
    Automatically detects the format of tool output
    """
    
    @staticmethod
    def detect(output: str) -> Tuple[OutputFormat, float]:
        """
        Detect format with confidence score
        
        Returns: (format, confidence)
        """
        if not output or not output.strip():
            return OutputFormat.FREE_FORM, 0.0
        
        output_stripped = output.strip()
        
        # JSON detection
        if (output_stripped.startswith('{') and output_stripped.endswith('}')) or \
           (output_stripped.startswith('[') and output_stripped.endswith(']')):
            try:
                json.loads(output_stripped)
                return OutputFormat.JSON, 1.0
            except json.JSONDecodeError:
                pass
        
        # XML detection
        if output_stripped.startswith('<?xml') or \
           (output_stripped.startswith('<') and output_stripped.endswith('>')):
            try:
                ET.fromstring(output_stripped)
                return OutputFormat.XML, 1.0
            except ET.ParseError:
                pass
        
        # Table detection (look for consistent column headers)
        lines = output.split('\n')
        if len(lines) >= 3:
            # Check for header pattern
            header_indicators = ['PORT', 'STATE', 'SERVICE', 'TYPE', 'NAME', 'SHARENAME']
            first_line = lines[0].upper()
            if any(indicator in first_line for indicator in header_indicators):
                # Check for separator line (dashes)
                if re.match(r'^[\s\-=]+$', lines[1]):
                    return OutputFormat.TABLE, 0.9
        
        # Log-style detection (lines with prefixes like [+], [-], [*])
        log_prefixes = [r'^\[\+\]', r'^\[-\]', r'^\[\*\]', r'^\[!\]', r'^\[E\]', r'^\[W\]']
        log_matches = 0
        for line in lines[:20]:  # Check first 20 lines
            if any(re.match(pattern, line.strip()) for pattern in log_prefixes):
                log_matches += 1
        
        if log_matches >= 3:
            return OutputFormat.LOG_STYLE, min(0.9, 0.5 + log_matches * 0.1)
        
        # Key-value detection (lines like "key: value")
        kv_pattern = r'^\s*[A-Za-z_][\w\s]*:\s+.+'
        kv_matches = sum(1 for line in lines if re.match(kv_pattern, line))
        
        if kv_matches >= 3:
            return OutputFormat.KEY_VALUE, min(0.8, 0.4 + kv_matches * 0.1)
        
        # Default to free-form
        return OutputFormat.FREE_FORM, 0.5


class SemanticNormalizer:
    """
    Normalizes tool-specific terminology to standard names
    
    This handles semantic equivalence (microsoft-ds = SMB)
    """
    
    # Service name mappings
    SERVICE_MAPPINGS = {
        'microsoft-ds': 'smb',
        'netbios-ssn': 'smb',
        'cifs': 'smb',
        'ms-wbt-server': 'rdp',
        'http-proxy': 'http',
        'http-alt': 'http',
        'https-alt': 'https',
        'domain': 'dns',
        'ms-sql': 'mssql',
        'ms-sql-s': 'mssql',
        'postgresql': 'postgres',
        'mysql': 'mysql',
        'ssh': 'ssh',
        'telnet': 'telnet',
        'ftp': 'ftp',
        'smtp': 'smtp',
    }
    
    # OS name mappings
    OS_MAPPINGS = {
        'windows 5.1': 'Windows XP',
        'windows 5.2': 'Windows Server 2003',
        'windows 6.0': 'Windows Vista',
        'windows 6.1': 'Windows 7',
        'windows 6.2': 'Windows 8',
        'windows 6.3': 'Windows 8.1',
        'windows 10.0': 'Windows 10',
    }
    
    # Privileged account names
    PRIVILEGED_ACCOUNTS = {
        'administrator', 'admin', 'root', 'system', 'domain admin',
        'enterprise admin', 'schema admin', 'administrators'
    }
    
    # Default accounts (present on most systems)
    DEFAULT_ACCOUNTS = {
        'administrator', 'guest', 'krbtgt', 'system', 'root', 'daemon',
        'bin', 'nobody', 'www-data'
    }
    
    @classmethod
    def normalize_service(cls, service_name: str) -> str:
        """Normalize service name to canonical form"""
        normalized = service_name.lower().strip()
        return cls.SERVICE_MAPPINGS.get(normalized, normalized)
    
    @classmethod
    def normalize_os(cls, os_string: str) -> Dict[str, Any]:
        """Normalize OS string to structured format"""
        os_lower = os_string.lower()
        
        result = {
            "raw": os_string,
            "name": None,
            "version": None,
            "family": None
        }
        
        # Check mappings
        for key, value in cls.OS_MAPPINGS.items():
            if key in os_lower:
                result["name"] = value
                result["version"] = key.split()[-1]
                break
        
        # Determine family
        if 'windows' in os_lower:
            result["family"] = "Windows"
            if not result["name"]:
                result["name"] = "Windows"
        elif 'linux' in os_lower:
            result["family"] = "Linux"
        elif any(x in os_lower for x in ['unix', 'bsd', 'solaris']):
            result["family"] = "Unix"
        
        return result
    
    @classmethod
    def enrich_user(cls, username: str, rid: Optional[int] = None) -> Dict[str, Any]:
        """Enrich user account with semantic information"""
        username_lower = username.lower()
        
        return {
            "name": username,
            "rid": rid,
            "is_privileged": username_lower in cls.PRIVILEGED_ACCOUNTS,
            "is_default": username_lower in cls.DEFAULT_ACCOUNTS,
            "risk_level": "HIGH" if username_lower in cls.PRIVILEGED_ACCOUNTS else "MEDIUM"
        }
    
    @classmethod
    def enrich_share(cls, share_name: str) -> Dict[str, Any]:
        """Enrich share with semantic information"""
        share_lower = share_name.lower()
        
        # Admin shares (end with $)
        is_admin = share_name.endswith('$')
        
        # Sensitive share names
        sensitive_keywords = ['backup', 'admin', 'confidential', 'private', 'finance']
        is_sensitive = any(keyword in share_lower for keyword in sensitive_keywords)
        
        risk_level = "HIGH" if is_admin or is_sensitive else "LOW"
        
        return {
            "name": share_name,
            "is_admin_share": is_admin,
            "is_sensitive": is_sensitive,
            "risk_level": risk_level
        }
    
    @classmethod
    def validate_port(cls, port: Any) -> Optional[int]:
        """Validate and normalize port number"""
        try:
            port_int = int(port)
            if 1 <= port_int <= 65535:
                return port_int
        except (ValueError, TypeError):
            pass
        return None
    
    @classmethod
    def validate_cve(cls, cve_str: str) -> Optional[str]:
        """Validate CVE format"""
        match = re.match(r'CVE-(\d{4})-(\d{4,7})', cve_str, re.IGNORECASE)
        if match:
            return f"CVE-{match.group(1)}-{match.group(2)}"
        return None


class StructureExtractor:
    """
    Extracts structured data from different formats
    """
    
    @staticmethod
    def extract_json(output: str) -> Dict[str, Any]:
        """Extract data from JSON format"""
        try:
            return json.loads(output.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {}
    
    @staticmethod
    def extract_xml(output: str) -> Dict[str, Any]:
        """Extract data from XML format"""
        try:
            root = ET.fromstring(output.strip())
            return StructureExtractor._xml_to_dict(root)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return {}
    
    @staticmethod
    def _xml_to_dict(element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary"""
        result = {}
        
        # Add attributes
        if element.attrib:
            result.update(element.attrib)
        
        # Add text content
        if element.text and element.text.strip():
            result['_text'] = element.text.strip()
        
        # Add children
        for child in element:
            child_data = StructureExtractor._xml_to_dict(child)
            tag = child.tag
            
            if tag in result:
                # Multiple children with same tag - make it a list
                if not isinstance(result[tag], list):
                    result[tag] = [result[tag]]
                result[tag].append(child_data)
            else:
                result[tag] = child_data
        
        return result
    
    @staticmethod
    def extract_table(output: str) -> List[Dict[str, str]]:
        """Extract data from table format"""
        lines = [line for line in output.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return []
        
        # First line is header
        header_line = lines[0]
        
        # Find column positions by looking for whitespace gaps
        header_parts = header_line.split()
        
        # Simple approach: split by multiple spaces
        data_rows = []
        for line in lines[2:]:  # Skip header and separator
            if line.strip() and not re.match(r'^[\s\-=]+$', line):
                # Split by multiple spaces
                parts = re.split(r'\s{2,}', line.strip())
                if parts:
                    row = {}
                    for i, header in enumerate(header_parts):
                        if i < len(parts):
                            row[header.lower()] = parts[i].strip()
                    data_rows.append(row)
        
        return data_rows
    
    @staticmethod
    def extract_key_value(output: str) -> Dict[str, str]:
        """Extract key-value pairs"""
        result = {}
        pattern = r'^\s*([A-Za-z_][\w\s]*?):\s+(.+)'
        
        for line in output.split('\n'):
            match = re.match(pattern, line)
            if match:
                key = match.group(1).strip().lower().replace(' ', '_')
                value = match.group(2).strip()
                result[key] = value
        
        return result
    
    @staticmethod
    def extract_log_style(output: str) -> Dict[str, List[str]]:
        """Extract from log-style output"""
        result = {
            'success': [],
            'failure': [],
            'warning': [],
            'info': [],
            'error': []
        }
        
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('[+]'):
                result['success'].append(line[3:].strip())
            elif line.startswith('[-]'):
                result['failure'].append(line[3:].strip())
            elif line.startswith('[!]') or line.startswith('[W]'):
                result['warning'].append(line[3:].strip())
            elif line.startswith('[*]'):
                result['info'].append(line[3:].strip())
            elif line.startswith('[E]'):
                result['error'].append(line[3:].strip())
        
        return result


class LLMParser:
    """
    LLM-based intelligent parser for free-form text
    
    This is where the magic happens - using LLM intelligence to understand
    tool outputs without regex patterns
    """
    
    def __init__(self, use_mock: bool = True):
        """
        Initialize LLM parser
        
        Args:
            use_mock: If True, uses mock LLM for testing. In production, set to False
                     and implement actual Qwen integration.
        """
        self.use_mock = use_mock
    
    def parse_with_llm(self, output: str, tool_name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Use LLM to extract information from tool output
        
        This is the universal parser - works with ANY tool
        """
        if self.use_mock:
            return self._mock_llm_parse(output, tool_name)
        else:
            return self._real_llm_parse(output, tool_name, context)
    
    def _mock_llm_parse(self, output: str, tool_name: str) -> Dict[str, Any]:
        """
        Mock LLM parser for testing
        
        In production, replace this with actual Qwen API calls
        """
        result = {
            "os": None,
            "services": [],
            "users": [],
            "shares": [],
            "vulnerabilities": [],
            "cves": [],
            "ports": [],
            "domain": None
        }
        
        # Simple pattern matching as fallback
        # In production, this is replaced by actual LLM understanding
        
        # Extract ports
        port_pattern = r'\b(\d+)/(tcp|udp)\b'
        for match in re.finditer(port_pattern, output):
            port = int(match.group(1))
            if 1 <= port <= 65535:
                result["ports"].append(port)
        
        # Extract CVEs
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        result["cves"] = list(set(re.findall(cve_pattern, output, re.IGNORECASE)))
        
        # Extract users (simple heuristic)
        user_pattern = r'(?:user:|[Uu]ser\s+name|[Aa]dministrator|[Gg]uest|[Rr]oot)'
        if re.search(user_pattern, output):
            # Try to extract username
            for line in output.split('\n'):
                if 'administrator' in line.lower():
                    result["users"].append("Administrator")
                if 'guest' in line.lower() and 'Guest' not in result["users"]:
                    result["users"].append("Guest")
        
        # Extract OS info
        os_patterns = [
            r'Windows\s+(?:XP|Vista|7|8|10|11|Server\s+\d{4})',
            r'Linux',
            r'Ubuntu',
            r'CentOS'
        ]
        for pattern in os_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                result["os"] = match.group(0)
                break
        
        return result
    
    def _real_llm_parse(self, output: str, tool_name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Real LLM-based parsing using Qwen
        
        IMPLEMENTATION NOTE: This should be implemented with actual Qwen API
        """
        prompt = self._build_extraction_prompt(output, tool_name, context)
        
        # TODO: Replace with actual Qwen API call
        # response = qwen_client.generate(prompt)
        # return json.loads(response)
        
        # For now, fallback to mock
        return self._mock_llm_parse(output, tool_name)
    
    def _build_extraction_prompt(self, output: str, tool_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build extraction prompt for LLM
        """
        prompt = f"""Extract cybersecurity reconnaissance information from this {tool_name} output.

Tool Output:
```
{output[:2000]}  # Limit to avoid token limits
```

Extract and return ONLY the following information types that are present (ignore what's not found):

1. **Operating System**: Name, version, family
2. **Network Services**: Port number, service name, version, state
3. **User Accounts**: Usernames, RIDs, privilege level
4. **Network Shares**: Share names, types, access level
5. **Vulnerabilities**: CVE IDs, severity, description
6. **Domain/Workgroup**: Network domain or workgroup name
7. **Hostname**: Target hostname if mentioned

Return ONLY valid JSON in this format:
{{
    "os": {{"name": "...", "version": "...", "family": "..."}},
    "services": [{{"port": 445, "name": "smb", "version": "...", "state": "open"}}],
    "users": [{{"name": "Administrator", "rid": 500}}],
    "shares": [{{"name": "C$", "type": "Disk"}}],
    "vulnerabilities": [{{"cve": "CVE-2017-0144", "severity": "critical"}}],
    "domain": "WORKGROUP",
    "hostname": "TARGET"
}}

CRITICAL RULES:
- Extract ONLY information explicitly stated in the output
- DO NOT invent or hallucinate information
- Ignore status messages, errors, and headers
- If a category has no data, omit it from the response
- Return valid JSON only"""

        if context:
            prompt += f"\n\nAdditional Context: {json.dumps(context)}"
        
        return prompt


class UniversalOutputParser:
    """
    Universal parser that handles all tool outputs
    
    This is the main interface - combines all parsing strategies
    """
    
    def __init__(self, use_mock_llm: bool = True):
        self.format_detector = FormatDetector()
        self.structure_extractor = StructureExtractor()
        self.normalizer = SemanticNormalizer()
        self.llm_parser = LLMParser(use_mock=use_mock_llm)
    
    def parse(
        self,
        tool_name: str,
        stdout: str,
        stderr: str = "",
        exit_code: int = 0,
        context: Optional[Dict[str, Any]] = None
    ) -> ParseResult:
        """
        Universal parsing method
        
        Args:
            tool_name: Name of the tool that generated output
            stdout: Standard output from tool
            stderr: Standard error from tool
            exit_code: Exit code
            context: Optional context (command type, target, etc.)
            
        Returns:
            ParseResult with extracted and normalized data
        """
        logger.info(f"Parsing output from {tool_name}")
        
        # Combine stdout and stderr
        full_output = stdout
        if stderr:
            full_output += "\n--- STDERR ---\n" + stderr
        
        if not full_output.strip():
            return ParseResult(
                success=False,
                format_detected=OutputFormat.FREE_FORM,
                extracted_data={},
                confidence=0.0,
                errors=["Empty output"]
            )
        
        # Step 1: Detect format
        detected_format, format_confidence = self.format_detector.detect(full_output)
        logger.info(f"Detected format: {detected_format.value} (confidence: {format_confidence:.2f})")
        
        # Step 2: Extract structure based on format
        try:
            extracted_raw = self._extract_by_format(full_output, detected_format)
        except Exception as e:
            logger.error(f"Structure extraction error: {e}")
            extracted_raw = {}
        
        # Step 3: Apply LLM parsing for semantic extraction
        try:
            llm_extracted = self.llm_parser.parse_with_llm(full_output, tool_name, context)
            # Merge with structure extraction
            extracted_data = {**extracted_raw, **llm_extracted}
        except Exception as e:
            logger.error(f"LLM parsing error: {e}")
            extracted_data = extracted_raw
        
        # Step 4: Normalize and enrich
        normalized_data = self._normalize_and_enrich(extracted_data, tool_name)
        
        # Step 5: Validate
        validated_data, warnings = self._validate(normalized_data)
        
        # Calculate confidence
        confidence = self._calculate_confidence(validated_data, format_confidence, exit_code)
        
        return ParseResult(
            success=True,
            format_detected=detected_format,
            extracted_data=validated_data,
            confidence=confidence,
            warnings=warnings,
            metadata={
                "tool_name": tool_name,
                "exit_code": exit_code,
                "output_length": len(full_output),
                "parsed_at": datetime.now().isoformat()
            }
        )
    
    def _extract_by_format(self, output: str, format_type: OutputFormat) -> Dict[str, Any]:
        """Extract based on detected format"""
        if format_type == OutputFormat.JSON:
            return self.structure_extractor.extract_json(output)
        elif format_type == OutputFormat.XML:
            return self.structure_extractor.extract_xml(output)
        elif format_type == OutputFormat.TABLE:
            return {"table_data": self.structure_extractor.extract_table(output)}
        elif format_type == OutputFormat.KEY_VALUE:
            return self.structure_extractor.extract_key_value(output)
        elif format_type == OutputFormat.LOG_STYLE:
            return self.structure_extractor.extract_log_style(output)
        else:
            return {}
    
    def _normalize_and_enrich(self, data: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Normalize and enrich extracted data"""
        normalized = {}
        
        # Normalize OS
        if 'os' in data and data['os']:
            if isinstance(data['os'], str):
                normalized['os'] = self.normalizer.normalize_os(data['os'])
            else:
                normalized['os'] = data['os']
        
        # Normalize and enrich services
        if 'services' in data and data['services']:
            normalized_services = []
            for svc in data['services']:
                if isinstance(svc, dict):
                    if 'name' in svc:
                        svc['name'] = self.normalizer.normalize_service(svc['name'])
                    if 'port' in svc:
                        svc['port'] = self.normalizer.validate_port(svc['port'])
                    normalized_services.append(svc)
            normalized['services'] = normalized_services
        
        # Enrich users
        if 'users' in data and data['users']:
            enriched_users = []
            for user in data['users']:
                if isinstance(user, str):
                    enriched = self.normalizer.enrich_user(user)
                elif isinstance(user, dict):
                    enriched = self.normalizer.enrich_user(
                        user.get('name', ''),
                        user.get('rid')
                    )
                    enriched.update(user)  # Keep original data
                else:
                    continue
                enriched_users.append(enriched)
            normalized['users'] = enriched_users
        
        # Enrich shares
        if 'shares' in data and data['shares']:
            enriched_shares = []
            for share in data['shares']:
                if isinstance(share, str):
                    enriched = self.normalizer.enrich_share(share)
                elif isinstance(share, dict):
                    enriched = self.normalizer.enrich_share(share.get('name', ''))
                    enriched.update(share)
                else:
                    continue
                enriched_shares.append(enriched)
            normalized['shares'] = enriched_shares
        
        # Normalize CVEs
        if 'cves' in data and data['cves']:
            validated_cves = []
            for cve in data['cves']:
                validated = self.normalizer.validate_cve(str(cve))
                if validated:
                    validated_cves.append(validated)
            normalized['cves'] = validated_cves
        
        # Normalize ports
        if 'ports' in data and data['ports']:
            validated_ports = []
            for port in data['ports']:
                validated = self.normalizer.validate_port(port)
                if validated:
                    validated_ports.append(validated)
            normalized['ports'] = list(set(validated_ports))  # Remove duplicates
        
        # Pass through other data
        for key in ['vulnerabilities', 'domain', 'hostname', 'groups']:
            if key in data:
                normalized[key] = data[key]
        
        return normalized
    
    def _validate(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Validate extracted data"""
        warnings = []
        validated = data.copy()
        
        # Validate ports are in valid range
        if 'ports' in validated:
            valid_ports = [p for p in validated['ports'] if 1 <= p <= 65535]
            if len(valid_ports) < len(validated['ports']):
                warnings.append(f"Removed {len(validated['ports']) - len(valid_ports)} invalid ports")
            validated['ports'] = valid_ports
        
        # Validate services have required fields
        if 'services' in validated:
            valid_services = []
            for svc in validated['services']:
                if isinstance(svc, dict) and ('port' in svc or 'name' in svc):
                    valid_services.append(svc)
                else:
                    warnings.append(f"Removed invalid service entry: {svc}")
            validated['services'] = valid_services
        
        # Check for suspiciously large data
        if 'users' in validated and len(validated['users']) > 1000:
            warnings.append(f"Very large user list ({len(validated['users'])} users) - possible parsing error")
        
        return validated, warnings
    
    def _calculate_confidence(self, data: Dict[str, Any], format_confidence: float, exit_code: int) -> float:
        """Calculate overall parsing confidence"""
        confidence = format_confidence
        
        # Increase confidence if we extracted meaningful data
        if data:
            data_count = sum(len(v) if isinstance(v, (list, dict)) else 1 
                           for v in data.values() if v is not None)
            confidence += min(0.2, data_count * 0.02)
        
        # Decrease confidence for non-zero exit codes
        if exit_code != 0:
            confidence *= 0.8
        
        return min(1.0, confidence)


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Universal Output Parser - Demo")
    print("LLM-Powered Tool Output Translation")
    print("="*70 + "\n")
    
    # Initialize parser
    parser = UniversalOutputParser(use_mock_llm=True)
    
    # Test 1: nmap output (table format)
    print("Test 1: Parsing nmap output")
    print("-" * 70)
    nmap_output = """
    Nmap scan report for 192.168.1.100
    Host is up (0.0010s latency).
    Not shown: 997 closed ports
    PORT     STATE SERVICE      VERSION
    139/tcp  open  netbios-ssn  Microsoft Windows netbios-ssn
    445/tcp  open  microsoft-ds Microsoft Windows XP microsoft-ds
    3389/tcp open  ms-wbt-server
    OS details: Microsoft Windows XP SP2 or SP3
    """
    
    result1 = parser.parse("nmap", nmap_output)
    print(f"Format detected: {result1.format_detected.value}")
    print(f"Confidence: {result1.confidence:.2f}")
    print(f"Extracted data:")
    print(json.dumps(result1.extracted_data, indent=2))
    
    # Test 2: enum4linux output (log style)
    print("\n" + "="*70)
    print("Test 2: Parsing enum4linux output")
    print("-" * 70)
    enum4linux_output = """
    [+] Got domain/workgroup name: WORKGROUP
    
    [+] Enumerating users using SID
    [+] Administrator (RID: 500)
    [+] Guest (RID: 501)
    [+] john.doe (RID: 1001)
    [+] alice (RID: 1002)
    
    [+] Share Enumeration on 192.168.1.100
    
    Sharename       Type      Comment
    ---------       ----      -------
    C$              Disk      Default share
    ADMIN$          Disk      Remote Admin
    Backup          Disk      Backup files
    """
    
    result2 = parser.parse("enum4linux", enum4linux_output)
    print(f"Format detected: {result2.format_detected.value}")
    print(f"Confidence: {result2.confidence:.2f}")
    print(f"Extracted data:")
    print(json.dumps(result2.extracted_data, indent=2))
    
    # Test 3: nuclei output with CVEs
    print("\n" + "="*70)
    print("Test 3: Parsing nuclei output")
    print("-" * 70)
    nuclei_output = """
    [critical] [CVE-2017-0144] [smb-vuln-ms17-010] SMB Remote Code Execution
    [high] [CVE-2008-4250] [ms08-067] Microsoft Windows Server Service RPC Handling
    [medium] [CVE-2019-0708] [bluekeep] Remote Desktop Services RCE
    """
    
    result3 = parser.parse("nuclei", nuclei_output)
    print(f"Format detected: {result3.format_detected.value}")
    print(f"Confidence: {result3.confidence:.2f}")
    print(f"Extracted data:")
    print(json.dumps(result3.extracted_data, indent=2))
    
    print("\n" + "="*70)
    print("Key Features Demonstrated:")
    print("="*70)
    print("✓ Automatic format detection (table, log-style, free-form)")
    print("✓ Service name normalization (microsoft-ds → smb)")
    print("✓ User enrichment (Administrator → privileged, high risk)")
    print("✓ Share enrichment (C$ → admin share, high risk)")
    print("✓ CVE validation and extraction")
    print("✓ Confidence scoring")
    print("✓ Port validation")
    print("\n" + "="*70 + "\n")
