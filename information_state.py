"""
Information State Tracking System

This module implements an intelligent tracking system that acts as the "memory" 
of the reconnaissance process. It tracks discovered information, prevents redundancy,
measures completeness, and guides the RL agent toward efficient reconnaissance.

Core Capabilities:
- Information ingestion from multiple tool formats
- Intelligent deduplication
- Value-based scoring
- Completeness measurement across multiple dimensions
- Gap detection and prioritization
- Temporal and source tracking
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from collections import defaultdict
import hashlib


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InformationType(Enum):
    """Types of information that can be discovered"""
    # Identity
    OS_NAME = "os_name"
    OS_VERSION = "os_version"
    HOSTNAME = "hostname"
    DOMAIN = "domain"
    
    # Network
    OPEN_PORT = "open_port"
    SERVICE = "service"
    PROTOCOL = "protocol"
    FIREWALL_RULE = "firewall_rule"
    
    # Access Control
    USER_ACCOUNT = "user_account"
    SHARE = "share"
    PERMISSION = "permission"
    GROUP = "group"
    
    # Security
    VULNERABILITY = "vulnerability"
    CVE = "cve"
    PATCH_LEVEL = "patch_level"
    SECURITY_SOFTWARE = "security_software"
    
    # Exploitability
    CREDENTIAL = "credential"
    EXPLOIT_PATH = "exploit_path"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class InformationValue(Enum):
    """Value/priority levels for different information types"""
    CRITICAL = 50  # Valid credentials, exploitable vulnerabilities
    HIGH = 20      # Service versions, exploitable CVEs
    MEDIUM = 10    # User lists, shares, OS details
    LOW = 2        # Hostname, default accounts


@dataclass
class ServiceInfo:
    """
    Detailed service information with normalization
    """
    port: int
    protocol: str = "tcp"
    state: str = "open"
    service_name: Optional[str] = None
    product: Optional[str] = None
    version: Optional[str] = None
    banner: Optional[str] = None
    cpe: Optional[str] = None  # Common Platform Enumeration
    
    discovered_by: List[str] = field(default_factory=list)
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    state_changes: List[Dict[str, str]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.first_seen:
            self.first_seen = datetime.now().isoformat()
        if not self.last_seen:
            self.last_seen = self.first_seen
    
    def update_state(self, new_state: str, tool: str):
        """Track state changes over time"""
        timestamp = datetime.now().isoformat()
        if new_state != self.state:
            self.state_changes.append({
                "time": timestamp,
                "old_state": self.state,
                "new_state": new_state,
                "detected_by": tool
            })
            self.state = new_state
        self.last_seen = timestamp
        if tool not in self.discovered_by:
            self.discovered_by.append(tool)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "port": self.port,
            "protocol": self.protocol,
            "state": self.state,
            "service_name": self.service_name,
            "product": self.product,
            "version": self.version,
            "banner": self.banner,
            "cpe": self.cpe,
            "discovered_by": self.discovered_by,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "state_changes": self.state_changes
        }


@dataclass
class InformationItem:
    """
    Represents a single piece of discovered information with confidence and temporal tracking
    """
    info_type: InformationType
    value: Any
    confidence: float = 1.0  # 0.0 to 1.0
    sources: List[str] = field(default_factory=list)  # Which tools discovered this
    timestamps: List[str] = field(default_factory=list)  # When discovered
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context
    
    # Enhanced tracking
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    confirmation_count: int = 1  # How many tools confirmed this
    
    # Relationships
    requires: List[str] = field(default_factory=list)  # Dependencies
    enables: List[str] = field(default_factory=list)  # What this unlocks
    related_items: List[str] = field(default_factory=list)  # Related information
    
    def __post_init__(self):
        if not self.sources:
            self.sources = []
        if not self.timestamps:
            now = datetime.now().isoformat()
            self.timestamps = [now]
            self.first_seen = now
            self.last_seen = now
        else:
            self.first_seen = self.timestamps[0]
            self.last_seen = self.timestamps[-1]
    
    def get_hash(self) -> str:
        """Generate unique hash for this information item"""
        # Use type and normalized value for hashing
        normalized = str(self.value).lower().strip()
        hash_input = f"{self.info_type.value}:{normalized}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def add_source(self, source: str, timestamp: Optional[str] = None):
        """Add a source that confirmed this information"""
        timestamp = timestamp or datetime.now().isoformat()
        if source not in self.sources:
            self.sources.append(source)
            self.timestamps.append(timestamp)
            self.confirmation_count += 1
            self.last_seen = timestamp
            # Increase confidence with multiple confirmations
            self.confidence = min(1.0, self.confidence + 0.05)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "type": self.info_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "sources": self.sources,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "confirmation_count": self.confirmation_count,
            "timestamps": self.timestamps,
            "metadata": self.metadata,
            "requires": self.requires,
            "enables": self.enables,
            "related_items": self.related_items
        }


@dataclass
class OSInfo:
    """Detailed OS information with confidence tracking"""
    name: Optional[str] = None
    family: Optional[str] = None  # Windows, Linux, Unix, etc.
    version: Optional[str] = None
    build: Optional[str] = None
    architecture: Optional[str] = None  # x86, x64, ARM
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    
    def update(self, name: Optional[str] = None, version: Optional[str] = None, 
               source: Optional[str] = None, confidence: float = 1.0):
        """Update OS information"""
        if name and (not self.name or confidence > self.confidence):
            self.name = name
            # Extract family
            name_lower = name.lower()
            if 'windows' in name_lower:
                self.family = 'Windows'
            elif 'linux' in name_lower:
                self.family = 'Linux'
            elif 'unix' in name_lower or 'bsd' in name_lower:
                self.family = 'Unix'
        
        if version and (not self.version or confidence > self.confidence):
            self.version = version
        
        if confidence > self.confidence:
            self.confidence = confidence
        
        if source and source not in self.sources:
            self.sources.append(source)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "version": self.version,
            "build": self.build,
            "architecture": self.architecture,
            "confidence": self.confidence,
            "sources": self.sources
        }


@dataclass
class TargetDossier:
    """
    Comprehensive dossier of all information about a target
    
    Organized into categories matching the completeness dimensions
    Enhanced with granular tracking and relationships
    """
    target_id: str  # IP, hostname, or identifier
    
    # Identity (enhanced)
    os_info: OSInfo = field(default_factory=OSInfo)
    hostname: Optional[str] = None
    domain: Optional[str] = None
    
    # Network (enhanced with ServiceInfo)
    open_ports: Set[int] = field(default_factory=set)
    services: Dict[int, ServiceInfo] = field(default_factory=dict)  # port -> ServiceInfo
    protocols: Set[str] = field(default_factory=set)
    closed_ports: Set[int] = field(default_factory=set)  # Track closed ports too
    
    # Access Control (with metadata)
    users: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # username -> metadata
    shares: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # share -> metadata
    permissions: Dict[str, List[str]] = field(default_factory=dict)
    groups: Set[str] = field(default_factory=set)
    
    # Security (enhanced)
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    cves: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # CVE -> details
    patch_level: Optional[str] = None
    security_software: Set[str] = field(default_factory=set)
    
    # Exploitability (with relationships)
    credentials: List[Dict[str, str]] = field(default_factory=list)
    exploit_paths: List[Dict[str, Any]] = field(default_factory=list)
    privilege_escalation: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    scan_count: int = 0
    
    def update_timestamp(self):
        """Update the last modified timestamp"""
        self.updated_at = datetime.now().isoformat()
        self.scan_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "target_id": self.target_id,
            "identity": {
                "os": self.os_info.to_dict(),
                "hostname": self.hostname,
                "domain": self.domain
            },
            "network": {
                "open_ports": sorted(list(self.open_ports)),
                "closed_ports": sorted(list(self.closed_ports)),
                "services": {port: svc.to_dict() for port, svc in self.services.items()},
                "protocols": list(self.protocols)
            },
            "access_control": {
                "users": self.users,
                "shares": self.shares,
                "permissions": self.permissions,
                "groups": list(self.groups)
            },
            "security": {
                "vulnerabilities": self.vulnerabilities,
                "cves": self.cves,
                "patch_level": self.patch_level,
                "security_software": list(self.security_software)
            },
            "exploitability": {
                "credentials": self.credentials,
                "exploit_paths": self.exploit_paths,
                "privilege_escalation": self.privilege_escalation
            },
            "metadata": {
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "scan_count": self.scan_count
            }
        }


@dataclass
class InformationDelta:
    """
    Tracks what changed after a tool execution
    """
    new_items: List[InformationItem] = field(default_factory=list)
    updated_items: List[Tuple[InformationItem, InformationItem]] = field(default_factory=list)  # (old, new)
    redundant_items: List[InformationItem] = field(default_factory=list)
    points_gained: int = 0
    breakdown: Dict[str, int] = field(default_factory=dict)
    
    def add_new(self, item: InformationItem, points: int):
        """Add a new information item"""
        self.new_items.append(item)
        self.points_gained += points
        item_type = item.info_type.value
        self.breakdown[item_type] = self.breakdown.get(item_type, 0) + points
    
    def add_updated(self, old_item: InformationItem, new_item: InformationItem, points: int):
        """Add an updated information item"""
        self.updated_items.append((old_item, new_item))
        self.points_gained += points
        item_type = new_item.info_type.value
        self.breakdown[item_type] = self.breakdown.get(item_type, 0) + points
    
    def add_redundant(self, item: InformationItem, penalty: int = -1):
        """Add a redundant information item"""
        self.redundant_items.append(item)
        self.points_gained += penalty
        self.breakdown["redundant"] = self.breakdown.get("redundant", 0) + penalty
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = []
        lines.append(f"Points Gained: {self.points_gained}")
        lines.append(f"New Items: {len(self.new_items)}")
        lines.append(f"Updated Items: {len(self.updated_items)}")
        lines.append(f"Redundant Items: {len(self.redundant_items)}")
        
        if self.breakdown:
            lines.append("\nBreakdown:")
            for key, value in sorted(self.breakdown.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {key}: {value:+d}")
        
        return "\n".join(lines)


@dataclass
class CompletenessScore:
    """
    Measures how complete the reconnaissance is across multiple dimensions
    """
    # Dimension scores (0.0 to 1.0)
    identity_score: float = 0.0
    network_score: float = 0.0
    access_control_score: float = 0.0
    security_score: float = 0.0
    exploitability_score: float = 0.0
    
    # Dimension weights (must sum to 1.0)
    identity_weight: float = 0.20
    network_weight: float = 0.20
    access_control_weight: float = 0.25
    security_weight: float = 0.20
    exploitability_weight: float = 0.15
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall completeness score"""
        return (
            self.identity_score * self.identity_weight +
            self.network_score * self.network_weight +
            self.access_control_score * self.access_control_weight +
            self.security_score * self.security_weight +
            self.exploitability_score * self.exploitability_weight
        )
    
    @property
    def percentage(self) -> float:
        """Overall score as percentage"""
        return self.overall_score * 100
    
    def get_weakest_dimension(self) -> Tuple[str, float]:
        """Return the dimension with lowest completeness"""
        dimensions = {
            "identity": self.identity_score,
            "network": self.network_score,
            "access_control": self.access_control_score,
            "security": self.security_score,
            "exploitability": self.exploitability_score
        }
        return min(dimensions.items(), key=lambda x: x[1])
    
    def get_missing_critical_items(self) -> List[str]:
        """Return list of critical missing items"""
        missing = []
        
        if self.identity_score < 0.5:
            missing.append("Basic target identity (OS, hostname)")
        if self.network_score < 0.3:
            missing.append("Network services enumeration")
        if self.access_control_score < 0.3:
            missing.append("User accounts and shares")
        if self.security_score < 0.3:
            missing.append("Vulnerability scanning")
        if self.exploitability_score < 0.1:
            missing.append("Exploitation vectors (credentials, exploits)")
        
        return missing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with detailed breakdown"""
        weakest = self.get_weakest_dimension()
        missing = self.get_missing_critical_items()
        
        return {
            "overall": self.overall_score,
            "percentage": f"{self.percentage:.1f}%",
            "dimensions": {
                "identity": {
                    "score": self.identity_score,
                    "percentage": f"{self.identity_score * 100:.1f}%",
                    "weight": self.identity_weight
                },
                "network": {
                    "score": self.network_score,
                    "percentage": f"{self.network_score * 100:.1f}%",
                    "weight": self.network_weight
                },
                "access_control": {
                    "score": self.access_control_score,
                    "percentage": f"{self.access_control_score * 100:.1f}%",
                    "weight": self.access_control_weight
                },
                "security": {
                    "score": self.security_score,
                    "percentage": f"{self.security_score * 100:.1f}%",
                    "weight": self.security_weight
                },
                "exploitability": {
                    "score": self.exploitability_score,
                    "percentage": f"{self.exploitability_score * 100:.1f}%",
                    "weight": self.exploitability_weight
                }
            },
            "analysis": {
                "weakest_dimension": weakest[0],
                "weakest_score": weakest[1],
                "missing_critical": missing
            }
        }


class InformationNormalizer:
    """
    Normalizes tool-specific outputs into standardized information schema
    
    This bridges the gap between raw tool output and our information model
    """
    
    @staticmethod
    def normalize_os_info(raw_text: str, tool_name: str) -> Dict[str, Any]:
        """
        Extract and normalize OS information from various tool formats
        
        Handles:
        - nmap: "Running: Microsoft Windows XP|2003"
        - enum4linux: "OS=[Windows 5.1]"
        - crackmapexec: "Windows XP Build 2600"
        """
        normalized = {
            "name": None,
            "family": None,
            "version": None,
            "confidence": 0.5
        }
        
        text_lower = raw_text.lower()
        
        # Windows detection
        if 'windows' in text_lower:
            normalized["family"] = "Windows"
            normalized["confidence"] = 0.9
            
            # Extract specific version
            if 'xp' in text_lower:
                normalized["name"] = "Windows XP"
                normalized["version"] = "5.1"
            elif 'vista' in text_lower:
                normalized["name"] = "Windows Vista"
                normalized["version"] = "6.0"
            elif '2003' in text_lower:
                normalized["name"] = "Windows Server 2003"
                normalized["version"] = "5.2"
            elif '2008' in text_lower:
                normalized["name"] = "Windows Server 2008"
                normalized["version"] = "6.0"
            elif '7' in text_lower or 'seven' in text_lower:
                normalized["name"] = "Windows 7"
                normalized["version"] = "6.1"
            elif '10' in text_lower or 'ten' in text_lower:
                normalized["name"] = "Windows 10"
                normalized["version"] = "10.0"
            elif '11' in text_lower:
                normalized["name"] = "Windows 11"
                normalized["version"] = "10.0"
            else:
                normalized["name"] = "Windows"
        
        # Linux detection
        elif 'linux' in text_lower:
            normalized["family"] = "Linux"
            normalized["confidence"] = 0.9
            
            # Extract distribution
            if 'ubuntu' in text_lower:
                normalized["name"] = "Ubuntu Linux"
            elif 'debian' in text_lower:
                normalized["name"] = "Debian Linux"
            elif 'centos' in text_lower:
                normalized["name"] = "CentOS Linux"
            elif 'red hat' in text_lower or 'redhat' in text_lower:
                normalized["name"] = "Red Hat Linux"
            else:
                normalized["name"] = "Linux"
        
        # Unix variants
        elif any(x in text_lower for x in ['unix', 'bsd', 'solaris', 'aix']):
            normalized["family"] = "Unix"
            normalized["name"] = raw_text
            normalized["confidence"] = 0.85
        
        return normalized
    
    @staticmethod
    def normalize_service(port: int, service_str: str, tool_name: str) -> ServiceInfo:
        """
        Extract and normalize service information
        
        Handles:
        - nmap: "445/tcp open microsoft-ds Microsoft Windows XP microsoft-ds"
        - masscan: "open tcp 445"
        - crackmapexec: "SMB    192.168.1.100    445    MACHINE"
        """
        service_info = ServiceInfo(port=port)
        
        parts = service_str.lower().split()
        
        # Extract protocol
        if 'tcp' in service_str:
            service_info.protocol = 'tcp'
        elif 'udp' in service_str:
            service_info.protocol = 'udp'
        
        # Extract state
        if 'open' in parts:
            service_info.state = 'open'
        elif 'closed' in parts:
            service_info.state = 'closed'
        elif 'filtered' in parts:
            service_info.state = 'filtered'
        
        # Extract service name
        common_services = {
            '21': 'ftp', '22': 'ssh', '23': 'telnet', '25': 'smtp',
            '53': 'dns', '80': 'http', '110': 'pop3', '139': 'netbios-ssn',
            '143': 'imap', '443': 'https', '445': 'microsoft-ds', '3306': 'mysql',
            '3389': 'rdp', '5432': 'postgresql', '8080': 'http-proxy'
        }
        
        # Try to identify service
        for svc in ['http', 'https', 'ssh', 'ftp', 'smb', 'rdp', 'mysql', 'postgresql']:
            if svc in service_str.lower():
                service_info.service_name = svc
                break
        
        if not service_info.service_name and str(port) in common_services:
            service_info.service_name = common_services[str(port)]
        
        # Extract version info
        version_match = re.search(r'(\d+\.[\d.]+)', service_str)
        if version_match:
            service_info.version = version_match.group(1)
        
        # Extract product
        if 'microsoft' in service_str.lower():
            service_info.product = 'Microsoft'
        elif 'apache' in service_str.lower():
            service_info.product = 'Apache'
        elif 'nginx' in service_str.lower():
            service_info.product = 'nginx'
        
        return service_info
    
    @staticmethod
    def normalize_users(user_str: str, tool_name: str) -> List[Dict[str, Any]]:
        """
        Extract and normalize user account information
        
        Handles:
        - enum4linux: "[+] Administrator (RID: 500)"
        - crackmapexec: "user:[Administrator] rid:[0x1f4]"
        - rpcclient: "S-1-5-21-xxx-500 Administrator"
        """
        users = []
        
        # Pattern 1: enum4linux format
        pattern1 = r'\[?\+?\]?\s*([A-Za-z0-9._-]+)\s*(?:\(RID:\s*(\d+)\))?'
        for match in re.finditer(pattern1, user_str):
            username, rid = match.groups()
            if username and username.lower() not in ['user', 'rid', 'sid']:
                users.append({
                    "name": username,
                    "rid": int(rid) if rid else None,
                    "type": "local" if rid and int(rid) < 1000 else "domain"
                })
        
        # Pattern 2: crackmapexec format
        pattern2 = r'user:\[([^\]]+)\]\s*rid:\[0x([0-9a-fA-F]+)\]'
        for match in re.finditer(pattern2, user_str):
            username, rid_hex = match.groups()
            users.append({
                "name": username,
                "rid": int(rid_hex, 16),
                "type": "local"
            })
        
        return users
    
    @staticmethod
    def normalize_cve(cve_str: str, context: str = "") -> Dict[str, Any]:
        """
        Normalize CVE information with severity and exploitability
        """
        cve_match = re.search(r'(CVE-\d{4}-\d+)', cve_str, re.IGNORECASE)
        if not cve_match:
            return None
        
        cve_id = cve_match.group(1).upper()
        
        # Extract severity
        severity = "unknown"
        context_lower = (cve_str + " " + context).lower()
        if 'critical' in context_lower:
            severity = 'critical'
        elif 'high' in context_lower:
            severity = 'high'
        elif 'medium' in context_lower:
            severity = 'medium'
        elif 'low' in context_lower:
            severity = 'low'
        
        # Check for exploit availability
        has_exploit = any(x in context_lower for x in ['exploit', 'metasploit', 'poc', 'proof of concept'])
        
        return {
            "cve_id": cve_id,
            "severity": severity,
            "has_exploit": has_exploit,
            "description": cve_str
        }


class InformationExtractor:
    """
    Extracts structured information from tool outputs
    
    Handles multiple tool formats and normalizes data using InformationNormalizer
    """
    
    def __init__(self):
        self.normalizer = InformationNormalizer()
    
    def extract_from_nmap(self, stdout: str) -> List[InformationItem]:
        """Extract information from nmap output with normalization"""
        items = []
        
        # Extract OS information (normalized)
        os_match = re.search(r'OS details?: (.+?)(?:\n|$)', stdout, re.IGNORECASE)
        if os_match:
            os_detail = os_match.group(1).strip()
            normalized_os = self.normalizer.normalize_os_info(os_detail, 'nmap')
            
            if normalized_os['name']:
                items.append(InformationItem(
                    info_type=InformationType.OS_NAME,
                    value=normalized_os['name'],
                    confidence=normalized_os['confidence'],
                    metadata={
                        "raw": os_detail,
                        "family": normalized_os['family'],
                        "version": normalized_os['version']
                    }
                ))
        
        # Extract open ports with enhanced service info
        port_pattern = r'(\d+)/(tcp|udp)\s+open\s+(\S+)(?:\s+(.+?))?(?:\n|$)'
        for match in re.finditer(port_pattern, stdout):
            port_num, protocol, service, version_info = match.groups()
            port = int(port_num)
            
            # Port item
            items.append(InformationItem(
                info_type=InformationType.OPEN_PORT,
                value=port,
                metadata={"protocol": protocol}
            ))
            
            # Normalized service item
            service_str = f"{port_num}/{protocol} open {service} {version_info or ''}"
            normalized_service = self.normalizer.normalize_service(port, service_str, 'nmap')
            
            items.append(InformationItem(
                info_type=InformationType.SERVICE,
                value=normalized_service.service_name or service,
                metadata={
                    "port": port,
                    "protocol": protocol,
                    "product": normalized_service.product,
                    "version": normalized_service.version or version_info or "unknown",
                    "service_info": normalized_service.to_dict()
                }
            ))
        
        # Extract hostname
        hostname_match = re.search(r'(?:Nmap scan report for|rDNS record for \d+\.\d+\.\d+\.\d+:)\s+(\S+)', stdout)
        if hostname_match:
            hostname = hostname_match.group(1)
            if not re.match(r'\d+\.\d+\.\d+\.\d+', hostname):  # Not an IP
                items.append(InformationItem(
                    info_type=InformationType.HOSTNAME,
                    value=hostname
                ))
        
        return items
    
    def extract_from_enum4linux(self, stdout: str) -> List[InformationItem]:
        """Extract information from enum4linux output with normalization"""
        items = []
        
        # Extract domain/workgroup
        domain_match = re.search(r'(?:Domain Name|Workgroup):\s+(\S+)', stdout, re.IGNORECASE)
        if domain_match:
            items.append(InformationItem(
                info_type=InformationType.DOMAIN,
                value=domain_match.group(1)
            ))
        
        # Extract users (normalized)
        normalized_users = self.normalizer.normalize_users(stdout, 'enum4linux')
        for user_data in normalized_users:
            items.append(InformationItem(
                info_type=InformationType.USER_ACCOUNT,
                value=user_data['name'],
                metadata=user_data
            ))
        
        # Extract shares with metadata
        share_pattern = r'Sharename\s+Type\s+Comment\s+\-+\s+(.+?)(?:\n\n|\Z)'
        share_section = re.search(share_pattern, stdout, re.DOTALL)
        if share_section:
            for line in share_section.group(1).split('\n'):
                parts = line.split()
                if parts and not line.strip().startswith('-'):
                    share_name = parts[0]
                    share_type = parts[1] if len(parts) > 1 else "unknown"
                    comment = ' '.join(parts[2:]) if len(parts) > 2 else ""
                    
                    # Determine relationships and risk
                    enables = []
                    risk_level = "LOW"
                    
                    if share_name.endswith('$'):
                        risk_level = "HIGH"  # Admin shares
                        enables.append("admin_access")
                    elif any(x in share_name.lower() for x in ['backup', 'data', 'share']):
                        risk_level = "MEDIUM"
                        enables.append("data_access")
                    
                    items.append(InformationItem(
                        info_type=InformationType.SHARE,
                        value=share_name,
                        metadata={
                            "type": share_type,
                            "comment": comment,
                            "risk_level": risk_level
                        },
                        enables=enables
                    ))
        
        # Extract OS information (normalized)
        os_match = re.search(r'OS:\s+(.+?)(?:\n|$)', stdout)
        if os_match:
            os_str = os_match.group(1).strip()
            normalized_os = self.normalizer.normalize_os_info(os_str, 'enum4linux')
            
            if normalized_os['name']:
                items.append(InformationItem(
                    info_type=InformationType.OS_NAME,
                    value=normalized_os['name'],
                    confidence=normalized_os['confidence'],
                    metadata={
                        "raw": os_str,
                        "family": normalized_os['family'],
                        "version": normalized_os['version']
                    }
                ))
        
        return items
    
    def extract_from_nuclei(self, stdout: str) -> List[InformationItem]:
        """Extract information from nuclei output with normalization"""
        items = []
        
        # Extract CVEs (normalized)
        cve_pattern = r'(CVE-\d{4}-\d+)'
        for match in re.finditer(cve_pattern, stdout):
            cve_context = stdout[max(0, match.start()-100):min(len(stdout), match.end()+100)]
            normalized_cve = self.normalizer.normalize_cve(match.group(1), cve_context)
            
            if normalized_cve:
                # Determine relationships
                requires = []
                enables = []
                
                if 'smb' in cve_context.lower():
                    requires.append("port_445_open")
                if 'http' in cve_context.lower() or 'web' in cve_context.lower():
                    requires.append("web_service")
                
                if normalized_cve['has_exploit']:
                    enables.append("remote_code_execution")
                
                if normalized_cve['severity'] in ['critical', 'high']:
                    enables.append("exploitation")
                
                items.append(InformationItem(
                    info_type=InformationType.CVE,
                    value=normalized_cve['cve_id'],
                    confidence=0.95,  # Nuclei is reliable
                    metadata=normalized_cve,
                    requires=requires,
                    enables=enables
                ))
        
        # Extract vulnerabilities
        vuln_pattern = r'\[([^\]]+)\]\s+\[([^\]]+)\].*?(http[s]?://[^\s]+)'
        for match in re.finditer(vuln_pattern, stdout):
            severity, template, url = match.groups()
            items.append(InformationItem(
                info_type=InformationType.VULNERABILITY,
                value=template,
                confidence=0.9,
                metadata={"severity": severity, "url": url, "scanner": "nuclei"}
            ))
        
        return items
    
    def extract_from_sqlmap(self, stdout: str) -> List[InformationItem]:
        """Extract information from sqlmap output"""
        items = []
        
        # Check if parameter is injectable
        if re.search(r'Parameter.*?is (?:vulnerable|injectable)', stdout, re.IGNORECASE):
            items.append(InformationItem(
                info_type=InformationType.VULNERABILITY,
                value="SQL Injection",
                metadata={"tool": "sqlmap", "type": "injection"}
            ))
        
        # Extract database names
        db_pattern = r'available databases.*?:(.*?)(?:\n\n|\Z)'
        db_match = re.search(db_pattern, stdout, re.DOTALL | re.IGNORECASE)
        if db_match:
            databases = re.findall(r'\[\*\]\s+(\S+)', db_match.group(1))
            for db in databases:
                items.append(InformationItem(
                    info_type=InformationType.VULNERABILITY,
                    value=f"Database: {db}",
                    metadata={"database": db, "access": "enumerated"}
                ))
        
        # Extract DBMS information
        dbms_match = re.search(r'back-end DBMS:\s+(.+?)(?:\n|$)', stdout, re.IGNORECASE)
        if dbms_match:
            items.append(InformationItem(
                info_type=InformationType.SERVICE,
                value=dbms_match.group(1).strip(),
                metadata={"type": "database"}
            ))
        
        return items
    
    def extract_generic(self, stdout: str, tool_name: str) -> List[InformationItem]:
        """Generic extraction for tools without specific extractors"""
        items = []
        
        # Extract IPs
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ips = set(re.findall(ip_pattern, stdout))
        
        # Extract common indicators
        if re.search(r'(?:success|found|vulnerable|exploitable)', stdout, re.IGNORECASE):
            items.append(InformationItem(
                info_type=InformationType.VULNERABILITY,
                value=f"Potential finding from {tool_name}",
                confidence=0.5,
                metadata={"tool": tool_name, "needs_review": True}
            ))
        
        return items


class InformationStateTracker:
    """
    Main state tracking system
    
    Manages target dossiers, deduplication, scoring, and completeness measurement
    """
    
    def __init__(self):
        self.dossiers: Dict[str, TargetDossier] = {}  # target_id -> dossier
        self.information_db: Dict[str, Dict[str, InformationItem]] = {}  # target_id -> {hash -> item}
        self.extractor = InformationExtractor()
        
        # Scoring configuration
        self.scoring_rules = self._initialize_scoring_rules()
    
    def _initialize_scoring_rules(self) -> Dict[InformationType, int]:
        """Initialize base scoring for different information types"""
        return {
            # Critical information
            InformationType.CREDENTIAL: 50,
            InformationType.EXPLOIT_PATH: 40,
            InformationType.CVE: 20,
            
            # High value
            InformationType.VULNERABILITY: 15,
            InformationType.SERVICE: 10,
            InformationType.PRIVILEGE_ESCALATION: 30,
            
            # Medium value
            InformationType.USER_ACCOUNT: 5,
            InformationType.SHARE: 8,
            InformationType.GROUP: 6,
            InformationType.OS_VERSION: 10,
            InformationType.PATCH_LEVEL: 12,
            
            # Low value
            InformationType.OPEN_PORT: 3,
            InformationType.OS_NAME: 5,
            InformationType.HOSTNAME: 2,
            InformationType.DOMAIN: 5,
            InformationType.PROTOCOL: 2,
        }
    
    def get_or_create_dossier(self, target_id: str) -> TargetDossier:
        """Get existing dossier or create new one"""
        if target_id not in self.dossiers:
            self.dossiers[target_id] = TargetDossier(target_id=target_id)
            self.information_db[target_id] = {}
            logger.info(f"Created new dossier for target: {target_id}")
        return self.dossiers[target_id]
    
    def ingest_tool_output(
        self,
        target_id: str,
        tool_name: str,
        stdout: str,
        stderr: str,
        timestamp: Optional[str] = None
    ) -> InformationDelta:
        """
        Ingest tool output and update state
        
        Returns delta showing what changed
        """
        timestamp = timestamp or datetime.now().isoformat()
        dossier = self.get_or_create_dossier(target_id)
        delta = InformationDelta()
        
        # Extract information based on tool
        extracted_items = self._extract_information(tool_name, stdout, stderr)
        
        # Add source and timestamp to all items
        for item in extracted_items:
            item.add_source(tool_name, timestamp)
        
        # Process each extracted item
        for item in extracted_items:
            self._process_information_item(target_id, item, delta)
        
        # Update dossier timestamp
        dossier.update_timestamp()
        
        logger.info(f"Ingested output from {tool_name} for {target_id}: {delta.points_gained} points")
        
        return delta
    
    def _extract_information(self, tool_name: str, stdout: str, stderr: str) -> List[InformationItem]:
        """Extract information using appropriate extractor"""
        tool_lower = tool_name.lower()
        
        if 'nmap' in tool_lower:
            return self.extractor.extract_from_nmap(stdout)
        elif 'enum4linux' in tool_lower:
            return self.extractor.extract_from_enum4linux(stdout)
        elif 'nuclei' in tool_lower:
            return self.extractor.extract_from_nuclei(stdout)
        elif 'sqlmap' in tool_lower:
            return self.extractor.extract_from_sqlmap(stdout)
        else:
            return self.extractor.extract_generic(stdout, tool_name)
    
    def _process_information_item(
        self,
        target_id: str,
        item: InformationItem,
        delta: InformationDelta
    ):
        """
        Process a single information item
        
        Handles deduplication and updates dossier
        """
        dossier = self.dossiers[target_id]
        info_db = self.information_db[target_id]
        item_hash = item.get_hash()
        
        # Check if this information already exists
        if item_hash in info_db:
            existing = info_db[item_hash]
            
            # Add new source
            if item.sources and item.sources[0] not in existing.sources:
                existing.add_source(item.sources[0], item.timestamps[0] if item.timestamps else None)
                delta.add_redundant(item, penalty=-1)
                logger.debug(f"Redundant info confirmed by new source: {item.info_type.value}={item.value}")
            else:
                delta.add_redundant(item, penalty=-2)
                logger.debug(f"Completely redundant info: {item.info_type.value}={item.value}")
            return
        
        # New information - add to database
        info_db[item_hash] = item
        
        # Calculate points
        base_points = self.scoring_rules.get(item.info_type, 5)
        points = int(base_points * item.confidence)
        
        # Update dossier with new information
        self._update_dossier(dossier, item)
        
        # Record in delta
        delta.add_new(item, points)
        logger.debug(f"New info: {item.info_type.value}={item.value} (+{points} points)")
    
    def _update_dossier(self, dossier: TargetDossier, item: InformationItem):
        """Update dossier with new information item (enhanced)"""
        info_type = item.info_type
        value = item.value
        source = item.sources[0] if item.sources else "unknown"
        
        # Identity updates (enhanced)
        if info_type == InformationType.OS_NAME:
            dossier.os_info.update(
                name=str(value),
                version=item.metadata.get('version'),
                source=source,
                confidence=item.confidence
            )
        elif info_type == InformationType.OS_VERSION:
            dossier.os_info.update(
                version=str(value),
                source=source,
                confidence=item.confidence
            )
        elif info_type == InformationType.HOSTNAME:
            dossier.hostname = str(value)
        elif info_type == InformationType.DOMAIN:
            dossier.domain = str(value)
        
        # Network updates (enhanced with ServiceInfo)
        elif info_type == InformationType.OPEN_PORT:
            dossier.open_ports.add(int(value))
        elif info_type == InformationType.SERVICE:
            port = item.metadata.get('port', 0)
            
            # Check if we have full service_info in metadata
            if 'service_info' in item.metadata:
                service_data = item.metadata['service_info']
                if port not in dossier.services:
                    # Create new ServiceInfo
                    dossier.services[port] = ServiceInfo(
                        port=port,
                        protocol=service_data.get('protocol', 'tcp'),
                        state=service_data.get('state', 'open'),
                        service_name=service_data.get('service_name'),
                        product=service_data.get('product'),
                        version=service_data.get('version'),
                        banner=service_data.get('banner')
                    )
                    dossier.services[port].discovered_by.append(source)
                else:
                    # Update existing service
                    existing = dossier.services[port]
                    if service_data.get('version') and not existing.version:
                        existing.version = service_data.get('version')
                    if service_data.get('product') and not existing.product:
                        existing.product = service_data.get('product')
                    existing.update_state(service_data.get('state', 'open'), source)
            else:
                # Fallback to simple dict
                if port not in dossier.services:
                    dossier.services[port] = ServiceInfo(
                        port=port,
                        service_name=str(value),
                        version=item.metadata.get('version', 'unknown')
                    )
                    dossier.services[port].discovered_by.append(source)
        
        elif info_type == InformationType.PROTOCOL:
            dossier.protocols.add(str(value))
        
        # Access Control updates (enhanced with metadata)
        elif info_type == InformationType.USER_ACCOUNT:
            username = str(value)
            if username not in dossier.users:
                dossier.users[username] = {
                    "rid": item.metadata.get('rid'),
                    "type": item.metadata.get('type', 'unknown'),
                    "discovered_by": [source],
                    "first_seen": item.first_seen
                }
            else:
                # Confirm existing user
                if source not in dossier.users[username]['discovered_by']:
                    dossier.users[username]['discovered_by'].append(source)
        
        elif info_type == InformationType.SHARE:
            sharename = str(value)
            if sharename not in dossier.shares:
                dossier.shares[sharename] = {
                    "type": item.metadata.get('type', 'unknown'),
                    "comment": item.metadata.get('comment', ''),
                    "risk_level": item.metadata.get('risk_level', 'LOW'),
                    "discovered_by": [source],
                    "enables": item.enables,
                    "first_seen": item.first_seen
                }
            else:
                if source not in dossier.shares[sharename]['discovered_by']:
                    dossier.shares[sharename]['discovered_by'].append(source)
        
        elif info_type == InformationType.GROUP:
            dossier.groups.add(str(value))
        
        # Security updates (enhanced)
        elif info_type == InformationType.CVE:
            cve_id = str(value)
            if cve_id not in dossier.cves:
                dossier.cves[cve_id] = {
                    **item.metadata,
                    "discovered_by": [source],
                    "first_seen": item.first_seen,
                    "requires": item.requires,
                    "enables": item.enables,
                    "confidence": item.confidence
                }
            else:
                if source not in dossier.cves[cve_id]['discovered_by']:
                    dossier.cves[cve_id]['discovered_by'].append(source)
        
        elif info_type == InformationType.VULNERABILITY:
            dossier.vulnerabilities.append({
                "name": str(value),
                "metadata": item.metadata,
                "discovered_at": item.timestamps[-1] if item.timestamps else None,
                "discovered_by": source,
                "confidence": item.confidence,
                "enables": item.enables
            })
        
        elif info_type == InformationType.PATCH_LEVEL:
            dossier.patch_level = str(value)
        
        elif info_type == InformationType.SECURITY_SOFTWARE:
            dossier.security_software.add(str(value))
        
        # Exploitability updates (enhanced with relationships)
        elif info_type == InformationType.CREDENTIAL:
            cred_data = {
                **item.metadata,
                "discovered_by": source,
                "first_seen": item.first_seen,
                "confidence": item.confidence
            }
            dossier.credentials.append(cred_data)
        
        elif info_type == InformationType.EXPLOIT_PATH:
            dossier.exploit_paths.append({
                "path": str(value),
                "metadata": item.metadata,
                "requires": item.requires,
                "discovered_by": source
            })
        
        elif info_type == InformationType.PRIVILEGE_ESCALATION:
            dossier.privilege_escalation.append(str(value))
    
    def calculate_completeness(self, target_id: str) -> CompletenessScore:
        """
        Calculate reconnaissance completeness across all dimensions (enhanced)
        """
        if target_id not in self.dossiers:
            return CompletenessScore()
        
        dossier = self.dossiers[target_id]
        score = CompletenessScore()
        
        # Identity dimension (4 items with confidence weighting)
        identity_items = 0.0
        if dossier.os_info.name:
            # Weight by confidence
            identity_items += dossier.os_info.confidence
        if dossier.os_info.version:
            identity_items += 1.0
        if dossier.hostname:
            identity_items += 1.0
        if dossier.domain:
            identity_items += 1.0
        score.identity_score = min(identity_items / 4.0, 1.0)
        
        # Network dimension (enhanced with service detail scoring)
        network_items = 0.0
        if dossier.open_ports:
            # More ports = more complete, up to 10
            port_score = min(len(dossier.open_ports) / 10.0, 1.0)
            network_items += port_score
        
        if dossier.services:
            # Score services by detail level
            service_score = 0.0
            for svc in dossier.services.values():
                detail_level = 0.2  # Base for just knowing service exists
                if svc.service_name: detail_level += 0.2
                if svc.version: detail_level += 0.3
                if svc.product: detail_level += 0.2
                if svc.banner: detail_level += 0.1
                service_score += detail_level
            network_items += min(service_score / 5.0, 1.0)  # Normalize to 5 services
        
        if dossier.protocols:
            network_items += 0.3
        
        score.network_score = min(network_items / 2.3, 1.0)
        
        # Access Control dimension (enhanced)
        access_items = 0.0
        if dossier.users:
            # Score by number and confirmation
            user_score = 0.0
            for user_data in dossier.users.values():
                base = 0.2
                if len(user_data.get('discovered_by', [])) > 1:
                    base += 0.1  # Confirmed by multiple tools
                user_score += base
            access_items += min(user_score / 5.0, 1.0)
        
        if dossier.shares:
            share_score = min(len(dossier.shares) / 3.0, 1.0)
            access_items += share_score
        
        if dossier.permissions:
            access_items += 0.5
        if dossier.groups:
            access_items += 0.5
        
        score.access_control_score = min(access_items / 3.0, 1.0)
        
        # Security dimension (enhanced with severity weighting)
        security_items = 0.0
        
        if dossier.cves:
            # Weight by severity
            cve_score = 0.0
            for cve_data in dossier.cves.values():
                severity = cve_data.get('severity', 'unknown')
                if severity == 'critical':
                    cve_score += 0.4
                elif severity == 'high':
                    cve_score += 0.3
                elif severity == 'medium':
                    cve_score += 0.2
                else:
                    cve_score += 0.1
            security_items += min(cve_score, 1.0)
        
        if dossier.vulnerabilities:
            security_items += min(len(dossier.vulnerabilities) / 5.0, 1.0)
        
        if dossier.patch_level:
            security_items += 0.5
        
        if dossier.security_software:
            security_items += 0.3
        
        score.security_score = min(security_items / 2.8, 1.0)
        
        # Exploitability dimension (most valuable)
        exploit_items = 0.0
        
        if dossier.credentials:
            # Credentials are critical - weight heavily
            exploit_items += min(len(dossier.credentials) * 0.5, 1.0)
        
        if dossier.exploit_paths:
            exploit_items += min(len(dossier.exploit_paths) * 0.4, 1.0)
        
        if dossier.privilege_escalation:
            exploit_items += min(len(dossier.privilege_escalation) * 0.3, 1.0)
        
        score.exploitability_score = min(exploit_items / 3.0, 1.0)
        
        return score
    
    def query_state(self, target_id: str, query: str) -> Any:
        """
        Query interface for RL agent (enhanced)
        
        Examples:
        - "os" -> Returns OS information with confidence
        - "users" -> Returns list of users with metadata
        - "completeness" -> Returns detailed completeness score
        """
        if target_id not in self.dossiers:
            return None
        
        dossier = self.dossiers[target_id]
        query_lower = query.lower()
        
        if query_lower in ['os', 'operating system']:
            return dossier.os_info.to_dict()
        
        elif query_lower in ['users', 'accounts']:
            return dossier.users
        
        elif query_lower in ['shares']:
            return dossier.shares
        
        elif query_lower in ['ports']:
            return {
                "open": sorted(list(dossier.open_ports)),
                "closed": sorted(list(dossier.closed_ports)),
                "total_found": len(dossier.open_ports) + len(dossier.closed_ports)
            }
        
        elif query_lower in ['services']:
            return {port: svc.to_dict() for port, svc in dossier.services.items()}
        
        elif query_lower in ['vulnerabilities', 'vulns', 'cves']:
            return {
                "cves": dossier.cves,
                "vulnerabilities": dossier.vulnerabilities,
                "count": {
                    "cves": len(dossier.cves),
                    "vulns": len(dossier.vulnerabilities)
                }
            }
        
        elif query_lower in ['credentials', 'creds']:
            return {
                "credentials": dossier.credentials,
                "count": len(dossier.credentials),
                "has_valid": len(dossier.credentials) > 0
            }
        
        elif query_lower in ['completeness', 'progress']:
            return self.calculate_completeness(target_id).to_dict()
        
        elif query_lower == 'summary':
            return dossier.to_dict()
        
        elif query_lower == 'statistics':
            completeness = self.calculate_completeness(target_id)
            return {
                "target_id": target_id,
                "scan_count": dossier.scan_count,
                "completeness": completeness.percentage,
                "items_discovered": {
                    "os": bool(dossier.os_info.name),
                    "ports": len(dossier.open_ports),
                    "services": len(dossier.services),
                    "users": len(dossier.users),
                    "shares": len(dossier.shares),
                    "cves": len(dossier.cves),
                    "credentials": len(dossier.credentials)
                },
                "created_at": dossier.created_at,
                "updated_at": dossier.updated_at
            }
        
        else:
            return None
    
    def get_missing_categories(self, target_id: str) -> List[Dict[str, Any]]:
        """
        Identify what information categories are still missing (enhanced)
        """
        if target_id not in self.dossiers:
            return []
        
        dossier = self.dossiers[target_id]
        completeness = self.calculate_completeness(target_id)
        missing = []
        
        # Check identity
        if not dossier.os_info.name:
            missing.append({
                "category": "identity",
                "item": "OS Name",
                "priority": "HIGH",
                "suggested_tools": ["nmap", "enum4linux"]
            })
        if not dossier.os_info.version:
            missing.append({
                "category": "identity",
                "item": "OS Version",
                "priority": "MEDIUM",
                "suggested_tools": ["nmap -O", "enum4linux"]
            })
        if not dossier.hostname:
            missing.append({
                "category": "identity",
                "item": "Hostname",
                "priority": "LOW",
                "suggested_tools": ["nmap", "nbtscan"]
            })
        
        # Check network
        if not dossier.open_ports:
            missing.append({
                "category": "network",
                "item": "Open Ports",
                "priority": "CRITICAL",
                "suggested_tools": ["nmap", "masscan", "rustscan"]
            })
        elif len(dossier.open_ports) < 5:
            missing.append({
                "category": "network",
                "item": "Additional Ports (only {} found)".format(len(dossier.open_ports)),
                "priority": "MEDIUM",
                "suggested_tools": ["nmap -p-", "masscan"]
            })
        
        if not dossier.services or len(dossier.services) < len(dossier.open_ports):
            missing.append({
                "category": "network",
                "item": "Service Version Detection",
                "priority": "HIGH",
                "suggested_tools": ["nmap -sV"]
            })
        
        # Check access control
        if not dossier.users:
            missing.append({
                "category": "access_control",
                "item": "User Accounts",
                "priority": "MEDIUM",
                "suggested_tools": ["enum4linux", "crackmapexec", "rpcclient"]
            })
        if not dossier.shares:
            missing.append({
                "category": "access_control",
                "item": "Network Shares",
                "priority": "MEDIUM",
                "suggested_tools": ["enum4linux", "smbclient", "crackmapexec"]
            })
        if not dossier.permissions:
            missing.append({
                "category": "access_control",
                "item": "Share Permissions",
                "priority": "LOW",
                "suggested_tools": ["crackmapexec", "smbmap"]
            })
        
        # Check security
        if not dossier.cves and not dossier.vulnerabilities:
            missing.append({
                "category": "security",
                "item": "Vulnerability Scan",
                "priority": "CRITICAL",
                "suggested_tools": ["nuclei", "nmap --script vuln", "openvas"]
            })
        elif len(dossier.cves) < 3 and completeness.security_score < 0.5:
            missing.append({
                "category": "security",
                "item": "Deep Vulnerability Analysis",
                "priority": "HIGH",
                "suggested_tools": ["nuclei", "nikto", "wpscan"]
            })
        
        if not dossier.patch_level:
            missing.append({
                "category": "security",
                "item": "Patch Level Assessment",
                "priority": "MEDIUM",
                "suggested_tools": ["enum4linux", "crackmapexec"]
            })
        
        # Check exploitability
        if not dossier.credentials:
            missing.append({
                "category": "exploitability",
                "item": "Valid Credentials",
                "priority": "CRITICAL",
                "suggested_tools": ["hydra", "medusa", "crackmapexec", "john"]
            })
        if not dossier.exploit_paths and len(dossier.cves) > 0:
            missing.append({
                "category": "exploitability",
                "item": "Exploit Verification",
                "priority": "HIGH",
                "suggested_tools": ["metasploit", "searchsploit"]
            })
        
        return missing
    
    def export_state(self, target_id: str) -> str:
        """Export complete state as JSON"""
        if target_id not in self.dossiers:
            return json.dumps({"error": "Target not found"})
        
        dossier = self.dossiers[target_id]
        completeness = self.calculate_completeness(target_id)
        missing = self.get_missing_categories(target_id)
        
        export_data = {
            "target_id": target_id,
            "dossier": dossier.to_dict(),
            "completeness": completeness.to_dict(),
            "missing_categories": missing,
            "information_count": len(self.information_db.get(target_id, {}))
        }
        
        return json.dumps(export_data, indent=2)


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ENHANCED Information State Tracking System - Demo")
    print("With Normalization, Confidence Tracking & Relationship Mapping")
    print("="*70 + "\n")
    
    # Initialize tracker
    tracker = InformationStateTracker()
    target = "192.168.1.100"
    
    # Simulate nmap scan
    print("Step 1: Running nmap scan...")
    nmap_output = """
    Nmap scan report for windows-xp.local (192.168.1.100)
    Host is up (0.0010s latency).
    Not shown: 997 closed ports
    PORT    STATE SERVICE      VERSION
    139/tcp open  netbios-ssn  Microsoft Windows netbios-ssn
    445/tcp open  microsoft-ds Microsoft Windows XP microsoft-ds
    3389/tcp open  ms-wbt-server
    OS details: Microsoft Windows XP SP2 or SP3
    """
    
    delta1 = tracker.ingest_tool_output(target, "nmap", nmap_output, "")
    print(f"\n{delta1.summary()}")
    
    completeness = tracker.calculate_completeness(target)
    print(f"\nCompleteness: {completeness.percentage:.1f}%")
    
    # Simulate enum4linux scan
    print("\n" + "-"*60)
    print("Step 2: Running enum4linux...")
    enum4linux_output = """
    Domain Name: WORKGROUP
    OS: Windows XP
    
    Users:
    user:[Administrator] rid:[0x1f4]
    user:[Guest] rid:[0x1f5]
    user:[john.doe] rid:[0x3e8]
    user:[alice] rid:[0x3e9]
    
    Sharename       Type      Comment
    ---------       ----      -------
    C$              Disk      Default share
    ADMIN$          Disk      Remote Admin
    Backup          Disk      Backup files
    """
    
    delta2 = tracker.ingest_tool_output(target, "enum4linux", enum4linux_output, "")
    print(f"\n{delta2.summary()}")
    
    completeness = tracker.calculate_completeness(target)
    print(f"\nCompleteness: {completeness.percentage:.1f}%")
    
    # Simulate redundant enum4linux run
    print("\n" + "-"*60)
    print("Step 3: Running enum4linux again (should be redundant)...")
    delta3 = tracker.ingest_tool_output(target, "enum4linux", enum4linux_output, "")
    print(f"\n{delta3.summary()}")
    
    # Simulate nuclei scan
    print("\n" + "-"*60)
    print("Step 4: Running nuclei vulnerability scan...")
    nuclei_output = """
    [critical] [CVE-2017-0144] SMB Remote Code Execution (EternalBlue)
    [high] [CVE-2008-4250] MS08-067 NetAPI Vulnerability
    """
    
    delta4 = tracker.ingest_tool_output(target, "nuclei", nuclei_output, "")
    print(f"\n{delta4.summary()}")
    
    completeness = tracker.calculate_completeness(target)
    print(f"\nCompleteness: {completeness.percentage:.1f}%")
    
    # Query examples
    print("\n" + "="*70)
    print("Query Examples - Enhanced Features")
    print("="*70)
    
    print("\nQ: What OS is this target? (with confidence)")
    os_info = tracker.query_state(target, 'os')
    print(f"A: {json.dumps(os_info, indent=2)}")
    
    print("\nQ: What users were found? (with metadata)")
    users = tracker.query_state(target, 'users')
    print(f"A: Found {len(users)} users:")
    for username, data in list(users.items())[:3]:
        print(f"  - {username}: RID={data.get('rid')}, Confirmed by {len(data.get('discovered_by', []))} tools")
    
    print("\nQ: What network services are running?")
    services = tracker.query_state(target, 'services')
    print(f"A: Found {len(services)} services:")
    for port, svc_data in list(services.items())[:3]:
        print(f"  - Port {port}: {svc_data.get('service_name')} " +
              f"(Version: {svc_data.get('version', 'unknown')}, State: {svc_data.get('state')})")
    
    print("\nQ: What vulnerabilities exist? (with severity)")
    vulns = tracker.query_state(target, 'vulnerabilities')
    print(f"A: Found {vulns['count']['cves']} CVEs:")
    for cve_id, cve_data in list(vulns['cves'].items())[:2]:
        print(f"  - {cve_id}: Severity={cve_data.get('severity')}, " +
              f"Exploit Available={cve_data.get('has_exploit')}")
    
    print("\nQ: What information is still missing? (with suggestions)")
    missing = tracker.get_missing_categories(target)
    print(f"A: Missing {len(missing)} items:")
    for item in missing[:5]:
        print(f"  - [{item['priority']}] {item['item']}")
        print(f"    Suggested tools: {', '.join(item.get('suggested_tools', []))}")
    
    print("\nQ: Get overall statistics")
    stats = tracker.query_state(target, 'statistics')
    print(f"A: {json.dumps(stats, indent=2)}")
    
    print("\n" + "="*70)
    print("Detailed Completeness Analysis")
    print("="*70)
    completeness = tracker.query_state(target, 'completeness')
    print(json.dumps(completeness, indent=2))
    
    print("\n" + "="*70)
    print("Key Insights:")
    print("="*70)
    print(f"1. Overall reconnaissance is {completeness['percentage']} complete")
    print(f"2. Weakest area: {completeness['analysis']['weakest_dimension']} " +
          f"({completeness['analysis']['weakest_score']*100:.1f}%)")
    print(f"3. Missing {len(completeness['analysis']['missing_critical'])} critical items")
    print("4. Tool execution generated intelligence with:")
    print(f"   - Information deduplication (prevented {delta3.summary().count('Redundant')} redundant discoveries)")
    print(f"   - Confidence-based scoring (OS detected with {os_info.get('confidence', 0)*100:.0f}% confidence)")
    print(f"   - Relationship mapping (CVEs linked to required conditions)")
    print("\n" + "="*70 + "\n")
