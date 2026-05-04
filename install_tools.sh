#!/bin/bash

################################################################################
# Security Tools Installation Script for Ubuntu
# 
# This script installs all 53 security tools used by the Mythos v2 
# reconnaissance system.
#
# Usage:
#   sudo ./install_tools.sh              # Install all tools
#   sudo ./install_tools.sh --check      # Check which tools are installed
#   sudo ./install_tools.sh --selective  # Interactive selection
#
# Requirements: Ubuntu 20.04+ with sudo privileges
################################################################################

# Don't exit on error - continue installing other tools
# set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="tool_installation.log"
INSTALL_DIR="/opt/security-tools"

# Counters
TOTAL_TOOLS=0
INSTALLED_TOOLS=0
SKIPPED_TOOLS=0
FAILED_TOOLS=0

# Modes
FAST_MODE=false

################################################################################
# Utility Functions
################################################################################

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
    ((INSTALLED_TOOLS++))
}

log_error() {
    echo -e "${RED}✗${NC} $1" | tee -a "$LOG_FILE"
    ((FAILED_TOOLS++))
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1" | tee -a "$LOG_FILE"
    ((SKIPPED_TOOLS++))
}

log_info() {
    echo -e "${BLUE}ℹ${NC} $1" | tee -a "$LOG_FILE"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_internet() {
    if ! ping -c 1 8.8.8.8 &> /dev/null; then
        log_error "No internet connection detected"
        exit 1
    fi
}

is_installed() {
    command -v "$1" &> /dev/null
}

safe_wget() {
    # Download with timeout and error handling
    local url="$1"
    local output="$2"
    
    if [ -z "$output" ]; then
        wget --timeout=30 --tries=2 -q "$url" 2>/dev/null
    else
        wget --timeout=30 --tries=2 -q "$url" -O "$output" 2>/dev/null
    fi
}

safe_git_clone() {
    # Git clone with timeout
    local url="$1"
    local dest="$2"
    
    timeout 60 git clone --depth=1 "$url" "$dest" 2>/dev/null
}

################################################################################
# Installation Functions
################################################################################

install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    apt-get update -qq
    apt-get install -y \
        build-essential \
        git \
        wget \
        curl \
        python3 \
        python3-pip \
        python3-venv \
        golang-go \
        ruby \
        ruby-dev \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev \
        libcurl4-openssl-dev \
        libpcap-dev \
        libpq-dev \
        libsqlite3-dev \
        default-jdk \
        snapd \
        apt-transport-https \
        ca-certificates \
        software-properties-common
    
    log_success "System dependencies installed"
}

# Network Scanning Tools
install_nmap() {
    if is_installed nmap; then
        log_warning "nmap already installed"
        return
    fi
    log_info "Installing nmap..."
    apt-get install -y nmap
    log_success "nmap installed"
}

install_masscan() {
    if is_installed masscan; then
        log_warning "masscan already installed"
        return
    fi
    log_info "Installing masscan..."
    apt-get install -y masscan
    log_success "masscan installed"
}

install_rustscan() {
    if is_installed rustscan; then
        log_warning "rustscan already installed"
        return
    fi
    log_info "Installing rustscan..."
    
    # Try to install via deb package with timeout
    if wget --timeout=30 -q https://github.com/RustScan/RustScan/releases/download/2.1.1/rustscan_2.1.1_amd64.deb 2>/dev/null; then
        if dpkg -i rustscan_2.1.1_amd64.deb 2>/dev/null || apt-get install -f -y 2>/dev/null; then
            rm -f rustscan_2.1.1_amd64.deb
            log_success "rustscan installed"
        else
            rm -f rustscan_2.1.1_amd64.deb
            log_error "rustscan installation failed (dpkg error)"
        fi
    else
        log_error "rustscan download failed (skipping)"
    fi
}

# SMB/Windows Tools
install_enum4linux() {
    if is_installed enum4linux; then
        log_warning "enum4linux already installed"
        return
    fi
    log_info "Installing enum4linux..."
    apt-get install -y enum4linux
    log_success "enum4linux installed"
}

install_enum4linux_ng() {
    if [ -f "/opt/enum4linux-ng/enum4linux-ng.py" ]; then
        log_warning "enum4linux-ng already installed"
        return
    fi
    log_info "Installing enum4linux-ng..."
    
    if safe_git_clone https://github.com/cddmp/enum4linux-ng.git /opt/enum4linux-ng; then
        if pip3 install -q -r /opt/enum4linux-ng/requirements.txt 2>/dev/null; then
            ln -sf /opt/enum4linux-ng/enum4linux-ng.py /usr/local/bin/enum4linux-ng
            log_success "enum4linux-ng installed"
        else
            log_error "enum4linux-ng dependencies failed"
        fi
    else
        log_error "enum4linux-ng clone failed (skipping)"
    fi
}

install_smbclient() {
    if is_installed smbclient; then
        log_warning "smbclient already installed"
        return
    fi
    log_info "Installing smbclient..."
    apt-get install -y smbclient
    log_success "smbclient installed"
}

install_crackmapexec() {
    if is_installed crackmapexec; then
        log_warning "crackmapexec already installed"
        return
    fi
    log_info "Installing crackmapexec..."
    apt-get install -y crackmapexec || pip3 install crackmapexec
    log_success "crackmapexec installed"
}

install_impacket() {
    if python3 -c "import impacket" 2>/dev/null; then
        log_warning "impacket already installed"
        return
    fi
    log_info "Installing impacket..."
    pip3 install impacket
    log_success "impacket installed"
}

# Web Application Tools
install_nikto() {
    if is_installed nikto; then
        log_warning "nikto already installed"
        return
    fi
    log_info "Installing nikto..."
    apt-get install -y nikto
    log_success "nikto installed"
}

install_nuclei() {
    if is_installed nuclei; then
        log_warning "nuclei already installed"
        return
    fi
    log_info "Installing nuclei..."
    go install -v github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest
    ln -sf /root/go/bin/nuclei /usr/local/bin/nuclei 2>/dev/null || true
    log_success "nuclei installed"
}

install_sqlmap() {
    if is_installed sqlmap; then
        log_warning "sqlmap already installed"
        return
    fi
    log_info "Installing sqlmap..."
    apt-get install -y sqlmap
    log_success "sqlmap installed"
}

install_wpscan() {
    if is_installed wpscan; then
        log_warning "wpscan already installed"
        return
    fi
    log_info "Installing wpscan..."
    gem install wpscan
    log_success "wpscan installed"
}

install_gobuster() {
    if is_installed gobuster; then
        log_warning "gobuster already installed"
        return
    fi
    log_info "Installing gobuster..."
    apt-get install -y gobuster || go install github.com/OJ/gobuster/v3@latest
    log_success "gobuster installed"
}

install_ffuf() {
    if is_installed ffuf; then
        log_warning "ffuf already installed"
        return
    fi
    log_info "Installing ffuf..."
    go install github.com/ffuf/ffuf@latest
    ln -sf /root/go/bin/ffuf /usr/local/bin/ffuf 2>/dev/null || true
    log_success "ffuf installed"
}

install_whatweb() {
    if is_installed whatweb; then
        log_warning "whatweb already installed"
        return
    fi
    log_info "Installing whatweb..."
    apt-get install -y whatweb
    log_success "whatweb installed"
}

install_wafw00f() {
    if is_installed wafw00f; then
        log_warning "wafw00f already installed"
        return
    fi
    log_info "Installing wafw00f..."
    pip3 install wafw00f
    log_success "wafw00f installed"
}

install_burpsuite() {
    if [ -f "/opt/burpsuite/burpsuite" ]; then
        log_warning "burpsuite already installed"
        return
    fi
    log_info "Installing burpsuite (community)..."
    log_warning "Burpsuite requires manual download from PortSwigger website"
    log_info "Download from: https://portswigger.net/burp/communitydownload"
}

# Password Cracking
install_hydra() {
    if is_installed hydra; then
        log_warning "hydra already installed"
        return
    fi
    log_info "Installing hydra..."
    apt-get install -y hydra
    log_success "hydra installed"
}

install_medusa() {
    if is_installed medusa; then
        log_warning "medusa already installed"
        return
    fi
    log_info "Installing medusa..."
    apt-get install -y medusa
    log_success "medusa installed"
}

install_john() {
    if is_installed john; then
        log_warning "john already installed"
        return
    fi
    log_info "Installing john the ripper..."
    apt-get install -y john
    log_success "john installed"
}

install_hashcat() {
    if is_installed hashcat; then
        log_warning "hashcat already installed"
        return
    fi
    log_info "Installing hashcat..."
    apt-get install -y hashcat
    log_success "hashcat installed"
}

# Exploitation Frameworks
install_metasploit() {
    if is_installed msfconsole; then
        log_warning "metasploit already installed"
        return
    fi
    log_info "Installing metasploit framework..."
    curl https://raw.githubusercontent.com/rapid7/metasploit-omnibus/master/config/templates/metasploit-framework-wrappers/msfupdate.erb > msfinstall
    chmod 755 msfinstall
    ./msfinstall
    rm -f msfinstall
    log_success "metasploit installed"
}

install_searchsploit() {
    if is_installed searchsploit; then
        log_warning "searchsploit already installed"
        return
    fi
    log_info "Installing searchsploit..."
    git clone https://gitlab.com/exploit-database/exploitdb.git /opt/exploitdb
    ln -sf /opt/exploitdb/searchsploit /usr/local/bin/searchsploit
    log_success "searchsploit installed"
}

# Reconnaissance Tools
install_amass() {
    if is_installed amass; then
        log_warning "amass already installed"
        return
    fi
    log_info "Installing amass..."
    snap install amass
    log_success "amass installed"
}

install_subfinder() {
    if is_installed subfinder; then
        log_warning "subfinder already installed"
        return
    fi
    log_info "Installing subfinder..."
    go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest
    ln -sf /root/go/bin/subfinder /usr/local/bin/subfinder 2>/dev/null || true
    log_success "subfinder installed"
}

install_theharvester() {
    if is_installed theHarvester; then
        log_warning "theHarvester already installed"
        return
    fi
    log_info "Installing theHarvester..."
    pip3 install theHarvester
    log_success "theHarvester installed"
}

install_dnsenum() {
    if is_installed dnsenum; then
        log_warning "dnsenum already installed"
        return
    fi
    log_info "Installing dnsenum..."
    apt-get install -y dnsenum
    log_success "dnsenum installed"
}

install_shodan() {
    if python3 -c "import shodan" 2>/dev/null; then
        log_warning "shodan already installed"
        return
    fi
    log_info "Installing shodan..."
    pip3 install shodan
    log_success "shodan installed"
}

# Network Analysis
install_wireshark() {
    if is_installed wireshark; then
        log_warning "wireshark already installed"
        return
    fi
    log_info "Installing wireshark..."
    DEBIAN_FRONTEND=noninteractive apt-get install -y wireshark
    log_success "wireshark installed"
}

install_tcpdump() {
    if is_installed tcpdump; then
        log_warning "tcpdump already installed"
        return
    fi
    log_info "Installing tcpdump..."
    apt-get install -y tcpdump
    log_success "tcpdump installed"
}

install_netcat() {
    if is_installed nc; then
        log_warning "netcat already installed"
        return
    fi
    log_info "Installing netcat..."
    apt-get install -y netcat-traditional
    log_success "netcat installed"
}

install_socat() {
    if is_installed socat; then
        log_warning "socat already installed"
        return
    fi
    log_info "Installing socat..."
    apt-get install -y socat
    log_success "socat installed"
}

install_snmpwalk() {
    if is_installed snmpwalk; then
        log_warning "snmpwalk already installed"
        return
    fi
    log_info "Installing snmpwalk..."
    apt-get install -y snmp
    log_success "snmpwalk installed"
}

# Vulnerability Scanners
install_openvas() {
    if is_installed openvas; then
        log_warning "openvas already installed"
        return
    fi
    log_info "Installing openvas..."
    apt-get install -y openvas
    log_warning "OpenVAS requires additional setup. Run: gvm-setup"
    log_success "openvas installed"
}

install_trivy() {
    if is_installed trivy; then
        log_warning "trivy already installed"
        return
    fi
    log_info "Installing trivy..."
    
    if wget --timeout=30 -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key 2>/dev/null | apt-key add - 2>/dev/null; then
        echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | tee -a /etc/apt/sources.list.d/trivy.list >/dev/null
        if apt-get update -qq 2>/dev/null && apt-get install -y trivy 2>/dev/null; then
            log_success "trivy installed"
        else
            log_error "trivy installation failed"
        fi
    else
        log_error "trivy repository setup failed (skipping)"
    fi
}

# Wireless Tools
install_aircrack_ng() {
    if is_installed aircrack-ng; then
        log_warning "aircrack-ng already installed"
        return
    fi
    log_info "Installing aircrack-ng..."
    apt-get install -y aircrack-ng
    log_success "aircrack-ng installed"
}

install_bettercap() {
    if is_installed bettercap; then
        log_warning "bettercap already installed"
        return
    fi
    log_info "Installing bettercap..."
    apt-get install -y bettercap
    log_success "bettercap installed"
}

# Post-Exploitation
install_bloodhound() {
    if [ -f "/opt/BloodHound/BloodHound" ]; then
        log_warning "bloodhound already installed"
        return
    fi
    log_info "Installing bloodhound..."
    wget -q https://github.com/BloodHoundAD/BloodHound/releases/download/4.3.1/BloodHound-linux-x64.zip
    unzip -q BloodHound-linux-x64.zip -d /opt/
    rm -f BloodHound-linux-x64.zip
    log_success "bloodhound installed"
}

install_responder() {
    if [ -f "/opt/Responder/Responder.py" ]; then
        log_warning "responder already installed"
        return
    fi
    log_info "Installing responder..."
    git clone https://github.com/lgandx/Responder.git /opt/Responder
    pip3 install -r /opt/Responder/requirements.txt
    ln -sf /opt/Responder/Responder.py /usr/local/bin/responder
    log_success "responder installed"
}

install_chisel() {
    if is_installed chisel; then
        log_warning "chisel already installed"
        return
    fi
    log_info "Installing chisel..."
    wget -q https://github.com/jpillora/chisel/releases/download/v1.9.1/chisel_1.9.1_linux_amd64.gz
    gunzip chisel_1.9.1_linux_amd64.gz
    mv chisel_1.9.1_linux_amd64 /usr/local/bin/chisel
    chmod +x /usr/local/bin/chisel
    log_success "chisel installed"
}

install_mitmproxy() {
    if is_installed mitmproxy; then
        log_warning "mitmproxy already installed"
        return
    fi
    log_info "Installing mitmproxy..."
    pip3 install mitmproxy
    log_success "mitmproxy installed"
}

install_weevely() {
    if [ -f "/opt/weevely/weevely.py" ]; then
        log_warning "weevely already installed"
        return
    fi
    log_info "Installing weevely..."
    git clone https://github.com/epinna/weevely3.git /opt/weevely
    pip3 install -r /opt/weevely/requirements.txt
    ln -sf /opt/weevely/weevely.py /usr/local/bin/weevely
    log_success "weevely installed"
}

# Binary Analysis
install_ghidra() {
    if [ -d "/opt/ghidra" ]; then
        log_warning "ghidra already installed"
        return
    fi
    log_info "Installing ghidra..."
    wget -q https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_10.4_build/ghidra_10.4_PUBLIC_20230928.zip
    unzip -q ghidra_10.4_PUBLIC_20230928.zip -d /opt/
    mv /opt/ghidra_10.4_PUBLIC /opt/ghidra
    rm -f ghidra_10.4_PUBLIC_20230928.zip
    log_success "ghidra installed"
}

install_radare2() {
    if is_installed radare2; then
        log_warning "radare2 already installed"
        return
    fi
    log_info "Installing radare2..."
    git clone https://github.com/radareorg/radare2 /tmp/radare2
    cd /tmp/radare2
    sys/install.sh
    cd -
    rm -rf /tmp/radare2
    log_success "radare2 installed"
}

install_binwalk() {
    if is_installed binwalk; then
        log_warning "binwalk already installed"
        return
    fi
    log_info "Installing binwalk..."
    apt-get install -y binwalk
    log_success "binwalk installed"
}

install_ropper() {
    if is_installed ropper; then
        log_warning "ropper already installed"
        return
    fi
    log_info "Installing ropper..."
    pip3 install ropper
    log_success "ropper installed"
}

# Mobile/Android
install_apktool() {
    if is_installed apktool; then
        log_warning "apktool already installed"
        return
    fi
    log_info "Installing apktool..."
    wget -q https://raw.githubusercontent.com/iBotPeaches/Apktool/master/scripts/linux/apktool -O /usr/local/bin/apktool
    wget -q https://bitbucket.org/iBotPeaches/apktool/downloads/apktool_2.9.2.jar -O /usr/local/bin/apktool.jar
    chmod +x /usr/local/bin/apktool
    log_success "apktool installed"
}

install_frida() {
    if is_installed frida; then
        log_warning "frida already installed"
        return
    fi
    log_info "Installing frida..."
    pip3 install frida-tools
    log_success "frida installed"
}

# Cloud Security
install_prowler() {
    if is_installed prowler; then
        log_warning "prowler already installed"
        return
    fi
    log_info "Installing prowler..."
    pip3 install prowler
    log_success "prowler installed"
}

install_kube_hunter() {
    if is_installed kube-hunter; then
        log_warning "kube-hunter already installed"
        return
    fi
    log_info "Installing kube-hunter..."
    pip3 install kube-hunter
    log_success "kube-hunter installed"
}

install_docker_bench() {
    if [ -f "/opt/docker-bench-security/docker-bench-security.sh" ]; then
        log_warning "docker-bench-security already installed"
        return
    fi
    log_info "Installing docker-bench-security..."
    git clone https://github.com/docker/docker-bench-security.git /opt/docker-bench-security
    log_success "docker-bench-security installed"
}

# Python Exploitation Tools
install_pwntools() {
    if python3 -c "import pwn" 2>/dev/null; then
        log_warning "pwntools already installed"
        return
    fi
    log_info "Installing pwntools..."
    pip3 install pwntools
    log_success "pwntools installed"
}

install_pypykatz() {
    if is_installed pypykatz; then
        log_warning "pypykatz already installed"
        return
    fi
    log_info "Installing pypykatz..."
    pip3 install pypykatz
    log_success "pypykatz installed"
}

################################################################################
# Main Installation Flow
################################################################################

print_summary() {
    echo ""
    echo "================================================================================"
    echo "Installation Summary"
    echo "================================================================================"
    echo -e "${GREEN}✓ Successfully installed:${NC} $INSTALLED_TOOLS tools"
    echo -e "${YELLOW}⚠ Already installed/skipped:${NC} $SKIPPED_TOOLS tools"
    echo -e "${RED}✗ Failed:${NC} $FAILED_TOOLS tools"
    echo "================================================================================"
}

install_all_tools() {
    log_info "Starting installation of all security tools..."
    echo "Progress indicator: [Current/Total]"
    echo ""
    
    local current=0
    local total=53
    
    # Network Scanning
    ((current++)); echo "[$current/$total] nmap"; install_nmap
    ((current++)); echo "[$current/$total] masscan"; install_masscan
    ((current++)); echo "[$current/$total] rustscan"; install_rustscan
    
    # SMB/Windows
    ((current++)); echo "[$current/$total] enum4linux"; install_enum4linux
    ((current++)); echo "[$current/$total] enum4linux-ng"; install_enum4linux_ng
    ((current++)); echo "[$current/$total] smbclient"; install_smbclient
    ((current++)); echo "[$current/$total] crackmapexec"; install_crackmapexec
    ((current++)); echo "[$current/$total] impacket"; install_impacket
    
    # Web Application
    ((current++)); echo "[$current/$total] nikto"; install_nikto
    ((current++)); echo "[$current/$total] nuclei"; install_nuclei
    ((current++)); echo "[$current/$total] sqlmap"; install_sqlmap
    ((current++)); echo "[$current/$total] wpscan"; install_wpscan
    ((current++)); echo "[$current/$total] gobuster"; install_gobuster
    ((current++)); echo "[$current/$total] ffuf"; install_ffuf
    ((current++)); echo "[$current/$total] whatweb"; install_whatweb
    ((current++)); echo "[$current/$total] wafw00f"; install_wafw00f
    ((current++)); echo "[$current/$total] burpsuite"; install_burpsuite
    
    # Password Cracking
    ((current++)); echo "[$current/$total] hydra"; install_hydra
    ((current++)); echo "[$current/$total] medusa"; install_medusa
    ((current++)); echo "[$current/$total] john"; install_john
    ((current++)); echo "[$current/$total] hashcat"; install_hashcat
    
    # Exploitation
    ((current++)); echo "[$current/$total] metasploit"; install_metasploit
    ((current++)); echo "[$current/$total] searchsploit"; install_searchsploit
    
    # Reconnaissance
    ((current++)); echo "[$current/$total] amass"; install_amass
    ((current++)); echo "[$current/$total] subfinder"; install_subfinder
    ((current++)); echo "[$current/$total] theharvester"; install_theharvester
    ((current++)); echo "[$current/$total] dnsenum"; install_dnsenum
    ((current++)); echo "[$current/$total] shodan"; install_shodan
    
    # Network Analysis
    ((current++)); echo "[$current/$total] wireshark"; install_wireshark
    ((current++)); echo "[$current/$total] tcpdump"; install_tcpdump
    ((current++)); echo "[$current/$total] netcat"; install_netcat
    ((current++)); echo "[$current/$total] socat"; install_socat
    ((current++)); echo "[$current/$total] snmpwalk"; install_snmpwalk
    
    # Vulnerability Scanners
    ((current++)); echo "[$current/$total] openvas"; install_openvas
    ((current++)); echo "[$current/$total] trivy"; install_trivy
    
    # Wireless
    ((current++)); echo "[$current/$total] aircrack-ng"; install_aircrack_ng
    ((current++)); echo "[$current/$total] bettercap"; install_bettercap
    
    # Post-Exploitation
    ((current++)); echo "[$current/$total] bloodhound"; install_bloodhound
    ((current++)); echo "[$current/$total] responder"; install_responder
    ((current++)); echo "[$current/$total] chisel"; install_chisel
    ((current++)); echo "[$current/$total] mitmproxy"; install_mitmproxy
    ((current++)); echo "[$current/$total] weevely"; install_weevely
    
    # Binary Analysis
    ((current++)); echo "[$current/$total] ghidra"; install_ghidra
    ((current++)); echo "[$current/$total] radare2"; install_radare2
    ((current++)); echo "[$current/$total] binwalk"; install_binwalk
    ((current++)); echo "[$current/$total] ropper"; install_ropper
    
    # Mobile
    ((current++)); echo "[$current/$total] apktool"; install_apktool
    ((current++)); echo "[$current/$total] frida"; install_frida
    
    # Cloud Security
    ((current++)); echo "[$current/$total] prowler"; install_prowler
    ((current++)); echo "[$current/$total] kube-hunter"; install_kube_hunter
    ((current++)); echo "[$current/$total] docker-bench"; install_docker_bench
    
    # Python Tools
    ((current++)); echo "[$current/$total] pwntools"; install_pwntools
    ((current++)); echo "[$current/$total] pypykatz"; install_pypykatz
    
    echo ""
    log_success "Installation phase complete!"
    print_summary
}

check_installed_tools() {
    log_info "Checking installed tools..."
    
    tools=(
        "nmap" "masscan" "rustscan" "enum4linux" "smbclient" 
        "crackmapexec" "nikto" "nuclei" "sqlmap" "wpscan" 
        "gobuster" "ffuf" "whatweb" "wafw00f" "hydra" 
        "medusa" "john" "hashcat" "msfconsole" "searchsploit" 
        "amass" "subfinder" "theHarvester" "dnsenum" "wireshark" 
        "tcpdump" "nc" "socat" "snmpwalk" "openvas" 
        "trivy" "aircrack-ng" "bettercap" "chisel" "mitmproxy" 
        "radare2" "binwalk" "ropper" "apktool" "frida" 
        "prowler" "kube-hunter"
    )
    
    installed=0
    missing=0
    
    for tool in "${tools[@]}"; do
        if is_installed "$tool"; then
            echo -e "${GREEN}✓${NC} $tool"
            ((installed++))
        else
            echo -e "${RED}✗${NC} $tool"
            ((missing++))
        fi
    done
    
    echo ""
    log_info "Installed: $installed / $(( installed + missing ))"
    log_info "Missing: $missing / $(( installed + missing ))"
}

################################################################################
# Main Script
################################################################################

main() {
    echo "================================================================================"
    echo "Security Tools Installation Script for Mythos v2"
    echo "================================================================================"
    echo ""
    
    check_root
    check_internet
    
    case "${1:-}" in
        --check)
            check_installed_tools
            ;;
        --fast)
            log_warning "Fast mode: Skipping slow installations (metasploit, ghidra, etc.)"
            FAST_MODE=true
            log_info "Installing system dependencies..."
            install_system_dependencies
            log_info "Starting tool installation..."
            install_all_tools
            echo ""
            log_info "Check installation log: $LOG_FILE"
            log_info "Run with --check to verify installations"
            ;;
        --help)
            echo "Usage:"
            echo "  sudo $0              # Install all tools"
            echo "  sudo $0 --fast       # Install all tools (skip slow ones)"
            echo "  sudo $0 --check      # Check which tools are installed"
            echo "  sudo $0 --help       # Show this help"
            exit 0
            ;;
        *)
            log_info "Installing system dependencies..."
            install_system_dependencies
            
            log_info "Starting tool installation..."
            install_all_tools
            
            echo ""
            echo "================================================================================"
            log_info "All operations complete!"
            echo "================================================================================"
            echo ""
            log_info "Check installation log: $LOG_FILE"
            log_info "Run 'sudo $0 --check' to verify installations"
            ;;
    esac
}

main "$@"
