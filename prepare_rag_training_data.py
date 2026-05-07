#!/usr/bin/env python3
"""
Prepare RAG-Focused Training Data

Instead of teaching the model to MEMORIZE CVEs, we teach it to REASON about
retrieved CVE information.

Key differences:
- OLD: Input -> Output (memorization)
- NEW: Context + Input -> Output (reasoning)

Training data includes the retrieved context so model learns to analyze it.
"""

import json
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import argparse

random.seed(42)


def generate_classify_with_context(row: pd.Series, cve_context: str = None) -> Dict[str, Any]:
    """
    [CLASSIFY] task - Binary code vulnerability detection
    
    With RAG: Provide relevant CVE context to help classification
    """
    code = row.get('code') or row.get('source') or row.get('functions', '')
    label = str(int(row['label']))  # "0" or "1"
    
    # Truncate long code
    if len(code) > 1500:
        code = code[:1500] + "\n// ... (truncated)"
    
    # System prompt emphasizing reasoning
    system = """You are a code security classifier. For [CLASSIFY] tasks, analyze the code for vulnerabilities and respond with ONLY '0' (safe) or '1' (vulnerable).

Use your security knowledge to identify common vulnerability patterns like:
- Buffer overflows (strcpy, sprintf without bounds)
- SQL injection (string concatenation in queries)
- Command injection (system calls with user input)
- Path traversal (file operations with unsanitized paths)
- XSS, CSRF, deserialization attacks

If relevant CVE context is provided, use it to identify similar patterns."""
    
    # User prompt (no context needed for binary classification, but include for consistency)
    user = f"[CLASSIFY] Is this code vulnerable?\n\n{code}"
    
    # Assistant response
    assistant = label
    
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ],
        "task_type": "CLASSIFY"
    }


def generate_cve_analysis_with_context(nvd_entry: Dict[str, Any], related_cves: List[str] = None) -> Dict[str, Any]:
    """
    [CVE_LOOKUP] task - Analyze provided CVE information (not memorize it)
    
    Key: CVE details are in the CONTEXT, model learns to analyze them
    """
    cve_id = nvd_entry.get('cve_id', '')
    description = nvd_entry.get('description', '')
    cvss_score = nvd_entry.get('cvss_v3_score', nvd_entry.get('cvss_v2_score', 'N/A'))
    cvss_vector = nvd_entry.get('cvss_v3_vector', '')
    affected = nvd_entry.get('affected_products', [])
    published = nvd_entry.get('published_date', '')
    
    if not cve_id or not description:
        return None
    
    # Build context section (simulates retrieved information)
    context = f"""[CONTEXT - Retrieved CVE Information]

CVE ID: {cve_id}
Description: {description}
CVSS Score: {cvss_score}/10.0
CVSS Vector: {cvss_vector}
Affected Products: {', '.join(affected[:5]) if affected else 'See description'}
Published: {published}
"""
    
    # Random query variants
    queries = [
        f"What is {cve_id}?",
        f"Tell me about {cve_id}",
        f"Explain the vulnerability {cve_id}",
        f"What is the severity of {cve_id}?",
    ]
    
    # Also add product-based queries
    if affected:
        product = affected[0].split()[0]  # First word of first product
        queries.append(f"What vulnerabilities affect {product}?")
    
    query = random.choice(queries)
    
    # Generate analysis (model learns to extract and reason about context)
    severity_label = "CRITICAL" if cvss_score >= 9.0 else \
                    "HIGH" if cvss_score >= 7.0 else \
                    "MEDIUM" if cvss_score >= 4.0 else "LOW"
    
    # Parse attack vector from CVSS vector
    attack_vector = "NETWORK" if "AV:N" in cvss_vector else \
                   "ADJACENT" if "AV:A" in cvss_vector else \
                   "LOCAL" if "AV:L" in cvss_vector else "UNKNOWN"
    
    # Extract vulnerability type keywords
    vuln_type = ""
    desc_lower = description.lower()
    if any(term in desc_lower for term in ["buffer", "overflow", "overrun"]):
        vuln_type = "Buffer Overflow"
    elif any(term in desc_lower for term in ["sql", "injection"]):
        vuln_type = "SQL Injection"
    elif any(term in desc_lower for term in ["xss", "cross-site"]):
        vuln_type = "Cross-Site Scripting (XSS)"
    elif any(term in desc_lower for term in ["path", "traversal", "directory"]):
        vuln_type = "Path Traversal"
    elif any(term in desc_lower for term in ["rce", "remote code", "command"]):
        vuln_type = "Remote Code Execution"
    elif any(term in desc_lower for term in ["denial", "dos"]):
        vuln_type = "Denial of Service"
    else:
        vuln_type = "Security Vulnerability"
    
    # Response teaches model to synthesize information
    response = f"""{cve_id}: {vuln_type}

**Severity:** {severity_label} (CVSS {cvss_score}/10.0)
**Attack Vector:** {attack_vector}

**Description:**
{description[:300]}{'...' if len(description) > 300 else ''}

**Impact:**
This vulnerability could allow attackers to compromise the security of affected systems.

**Recommendation:**
Update to the latest patched version. Monitor vendor security advisories."""
    
    return {
        "messages": [
            {"role": "system", "content": "You are a CVE database expert. Analyze the provided CVE information and explain it clearly."},
            {"role": "user", "content": f"{context}\n\n[QUERY]\n{query}"},
            {"role": "assistant", "content": response}
        ],
        "task_type": "CVE_LOOKUP"
    }


def generate_code_analysis_with_pattern_context(row: pd.Series) -> Dict[str, Any]:
    """
    [CODE_ANALYSIS] task - Deep vulnerability analysis
    
    Teach model to identify vulnerability patterns and relate to CVEs
    """
    code = row.get('code') or row.get('vuln_code', '')
    
    if not code or len(code) < 20:
        return None
    
    if len(code) > 1500:
        code = code[:1500] + "\n// ... (truncated)"
    
    # Identify vulnerability pattern (for training labels)
    pattern = identify_vulnerability_pattern(code)
    
    if not pattern:
        return None
    
    # Context: Common patterns and related CVEs (teaches pattern recognition)
    context = f"""[SECURITY ANALYSIS CONTEXT]

Common {pattern['type']} patterns:
{pattern['description']}

Example CVEs: {', '.join(pattern['example_cves'])}"""
    
    query = f"[CODE_ANALYSIS] Analyze this code for vulnerabilities:\n\n{code}"
    
    # Response format teaches structured analysis
    response = f"""VULNERABILITY DETECTED

**Type:** {pattern['type']}
**CWE:** {pattern['cwe']}
**Severity:** {pattern['severity']}

**Issue:** {pattern['issue']}

**Impact:** {pattern['impact']}

**Recommendation:** {pattern['recommendation']}"""
    
    return {
        "messages": [
            {"role": "system", "content": "You are a code security analyst. Analyze code for vulnerabilities using security knowledge."},
            {"role": "user", "content": f"{context}\n\n{query}"},
            {"role": "assistant", "content": response}
        ],
        "task_type": "CODE_ANALYSIS"
    }


def generate_fix_with_examples(row: pd.Series) -> Dict[str, Any]:
    """
    [FIX] task - Security remediation guidance
    
    Teach model to generate secure alternatives
    """
    vuln_code = row.get('vuln_code') or row.get('code', '')
    safe_code = row.get('safe_code', '')
    
    if not vuln_code or not safe_code:
        return None
    
    if len(vuln_code) > 1000:
        vuln_code = vuln_code[:1000] + "\n// ... (truncated)"
    if len(safe_code) > 1000:
        safe_code = safe_code[:1000] + "\n// ... (truncated)"
    
    # Context: General secure coding principles
    context = """[SECURE CODING PRINCIPLES]

Key remediation strategies:
- Input validation and sanitization
- Use parameterized queries (not string concatenation)
- Bounds checking for buffer operations
- Least privilege principle
- Fail securely by default"""
    
    query = f"[FIX] How to fix this vulnerability?\n\n{vuln_code}"
    
    response = f"""VULNERABILITY FIX

**Vulnerable Code:**
```
{vuln_code}
```

**Secure Code:**
```
{safe_code}
```

**Explanation:**
The fix addresses the vulnerability by implementing proper security controls. The vulnerable code lacked necessary validation/sanitization, which the secure version now includes."""
    
    return {
        "messages": [
            {"role": "system", "content": "You are a security remediation expert. Provide secure code alternatives with explanations."},
            {"role": "user", "content": f"{context}\n\n{query}"},
            {"role": "assistant", "content": response}
        ],
        "task_type": "FIX"
    }


def identify_vulnerability_pattern(code: str) -> Dict[str, str]:
    """Identify vulnerability pattern in code"""
    code_lower = code.lower()
    
    patterns = {
        "strcpy|sprintf|gets": {
            "type": "Buffer Overflow",
            "cwe": "CWE-120",
            "severity": "HIGH",
            "description": "Unsafe string operations without bounds checking",
            "example_cves": ["CVE-2021-3156", "CVE-2019-14287"],
            "issue": "Code uses unsafe string functions that don't check buffer boundaries",
            "impact": "Buffer overflow could lead to code execution or crash",
            "recommendation": "Use safe alternatives like strncpy, snprintf with explicit bounds"
        },
        "system\\(|exec\\(|popen\\(": {
            "type": "Command Injection",
            "cwe": "CWE-78",
            "severity": "CRITICAL",
            "description": "Executing commands with unsanitized user input",
            "example_cves": ["CVE-2021-44228", "CVE-2022-22965"],
            "issue": "Code executes system commands with user-controllable input",
            "impact": "Arbitrary command execution with application privileges",
            "recommendation": "Use subprocess with shell=False and input validation"
        },
        "select.*from.*where.*\\+|query.*=.*\\+": {
            "type": "SQL Injection",
            "cwe": "CWE-89",
            "severity": "CRITICAL",
            "description": "SQL queries built with string concatenation",
            "example_cves": ["CVE-2021-41773", "CVE-2022-0847"],
            "issue": "SQL query constructed using string concatenation with user input",
            "impact": "Database compromise, data theft, authentication bypass",
            "recommendation": "Use parameterized queries or prepared statements"
        },
        "eval\\(|exec\\(.*input": {
            "type": "Code Injection",
            "cwe": "CWE-94",
            "severity": "CRITICAL",
            "description": "Dynamic code evaluation with user input",
            "example_cves": ["CVE-2021-44228", "CVE-2022-22963"],
            "issue": "Code dynamically evaluates user input as code",
            "impact": "Arbitrary code execution",
            "recommendation": "Never use eval/exec with user input, use safe alternatives"
        },
    }
    
    import re
    for pattern_re, details in patterns.items():
        if re.search(pattern_re, code_lower):
            return details
    
    # Generic pattern
    return {
        "type": "Security Vulnerability",
        "cwe": "CWE-Other",
        "severity": "MEDIUM",
        "description": "Potential security issue detected",
        "example_cves": ["Multiple"],
        "issue": "Code contains potential security weakness",
        "impact": "Application security may be compromised",
        "recommendation": "Review code for security best practices"
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare RAG-focused training data")
    parser.add_argument('--nvd_csv', type=str,
                       default='vuln_data/processed/nvd_cves.csv',
                       help='NVD CSV file')
    parser.add_argument('--d2a_csv', type=str,
                       default='vuln_data/datasets/d2a_code_vulnerabilities.csv',
                       help='D2A code CSV file')
    parser.add_argument('--pyresbugs_csv', type=str,
                       default='vuln_data/datasets/pyresbugs_vulnerabilities.csv',
                       help='PyResBugs CSV file')
    parser.add_argument('--output_dir', type=str, default='vuln_data/rag_training_data/')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='Sample ratio (0.1 = 10%)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("RAG-Focused Training Data Preparation")
    print("="*60)
    print(f"Sample ratio: {args.sample_ratio*100}%")
    print(f"Output: {output_dir}")
    print("="*60)
    
    all_examples = []
    
    # 1. CVE_LOOKUP examples (reasoning about retrieved CVEs)
    print("\n1. Processing NVD CSV data for CVE_LOOKUP...")
    nvd_df = pd.read_csv(args.nvd_csv)
    sampled_nvd = nvd_df.sample(frac=args.sample_ratio, random_state=42)
    
    for _, row in tqdm(sampled_nvd.iterrows(), total=len(sampled_nvd), desc="CVE_LOOKUP"):
        # Convert CSV row to dict format expected by function
        entry = {
            'cve_id': row.get('cve_id', ''),
            'description': row.get('description', ''),
            'cvss_v3_score': row.get('cvss_score', 0.0),
            'cvss_v3_vector': f"AV:{row.get('attack_vector', 'UNKNOWN')}/AC:{row.get('attack_complexity', 'UNKNOWN')}",
            'affected_products': [p.strip() for p in str(row.get('affected_products', '')).split(';') if p.strip()],
            'published_date': row.get('published_date', '')
        }
        example = generate_cve_analysis_with_context(entry)
        if example:
            all_examples.append(example)
    
    print(f"✓ Generated {len([e for e in all_examples if e['task_type']=='CVE_LOOKUP'])} CVE_LOOKUP examples")
    
    # 2. CLASSIFY examples
    print("\n2. Processing D2A CSV data for CLASSIFY...")
    d2a_df = pd.read_csv(args.d2a_csv)
    sampled_d2a = d2a_df.sample(frac=args.sample_ratio, random_state=42)
    
    for _, row in tqdm(sampled_d2a.iterrows(), total=len(sampled_d2a), desc="CLASSIFY"):
        # Rename 'functions' column to 'code' if needed
        if 'code' not in row and 'functions' in row:
            row['code'] = row['functions']
        example = generate_classify_with_context(row)
        if example:
            all_examples.append(example)
    
    print(f"✓ Generated {len([e for e in all_examples if e['task_type']=='CLASSIFY'])} CLASSIFY examples")
    
    # 3. CODE_ANALYSIS and FIX from PyResBugs
    print("\n3. Processing PyResBugs CSV for CODE_ANALYSIS and FIX...")
    pyres_df = pd.read_csv(args.pyresbugs_csv)
    sampled_pyres = pyres_df.sample(frac=args.sample_ratio, random_state=42)
    
    for _, row in tqdm(sampled_pyres.iterrows(), total=len(sampled_pyres), desc="CODE_ANALYSIS/FIX"):
        # CODE_ANALYSIS - use faulty_code
        if pd.notna(row.get('faulty_code', '')):
            row_dict = {'code': row['faulty_code'], 'vuln_code': row['faulty_code']}
            analysis_ex = generate_code_analysis_with_pattern_context(pd.Series(row_dict))
            if analysis_ex:
                all_examples.append(analysis_ex)
        
        # FIX - use both faulty and fault_free code
        if pd.notna(row.get('faulty_code', '')) and pd.notna(row.get('fault_free_code', '')):
            row_dict = {
                'vuln_code': row['faulty_code'],
                'safe_code': row['fault_free_code']
            }
            fix_ex = generate_fix_with_examples(pd.Series(row_dict))
            if fix_ex:
                all_examples.append(fix_ex)
    
    print(f"✓ Generated {len([e for e in all_examples if e['task_type']=='CODE_ANALYSIS'])} CODE_ANALYSIS examples")
    print(f"✓ Generated {len([e for e in all_examples if e['task_type']=='FIX'])} FIX examples")
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Split train/val (90/10)
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    # Save
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    
    with open(train_path, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    
    with open(val_path, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')
    
    print("\n" + "="*60)
    print("✅ RAG Training Data Generation Complete!")
    print("="*60)
    print(f"Training examples: {len(train_examples):,}")
    print(f"Validation examples: {len(val_examples):,}")
    print(f"Total: {len(all_examples):,}")
    print(f"\nTask distribution:")
    for task in ["CLASSIFY", "CVE_LOOKUP", "CODE_ANALYSIS", "FIX"]:
        count = len([e for e in all_examples if e['task_type']==task])
        pct = (count / len(all_examples)) * 100
        print(f"  {task:20s}: {count:6,} ({pct:5.1f}%)")
    print("="*60)
    print(f"\nSaved to:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print("\nNext step:")
    print(f"  deepspeed --num_gpus=8 train_qwen_zero3.py --config config/multitask_training_config_rag.yaml")


if __name__ == '__main__':
    main()
