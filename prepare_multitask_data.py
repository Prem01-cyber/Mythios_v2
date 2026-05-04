#!/usr/bin/env python3
"""
Multi-Task Training Data Preparation for Qwen Security Expert

Transforms 3 datasets into 4 training tasks:
1. [CLASSIFY] - Binary vulnerability classification (D2A)
2. [CVE_LOOKUP] - CVE identification (NVD)
3. [CODE_ANALYSIS] - Deep vulnerability analysis (PyResBugs + D2A subset)
4. [FIX] - Remediation guidance (PyResBugs)

Implements balanced sampling to prevent task imbalance.
"""

import pandas as pd
import json
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import hashlib


class MultiTaskDataGenerator:
    """
    Generates balanced multi-task training data from 3 datasets
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
        # System prompts for each task
        self.system_prompts = {
            "CLASSIFY": (
                "You are a code security classifier. For [CLASSIFY] tasks, "
                "respond with ONLY '0' (safe) or '1' (vulnerable). "
                "No explanations, just the number."
            ),
            "CVE_LOOKUP": (
                "You are a CVE database expert. For [CVE_LOOKUP] tasks, "
                "provide detailed vulnerability information including CVE IDs, "
                "CVSS scores, attack vectors, and actionable recommendations."
            ),
            "CODE_ANALYSIS": (
                "You are a code security analyst. For [CODE_ANALYSIS] tasks, "
                "provide comprehensive vulnerability analysis including type, "
                "severity, CVEs (if applicable), impact, and proof-of-concept."
            ),
            "FIX": (
                "You are a security remediation expert. For [FIX] tasks, "
                "provide before/after code comparisons with clear explanations "
                "of why the fix works."
            )
        }
        
        # Question templates for variety
        self.classify_templates = [
            "[CLASSIFY] Is this code vulnerable?\n\n{code}",
            "[CLASSIFY] Vulnerability check:\n\n{code}",
            "[CLASSIFY] Does this code have security issues?\n\n{code}",
            "[CLASSIFY] Security scan:\n\n{code}",
        ]
        
        self.cve_lookup_templates = [
            "[CVE_LOOKUP] What vulnerabilities affect {product}?",
            "[CVE_LOOKUP] Are there any CVEs for {product}?",
            "[CVE_LOOKUP] Security issues in {product}?",
            "[CVE_LOOKUP] Tell me about {product} vulnerabilities",
        ]
        
        self.code_analysis_templates = [
            "[CODE_ANALYSIS] Analyze this code:\n\n{code}",
            "[CODE_ANALYSIS] Security review:\n\n{code}",
            "[CODE_ANALYSIS] What vulnerabilities exist in:\n\n{code}",
            "[CODE_ANALYSIS] Deep security analysis:\n\n{code}",
        ]
        
        self.fix_templates = [
            "[FIX] How to fix this vulnerability?\n\n{code}",
            "[FIX] Remediate:\n\n{code}",
            "[FIX] How to secure this code?\n\n{code}",
            "[FIX] What's the proper fix for:\n\n{code}",
        ]
    
    # ============================================
    # TASK 1: Binary Classification (D2A)
    # ============================================
    
    def generate_classify_example(self, row: pd.Series) -> Dict[str, Any]:
        """
        Generate [CLASSIFY] task from D2A dataset
        
        Input: Code snippet
        Output: "0" or "1"
        """
        code = row.get('code') or row.get('source') or row.get('functions', '')
        label = str(int(row['label']))  # Ensure it's "0" or "1"
        
        # Truncate very long code
        if len(code) > 2000:
            code = code[:2000] + "\n// ... (truncated)"
        
        instruction = random.choice(self.classify_templates).format(code=code.strip())
        
        return {
            "messages": [
                {"role": "system", "content": self.system_prompts["CLASSIFY"]},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": label}
            ],
            "task_type": "CLASSIFY",
            "metadata": {"source": row.get('source', 'D2A')}
        }
    
    # ============================================
    # TASK 2: CVE Lookup (NVD)
    # ============================================
    
    def generate_cve_lookup_example(self, row: pd.Series) -> Dict[str, Any]:
        """
        Generate [CVE_LOOKUP] task from NVD dataset
        
        Input: Product/service name
        Output: Detailed CVE information
        """
        cve_id = row['cve_id']
        description = row.get('description', '')
        cvss = row.get('cvss_score', 'N/A')
        severity = row.get('severity', 'UNKNOWN')
        attack_vector = row.get('attack_vector', 'UNKNOWN')
        products = row.get('affected_products', '')
        exploitability = row.get('exploitability', 0)
        
        # Extract product name for question
        product = products.split(';')[0].strip() if products else "this software"
        
        instruction = random.choice(self.cve_lookup_templates).format(product=product)
        
        # Build detailed response
        response = f"{cve_id}: {description}\n\n"
        response += f"**Severity:** {severity} (CVSS {cvss})\n"
        
        if attack_vector != 'UNKNOWN':
            response += f"**Attack Vector:** {attack_vector}\n"
        
        if products:
            response += f"**Affected:** {products}\n"
        
        # Exploitability assessment
        if exploitability > 0.7:
            response += "\n**Exploitability:** HIGH - Public exploits available\n"
            response += "**Priority:** CRITICAL - Immediate action required"
        elif exploitability > 0.4:
            response += "\n**Exploitability:** MEDIUM - Proof-of-concept exists\n"
            response += "**Priority:** HIGH - Plan remediation soon"
        else:
            response += "\n**Exploitability:** LOW - Difficult to exploit\n"
            response += "**Priority:** MEDIUM - Include in patch cycle"
        
        return {
            "messages": [
                {"role": "system", "content": self.system_prompts["CVE_LOOKUP"]},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response.strip()}
            ],
            "task_type": "CVE_LOOKUP",
            "metadata": {"cve_id": cve_id}
        }
    
    # ============================================
    # TASK 3: Code Analysis (PyResBugs + D2A)
    # ============================================
    
    def generate_code_analysis_from_pyresbugs(self, row: pd.Series) -> Dict[str, Any]:
        """
        Generate [CODE_ANALYSIS] from PyResBugs (has CVE and detailed info)
        """
        faulty_code = row.get('faulty_code', '')
        cve_id = row.get('cve_id', '')
        bug_type = row.get('bug_type', 'security')
        bug_description = row.get('bug_description', '')
        
        if len(faulty_code) > 2000:
            faulty_code = faulty_code[:2000] + "\n# ... (truncated)"
        
        instruction = random.choice(self.code_analysis_templates).format(code=faulty_code.strip())
        
        # Build comprehensive analysis
        response = "VULNERABILITY DETECTED\n\n"
        
        if cve_id:
            response += f"**CVE:** {cve_id}\n"
        
        response += f"**Type:** {bug_type.title()}\n"
        response += f"**Severity:** {'CRITICAL' if cve_id else 'HIGH'}\n\n"
        
        if bug_description:
            # Clean up description
            desc = bug_description.replace('\n\n', ' ').replace('\n', ' ')[:500]
            response += f"**Issue:** {desc}\n\n"
        
        response += "**Impact:** This vulnerability could allow attackers to compromise security.\n"
        response += "**Recommendation:** Apply the security fix immediately."
        
        return {
            "messages": [
                {"role": "system", "content": self.system_prompts["CODE_ANALYSIS"]},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response.strip()}
            ],
            "task_type": "CODE_ANALYSIS",
            "metadata": {"cve_id": cve_id, "source": "PyResBugs"}
        }
    
    def generate_code_analysis_from_d2a(self, row: pd.Series) -> Dict[str, Any]:
        """
        Generate [CODE_ANALYSIS] from D2A vulnerable code (label=1)
        """
        code = row.get('code') or row.get('source') or row.get('functions', '')
        
        if len(code) > 2000:
            code = code[:2000] + "\n// ... (truncated)"
        
        instruction = random.choice(self.code_analysis_templates).format(code=code.strip())
        
        # Infer vulnerability type from code patterns
        vuln_type = self._infer_vulnerability_type(code)
        
        response = f"VULNERABILITY DETECTED\n\n"
        response += f"**Type:** {vuln_type['name']}\n"
        response += f"**CWE:** {vuln_type['cwe']}\n"
        response += f"**Severity:** {vuln_type['severity']}\n\n"
        response += f"**Issue:** {vuln_type['description']}\n\n"
        response += f"**Impact:** {vuln_type['impact']}\n"
        response += f"**Recommendation:** {vuln_type['fix']}"
        
        return {
            "messages": [
                {"role": "system", "content": self.system_prompts["CODE_ANALYSIS"]},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response.strip()}
            ],
            "task_type": "CODE_ANALYSIS",
            "metadata": {"source": "D2A", "inferred": True}
        }
    
    def _infer_vulnerability_type(self, code: str) -> Dict[str, str]:
        """Infer vulnerability type from code patterns"""
        code_lower = code.lower()
        
        # Pattern matching for common vulnerabilities
        if 'strcpy' in code_lower or 'strcat' in code_lower:
            return {
                "name": "Buffer Overflow",
                "cwe": "CWE-120",
                "severity": "HIGH",
                "description": "Unsafe string function that doesn't check buffer bounds",
                "impact": "Memory corruption, arbitrary code execution possible",
                "fix": "Use strncpy() or strlcpy() with proper size limits"
            }
        elif 'eval(' in code_lower:
            return {
                "name": "Code Injection",
                "cwe": "CWE-95",
                "severity": "CRITICAL",
                "description": "eval() executes arbitrary code from potentially untrusted input",
                "impact": "Complete system compromise, arbitrary command execution",
                "fix": "Never use eval() on user input. Use ast.literal_eval() for literals only"
            }
        elif 'sql' in code_lower and ('+' in code or 'concat' in code_lower):
            return {
                "name": "SQL Injection",
                "cwe": "CWE-89",
                "severity": "CRITICAL",
                "description": "SQL query constructed with string concatenation of user input",
                "impact": "Database compromise, data theft, privilege escalation",
                "fix": "Use parameterized queries or prepared statements"
            }
        elif 'system(' in code_lower or 'exec(' in code_lower or 'shell' in code_lower:
            return {
                "name": "Command Injection",
                "cwe": "CWE-78",
                "severity": "CRITICAL",
                "description": "System command execution with unsanitized user input",
                "impact": "Arbitrary command execution, full system compromise",
                "fix": "Use subprocess with argument list, validate all input"
            }
        elif 'gets(' in code_lower:
            return {
                "name": "Buffer Overflow (gets)",
                "cwe": "CWE-120",
                "severity": "CRITICAL",
                "description": "gets() has no bounds checking and is inherently unsafe",
                "impact": "Stack buffer overflow, code execution",
                "fix": "Use fgets() with size limit instead"
            }
        else:
            return {
                "name": "Security Vulnerability",
                "cwe": "CWE-Other",
                "severity": "MEDIUM",
                "description": "Code contains potential security weakness",
                "impact": "System security may be compromised",
                "fix": "Review and validate all inputs, follow secure coding practices"
            }
    
    # ============================================
    # TASK 4: Remediation (PyResBugs)
    # ============================================
    
    def generate_fix_example(self, row: pd.Series) -> Dict[str, Any]:
        """
        Generate [FIX] task from PyResBugs
        
        Input: Vulnerable code
        Output: Before/after comparison with explanation
        """
        faulty_code = row.get('faulty_code', '')
        fixed_code = row.get('fault_free_code', '')
        bug_description = row.get('bug_description', '')
        cve_id = row.get('cve_id', '')
        
        if len(faulty_code) > 1000:
            faulty_code = faulty_code[:1000] + "\n# ... (truncated)"
        if len(fixed_code) > 1000:
            fixed_code = fixed_code[:1000] + "\n# ... (truncated)"
        
        # Create instruction
        if cve_id:
            instruction = f"[FIX] How to fix {cve_id}?\n\n{faulty_code.strip()}"
        else:
            instruction = random.choice(self.fix_templates).format(code=faulty_code.strip())
        
        # Build response with before/after
        response = "**VULNERABLE CODE:**\n```\n"
        response += faulty_code.strip()
        response += "\n```\n\n"
        
        response += "**FIXED CODE:**\n```\n"
        response += fixed_code.strip()
        response += "\n```\n\n"
        
        response += "**Explanation:**\n"
        if bug_description:
            # Extract key fix info from description
            desc = bug_description.split('\n')[:3]  # First 3 lines usually most relevant
            response += ' '.join(desc)
        else:
            response += "The fix addresses the security vulnerability by properly validating and sanitizing inputs."
        
        return {
            "messages": [
                {"role": "system", "content": self.system_prompts["FIX"]},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response.strip()}
            ],
            "task_type": "FIX",
            "metadata": {"cve_id": cve_id, "source": "PyResBugs"}
        }


def load_and_sample_datasets(
    nvd_path: str,
    d2a_code_path: str,
    d2a_func_path: str,
    pyresbugs_path: str,
    target_distribution: Dict[str, int],
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Load datasets and sample according to target distribution
    
    Target distribution example:
    {
        "CVE_LOOKUP": 1_000_000,     # 40%
        "CODE_ANALYSIS": 750_000,     # 30%
        "CLASSIFY": 500_000,          # 20%
        "FIX": 250_000                # 10%
    }
    """
    random.seed(seed)
    
    print("Loading datasets...")
    
    # Load NVD data (already split)
    nvd_df = pd.read_csv(nvd_path)
    print(f"  NVD: {len(nvd_df):,} samples")
    
    # Load D2A datasets
    d2a_code_df = pd.read_csv(d2a_code_path)
    d2a_func_df = pd.read_csv(d2a_func_path)
    print(f"  D2A Code: {len(d2a_code_df):,} samples")
    print(f"  D2A Function: {len(d2a_func_df):,} samples")
    
    # Combine D2A datasets
    d2a_df = pd.concat([d2a_code_df, d2a_func_df], ignore_index=True)
    print(f"  D2A Combined: {len(d2a_df):,} samples")
    
    # Load PyResBugs
    pyresbugs_df = pd.read_csv(pyresbugs_path)
    print(f"  PyResBugs: {len(pyresbugs_df):,} samples")
    
    print("\nApplying balanced sampling...")
    
    # Sample datasets according to target distribution
    sampled = {}
    
    # CVE_LOOKUP: Use full NVD + augmentation if needed
    cve_target = target_distribution.get("CVE_LOOKUP", len(nvd_df))
    if len(nvd_df) >= cve_target:
        sampled["CVE_LOOKUP"] = nvd_df.sample(n=cve_target, random_state=seed)
    else:
        # Oversample to reach target
        sampled["CVE_LOOKUP"] = nvd_df.sample(n=cve_target, replace=True, random_state=seed)
    print(f"  CVE_LOOKUP: {len(sampled['CVE_LOOKUP']):,} samples")
    
    # CLASSIFY: Sample from D2A
    classify_target = target_distribution.get("CLASSIFY", 500_000)
    sampled["CLASSIFY"] = d2a_df.sample(n=min(classify_target, len(d2a_df)), random_state=seed)
    print(f"  CLASSIFY: {len(sampled['CLASSIFY']):,} samples")
    
    # CODE_ANALYSIS: PyResBugs + D2A vulnerable samples (label=1)
    analysis_target = target_distribution.get("CODE_ANALYSIS", 100_000)
    d2a_vulnerable = d2a_df[d2a_df['label'] == 1]
    
    # Use all PyResBugs + sample from D2A vulnerable
    n_from_d2a = analysis_target - len(pyresbugs_df)
    if n_from_d2a > 0:
        d2a_sample = d2a_vulnerable.sample(n=min(n_from_d2a, len(d2a_vulnerable)), random_state=seed)
        sampled["CODE_ANALYSIS_pyres"] = pyresbugs_df
        sampled["CODE_ANALYSIS_d2a"] = d2a_sample
    else:
        sampled["CODE_ANALYSIS_pyres"] = pyresbugs_df.sample(n=analysis_target, random_state=seed)
        sampled["CODE_ANALYSIS_d2a"] = pd.DataFrame()
    
    print(f"  CODE_ANALYSIS: {len(pyresbugs_df):,} (PyResBugs) + {len(sampled.get('CODE_ANALYSIS_d2a', [])):,} (D2A) samples")
    
    # FIX: Use PyResBugs with oversampling if needed
    fix_target = target_distribution.get("FIX", len(pyresbugs_df))
    if len(pyresbugs_df) >= fix_target:
        sampled["FIX"] = pyresbugs_df.sample(n=fix_target, random_state=seed)
    else:
        # Oversample to reach target
        sampled["FIX"] = pyresbugs_df.sample(n=fix_target, replace=True, random_state=seed)
    print(f"  FIX: {len(sampled['FIX']):,} samples")
    
    return sampled


def generate_multitask_dataset(
    sampled_datasets: Dict[str, pd.DataFrame],
    output_path: str,
    shuffle: bool = True,
    seed: int = 42
):
    """
    Generate final multi-task training dataset with task prefixes
    """
    generator = MultiTaskDataGenerator(seed=seed)
    all_examples = []
    
    print("\nGenerating multi-task examples...")
    
    # Task 1: CVE Lookup
    print("  Generating CVE_LOOKUP examples...")
    for _, row in tqdm(sampled_datasets["CVE_LOOKUP"].iterrows(), total=len(sampled_datasets["CVE_LOOKUP"])):
        try:
            example = generator.generate_cve_lookup_example(row)
            all_examples.append(example)
        except Exception as e:
            continue
    
    # Task 2: Binary Classification
    print("  Generating CLASSIFY examples...")
    for _, row in tqdm(sampled_datasets["CLASSIFY"].iterrows(), total=len(sampled_datasets["CLASSIFY"])):
        try:
            example = generator.generate_classify_example(row)
            all_examples.append(example)
        except Exception as e:
            continue
    
    # Task 3: Code Analysis (from PyResBugs)
    if "CODE_ANALYSIS_pyres" in sampled_datasets and len(sampled_datasets["CODE_ANALYSIS_pyres"]) > 0:
        print("  Generating CODE_ANALYSIS examples (PyResBugs)...")
        for _, row in tqdm(sampled_datasets["CODE_ANALYSIS_pyres"].iterrows(), 
                          total=len(sampled_datasets["CODE_ANALYSIS_pyres"])):
            try:
                example = generator.generate_code_analysis_from_pyresbugs(row)
                all_examples.append(example)
            except Exception as e:
                continue
    
    # Task 3: Code Analysis (from D2A)
    if "CODE_ANALYSIS_d2a" in sampled_datasets and len(sampled_datasets["CODE_ANALYSIS_d2a"]) > 0:
        print("  Generating CODE_ANALYSIS examples (D2A)...")
        for _, row in tqdm(sampled_datasets["CODE_ANALYSIS_d2a"].iterrows(),
                          total=len(sampled_datasets["CODE_ANALYSIS_d2a"])):
            try:
                example = generator.generate_code_analysis_from_d2a(row)
                all_examples.append(example)
            except Exception as e:
                continue
    
    # Task 4: Remediation
    print("  Generating FIX examples...")
    for _, row in tqdm(sampled_datasets["FIX"].iterrows(), total=len(sampled_datasets["FIX"])):
        try:
            example = generator.generate_fix_example(row)
            all_examples.append(example)
        except Exception as e:
            continue
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(all_examples)
    
    # Save to JSONL
    print(f"\nSaving {len(all_examples):,} examples to {output_path}...")
    with open(output_path, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    # Print statistics
    task_counts = {}
    for ex in all_examples:
        task = ex['task_type']
        task_counts[task] = task_counts.get(task, 0) + 1
    
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(f"Total examples: {len(all_examples):,}")
    print("\nPer-task distribution:")
    for task, count in sorted(task_counts.items()):
        percentage = (count / len(all_examples)) * 100
        print(f"  {task:20s}: {count:8,} ({percentage:5.1f}%)")
    
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\nOutput file size: {size_mb:.2f} MB")
    print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare multi-task training data')
    parser.add_argument('--nvd_train', type=str,
                       default='vuln_data/processed/train.csv',
                       help='NVD training data')
    parser.add_argument('--nvd_val', type=str,
                       default='vuln_data/processed/val.csv',
                       help='NVD validation data')
    parser.add_argument('--d2a_code', type=str,
                       default='vuln_data/datasets/d2a_code_vulnerabilities.csv',
                       help='D2A code vulnerabilities')
    parser.add_argument('--d2a_func', type=str,
                       default='vuln_data/datasets/d2a_function_vulnerabilities.csv',
                       help='D2A function vulnerabilities')
    parser.add_argument('--pyresbugs', type=str,
                       default='vuln_data/datasets/pyresbugs_vulnerabilities.csv',
                       help='PyResBugs data')
    parser.add_argument('--output_dir', type=str,
                       default='vuln_data/multitask_data',
                       help='Output directory')
    
    # Sampling targets (total ~2.5M samples)
    parser.add_argument('--cve_lookup_samples', type=int, default=1_000_000,
                       help='Target samples for CVE lookup (40%%)')
    parser.add_argument('--code_analysis_samples', type=int, default=750_000,
                       help='Target samples for code analysis (30%%)')
    parser.add_argument('--classify_samples', type=int, default=500_000,
                       help='Target samples for classification (20%%)')
    parser.add_argument('--fix_samples', type=int, default=250_000,
                       help='Target samples for remediation (10%%)')
    
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define target distribution
    target_distribution = {
        "CVE_LOOKUP": args.cve_lookup_samples,
        "CODE_ANALYSIS": args.code_analysis_samples,
        "CLASSIFY": args.classify_samples,
        "FIX": args.fix_samples
    }
    
    print("="*60)
    print("Multi-Task Training Data Preparation")
    print("="*60)
    print("\nTarget Distribution:")
    total = sum(target_distribution.values())
    for task, count in target_distribution.items():
        pct = (count / total) * 100
        print(f"  {task:20s}: {count:8,} ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total:8,} (100.0%)")
    print("="*60)
    
    # Process training data
    print("\n" + "="*60)
    print("Processing Training Data")
    print("="*60)
    
    sampled_train = load_and_sample_datasets(
        nvd_path=args.nvd_train,
        d2a_code_path=args.d2a_code,
        d2a_func_path=args.d2a_func,
        pyresbugs_path=args.pyresbugs,
        target_distribution=target_distribution,
        seed=args.seed
    )
    
    generate_multitask_dataset(
        sampled_datasets=sampled_train,
        output_path=output_dir / 'train.jsonl',
        shuffle=True,
        seed=args.seed
    )
    
    # Process validation data (smaller, same proportions)
    print("\n" + "="*60)
    print("Processing Validation Data")
    print("="*60)
    
    val_distribution = {k: v // 10 for k, v in target_distribution.items()}  # 10% of training
    
    sampled_val = load_and_sample_datasets(
        nvd_path=args.nvd_val,
        d2a_code_path=args.d2a_code,
        d2a_func_path=args.d2a_func,
        pyresbugs_path=args.pyresbugs,
        target_distribution=val_distribution,
        seed=args.seed + 1  # Different seed for val
    )
    
    generate_multitask_dataset(
        sampled_datasets=sampled_val,
        output_path=output_dir / 'val.jsonl',
        shuffle=True,
        seed=args.seed + 1
    )
    
    print("\n" + "="*60)
    print("✓ Multi-task data preparation complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'train.jsonl'}")
    print(f"  - {output_dir / 'val.jsonl'}")
    print(f"\nNext step: Train with multi-task curriculum learning")


if __name__ == '__main__':
    main()
