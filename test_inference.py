#!/usr/bin/env python3
"""
Comprehensive Inference Testing Script

Tests the trained model on diverse real-world examples across all 4 tasks.
Shows detailed outputs to understand model capabilities and gaps.

Usage:
    python test_inference.py --checkpoint checkpoints/qwen-security-expert-multitask/checkpoint-epoch3-step2814
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from pathlib import Path
from typing import List, Dict
import time


class InferenceTester:
    """Test model on comprehensive examples"""
    
    def __init__(self, checkpoint_path: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
        print(f"\n{'='*80}")
        print("Loading Model for Inference Testing")
        print(f"{'='*80}")
        print(f"Base model: {base_model}")
        print(f"Checkpoint: {checkpoint_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base + LoRA
        print("Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(base, checkpoint_path)
        self.model.eval()
        print(f"✓ Model loaded\n")
    
    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = 0.1) -> str:
        """Generate response"""
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def test_classify(self):
        """Test binary code classification"""
        print("\n" + "="*80)
        print("TASK 1: CLASSIFY - Binary Code Vulnerability Classification")
        print("="*80)
        
        test_cases = [
            {
                "name": "Buffer Overflow (Obvious)",
                "code": "strcpy(buffer, user_input);",
                "expected": "1",
                "difficulty": "Easy"
            },
            {
                "name": "Safe String Copy",
                "code": "strncpy(buffer, user_input, sizeof(buffer) - 1);\nbuffer[sizeof(buffer) - 1] = '\\0';",
                "expected": "0",
                "difficulty": "Easy"
            },
            {
                "name": "Command Injection",
                "code": "system(\"ping \" + user_input);",
                "expected": "1",
                "difficulty": "Easy"
            },
            {
                "name": "SQL Injection",
                "code": "query = \"SELECT * FROM users WHERE id=\" + request.GET['id']",
                "expected": "1",
                "difficulty": "Medium"
            },
            {
                "name": "Parameterized Query (Safe)",
                "code": "cursor.execute(\"SELECT * FROM users WHERE id=?\", (user_id,))",
                "expected": "0",
                "difficulty": "Medium"
            },
            {
                "name": "Code Injection (eval)",
                "code": "eval(request.POST['expression'])",
                "expected": "1",
                "difficulty": "Easy"
            },
            {
                "name": "Path Traversal",
                "code": "with open(\"/data/\" + filename, 'r') as f:\n    return f.read()",
                "expected": "1",
                "difficulty": "Medium"
            },
            {
                "name": "Safe Path (Validated)",
                "code": "if not '..' in filename:\n    with open(os.path.join(SAFE_DIR, os.path.basename(filename)), 'r') as f:\n        return f.read()",
                "expected": "0",
                "difficulty": "Hard"
            }
        ]
        
        correct = 0
        for i, test in enumerate(test_cases, 1):
            messages = [
                {"role": "system", "content": "You are a code security classifier. For [CLASSIFY] tasks, respond with ONLY '0' (safe) or '1' (vulnerable). No explanations, just the number."},
                {"role": "user", "content": f"[CLASSIFY] Is this code vulnerable?\n\n{test['code']}"}
            ]
            
            response = self.generate(messages, max_tokens=10, temperature=0)
            predicted = response.strip()[0] if response and response[0] in ['0', '1'] else '?'
            is_correct = predicted == test['expected']
            correct += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"\n{i}. {test['name']} ({test['difficulty']})")
            print(f"   Code: {test['code'][:60]}...")
            print(f"   Expected: {test['expected']} | Predicted: {predicted} {status}")
        
        accuracy = correct / len(test_cases)
        print(f"\n{'='*80}")
        print(f"CLASSIFY Accuracy: {correct}/{len(test_cases)} = {accuracy:.1%}")
        print(f"{'='*80}")
        return accuracy
    
    def test_cve_lookup(self):
        """Test CVE identification"""
        print("\n" + "="*80)
        print("TASK 2: CVE_LOOKUP - CVE Identification")
        print("="*80)
        
        test_cases = [
            {
                "query": "What vulnerabilities affect Apache httpd 2.4.49?",
                "expected_cves": ["CVE-2021-41773", "CVE-2021-42013"],
                "difficulty": "Easy"
            },
            {
                "query": "What is CVE-2017-0144?",
                "expected_keywords": ["EternalBlue", "SMB", "Windows"],
                "difficulty": "Easy"
            },
            {
                "query": "What vulnerabilities are in Apache Log4j 2.14.1?",
                "expected_cves": ["CVE-2021-44228"],
                "expected_keywords": ["Log4Shell", "JNDI"],
                "difficulty": "Easy"
            },
            {
                "query": "Security issues in OpenSSL 1.0.1?",
                "expected_keywords": ["Heartbleed", "CVE-2014-0160"],
                "difficulty": "Medium"
            },
            {
                "query": "Windows XP SMB vulnerabilities",
                "expected_cves": ["CVE-2017-0144", "CVE-2008-4250"],
                "difficulty": "Medium"
            },
            {
                "query": "What is the severity of CVE-2021-44228?",
                "expected_keywords": ["Critical", "10.0", "CVSS"],
                "difficulty": "Easy"
            }
        ]
        
        total_score = 0
        for i, test in enumerate(test_cases, 1):
            messages = [
                {"role": "system", "content": "You are a CVE database expert. For [CVE_LOOKUP] tasks, provide detailed vulnerability information including CVE IDs, CVSS scores, attack vectors, and actionable recommendations."},
                {"role": "user", "content": f"[CVE_LOOKUP] {test['query']}"}
            ]
            
            response = self.generate(messages, max_tokens=256)
            
            score = 0
            feedback = []
            
            # Check for expected CVEs
            if 'expected_cves' in test:
                found_cves = [cve for cve in test['expected_cves'] if cve in response]
                if found_cves:
                    score += 0.5
                    feedback.append(f"✅ Found CVEs: {', '.join(found_cves)}")
                else:
                    feedback.append(f"❌ Missing CVEs: {', '.join(test['expected_cves'])}")
            
            # Check for expected keywords
            if 'expected_keywords' in test:
                found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in response.lower()]
                if found_keywords:
                    score += 0.5
                    feedback.append(f"✅ Found keywords: {', '.join(found_keywords)}")
                else:
                    feedback.append(f"❌ Missing keywords: {', '.join(test['expected_keywords'])}")
            
            total_score += score
            
            print(f"\n{i}. {test['query']} ({test['difficulty']})")
            print(f"   Response: {response[:150]}...")
            for fb in feedback:
                print(f"   {fb}")
            print(f"   Score: {score:.1f}/1.0")
        
        avg_score = total_score / len(test_cases)
        print(f"\n{'='*80}")
        print(f"CVE_LOOKUP Average Score: {avg_score:.1%}")
        print(f"{'='*80}")
        return avg_score
    
    def test_code_analysis(self):
        """Test vulnerability analysis"""
        print("\n" + "="*80)
        print("TASK 3: CODE_ANALYSIS - Deep Vulnerability Analysis")
        print("="*80)
        
        test_cases = [
            {
                "code": "eval(request.GET['cmd'])",
                "expected_keywords": ["code execution", "eval", "injection", "critical"],
                "difficulty": "Easy"
            },
            {
                "code": "password = request.GET['pwd']\nif password == 'admin123':\n    grant_access()",
                "expected_keywords": ["hardcoded", "password", "credential", "authentication"],
                "difficulty": "Medium"
            },
            {
                "code": "$query = \"SELECT * FROM users WHERE name='\" . $_GET['name'] . \"'\";",
                "expected_keywords": ["sql injection", "injection", "query", "database"],
                "difficulty": "Easy"
            },
            {
                "code": "import pickle\ndata = pickle.loads(user_data)",
                "expected_keywords": ["deserialization", "pickle", "code execution", "untrusted"],
                "difficulty": "Hard"
            }
        ]
        
        total_score = 0
        for i, test in enumerate(test_cases, 1):
            messages = [
                {"role": "system", "content": "You are a code security analyst. For [CODE_ANALYSIS] tasks, provide comprehensive vulnerability analysis including type, severity, CVEs (if applicable), impact, and proof-of-concept."},
                {"role": "user", "content": f"[CODE_ANALYSIS] Analyze this code:\n\n{test['code']}"}
            ]
            
            response = self.generate(messages, max_tokens=256)
            
            found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in response.lower()]
            score = len(found_keywords) / len(test['expected_keywords'])
            total_score += score
            
            print(f"\n{i}. {test['code'][:50]}... ({test['difficulty']})")
            print(f"   Response: {response[:200]}...")
            print(f"   Found: {', '.join(found_keywords) if found_keywords else 'None'}")
            print(f"   Score: {score:.1%}")
        
        avg_score = total_score / len(test_cases)
        print(f"\n{'='*80}")
        print(f"CODE_ANALYSIS Average Score: {avg_score:.1%}")
        print(f"{'='*80}")
        return avg_score
    
    def test_fix(self):
        """Test code remediation"""
        print("\n" + "="*80)
        print("TASK 4: FIX - Code Remediation")
        print("="*80)
        
        test_cases = [
            {
                "vuln_code": "query = \"SELECT * FROM users WHERE id=\" + user_id",
                "expected_keywords": ["parameterized", "?", "prepare", "bind"],
                "difficulty": "Easy"
            },
            {
                "vuln_code": "eval(user_input)",
                "expected_keywords": ["ast.literal_eval", "json", "avoid eval", "whitelist"],
                "difficulty": "Medium"
            },
            {
                "vuln_code": "strcpy(buffer, input);",
                "expected_keywords": ["strncpy", "strlcpy", "bounds check", "sizeof"],
                "difficulty": "Easy"
            },
            {
                "vuln_code": "os.system('ping ' + host)",
                "expected_keywords": ["subprocess", "shell=False", "list", "sanitize"],
                "difficulty": "Medium"
            }
        ]
        
        total_score = 0
        for i, test in enumerate(test_cases, 1):
            messages = [
                {"role": "system", "content": "You are a security remediation expert. For [FIX] tasks, provide before/after code comparisons with clear explanations of why the fix works."},
                {"role": "user", "content": f"[FIX] How to fix this vulnerability?\n\n{test['vuln_code']}"}
            ]
            
            response = self.generate(messages, max_tokens=512)
            
            score = 0
            feedback = []
            
            # Check for code blocks
            has_code = '```' in response or '`' in response
            if has_code:
                score += 0.5
                feedback.append("✅ Contains code blocks")
            else:
                feedback.append("❌ No code blocks")
            
            # Check for fix keywords
            found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in response.lower()]
            if found_keywords:
                score += 0.5 * (len(found_keywords) / len(test['expected_keywords']))
                feedback.append(f"✅ Mentions: {', '.join(found_keywords)}")
            else:
                feedback.append(f"❌ Missing fix techniques")
            
            total_score += score
            
            print(f"\n{i}. {test['vuln_code'][:50]}... ({test['difficulty']})")
            print(f"   Response: {response[:200]}...")
            for fb in feedback:
                print(f"   {fb}")
            print(f"   Score: {score:.1f}/1.0")
        
        avg_score = total_score / len(test_cases)
        print(f"\n{'='*80}")
        print(f"FIX Average Score: {avg_score:.1%}")
        print(f"{'='*80}")
        return avg_score
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE INFERENCE TEST")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        results = {}
        results['classify'] = self.test_classify()
        results['cve_lookup'] = self.test_cve_lookup()
        results['code_analysis'] = self.test_code_analysis()
        results['fix'] = self.test_fix()
        
        elapsed = time.time() - start_time
        
        # Final summary
        print(f"\n\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"\n✅ CLASSIFY:       {results['classify']:.1%}")
        print(f"✅ CVE_LOOKUP:     {results['cve_lookup']:.1%}")
        print(f"✅ CODE_ANALYSIS:  {results['code_analysis']:.1%}")
        print(f"✅ FIX:            {results['fix']:.1%}")
        print(f"\n⚡ Overall Average: {sum(results.values())/len(results):.1%}")
        print(f"⏱️  Time taken: {elapsed:.1f}s")
        print(f"\n{'='*80}")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        print("-" * 80)
        
        if results['classify'] < 0.7:
            print("❌ CLASSIFY: <70% - Model needs more training on binary classification")
        elif results['classify'] < 0.85:
            print("⚠️  CLASSIFY: 70-85% - Decent but room for improvement")
        else:
            print("✅ CLASSIFY: >85% - Good performance!")
        
        if results['cve_lookup'] < 0.5:
            print("❌ CVE_LOOKUP: <50% - Model has weak CVE knowledge, needs more CVE data")
        elif results['cve_lookup'] < 0.7:
            print("⚠️  CVE_LOOKUP: 50-70% - Learning but needs more training")
        else:
            print("✅ CVE_LOOKUP: >70% - Good CVE knowledge!")
        
        if results['code_analysis'] < 0.5:
            print("❌ CODE_ANALYSIS: <50% - Struggling with vulnerability analysis")
        elif results['code_analysis'] < 0.7:
            print("⚠️  CODE_ANALYSIS: 50-70% - Improving but needs refinement")
        else:
            print("✅ CODE_ANALYSIS: >70% - Good analysis capability!")
        
        if results['fix'] < 0.5:
            print("❌ FIX: <50% - Weak remediation guidance")
        elif results['fix'] < 0.7:
            print("⚠️  FIX: 50-70% - Can provide fixes but needs improvement")
        else:
            print("✅ FIX: >70% - Good remediation knowledge!")
        
        overall = sum(results.values()) / len(results)
        print(f"\n{'='*80}")
        if overall < 0.6:
            print("❌ OVERALL: Model needs significant more training on full dataset")
            print("   Recommendation: Train on 50-100% of full data")
        elif overall < 0.75:
            print("⚠️  OVERALL: Model is learning but not production-ready")
            print("   Recommendation: Train on 50% of full data for better results")
        else:
            print("✅ OVERALL: Model shows good capability!")
            print("   Recommendation: Fine-tune on full dataset for production quality")
        print(f"{'='*80}\n")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Test model inference on comprehensive examples')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='Base model name')
    
    args = parser.parse_args()
    
    tester = InferenceTester(args.checkpoint, args.base_model)
    results = tester.run_all_tests()


if __name__ == '__main__':
    main()
