#!/usr/bin/env python3
"""
Evaluation Script for Fine-Tuned Qwen CVE Expert

Tests the model's ability to:
1. Identify CVEs for given services/products
2. Assess severity correctly
3. Provide actionable tool recommendations
4. Prioritize vulnerabilities
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import argparse
from dataclasses import dataclass
import re


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    cve_recall_at_1: float = 0.0
    cve_recall_at_5: float = 0.0
    severity_accuracy: float = 0.0
    response_quality: float = 0.0
    total_samples: int = 0


class QwenCVEEvaluator:
    """
    Evaluates fine-tuned Qwen on CVE tasks
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        print(f"Loading model from {model_path}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model.eval()
        print(f"Model loaded on {device}")
    
    def generate_response(
        self,
        query: str,
        system_prompt: str = None,
        max_length: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """Generate response for a query"""
        
        if system_prompt is None:
            system_prompt = (
                "You are a cybersecurity expert specializing in vulnerability analysis. "
                "Provide specific, actionable security guidance based on CVE databases."
            )
        
        # Format conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Format for Qwen
        formatted_text = ""
        for message in messages:
            role = message['role']
            content = message['content']
            formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Add assistant start token
        formatted_text += "<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response (after the formatted input)
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        
        return response.strip()
    
    def evaluate_cve_recall(
        self,
        test_data: List[Dict[str, Any]],
        k_values: List[int] = [1, 5]
    ) -> Dict[str, float]:
        """
        Evaluate CVE recall@K
        
        Given a product/service, does model predict correct CVE in top K?
        """
        results = {f"recall@{k}": [] for k in k_values}
        
        print("Evaluating CVE Recall...")
        for sample in tqdm(test_data):
            # Create query
            product = sample.get('product', 'unknown')
            ground_truth_cve = sample['cve_id']
            
            query = f"What vulnerabilities affect {product}?"
            
            # Generate response
            response = self.generate_response(query)
            
            # Extract CVE IDs from response
            predicted_cves = re.findall(r'CVE-\d{4}-\d{4,7}', response)
            
            # Check recall@K
            for k in k_values:
                top_k = predicted_cves[:k]
                hit = 1.0 if ground_truth_cve in top_k else 0.0
                results[f"recall@{k}"].append(hit)
        
        # Calculate averages
        metrics = {}
        for k in k_values:
            metrics[f"recall@{k}"] = sum(results[f"recall@{k}"]) / len(results[f"recall@{k}"])
        
        return metrics
    
    def evaluate_severity_classification(
        self,
        test_data: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate severity classification accuracy
        
        Does model correctly identify severity (CRITICAL/HIGH/MEDIUM/LOW)?
        """
        correct = 0
        total = 0
        
        print("Evaluating Severity Classification...")
        for sample in tqdm(test_data):
            cve_id = sample['cve_id']
            ground_truth_severity = sample['severity'].upper()
            
            query = f"What is the severity of {cve_id}?"
            response = self.generate_response(query)
            
            # Check if correct severity is in response
            if ground_truth_severity in response.upper():
                correct += 1
            
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_tool_recommendations(
        self,
        test_samples: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate quality of tool recommendations
        
        Does model provide specific, actionable tool names?
        """
        quality_scores = []
        
        # Common security tools we expect to see
        expected_tools = {
            'nmap', 'nuclei', 'metasploit', 'crackmapexec', 'sqlmap',
            'nikto', 'burp', 'wireshark', 'hydra', 'john', 'hashcat',
            'enum4linux', 'smbclient', 'responder', 'bloodhound'
        }
        
        print("Evaluating Tool Recommendations...")
        for sample in tqdm(test_samples):
            cve_id = sample['cve_id']
            
            query = f"How can I check for {cve_id}?"
            response = self.generate_response(query).lower()
            
            # Check if response contains specific tool names
            mentioned_tools = sum(1 for tool in expected_tools if tool in response)
            
            # Check for command examples (presence of backticks or specific syntax)
            has_commands = '`' in response or 'command:' in response.lower()
            
            # Scoring
            score = 0.0
            if mentioned_tools > 0:
                score += 0.5
            if mentioned_tools >= 2:
                score += 0.25
            if has_commands:
                score += 0.25
            
            quality_scores.append(score)
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    def run_interactive_demo(self):
        """Interactive demo mode"""
        print("\n" + "="*60)
        print("Qwen CVE Expert - Interactive Demo")
        print("="*60)
        print("Ask questions about vulnerabilities, CVEs, or security tools.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                print("\nQwen CVE Expert:")
                response = self.generate_response(query)
                print(response)
                print("\n" + "-"*60 + "\n")
                
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")
    
    def run_benchmark_examples(self):
        """Run a set of benchmark examples"""
        examples = [
            {
                "query": "I found Apache httpd 2.4.49 running on port 80. What should I check?",
                "expected_cves": ["CVE-2021-41773", "CVE-2021-42013"]
            },
            {
                "query": "What is CVE-2017-0144?",
                "expected_keywords": ["EternalBlue", "SMB", "Windows", "Critical"]
            },
            {
                "query": "How can I verify EternalBlue vulnerability?",
                "expected_tools": ["nmap", "metasploit", "smb-vuln-ms17-010"]
            },
            {
                "query": "Target has Windows XP with SMB open on port 445. What vulnerabilities should I look for?",
                "expected_cves": ["CVE-2017-0144", "CVE-2008-4250"]
            },
            {
                "query": "Is CVE-2021-44228 critical?",
                "expected_keywords": ["Log4Shell", "Critical", "CVSS 10.0", "RCE"]
            }
        ]
        
        print("\n" + "="*60)
        print("Running Benchmark Examples")
        print("="*60 + "\n")
        
        for i, example in enumerate(examples, 1):
            print(f"\nExample {i}:")
            print(f"Query: {example['query']}")
            print("-"*60)
            
            response = self.generate_response(example['query'])
            print(f"Response:\n{response}")
            
            # Simple validation
            if 'expected_cves' in example:
                found_cves = [cve for cve in example['expected_cves'] if cve in response]
                print(f"\n✓ Found CVEs: {found_cves}")
            
            if 'expected_keywords' in example:
                found_keywords = [kw for kw in example['expected_keywords'] if kw.lower() in response.lower()]
                print(f"✓ Found keywords: {found_keywords}")
            
            if 'expected_tools' in example:
                found_tools = [tool for tool in example['expected_tools'] if tool.lower() in response.lower()]
                print(f"✓ Found tools: {found_tools}")
            
            print("\n" + "="*60)


def load_test_data(test_csv_path: str, max_samples: int = 1000) -> List[Dict[str, Any]]:
    """Load test data from CSV"""
    df = pd.read_csv(test_csv_path)
    
    # Sample if needed
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    # Convert to list of dicts
    test_data = []
    for _, row in df.iterrows():
        # Extract product from affected_products
        product = row.get('affected_products', '')
        if pd.notna(product) and product:
            # Take first product if multiple
            product = product.split(';')[0].strip()
        
        test_data.append({
            'cve_id': row['cve_id'],
            'product': product,
            'severity': row['severity'],
            'cvss_score': row.get('cvss_score', 0),
            'description': row['description']
        })
    
    return test_data


def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen CVE Expert')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to fine-tuned model checkpoint')
    parser.add_argument('--test_data', type=str,
                       default='vuln_data/processed/test.csv',
                       help='Path to test data CSV')
    parser.add_argument('--mode', type=str, 
                       choices=['full', 'benchmark', 'interactive'],
                       default='benchmark',
                       help='Evaluation mode')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum test samples for full evaluation')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for metrics')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = QwenCVEEvaluator(args.model_path)
    
    if args.mode == 'interactive':
        # Interactive demo
        evaluator.run_interactive_demo()
    
    elif args.mode == 'benchmark':
        # Run benchmark examples
        evaluator.run_benchmark_examples()
    
    elif args.mode == 'full':
        # Full evaluation on test set
        print(f"Loading test data from {args.test_data}...")
        test_data = load_test_data(args.test_data, args.max_samples)
        print(f"Loaded {len(test_data)} test samples")
        
        # Run evaluations
        results = {}
        
        # CVE Recall
        recall_metrics = evaluator.evaluate_cve_recall(test_data[:min(500, len(test_data))])
        results.update(recall_metrics)
        
        # Severity Classification
        severity_acc = evaluator.evaluate_severity_classification(test_data[:min(500, len(test_data))])
        results['severity_accuracy'] = severity_acc
        
        # Tool Recommendations
        tool_quality = evaluator.evaluate_tool_recommendations(test_data[:min(200, len(test_data))])
        results['tool_recommendation_quality'] = tool_quality
        
        # Print results
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
