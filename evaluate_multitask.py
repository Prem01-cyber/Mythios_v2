#!/usr/bin/env python3
"""
Multi-Task Evaluation Script for Qwen Security Expert

Evaluates the model on all 4 training tasks:
1. [CLASSIFY] - Binary code vulnerability classification
2. [CVE_LOOKUP] - CVE identification from product/service
3. [CODE_ANALYSIS] - Deep vulnerability analysis
4. [FIX] - Code remediation recommendations

Usage:
    # BASELINE: Evaluate raw Qwen (no fine-tuning) to establish baseline
    python evaluate_multitask.py --baseline --max_samples 100 --output baseline_results.json
    
    # FINE-TUNED: Evaluate checkpoint (all tasks on validation set)
    python evaluate_multitask.py --model_path checkpoints/qwen-security-expert-multitask/checkpoint-epoch1-step2000
    
    # Quick test (100 samples per task)
    python evaluate_multitask.py --model_path checkpoints/.../checkpoint-... --max_samples 100
    
    # Custom test data
    python evaluate_multitask.py --model_path checkpoints/.../... --test_data custom_test.jsonl
    
    # Compare baseline vs fine-tuned
    python evaluate_multitask.py --baseline --max_samples 100 --output baseline.json
    python evaluate_multitask.py --model_path checkpoints/.../... --max_samples 100 --output finetuned.json
    # Then compare baseline.json vs finetuned.json
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import argparse
from dataclasses import dataclass
import re
from collections import defaultdict


@dataclass
class TaskMetrics:
    """Container for task-specific metrics"""
    # CLASSIFY metrics
    classify_accuracy: float = 0.0
    classify_precision: float = 0.0
    classify_recall: float = 0.0
    classify_f1: float = 0.0
    classify_samples: int = 0
    
    # CVE_LOOKUP metrics
    cve_recall_at_1: float = 0.0
    cve_recall_at_5: float = 0.0
    cve_precision: float = 0.0
    cve_samples: int = 0
    
    # CODE_ANALYSIS metrics
    code_analysis_quality: float = 0.0
    code_analysis_cve_accuracy: float = 0.0
    code_analysis_samples: int = 0
    
    # FIX metrics
    fix_quality: float = 0.0
    fix_has_before_after: float = 0.0
    fix_samples: int = 0


class MultiTaskEvaluator:
    """Evaluates fine-tuned Qwen on all 4 security tasks"""
    
    def __init__(
        self,
        model_path: str,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        use_lora: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        print(f"Loading model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            padding_side='right'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if use_lora:
            # Load base model
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(base, model_path)
        else:
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
        messages: List[Dict[str, str]],
        max_length: int = 512,
        temperature: float = 0.1
    ) -> str:
        """Generate response for a conversational input"""
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def evaluate_classify(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate CLASSIFY task - Binary code vulnerability classification"""
        print(f"\n[CLASSIFY] Evaluating {len(test_data)} samples...")
        
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        
        for sample in tqdm(test_data, desc="CLASSIFY"):
            messages = sample['messages']
            expected_label = messages[-1]['content'].strip()
            
            # Generate prediction
            response = self.generate_response(messages[:-1], max_length=10, temperature=0)
            
            # Extract first digit
            pred_match = re.search(r'[01]', response)
            if not pred_match:
                continue
            
            predicted = pred_match.group(0)
            
            # Count metrics
            if expected_label == '1' and predicted == '1':
                true_pos += 1
            elif expected_label == '0' and predicted == '0':
                true_neg += 1
            elif expected_label == '0' and predicted == '1':
                false_pos += 1
            elif expected_label == '1' and predicted == '0':
                false_neg += 1
        
        # Calculate metrics
        accuracy = (true_pos + true_neg) / len(test_data) if len(test_data) > 0 else 0
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total': len(test_data)
        }
    
    def evaluate_cve_lookup(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate CVE_LOOKUP task - CVE identification"""
        print(f"\n[CVE_LOOKUP] Evaluating {len(test_data)} samples...")
        
        recall_at_1 = []
        recall_at_5 = []
        precision_scores = []
        
        for sample in tqdm(test_data[:500], desc="CVE_LOOKUP"):  # Limit for speed
            messages = sample['messages']
            ground_truth = sample['metadata'].get('cve_id', '')
            
            if not ground_truth or ground_truth == 'NaN':
                continue
            
            # Generate prediction
            response = self.generate_response(messages[:-1], max_length=256)
            
            # Extract CVE IDs from response
            predicted_cves = re.findall(r'CVE-\d{4}-\d{4,7}', response)
            
            if not predicted_cves:
                recall_at_1.append(0)
                recall_at_5.append(0)
                continue
            
            # Recall@1
            recall_at_1.append(1 if ground_truth in predicted_cves[:1] else 0)
            
            # Recall@5
            recall_at_5.append(1 if ground_truth in predicted_cves[:5] else 0)
            
            # Precision (1 if any predicted CVE is correct)
            precision_scores.append(1 if ground_truth in predicted_cves else 0)
        
        return {
            'recall@1': sum(recall_at_1) / len(recall_at_1) if recall_at_1 else 0,
            'recall@5': sum(recall_at_5) / len(recall_at_5) if recall_at_5 else 0,
            'precision': sum(precision_scores) / len(precision_scores) if precision_scores else 0,
            'total': len(recall_at_1)
        }
    
    def evaluate_code_analysis(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate CODE_ANALYSIS task - Deep vulnerability analysis"""
        print(f"\n[CODE_ANALYSIS] Evaluating {len(test_data)} samples...")
        
        quality_scores = []
        cve_correct = []
        
        for sample in tqdm(test_data, desc="CODE_ANALYSIS"):
            messages = sample['messages']
            ground_truth_cve = sample['metadata'].get('cve_id', '')
            
            # Generate prediction
            response = self.generate_response(messages[:-1], max_length=256)
            
            # Quality scoring
            score = 0
            if 'vulnerability' in response.lower() or 'vuln' in response.lower():
                score += 0.25
            if 'cve' in response.lower() or 'CVE-' in response:
                score += 0.25
            if any(word in response.lower() for word in ['critical', 'high', 'medium', 'low', 'severity']):
                score += 0.25
            if any(word in response.lower() for word in ['impact', 'exploit', 'attack', 'fix', 'patch']):
                score += 0.25
            
            quality_scores.append(score)
            
            # CVE accuracy (if ground truth has CVE)
            if ground_truth_cve and ground_truth_cve != 'NaN':
                predicted_cves = re.findall(r'CVE-\d{4}-\d{4,7}', response)
                cve_correct.append(1 if ground_truth_cve in predicted_cves else 0)
        
        return {
            'quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'cve_accuracy': sum(cve_correct) / len(cve_correct) if cve_correct else 0,
            'total': len(test_data)
        }
    
    def evaluate_fix(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate FIX task - Code remediation"""
        print(f"\n[FIX] Evaluating {len(test_data)} samples...")
        
        quality_scores = []
        has_before_after = []
        
        for sample in tqdm(test_data[:200], desc="FIX"):  # Limit for speed
            messages = sample['messages']
            
            # Generate prediction
            response = self.generate_response(messages[:-1], max_length=512)
            
            # Quality scoring
            score = 0
            
            # Check for code blocks
            code_blocks = len(re.findall(r'```|`[^`]+`', response))
            if code_blocks >= 2:
                score += 0.4
                has_before_after.append(1)
            else:
                has_before_after.append(0)
            
            # Check for keywords
            if any(word in response.lower() for word in ['vulnerable', 'fixed', 'secure', 'safe']):
                score += 0.2
            if any(word in response.lower() for word in ['before', 'after', 'was', 'now']):
                score += 0.2
            if 'explanation' in response.lower() or 'why' in response.lower():
                score += 0.2
            
            quality_scores.append(score)
        
        return {
            'quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'has_before_after': sum(has_before_after) / len(has_before_after) if has_before_after else 0,
            'total': len(quality_scores)
        }


def load_test_data(test_jsonl: str, max_samples: Optional[int] = None) -> Dict[str, List[Dict]]:
    """Load test data and split by task type"""
    task_data = defaultdict(list)
    
    with open(test_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line)
            task_type = data.get('task_type')
            if task_type:
                task_data[task_type].append(data)
    
    # Limit samples if specified
    if max_samples:
        for task in task_data:
            if len(task_data[task]) > max_samples:
                task_data[task] = task_data[task][:max_samples]
    
    return dict(task_data)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen Multi-Task Security Expert')
    parser.add_argument('--model_path', type=str,
                       help='Path to fine-tuned model checkpoint (LoRA adapters). Omit for baseline.')
    parser.add_argument('--baseline', action='store_true',
                       help='Evaluate raw base model (no fine-tuning) to establish baseline')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='Base model name')
    parser.add_argument('--test_data', type=str,
                       default='vuln_data/multitask_data/val.jsonl',
                       help='Path to test data (JSONL with all tasks)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples per task (for quick testing)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--use_lora', action='store_true', default=True,
                       help='Load as LoRA model (default: True for checkpoints)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.baseline and not args.model_path:
        parser.error("Either --model_path or --baseline must be specified")
    
    if args.baseline and args.model_path:
        print("Warning: Both --baseline and --model_path specified. Using baseline mode.")
        args.model_path = None
    
    # Initialize evaluator
    if args.baseline:
        print("\n" + "="*70)
        print("BASELINE EVALUATION MODE")
        print("="*70)
        print(f"Evaluating raw {args.base_model} (no fine-tuning)")
        print("This establishes baseline performance before training.")
        print("="*70 + "\n")
        
        evaluator = MultiTaskEvaluator(
            args.base_model,  # Use base model directly
            base_model=args.base_model,
            use_lora=False  # No LoRA for baseline
        )
    else:
        print("\n" + "="*70)
        print("FINE-TUNED MODEL EVALUATION")
        print("="*70)
        print(f"Base model: {args.base_model}")
        print(f"Checkpoint: {args.model_path}")
        print(f"LoRA mode: {args.use_lora}")
        print("="*70 + "\n")
        
        evaluator = MultiTaskEvaluator(
            args.model_path,
            base_model=args.base_model,
            use_lora=args.use_lora
        )
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    task_data = load_test_data(args.test_data, args.max_samples)
    
    print(f"\nTest data loaded:")
    for task, samples in task_data.items():
        print(f"  {task:20s}: {len(samples):6,} samples")
    
    # Run evaluations
    results = {}
    
    # CLASSIFY
    if 'CLASSIFY' in task_data and len(task_data['CLASSIFY']) > 0:
        classify_metrics = evaluator.evaluate_classify(task_data['CLASSIFY'])
        results['CLASSIFY'] = classify_metrics
    
    # CVE_LOOKUP
    if 'CVE_LOOKUP' in task_data and len(task_data['CVE_LOOKUP']) > 0:
        cve_metrics = evaluator.evaluate_cve_lookup(task_data['CVE_LOOKUP'])
        results['CVE_LOOKUP'] = cve_metrics
    
    # CODE_ANALYSIS
    if 'CODE_ANALYSIS' in task_data and len(task_data['CODE_ANALYSIS']) > 0:
        code_metrics = evaluator.evaluate_code_analysis(task_data['CODE_ANALYSIS'])
        results['CODE_ANALYSIS'] = code_metrics
    
    # FIX
    if 'FIX' in task_data and len(task_data['FIX']) > 0:
        fix_metrics = evaluator.evaluate_fix(task_data['FIX'])
        results['FIX'] = fix_metrics
    
    # Print results
    print("\n" + "="*70)
    if args.baseline:
        print("BASELINE RESULTS (Raw Qwen - No Fine-Tuning)")
    else:
        print("FINE-TUNED MODEL RESULTS")
    print("="*70)
    
    for task, metrics in results.items():
        print(f"\n[{task}]")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric:25s}: {value:.4f}")
            else:
                print(f"  {metric:25s}: {value}")
    
    # Add metadata to results
    results['_metadata'] = {
        'mode': 'baseline' if args.baseline else 'fine-tuned',
        'model': args.base_model,
        'checkpoint': args.model_path if not args.baseline else None,
        'test_data': args.test_data
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
