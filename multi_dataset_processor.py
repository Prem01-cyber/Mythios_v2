import pandas as pd
from datasets import load_dataset
from pathlib import Path
import json
from typing import Dict, List

class MultiDatasetProcessor:
    """Process and integrate multiple vulnerability datasets"""
    
    def __init__(self, base_dir="vuln_data"):
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        
    def process_d2a_dataset(self):
        """Process D2A dataset - actual vulnerable code"""
        print("\n" + "="*60)
        print("Processing D2A Dataset (Code-level Vulnerabilities)")
        print("="*60)
        
        # Load all three D2A variants
        datasets = {}
        
        for variant in ["code", "function"]:  # Skip code_trace for now (large)
            print(f"\nLoading D2A variant: {variant}")
            try:
                ds = load_dataset("claudios/D2A", variant)
                datasets[variant] = ds
                print(f"  Loaded {len(ds['train'])} training samples")
            except Exception as e:
                print(f"  Error loading {variant}: {e}")
        
        # Process 'code' variant - full files with vulnerabilities
        if 'code' in datasets:
            code_ds = datasets['code']['train']
            
            if len(code_ds) > 0:
                print(f"\nAvailable fields in code dataset: {list(code_ds[0].keys())}")
                print(f"Sample data: {code_ds[0]}")
            
            code_records = []
            for i, sample in enumerate(code_ds):
                if i % 1000 == 0:
                    print(f"  Processing code samples: {i}/{len(code_ds)}")
                
                record = {
                    'source': 'D2A_code',
                    'id': sample.get('id', ''),
                    'label': sample.get('label', 0),
                    'bug_url': sample.get('bug_url', ''),
                    'bug_function': sample.get('bug_function', ''),
                    'functions': str(sample.get('functions', '')),
                }
                code_records.append(record)
            
            df_code = pd.DataFrame(code_records)
            output_path = self.datasets_dir / "d2a_code_vulnerabilities.csv"
            df_code.to_csv(output_path, index=False)
            print(f"\nSaved {len(df_code)} code-level vulnerabilities to {output_path}")
        
        # Process 'function' variant - isolated vulnerable functions
        if 'function' in datasets:
            func_ds = datasets['function']['train']
            
            if len(func_ds) > 0:
                print(f"\nAvailable fields in function dataset: {list(func_ds[0].keys())}")
                print(f"Sample data: {func_ds[0]}")
            
            func_records = []
            for i, sample in enumerate(func_ds):
                if i % 1000 == 0:
                    print(f"  Processing function samples: {i}/{len(func_ds)}")
                
                record = {
                    'source': 'D2A_function',
                    'id': sample.get('id', ''),
                    'label': sample.get('label', 0),
                    'code': sample.get('code', ''),
                }
                func_records.append(record)
            
            df_func = pd.DataFrame(func_records)
            output_path = self.datasets_dir / "d2a_function_vulnerabilities.csv"
            df_func.to_csv(output_path, index=False)
            print(f"Saved {len(df_func)} function-level vulnerabilities to {output_path}")
        
        return datasets
    
    def process_pyresbugs_dataset(self):
        """Process PyResBugs - Python residual bugs"""
        print("\n" + "="*60)
        print("Processing PyResBugs Dataset (Python Residual Bugs)")
        print("="*60)
        
        try:
            ds = load_dataset("OSS-forge/PyResBugs")
            
            records = []
            for split in ['train', 'test']:
                if split in ds:
                    print(f"\nProcessing {split} split: {len(ds[split])} samples")
                    
                    if len(ds[split]) > 0:
                        print(f"Available fields: {list(ds[split][0].keys())}")
                        print(f"Sample data: {ds[split][0]}")
                    
                    for i, sample in enumerate(ds[split]):
                        if i % 1000 == 0:
                            print(f"  Progress: {i}/{len(ds[split])}")
                        
                        record = {
                            'source': f'PyResBugs_{split}',
                            'cve_id': sample.get('CVE-ID', ''),
                            'faulty_code': sample.get('Faulty Code', ''),
                            'fault_free_code': sample.get('Fault Free Code', ''),
                            'bug_type': sample.get('Bug_Type', ''),
                            'bug_description': sample.get('Bug_Description', ''),
                            'project': sample.get('Project', ''),
                            'test_file_path': sample.get('Test_File_Path', ''),
                            'commit_url': sample.get('Commit_URL', ''),
                            'fault_acronym': sample.get('Fault_Acronym', ''),
                            'python_version': sample.get('Python_Version', ''),
                        }
                        records.append(record)
            
            df = pd.DataFrame(records)
            output_path = self.datasets_dir / "pyresbugs_vulnerabilities.csv"
            df.to_csv(output_path, index=False)
            print(f"\nSaved {len(df)} Python residual bugs to {output_path}")
            
            return df
            
        except Exception as e:
            print(f"Error processing PyResBugs: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def create_unified_knowledge_base(self):
        """Combine all datasets into unified knowledge base"""
        print("\n" + "="*60)
        print("Creating Unified Knowledge Base")
        print("="*60)
        
        knowledge_base = {
            'cve_knowledge': {},  # From NVD
            'code_patterns': {},  # From D2A
            'residual_bugs': {},  # From PyResBugs
        }
        
        # 1. Load NVD CVEs
        nvd_path = self.base_dir / "processed" / "nvd_cves.csv"
        if nvd_path.exists():
            df_nvd = pd.read_csv(nvd_path)
            
            # Group by CWE
            for cwe in df_nvd['cwe_id'].unique():
                cwe_df = df_nvd[df_nvd['cwe_id'] == cwe]
                
                knowledge_base['cve_knowledge'][cwe] = {
                    'count': len(cwe_df),
                    'avg_cvss': float(cwe_df['cvss_score'].mean()),
                    'max_cvss': float(cwe_df['cvss_score'].max()),
                    'severity_dist': cwe_df['severity'].value_counts().to_dict(),
                    'example_descriptions': cwe_df['description'].head(5).tolist(),
                }
            
            print(f"Loaded {len(df_nvd)} CVEs, {len(knowledge_base['cve_knowledge'])} CWE types")
        
        # 2. Load D2A code patterns
        d2a_code_path = self.datasets_dir / "d2a_code_vulnerabilities.csv"
        if d2a_code_path.exists():
            df_d2a = pd.read_csv(d2a_code_path)
            
            for label in df_d2a['label'].unique():
                label_df = df_d2a[df_d2a['label'] == label]
                
                knowledge_base['code_patterns'][f'label_{label}'] = {
                    'count': len(label_df),
                    'bug_urls': label_df['bug_url'].head(5).tolist() if 'bug_url' in label_df.columns else [],
                    'example_functions': label_df['bug_function'].head(2).tolist() if 'bug_function' in label_df.columns else [],
                }
            
            print(f"Loaded {len(df_d2a)} code-level vulnerabilities, {len(knowledge_base['code_patterns'])} pattern types")
        
        # 3. Load PyResBugs
        pyres_path = self.datasets_dir / "pyresbugs_vulnerabilities.csv"
        if pyres_path.exists():
            df_pyres = pd.read_csv(pyres_path)
            
            for bug_type in df_pyres['bug_type'].unique():
                if pd.notna(bug_type):
                    type_df = df_pyres[df_pyres['bug_type'] == bug_type]
                    
                    knowledge_base['residual_bugs'][bug_type] = {
                        'count': len(type_df),
                        'projects': type_df['project'].value_counts().to_dict() if 'project' in type_df.columns else {},
                        'fault_acronyms': type_df['fault_acronym'].value_counts().to_dict() if 'fault_acronym' in type_df.columns else {},
                        'examples': type_df[['faulty_code', 'fault_free_code', 'bug_description']].head(3).to_dict('records') if all(col in type_df.columns for col in ['faulty_code', 'fault_free_code', 'bug_description']) else [],
                    }
            
            print(f"Loaded {len(df_pyres)} residual bugs, {len(knowledge_base['residual_bugs'])} bug types")
        
        # Save unified knowledge base
        kb_path = self.datasets_dir / "unified_knowledge_base.json"
        with open(kb_path, 'w') as f:
            json.dump(knowledge_base, f, indent=2)
        
        print(f"\nSaved unified knowledge base to {kb_path}")
        
        # Generate statistics
        stats = {
            'total_cves': len(df_nvd) if nvd_path.exists() else 0,
            'total_code_vulns': len(df_d2a) if d2a_code_path.exists() else 0,
            'total_residual_bugs': len(df_pyres) if pyres_path.exists() else 0,
            'cwe_coverage': len(knowledge_base['cve_knowledge']),
            'code_pattern_types': len(knowledge_base['code_patterns']),
            'residual_bug_types': len(knowledge_base['residual_bugs']),
        }
        
        stats_path = self.datasets_dir / "multi_dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved statistics to {stats_path}")
        
        return knowledge_base

if __name__ == "__main__":
    processor = MultiDatasetProcessor()
    
    # Process each dataset
    processor.process_d2a_dataset()
    processor.process_pyresbugs_dataset()
    
    # Create unified knowledge base
    kb = processor.create_unified_knowledge_base()
    
    print("\n" + "="*60)
    print("Multi-Dataset Processing Complete!")
    print("="*60)