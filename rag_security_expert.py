#!/usr/bin/env python3
"""
RAG Security Expert - Hybrid Retrieval + Generation

Combines:
1. CVE retrieval from vector database (facts)
2. LoRA-tuned security model (reasoning)

For accurate, cited security analysis.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from cve_vector_store import CVEVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    sources: List[str]  # CVE IDs cited
    retrieved_context: List[Dict[str, Any]]
    confidence: float


class RAGSecurityExpert:
    """
    Retrieval-Augmented Generation Security Expert
    
    Retrieves relevant CVEs and uses LoRA-tuned model to reason about them.
    """
    
    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_checkpoint: Optional[str] = None,
        vector_db_path: str = "./cve_vector_db",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize RAG system
        
        Args:
            base_model: Base model name
            lora_checkpoint: Path to LoRA checkpoint (if trained)
            vector_db_path: Path to CVE vector database
            device: Device to run model on
        """
        self.device = device
        
        # 1. Load CVE vector store
        logger.info("Loading CVE vector database...")
        self.vector_store = CVEVectorStore(db_path=vector_db_path)
        stats = self.vector_store.get_stats()
        logger.info(f"✓ Loaded {stats['total_cves']:,} CVEs")
        
        # 2. Load tokenizer
        logger.info(f"Loading tokenizer: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 3. Load model
        logger.info(f"Loading model: {base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # 4. Load LoRA if available
        if lora_checkpoint:
            logger.info(f"Loading LoRA adapters from {lora_checkpoint}")
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint)
            logger.info("✓ LoRA adapters loaded")
        else:
            logger.info("No LoRA checkpoint provided, using base model")
        
        self.model.eval()
        logger.info(f"✓ RAG Security Expert ready on {device}")
    
    def answer_query(
        self,
        query: str,
        task_type: Optional[str] = None,
        top_k: int = 5,
        max_tokens: int = 512,
        temperature: float = 0.1
    ) -> RAGResponse:
        """
        Answer a security query using RAG
        
        Args:
            query: User query
            task_type: Task type hint ([CLASSIFY], [CVE_LOOKUP], etc.)
            top_k: Number of CVEs to retrieve
            max_tokens: Max generation tokens
            temperature: Sampling temperature
        
        Returns:
            RAGResponse with answer and sources
        """
        # Step 1: Retrieve relevant CVEs
        retrieved_docs = self._retrieve_context(query, top_k)
        
        # Step 2: Build prompt with context
        prompt = self._build_prompt(query, retrieved_docs, task_type)
        
        # Step 3: Generate with model
        answer = self._generate(prompt, max_tokens, temperature)
        
        # Step 4: Extract sources
        sources = [doc['id'] for doc in retrieved_docs]
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_context=retrieved_docs,
            confidence=self._estimate_confidence(retrieved_docs, answer)
        )
    
    def _retrieve_context(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve relevant CVEs for query"""
        
        # Enhance query for better retrieval
        # Add security-specific terms if not present
        enhanced_query = query
        if "vulnerability" not in query.lower() and "cve" not in query.lower():
            enhanced_query = f"{query} vulnerability security"
        
        # Retrieve
        results = self.vector_store.retrieve(enhanced_query, top_k=top_k)
        
        logger.debug(f"Retrieved {len(results)} CVEs for query: {query}")
        
        return results
    
    def _build_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        task_type: Optional[str] = None
    ) -> str:
        """Build prompt with retrieved context"""
        
        # System prompt
        system_prompt = """You are an expert security analyst. Your task is to analyze vulnerabilities and provide detailed, accurate security assessments.

IMPORTANT INSTRUCTIONS:
- Use ONLY the information provided in the CONTEXT section below
- Cite CVE IDs when referencing specific vulnerabilities
- If the context doesn't contain relevant information, state that clearly
- Provide actionable recommendations when applicable
- Be precise about CVSS scores and severity levels"""
        
        # Context section
        context_section = "\n[CONTEXT - Retrieved CVE Database Information]\n"
        for i, doc in enumerate(retrieved_docs, 1):
            context_section += f"\n--- CVE {i} ---\n"
            context_section += doc['document']
            context_section += "\n"
        
        # Task-specific instructions
        task_instruction = ""
        if task_type == "[CLASSIFY]":
            task_instruction = "\n[TASK]\nAnalyze if the provided code is vulnerable (1) or safe (0). Respond with only '0' or '1'.\n"
        elif task_type == "[CVE_LOOKUP]":
            task_instruction = "\n[TASK]\nProvide detailed information about the CVEs from the context, including severity, impact, and recommendations.\n"
        elif task_type == "[CODE_ANALYSIS]":
            task_instruction = "\n[TASK]\nAnalyze the code for vulnerabilities, identify the type, severity, and potential CVEs from the context.\n"
        elif task_type == "[FIX]":
            task_instruction = "\n[TASK]\nProvide secure code remediation with before/after comparison and explanation.\n"
        
        # User query
        query_section = f"\n[QUERY]\n{query}\n"
        
        # Combine
        full_prompt = system_prompt + context_section + task_instruction + query_section
        
        return full_prompt
    
    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate response using model"""
        
        # Format as messages for Qwen
        messages = [
            {"role": "system", "content": "You are a security expert."},
            {"role": "user", "content": prompt}
        ]
        
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
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _estimate_confidence(
        self,
        retrieved_docs: List[Dict[str, Any]],
        answer: str
    ) -> float:
        """Estimate confidence in answer based on retrieval quality"""
        
        if not retrieved_docs:
            return 0.0
        
        # Check if top result is highly relevant (low distance)
        top_distance = retrieved_docs[0].get('distance', 1.0)
        
        # Lower distance = higher confidence
        # Typical distances: 0.3-0.8 for relevant, >0.8 for irrelevant
        if top_distance < 0.4:
            confidence = 0.9
        elif top_distance < 0.6:
            confidence = 0.7
        elif top_distance < 0.8:
            confidence = 0.5
        else:
            confidence = 0.3
        
        # Boost if multiple CVE IDs cited in answer
        cve_count = sum(1 for doc in retrieved_docs if doc['id'] in answer)
        if cve_count >= 2:
            confidence = min(1.0, confidence + 0.1)
        
        return confidence


def demo_rag_system():
    """Demo the RAG system"""
    
    print("="*60)
    print("RAG Security Expert Demo")
    print("="*60)
    
    # Initialize (use base model if no LoRA trained yet)
    expert = RAGSecurityExpert(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_checkpoint=None,  # Set to checkpoint path once trained
        vector_db_path="./cve_vector_db"
    )
    
    # Test queries
    queries = [
        "What vulnerabilities affect Apache httpd 2.4.49?",
        "Tell me about EternalBlue (MS17-010)",
        "What is CVE-2021-44228 and how severe is it?",
        "Security issues in OpenSSL 1.0.1"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        response = expert.answer_query(
            query=query,
            task_type="[CVE_LOOKUP]",
            top_k=3
        )
        
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nSources: {', '.join(response.sources)}")
        print(f"Confidence: {response.confidence:.1%}")


if __name__ == '__main__':
    demo_rag_system()
