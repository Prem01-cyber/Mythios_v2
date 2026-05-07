#!/usr/bin/env python3
"""
CVE Vector Store - RAG Knowledge Base

Builds a vector database of CVEs for retrieval-augmented generation.
Uses ChromaDB for fast semantic search over 200K+ CVEs.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CVEDocument:
    """CVE document for vector storage"""
    cve_id: str
    description: str
    cvss_score: float
    cvss_vector: str
    affected_products: List[str]
    published_date: str
    cwe_ids: List[str]
    references: List[str]
    
    def to_text(self) -> str:
        """Convert to searchable text representation"""
        products_str = ", ".join(self.affected_products[:10])  # Limit to avoid too long
        cwe_str = ", ".join(self.cwe_ids)
        
        return f"""CVE ID: {self.cve_id}

Description: {self.description}

Severity: CVSS {self.cvss_score}/10.0 ({self.cvss_vector})

Affected Products: {products_str}

Weakness Types (CWE): {cwe_str}

Published: {self.published_date}

References: {len(self.references)} available"""
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for storage"""
        return {
            "cve_id": self.cve_id,
            "cvss_score": self.cvss_score,
            "cvss_vector": self.cvss_vector,
            "affected_products": json.dumps(self.affected_products[:50]),  # Limit size
            "published_date": self.published_date,
            "cwe_ids": json.dumps(self.cwe_ids),
            "num_references": len(self.references)
        }


class CVEVectorStore:
    """
    Vector database for CVE retrieval
    
    Uses ChromaDB for efficient semantic search over CVE data.
    """
    
    def __init__(
        self,
        db_path: str = "./cve_vector_db",
        collection_name: str = "cves",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store
        
        Args:
            db_path: Path to store database
            collection_name: Name of the collection
            embedding_model: Sentence transformer model for embeddings
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "CVE vulnerability database"}
        )
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        logger.info(f"✓ CVE Vector Store initialized")
        logger.info(f"  Database: {self.db_path}")
        logger.info(f"  Collection: {collection_name}")
        logger.info(f"  Current CVE count: {self.collection.count()}")
    
    def ingest_nvd_csv(self, csv_path: str, batch_size: int = 1000):
        """
        Ingest CVEs from NVD CSV file (actual format in vuln_data)
        
        Args:
            csv_path: Path to NVD CSV file
            batch_size: Batch size for ingestion
        """
        import pandas as pd
        
        logger.info(f"Ingesting CVEs from CSV: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} CVEs from CSV")
        
        # Convert to CVEDocuments
        cves = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing CVEs"):
            try:
                cve_doc = self._parse_csv_row(row)
                if cve_doc:
                    cves.append(cve_doc)
            except Exception as e:
                logger.warning(f"Failed to parse CVE {row.get('cve_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Parsed {len(cves)} CVEs successfully")
        
        # Ingest in batches
        for i in tqdm(range(0, len(cves), batch_size), desc="Ingesting batches"):
            batch = cves[i:i+batch_size]
            self._ingest_batch(batch)
        
        logger.info(f"✓ Ingestion complete. Total CVEs: {self.collection.count()}")
    
    def _parse_csv_row(self, row) -> Optional[CVEDocument]:
        """
        Parse CSV row from actual NVD data format
        
        CSV columns: cve_id, description, published_date, cvss_score, severity,
                    attack_vector, attack_complexity, cwe_id, cwe_list, 
                    affected_products, is_windows, exploitability
        """
        cve_id = row.get('cve_id', '')
        if not cve_id or pd.isna(cve_id):
            return None
        
        description = row.get('description', '')
        if pd.isna(description):
            description = ''
        
        # CVSS score
        cvss_score = row.get('cvss_score', 0.0)
        if pd.isna(cvss_score):
            cvss_score = 0.0
        
        # Build CVSS vector from available data
        attack_vector = row.get('attack_vector', 'UNKNOWN')
        attack_complexity = row.get('attack_complexity', 'UNKNOWN')
        severity = row.get('severity', 'UNKNOWN')
        cvss_vector = f"AV:{attack_vector}/AC:{attack_complexity}/S:{severity}"
        
        # Affected products
        affected_str = row.get('affected_products', '')
        if pd.isna(affected_str) or not affected_str:
            affected = []
        else:
            affected = [p.strip() for p in str(affected_str).split(';') if p.strip()]
        
        # Published date
        published = row.get('published_date', '')
        if pd.isna(published):
            published = ''
        
        # CWE IDs
        cwe_id = row.get('cwe_id', '')
        cwe_list = row.get('cwe_list', '')
        cwe_ids = []
        if not pd.isna(cwe_id) and cwe_id:
            cwe_ids.append(str(cwe_id))
        if not pd.isna(cwe_list) and cwe_list:
            cwe_ids.extend([c.strip() for c in str(cwe_list).split(',') if c.strip()])
        
        # References (not in CSV, use empty list)
        references = []
        
        return CVEDocument(
            cve_id=str(cve_id),
            description=str(description),
            cvss_score=float(cvss_score),
            cvss_vector=cvss_vector,
            affected_products=affected,
            published_date=str(published),
            cwe_ids=cwe_ids,
            references=references
        )
    
    def _ingest_batch(self, batch: List[CVEDocument]):
        """Ingest a batch of CVEs"""
        
        # Prepare data
        ids = [cve.cve_id for cve in batch]
        documents = [cve.to_text() for cve in batch]
        metadatas = [cve.to_metadata() for cve in batch]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            documents,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant CVEs for a query
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filters (e.g., {"cvss_score": {"$gt": 7.0}})
        
        Returns:
            List of dicts with keys: id, document, metadata, distance
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def get_by_id(self, cve_id: str) -> Optional[Dict[str, Any]]:
        """Get specific CVE by ID"""
        try:
            result = self.collection.get(
                ids=[cve_id],
                include=["documents", "metadatas"]
            )
            
            if result['ids']:
                return {
                    "id": result['ids'][0],
                    "document": result['documents'][0],
                    "metadata": result['metadatas'][0]
                }
        except:
            pass
        
        return None
    
    def search_by_product(
        self,
        product_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search CVEs affecting a specific product"""
        query = f"vulnerabilities affecting {product_name}"
        return self.retrieve(query, top_k=top_k)
    
    def search_high_severity(
        self,
        min_cvss: float = 7.0,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search high severity CVEs"""
        # Note: ChromaDB metadata filtering syntax
        filter_dict = {"cvss_score": {"$gte": min_cvss}}
        return self.retrieve("high severity vulnerabilities", top_k=top_k, filter_dict=filter_dict)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        count = self.collection.count()
        
        return {
            "total_cves": count,
            "database_path": str(self.db_path),
            "collection_name": self.collection.name
        }


def build_cve_database():
    """
    Build CVE vector database from actual NVD CSV data
    
    Run this once to populate the database.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Build CVE vector database")
    parser.add_argument('--nvd_csv', type=str,
                       default='vuln_data/processed/nvd_cves.csv',
                       help='Path to NVD CSV file (default: vuln_data/processed/nvd_cves.csv)')
    parser.add_argument('--db_path', type=str, default='./cve_vector_db',
                       help='Path to store vector database')
    parser.add_argument('--sample', type=float, default=None,
                       help='Sample fraction for testing (e.g., 0.01 for 1%%)')
    
    args = parser.parse_args()
    
    # Initialize store
    print("Initializing vector store...")
    store = CVEVectorStore(db_path=args.db_path)
    
    # Check if data already exists
    existing_count = store.collection.count()
    if existing_count > 0:
        print(f"\n⚠️  Database already contains {existing_count:,} CVEs")
        response = input("Do you want to clear and rebuild? (yes/no): ")
        if response.lower() == 'yes':
            store.client.delete_collection(store.collection.name)
            store.collection = store.client.create_collection(
                name="cves",
                metadata={"description": "CVE vulnerability database"}
            )
            print("✓ Database cleared")
        else:
            print("Keeping existing data. Exiting.")
            return
    
    # Load and sample if requested
    if args.sample:
        print(f"\n📊 Sampling {args.sample*100}% of data for testing...")
        df = pd.read_csv(args.nvd_csv)
        df = df.sample(frac=args.sample, random_state=42)
        temp_path = '/tmp/nvd_sample.csv'
        df.to_csv(temp_path, index=False)
        args.nvd_csv = temp_path
        print(f"✓ Sampled {len(df):,} CVEs")
    
    # Ingest data
    store.ingest_nvd_csv(args.nvd_csv)
    
    # Show stats
    stats = store.get_stats()
    print(f"\n{'='*60}")
    print("CVE Database Built Successfully!")
    print(f"{'='*60}")
    print(f"Total CVEs: {stats['total_cves']:,}")
    print(f"Database: {stats['database_path']}")
    print(f"{'='*60}")
    
    # Test retrieval
    print("\n🔍 Testing retrieval...")
    test_queries = [
        ("Apache httpd path traversal", "Apache vulnerabilities"),
        ("Windows SMB vulnerabilities", "EternalBlue"),
        ("buffer overflow", "Memory corruption")
    ]
    
    for query, desc in test_queries:
        print(f"\nQuery: {query} ({desc})")
        results = store.retrieve(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                cvss = result['metadata'].get('cvss_score', 'N/A')
                print(f"  {i}. {result['id']} (CVSS {cvss}, distance: {result.get('distance', 0):.3f})")
        else:
            print("  No results found")
    
    print(f"\n{'='*60}")
    print("✅ Database ready for RAG system!")
    print(f"{'='*60}")


if __name__ == '__main__':
    build_cve_database()
