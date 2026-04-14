"""
Retrieval Module
High-level RAG pipeline combining PDF ingestion, chunking, and vector search.
"""

from typing import List, Dict, Optional, Tuple
from modules.pdf_ingestion import PDFIngestionManager, PDFDocument
from modules.chunking import ChunkingPipeline, DocumentChunk
from modules.vector_db import VectorStore, RAGRetriever


class RAGPipeline:
    """
    Complete RAG pipeline: ingest PDFs → chunk → embed → retrieve → cite.
    """
    
    def __init__(self, pdf_dir: str = "data/pdfs", 
                 db_dir: str = "data/chroma_db",
                 chunk_size: int = 800,
                 chunk_overlap: int = 100):
        """
        Initialize RAG pipeline.
        
        Args:
            pdf_dir: Directory for PDF storage
            db_dir: Directory for Chroma persistence
            chunk_size: Tokens per chunk
            chunk_overlap: Token overlap
        """
        self.pdf_manager = PDFIngestionManager(storage_dir=pdf_dir)
        self.chunking_pipeline = ChunkingPipeline(chunk_size, chunk_overlap)
        self.vector_store = VectorStore(persist_dir=db_dir)
        self.retriever = RAGRetriever(self.vector_store)
    
    def ingest_pdf(self, file_path: str, title: Optional[str] = None) -> PDFDocument:
        """
        Ingest a local PDF: upload → chunk → embed → store.
        
        Args:
            file_path: Path to PDF file
            title: Optional document title
        
        Returns:
            PDFDocument with ingestion metadata
        """
        # Upload and store PDF
        doc = self.pdf_manager.upload_pdf(file_path, title)
        
        # Extract text and chunk
        text = doc.extract_text()
        chunks = self.chunking_pipeline.process_document(text, doc.title)
        
        # Add to vector store
        self.vector_store.add_chunks(chunks)
        
        print(f"✓ Ingested '{doc.title}': {len(chunks)} chunks, "
              f"{sum(c.token_count for c in chunks)} tokens total")
        
        return doc
    
    def ingest_arxiv(self, arxiv_id: str, title: Optional[str] = None) -> PDFDocument:
        """
        Ingest a paper from arXiv: fetch → chunk → embed → store.
        
        Args:
            arxiv_id: arXiv ID (e.g., "2301.12345")
            title: Optional document title
        
        Returns:
            PDFDocument with ingestion metadata
        """
        # Fetch from arXiv
        doc = self.pdf_manager.fetch_arxiv_paper(arxiv_id, title)
        
        # Extract text and chunk
        text = doc.extract_text()
        chunks = self.chunking_pipeline.process_document(text, doc.title)
        
        # Add to vector store
        self.vector_store.add_chunks(chunks)
        
        print(f"✓ Ingested arXiv paper '{doc.title}': {len(chunks)} chunks, "
              f"{sum(c.token_count for c in chunks)} tokens total")
        
        return doc
    
    def search_arxiv(self, query: str, max_results: int = 5) -> List[Tuple]:
        """
        Search arXiv for papers.
        
        Returns:
            List of (arxiv_id, title, authors)
        """
        return self.pdf_manager.search_arxiv(query, max_results)
    
    def retrieve(self, query: str, k: int = 5) -> Dict:
        """
        Retrieve context for a query with citations.
        
        Args:
            query: User query
            k: Number of chunks to retrieve
        
        Returns:
            Dict with "context", "chunks", "citations", etc.
        """
        return self.retriever.retrieve_for_query(query, k=k, include_citations=True)
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        chunking_stats = self.chunking_pipeline.get_stats()
        vector_stats = self.vector_store.get_stats()
        
        return {
            **chunking_stats,
            **vector_stats,
            "ingested_documents": len(self.pdf_manager.documents)
        }
    
    def list_documents(self) -> List[Dict]:
        """List all ingested documents."""
        docs = self.pdf_manager.list_documents()
        return [
            {
                "title": doc.title,
                "source": doc.source,
                "url": doc.url,
                "metadata": doc.get_metadata()
            }
            for doc in docs
        ]
