"""
Vector Database Module
Manages embeddings and semantic search using Chroma + OpenAI embeddings.
"""

from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
import os


class VectorStore:
    """
    Wrapper around Chroma vector database for RAG.
    Handles embeddings, storage, and retrieval.
    """
    
    def __init__(self, persist_dir: str = "data/chroma_db", 
                 embedding_model: str = "text-embedding-3-small", 
                 collection_name: str = "research_papers"):
        """
        Initialize vector store.
        
        Args:
            persist_dir: Directory to persist Chroma database
            embedding_model: OpenAI embedding model to use
            collection_name: Name of the collection in Chroma
        """
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize Chroma client with persistence
        settings = Settings(
            is_persistent=True,
            persist_directory=persist_dir,
            anonymized_telemetry=False
        )
        self.client = chromadb.Client(settings)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: List) -> None:
        """
        Add document chunks to vector store.
        
        Args:
            chunks: List of DocumentChunk objects from chunking module
        """
        if not chunks:
            return
        
        # Extract chunk data
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        
        # Create metadata for each chunk
        metadatas = [
            {
                "source_title": chunk.source_title,
                "source_section": chunk.source_section,
                "page_number": chunk.page_number,
                "token_count": chunk.token_count
            }
            for chunk in chunks
        ]
        
        # Generate embeddings (OpenAI handles this)
        embeddings_list = self.embeddings.embed_documents(texts)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(self, query: str, k: int = 5, 
               similarity_threshold: float = 0.0) -> List[Dict]:
        """
        Semantic search for query in vector store.
        
        Args:
            query: Search query string
            k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of retrieved chunks with scores and metadata
        """
        # Embed the query
        query_embedding = self.embeddings.embed_query(query)
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results
        retrieved = []
        if results["documents"] and len(results["documents"]) > 0:
            for idx, doc in enumerate(results["documents"][0]):
                # Chroma returns distances (not similarities)
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][idx] if results["distances"] else 0
                similarity = 1 - distance
                
                # Filter by threshold
                if similarity >= similarity_threshold:
                    metadata = results["metadatas"][0][idx] if results["metadatas"] else {}
                    
                    retrieved.append({
                        "text": doc,
                        "score": float(similarity),
                        "source_title": metadata.get("source_title", "Unknown"),
                        "source_section": metadata.get("source_section", "Unknown"),
                        "page_number": metadata.get("page_number", 1),
                        "token_count": metadata.get("token_count", 0)
                    })
        
        return retrieved
    
    def search_with_citations(self, query: str, k: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """
        Search and return results with proper citation format.
        
        Returns:
            Tuple of (retrieved_chunks, citations)
        """
        chunks = self.search(query, k=k)
        
        # Format citations
        citations = []
        seen_sources = set()
        
        for chunk in chunks:
            source_key = (chunk["source_title"], chunk["source_section"], chunk["page_number"])
            if source_key not in seen_sources:
                citation = {
                    "title": chunk["source_title"],
                    "section": chunk["source_section"],
                    "page": chunk["page_number"],
                    "relevance_score": chunk["score"]
                }
                citations.append(citation)
                seen_sources.add(source_key)
        
        return chunks, citations
    
    def get_stats(self) -> Dict:
        """Return statistics about the vector store."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "embedding_model": self.embedding_model,
            "persist_dir": self.persist_dir
        }
    
    def delete_collection(self) -> None:
        """Delete the entire collection (useful for testing)."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.delete_collection()


class RAGRetriever:
    """
    High-level RAG retriever that combines vector search with response formatting.
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def retrieve_for_query(self, query: str, k: int = 5, 
                          include_citations: bool = True) -> Dict:
        """
        Retrieve context for a query in RAG-friendly format.
        
        Returns:
            Dict with "context", "citations", and "sources"
        """
        if include_citations:
            chunks, citations = self.vector_store.search_with_citations(query, k=k)
        else:
            chunks = self.vector_store.search(query, k=k)
            citations = []
        
        # Format context as continuous text
        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {idx}: {chunk['source_title']}, "
                f"{chunk['source_section']}, "
                f"Page {chunk['page_number']}]\n"
                f"{chunk['text']}\n"
            )
        
        context = "\n".join(context_parts)
        
        return {
            "context": context,
            "chunks": chunks,
            "citations": citations,
            "num_sources": len(citations),
            "avg_relevance": sum(c["score"] for c in chunks) / len(chunks) if chunks else 0
        }
