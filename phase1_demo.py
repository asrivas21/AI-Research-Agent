#!/usr/bin/env python3
"""
Phase 1 Demo: PDF Ingestion + Vector Database + RAG Retrieval
Shows the complete RAG pipeline in action.
"""

from modules.retrieval import RAGPipeline
import os


def demo_basic_workflow():
    """Demonstrate basic RAG workflow."""
    print("=" * 70)
    print("PHASE 1 DEMO: RAG Pipeline (PDF Ingestion → Chunking → Retrieval)")
    print("=" * 70)
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        pdf_dir="data/pdfs",
        db_dir="data/chroma_db",
        chunk_size=800,
        chunk_overlap=100
    )
    
    print("\n✓ RAG Pipeline initialized")
    
    # Show stats
    stats = rag.get_stats()
    print(f"\n📊 Current Pipeline Stats:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Documents ingested: {stats['ingested_documents']}")
    
    return rag


def demo_pdf_upload(rag: RAGPipeline):
    """Demo: Upload a local PDF."""
    print("\n" + "=" * 70)
    print("DEMO: Uploading Local PDF")
    print("=" * 70)
    
    # Check if there's a sample PDF to upload
    sample_pdf_path = "data/sample.pdf"  # User would provide this
    
    if os.path.exists(sample_pdf_path):
        print(f"\nUploading: {sample_pdf_path}")
        try:
            doc = rag.ingest_pdf(sample_pdf_path)
            print(f"✓ Successfully ingested: {doc.title}")
            print(f"  Source: {doc.source}")
            print(f"  Metadata: {doc.get_metadata()}")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print(f"\nℹ️  No sample PDF found at {sample_pdf_path}")
        print("   To test PDF upload, place a PDF file there and run this demo again.")


def demo_arxiv_search_and_ingest(rag: RAGPipeline):
    """Demo: Search and ingest from arXiv."""
    print("\n" + "=" * 70)
    print("DEMO: arXiv Search and Ingestion")
    print("=" * 70)
    
    query = "attention mechanisms"
    print(f"\nSearching arXiv for: '{query}'")
    
    try:
        results = rag.search_arxiv(query, max_results=3)
        if results:
            print(f"\n📄 Found {len(results)} papers:")
            for arxiv_id, title, authors in results:
                print(f"\n  • {title}")
                print(f"    arXiv ID: {arxiv_id}")
                print(f"    First author: {authors}")
            
            # Ingest the first result
            if results:
                arxiv_id, title, _ = results[0]
                print(f"\nIngesting first result: {arxiv_id}")
                try:
                    doc = rag.ingest_arxiv(arxiv_id)
                    print(f"✓ Successfully ingested from arXiv")
                    print(f"  Paper: {doc.title}")
                    print(f"  URL: {doc.url}")
                except Exception as e:
                    print(f"✗ Error ingesting: {e}")
        else:
            print("No results found")
    
    except Exception as e:
        print(f"✗ Search error: {e}")


def demo_retrieval(rag: RAGPipeline):
    """Demo: Semantic search and retrieval."""
    print("\n" + "=" * 70)
    print("DEMO: Semantic Retrieval with Citations")
    print("=" * 70)
    
    # Check if we have any documents
    stats = rag.get_stats()
    if stats['total_chunks'] == 0:
        print("\nℹ️  No documents in knowledge base yet.")
        print("   Upload a PDF or arXiv paper first to test retrieval.")
        return
    
    queries = [
        "What are the key components of transformer models?",
        "How does attention work?",
        "What is the significance of this research?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        try:
            results = rag.retrieve(query, k=3)
            
            print(f"\n   Results: {results['num_sources']} sources found")
            print(f"   Average relevance: {results['avg_relevance']:.3f}")
            
            print(f"\n   📝 Context:")
            print("   " + "-" * 60)
            for line in results['context'].split('\n')[:5]:  # Show first 5 lines
                print(f"   {line}")
            print("   ...")
            
            print(f"\n   📚 Citations:")
            for idx, citation in enumerate(results['citations'], 1):
                print(f"   [{idx}] {citation['title']} ({citation['section']}, p.{citation['page']})")
        
        except Exception as e:
            print(f"   ✗ Error: {e}")


def demo_pipeline_stats(rag: RAGPipeline):
    """Show final pipeline statistics."""
    print("\n" + "=" * 70)
    print("FINAL PIPELINE STATISTICS")
    print("=" * 70)
    
    stats = rag.get_stats()
    docs = rag.list_documents()
    
    print(f"\n📊 Chunks & Tokens:")
    print(f"   Total chunks stored: {stats['total_chunks']}")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Unique documents: {stats['unique_documents']}")
    
    if stats['total_chunks'] > 0:
        avg_chunk_size = stats['total_tokens'] / stats['total_chunks']
        print(f"   Avg tokens/chunk: {avg_chunk_size:.1f}")
    
    print(f"\n📚 Ingested Documents:")
    if docs:
        for doc in docs:
            print(f"   • {doc['title']}")
            print(f"     Source: {doc['source']}")
            if doc['url']:
                print(f"     URL: {doc['url']}")
    else:
        print("   (None yet)")
    
    print(f"\n🗃️  Storage:")
    print(f"   PDF directory: {rag.pdf_manager.storage_dir}")
    print(f"   Vector DB: {rag.vector_store.persist_dir}")


def interactive_mode(rag: RAGPipeline):
    """Simple interactive query mode."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE: Query Your Knowledge Base")
    print("=" * 70)
    
    stats = rag.get_stats()
    if stats['total_chunks'] == 0:
        print("\nℹ️  No documents available for querying.")
        return
    
    print("\n(Type 'quit' to exit)\n")
    
    while True:
        query = input("Your query: ").strip()
        if query.lower() in ['quit', 'exit']:
            break
        
        if not query:
            continue
        
        try:
            results = rag.retrieve(query, k=5)
            print(f"\n✓ Retrieved {results['num_sources']} sources (relevance: {results['avg_relevance']:.3f})")
            
            print("\n--- TOP RESULT ---")
            if results['chunks']:
                chunk = results['chunks'][0]
                print(f"Source: {chunk['source_title']} ({chunk['source_section']}, p.{chunk['page_number']})")
                print(f"Relevance: {chunk['score']:.3f}\n")
                print(chunk['text'][:500] + "...")
            
            print("\n--- ALL SOURCES ---")
            for idx, chunk in enumerate(results['chunks'], 1):
                print(f"[{idx}] {chunk['source_title']} - {chunk['source_section']} (p.{chunk['page_number']}) - {chunk['score']:.3f}")
        
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print()


def main():
    """Run the Phase 1 demo."""
    # Initialize RAG
    rag = demo_basic_workflow()
    
    # Demo workflows
    demo_pdf_upload(rag)
    demo_arxiv_search_and_ingest(rag)
    demo_retrieval(rag)
    demo_pipeline_stats(rag)
    
    # Optional: Interactive mode
    print("\n" + "=" * 70)
    response = input("\nWould you like to try interactive query mode? (y/n): ").strip().lower()
    if response == 'y':
        interactive_mode(rag)
    
    print("\n✓ Phase 1 demo complete!")
    print("\nNext steps:")
    print("  1. Upload real research papers (PDFs or arXiv)")
    print("  2. Test retrieval with domain-specific queries")
    print("  3. In main.py, use 'from main import ingest_document' to integrate into your workflow")


if __name__ == "__main__":
    main()
