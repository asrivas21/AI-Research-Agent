#!/usr/bin/env python3
"""
Simple validation test for Phase 1 modules (without requiring OpenAI API key).
Tests module structure and basic functionality.
"""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from modules.pdf_ingestion import PDFIngestionManager, PDFDocument
        print("  ✓ pdf_ingestion imported")
    except ImportError as e:
        print(f"  ✗ pdf_ingestion failed: {e}")
        return False
    
    try:
        from modules.chunking import SemanticChunker, ChunkingPipeline, DocumentChunk
        print("  ✓ chunking imported")
    except ImportError as e:
        print(f"  ✗ chunking failed: {e}")
        return False
    
    try:
        from modules.vector_db import VectorStore, RAGRetriever
        print("  ✓ vector_db imported")
    except ImportError as e:
        print(f"  ✗ vector_db failed: {e}")
        return False
    
    try:
        from modules.retrieval import RAGPipeline
        print("  ✓ retrieval imported")
    except ImportError as e:
        print(f"  ✗ retrieval failed: {e}")
        return False
    
    return True


def test_pdf_ingestion():
    """Test PDF ingestion components."""
    print("\nTesting PDF ingestion components...")
    
    from modules.pdf_ingestion import PDFIngestionManager
    
    try:
        manager = PDFIngestionManager(storage_dir="/tmp/test_pdfs")
        print("  ✓ PDFIngestionManager instantiated")
        docs = manager.list_documents()
        print(f"  ✓ Can list documents: {len(docs)} docs")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_chunking():
    """Test chunking components."""
    print("\nTesting chunking components...")
    
    from modules.chunking import SemanticChunker, ChunkingPipeline
    
    try:
        chunker = SemanticChunker(chunk_size=800, chunk_overlap=100)
        print("  ✓ SemanticChunker instantiated")
        
        # Test on sample text
        sample_text = "This is a test document. " * 100  # Make it long enough
        chunks = chunker.chunk_document(sample_text, "Test Document")
        print(f"  ✓ Chunked sample text: {len(chunks)} chunks")
        
        pipeline = ChunkingPipeline()
        print("  ✓ ChunkingPipeline instantiated")
        stats = pipeline.get_stats()
        print(f"  ✓ Pipeline stats: {stats}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_integration():
    """Test that main.py integrates RAG properly."""
    print("\nTesting main.py integration...")
    
    try:
        # Just check that main.py can be parsed (don't run it)
        with open("main.py", "r") as f:
            code = f.read()
        
        # Check for RAG imports and initialization
        if "from modules.retrieval import RAGPipeline" in code:
            print("  ✓ RAGPipeline imported in main.py")
        else:
            print("  ✗ RAGPipeline not found in imports")
            return False
        
        if "rag_pipeline = RAGPipeline(" in code:
            print("  ✓ RAGPipeline initialized in main.py")
        else:
            print("  ✗ RAGPipeline not initialized")
            return False
        
        if "def ingest_document" in code:
            print("  ✓ ingest_document helper function exists")
        else:
            print("  ✗ ingest_document function missing")
            return False
        
        if "use_rag" in code:
            print("  ✓ RAG integration in main function")
        else:
            print("  ✗ RAG integration missing from main")
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("PHASE 1 VALIDATION TEST")
    print("=" * 70)
    
    all_pass = True
    
    if not test_imports():
        all_pass = False
    
    if not test_pdf_ingestion():
        all_pass = False
    
    if not test_chunking():
        all_pass = False
    
    if not test_main_integration():
        all_pass = False
    
    print("\n" + "=" * 70)
    if all_pass:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nPhase 1 is ready! Next steps:")
        print("1. Set your OPENAI_API_KEY environment variable")
        print("2. Run: python3 phase1_demo.py")
        print("3. Upload PDFs or arXiv papers to test RAG retrieval")
        print("4. Run: python3 main.py  (to test RAG + agent integration)")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
