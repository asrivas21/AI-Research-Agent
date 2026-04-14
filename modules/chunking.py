"""
Text Chunking Module
Implements semantic-aware text splitting for RAG with configurable overlap.
"""

import re
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


class DocumentChunk:
    """Represents a chunk of text with metadata."""
    
    def __init__(self, text: str, chunk_id: str, source_title: str, 
                 source_section: str = "Unknown", page_number: int = 1,
                 token_count: int = 0):
        self.text = text
        self.chunk_id = chunk_id
        self.source_title = source_title
        self.source_section = source_section
        self.page_number = page_number
        self.token_count = token_count
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary for storage."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "source_title": self.source_title,
            "source_section": self.source_section,
            "page_number": self.page_number,
            "token_count": self.token_count
        }


class SemanticChunker:
    """
    Breaks documents into semantic chunks suitable for embedding.
    Targets 500-1000 tokens per chunk with ~100 token overlap.
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100, 
                 model: str = "gpt2"):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target tokens per chunk (default 800)
            chunk_overlap: Token overlap between chunks (default 100)
            model: Tokenizer model for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 tokenizer
        
        # Multi-level splitting: paragraph → sentence → character
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            is_separator_regex=False
        )
    
    def chunk_document(self, text: str, document_title: str,
                      extract_sections: bool = True) -> List[DocumentChunk]:
        """
        Chunk a full document into semantic pieces.
        
        Args:
            text: Full document text (may include page markers and section headers)
            document_title: Title of the source document
            extract_sections: Whether to detect section headers
        
        Returns:
            List of DocumentChunk objects with metadata
        """
        chunks = []
        
        # Extract sections if requested
        sections = self._extract_sections(text) if extract_sections else [
            {"section": "Full Document", "text": text, "page": 1}
        ]
        
        chunk_counter = 0
        
        for section_info in sections:
            section_text = section_info["text"]
            section_name = section_info["section"]
            page_num = section_info.get("page", 1)
            
            # Split section into chunks
            split_texts = self.splitter.split_text(section_text)
            
            for idx, chunk_text in enumerate(split_texts):
                chunk_id = f"{document_title.replace(' ', '_')}_{chunk_counter:04d}"
                token_count = self._count_tokens(chunk_text)
                
                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source_title=document_title,
                    source_section=section_name,
                    page_number=page_num,
                    token_count=token_count
                )
                chunks.append(chunk)
                chunk_counter += 1
        
        return chunks
    
    def _extract_sections(self, text: str) -> List[Dict]:
        """
        Extract sections from document using common header patterns.
        
        Returns:
            List of dicts with "section", "text", and "page" keys
        """
        sections = []
        
        # Split on common markdown/paper headers
        # Pattern: lines starting with #, ##,###, etc. or numbered sections (1., 1.1, etc.)
        header_pattern = r'(?:^#+\s+|^[\d\.]+\s+)'
        
        # Also split on page markers (--- Page N ---)
        text_with_markers = text
        current_page = 1
        page_pattern = r'--- Page (\d+) ---'
        page_matches = list(re.finditer(page_pattern, text))
        
        if not page_matches:
            # No page markers, extract by headers only
            parts = re.split(header_pattern, text, flags=re.MULTILINE)
            
            if len(parts) == 1:
                # No sections detected, treat as single section
                sections.append({
                    "section": "Full Document",
                    "text": text,
                    "page": 1
                })
            else:
                # First part is pre-header text (intro)
                if parts[0].strip():
                    sections.append({
                        "section": "Introduction",
                        "text": parts[0],
                        "page": 1
                    })
                
                # Remaining parts are sections
                for i in range(1, len(parts), 2):
                    if i < len(parts):
                        section_name = parts[i].strip() if i < len(parts) else "Unknown"
                        section_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
                        
                        if section_text:
                            sections.append({
                                "section": section_name,
                                "text": section_text,
                                "page": 1
                            })
        else:
            # We have page markers, extract by page + headers
            for page_idx, page_match in enumerate(page_matches):
                page_num = int(page_match.group(1))
                start_pos = page_match.end()
                end_pos = page_matches[page_idx + 1].start() if page_idx + 1 < len(page_matches) else len(text)
                
                page_text = text[start_pos:end_pos].strip()
                
                if page_text:
                    sections.append({
                        "section": f"Page {page_num}",
                        "text": page_text,
                        "page": page_num
                    })
        
        return sections if sections else [{
            "section": "Full Document",
            "text": text,
            "page": 1
        }]
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using GPT tokenizer."""
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback: rough estimate (1 token ≈ 4 chars)
            return len(text) // 4


class ChunkingPipeline:
    """High-level pipeline for chunking multiple documents."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunker = SemanticChunker(chunk_size, chunk_overlap)
        self.all_chunks = []
    
    def process_document(self, text: str, document_title: str) -> List[DocumentChunk]:
        """Process a single document and add chunks to pipeline."""
        chunks = self.chunker.chunk_document(text, document_title)
        self.all_chunks.extend(chunks)
        return chunks
    
    def get_stats(self) -> Dict:
        """Return statistics about chunked documents."""
        total_tokens = sum(c.token_count for c in self.all_chunks)
        avg_tokens = total_tokens / len(self.all_chunks) if self.all_chunks else 0
        
        return {
            "total_chunks": len(self.all_chunks),
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": avg_tokens,
            "unique_documents": len(set(c.source_title for c in self.all_chunks))
        }
