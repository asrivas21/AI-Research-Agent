"""
PDF Ingestion Module
Handles local PDF uploads and arXiv paper fetching with metadata extraction.
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Tuple
import requests
import fitz  # PyMuPDF
from datetime import datetime


class PDFDocument:
    """Represents an ingested PDF document with metadata."""
    
    def __init__(self, file_path: str, title: str, source: str = "local", 
                 url: Optional[str] = None, metadata: Optional[Dict] = None):
        self.file_path = file_path
        self.title = title
        self.source = source  # "local", "arxiv", etc.
        self.url = url
        self.metadata = metadata or {}
        self.text_content = None
    
    def extract_text(self) -> str:
        """Extract full text from PDF."""
        if self.text_content:
            return self.text_content
        
        doc = fitz.open(self.file_path)
        full_text = ""
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        
        doc.close()
        self.text_content = full_text
        return full_text
    
    def get_metadata(self) -> Dict:
        """Return document metadata including extracted title."""
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "file_path": self.file_path,
            "ingestion_date": datetime.now().isoformat(),
            **self.metadata
        }


class PDFIngestionManager:
    """Manages PDF upload, arXiv fetching, and storage."""
    
    def __init__(self, storage_dir: str = "data/pdfs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.documents = {}
    
    def upload_pdf(self, file_path: str, title: Optional[str] = None) -> PDFDocument:
        """
        Upload a local PDF file.
        
        Args:
            file_path: Path to the PDF file
            title: Optional title; extracted from PDF if not provided
        
        Returns:
            PDFDocument instance
        """
        source_path = Path(file_path)
        if not source_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Generate unique filename
        filename = f"{source_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        dest_path = self.storage_dir / filename
        
        # Copy file to storage
        with open(source_path, "rb") as src:
            with open(dest_path, "wb") as dst:
                dst.write(src.read())
        
        # Extract title if not provided
        if not title:
            title = self._extract_title_from_pdf(str(dest_path))
        
        # Create document
        doc = PDFDocument(
            file_path=str(dest_path),
            title=title,
            source="local"
        )
        
        self.documents[title] = doc
        return doc
    
    def fetch_arxiv_paper(self, arxiv_id: str, title: Optional[str] = None) -> PDFDocument:
        """
        Fetch a paper from arXiv by ID (e.g., "2301.12345").
        
        Args:
            arxiv_id: arXiv paper ID (without "arXiv:" prefix)
            title: Optional title; fetched from arXiv if not provided
        
        Returns:
            PDFDocument instance
        """
        # Normalize arXiv ID (remove trailing version if present)
        arxiv_id = arxiv_id.split("v")[0]
        
        # Get paper metadata from arXiv API
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise IOError(f"Failed to fetch from arXiv: {e}")
        
        # Parse metadata
        metadata = self._parse_arxiv_response(response.text, arxiv_id)
        if not title:
            title = metadata.get("title", f"arXiv_{arxiv_id}")
        
        # Download PDF
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        filename = f"{title.replace(' ', '_')[:50]}_{arxiv_id}.pdf"
        file_path = self.storage_dir / filename
        
        try:
            pdf_response = requests.get(pdf_url, timeout=30)
            pdf_response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(pdf_response.content)
        except requests.RequestException as e:
            raise IOError(f"Failed to download PDF from arXiv: {e}")
        
        # Create document
        doc = PDFDocument(
            file_path=str(file_path),
            title=title,
            source="arxiv",
            url=f"https://arxiv.org/abs/{arxiv_id}",
            metadata=metadata
        )
        
        self.documents[title] = doc
        return doc
    
    def search_arxiv(self, query: str, max_results: int = 5) -> list:
        """
        Search arXiv for papers matching query.
        
        Args:
            query: Search query
            max_results: Max number of results
        
        Returns:
            List of (arxiv_id, title, authors) tuples
        """
        api_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise IOError(f"Failed to search arXiv: {e}")
        
        results = []
        # Simple XML parsing
        for match in re.finditer(
            r'<id>http://arxiv\.org/abs/([\d.]+)</id>.*?<title>(.*?)</title>.*?<author>.*?<name>(.*?)</name>',
            response.text,
            re.DOTALL
        ):
            arxiv_id = match.group(1)
            title = match.group(2).strip()
            author = match.group(3).strip()
            results.append((arxiv_id, title, author))
        
        return results[:max_results]
    
    def _extract_title_from_pdf(self, file_path: str, max_chars: int = 100) -> str:
        """Extract title from first page of PDF."""
        try:
            doc = fitz.open(file_path)
            first_page_text = doc[0].get_text()
            doc.close()
            
            # Extract first line as a heuristic
            lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
            title = lines[0][:max_chars] if lines else Path(file_path).stem
            return title
        except Exception:
            return Path(file_path).stem
    
    @staticmethod
    def _parse_arxiv_response(xml_response: str, arxiv_id: str) -> Dict:
        """Parse arXiv API XML response."""
        metadata = {"arxiv_id": arxiv_id}
        
        # Extract title
        title_match = re.search(r'<title>(.*?)</title>', xml_response)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        # Extract authors
        authors = re.findall(r'<name>(.*?)</name>', xml_response)
        if authors:
            metadata["authors"] = authors
        
        # Extract published date
        published_match = re.search(r'<published>([\d-]+)T', xml_response)
        if published_match:
            metadata["published_date"] = published_match.group(1)
        
        # Extract summary
        summary_match = re.search(r'<summary>(.*?)</summary>', xml_response, re.DOTALL)
        if summary_match:
            metadata["abstract"] = summary_match.group(1).strip()
        
        return metadata
    
    def list_documents(self) -> list:
        """Return list of ingested documents."""
        return list(self.documents.values())
    
    def get_document(self, title: str) -> Optional[PDFDocument]:
        """Retrieve a document by title."""
        return self.documents.get(title)
