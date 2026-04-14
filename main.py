from dotenv import load_dotenv
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, ValidationError
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool, wiki_tool, save_tool
from modules.retrieval import RAGPipeline

load_dotenv()

# ---------- Schema ----------
class EvidenceItem(BaseModel):
    title: Optional[str] = Field(None, description="Title of the source")
    url: str = Field(..., description="Direct link to the source")
    published_date: Optional[str] = Field(None, description="ISO date or YYYY-MM-DD when available")
    snippet: Optional[str] = Field(None, description="1–3 sentence extract")

class Argument(BaseModel):
    claim: str = Field(..., description="Key claim or point")
    evidence: Optional[str] = Field(None, description="1–3 sentence support summary")
    sources: List[str] = Field(default_factory=list, description="URLs supporting the claim")

class ResearchResponse(BaseModel):
    research_question: str = Field(..., description="The concrete research question")
    key_arguments: List[Argument] = Field(default_factory=list, description="Bulleted, source-backed arguments")
    synthesis: str = Field(..., description="Neutral synthesis (6–10 sentences)")
    citations: List[EvidenceItem] = Field(default_factory=list, description="High-quality sources with metadata")
    further_readings: List[str] = Field(default_factory=list, description="Seminal works, surveys, or textbooks")
    tools_used: List[str] = Field(default_factory=list, description="Names of tools actually used")

# ---------- RAG Pipeline ----------
# Initialize RAG for grounding answers in academic papers
rag_pipeline = RAGPipeline(
    pdf_dir="data/pdfs",
    db_dir="data/chroma_db",
    chunk_size=800,
    chunk_overlap=100
)

# ---------- LLM + parser ----------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# ---------- Prompt ----------
# Enhanced system prompt for RAG-grounded responses
system_prompt = """You are an academic research assistant with access to a curated knowledge base of research papers.
You have TWO sources of information:
1. RETRIEVED PAPERS: Summaries and excerpts from academic papers in your knowledge base
2. WEB TOOLS: Real-time web search and Wikipedia

INSTRUCTIONS:
- ALWAYS: (1) clarify and restate the research question, 
- (2) produce 3–6 key, source-backed arguments citing both papers AND web sources,
- (3) write a neutral synthesis (6–10 sentences),
- (4) include high-quality citations with URLs, paper titles, and dates if available,
- (5) suggest 3–6 further readings.

CRITICAL: When retrieved papers are provided, prioritize them as primary sources. 
Only use web search if papers don't cover the topic or for recent developments.

Your FINAL message must be ONLY valid JSON matching the schema below. Do NOT include backticks, markdown, or any extra text.

{format_instructions}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# ---------- Agent ----------
tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    max_iterations=6,
)

# ---------- RAG Helper Functions ----------
def ingest_document(file_path: str, title: Optional[str] = None) -> Dict:
    """
    Ingest a PDF into the RAG system.
    
    Args:
        file_path: Path to PDF file
        title: Optional document title
    
    Returns:
        Document metadata
    """
    try:
        doc = rag_pipeline.ingest_pdf(file_path, title)
        stats = rag_pipeline.get_stats()
        return {
            "status": "success",
            "document": doc.title,
            "chunks_added": stats["total_chunks"],
            "total_tokens": stats["total_tokens"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def ingest_from_arxiv(arxiv_id: str, title: Optional[str] = None) -> Dict:
    """
    Fetch and ingest a paper from arXiv.
    
    Args:
        arxiv_id: arXiv ID (e.g., "2301.12345")
        title: Optional document title
    
    Returns:
        Document metadata
    """
    try:
        doc = rag_pipeline.ingest_arxiv(arxiv_id, title)
        stats = rag_pipeline.get_stats()
        return {
            "status": "success",
            "document": doc.title,
            "chunks_added": stats["total_chunks"],
            "arxiv_url": doc.url
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def search_documents(query: str, k: int = 5) -> Dict:
    """
    Search ingested documents for relevant content.
    
    Args:
        query: Search query
        k: Number of chunks to retrieve
    
    Returns:
        Retrieved chunks with citations
    """
    try:
        results = rag_pipeline.retrieve(query, k=k)
        return {
            "status": "success",
            "num_results": len(results["chunks"]),
            "avg_relevance": results["avg_relevance"],
            "context": results["context"],
            "citations": results["citations"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ---------- Run ----------
def main(use_rag: bool = True):
    """
    Main research assistant loop.
    
    Args:
        use_rag: Whether to use RAG context (default: True)
    """
    query = input("What can I help you research? ")
    
    # Optionally retrieve RAG context
    rag_context = None
    if use_rag:
        stats = rag_pipeline.get_stats()
        if stats["total_chunks"] > 0:
            print("\n🔍 Searching knowledge base...")
            retrieval_result = rag_pipeline.retrieve(query, k=5)
            rag_context = retrieval_result
            print(f"📚 Found {rag_context['num_sources']} relevant sources")
            print(f"   Average relevance: {rag_context['avg_relevance']:.2f}")
    
    # Enhance query with RAG context if available
    enhanced_query = query
    if rag_context:
        enhanced_query = (
            f"<RETRIEVED_PAPERS>\n{rag_context['context']}\n</RETRIEVED_PAPERS>\n\n"
            f"User Query: {query}\n\n"
            f"First, synthesize insights from the papers above. If papers don't fully answer the query, use web search to supplement."
        )

    # Provide the chat_history placeholder expected by the prompt
    print("\n⏳ Processing query with RAG + tools..." if rag_context else "\n⏳ Processing query with tools...")
    raw_response = agent_executor.invoke({"query": enhanced_query, "chat_history": []})
    output_text = raw_response.get("output", "")

    def clean_json_text(text: str) -> str:
        return (
            text.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

    # 1) First attempt
    try:
        structured_response = parser.parse(output_text)
    except ValidationError:
        # 2) Try again after stripping code fences
        try:
            structured_response = parser.parse(clean_json_text(output_text))
        except ValidationError:
            # 3) Final repair pass: ask the model to convert to valid JSON
            repair_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a converter. Return ONLY valid JSON that matches this schema:\n{format_instructions}\n"
                        "No markdown, backticks, or commentary—JSON only."
                    ),
                    ("human", "Raw text to convert:\n{raw}")
                ]
            ).partial(format_instructions=parser.get_format_instructions())

            repaired = llm.invoke(
                repair_prompt.format_messages(raw=output_text)
            ).content

            structured_response = parser.parse(clean_json_text(repaired))

    # Optional: backfill research_question if missing
    payload = structured_response.model_dump()
    if not payload.get("research_question"):
        payload["research_question"] = (query or "").strip()

    # --- Enrich with actual tools used + URLs from intermediate steps ---
    steps = raw_response.get("intermediate_steps", [])
    tools_used: List[str] = []
    urls: List[str] = []

    def _collect_urls(obj):
        import re, json
        urls_local = []
        def walk(x):
            if isinstance(x, dict):
                for v in x.values(): walk(v)
            elif isinstance(x, list):
                for v in x: walk(v)
            elif isinstance(x, str):
                try:
                    j = json.loads(x)
                    walk(j)
                except Exception:
                    pass
                urls_local.extend(re.findall(r'https?://\S+', x))
        walk(obj)
        return urls_local

    for action, result in steps:
        tools_used.append(action.tool)
        urls.extend(_collect_urls(result))

    tools_used = sorted(set(tools_used))
    urls = sorted(set(urls))

    from urllib.parse import urlparse
    def _title_from_url(u: str) -> str:
        host = urlparse(u).netloc
        return host if host else "Source"

    auto_citations = [
        {"title": _title_from_url(u), "url": u, "published_date": None, "snippet": None}
        for u in urls
    ][:12]
    
    # Add RAG citations if available
    if rag_context and rag_context["citations"]:
        for citation in rag_context["citations"]:
            auto_citations.append({
                "title": citation["title"],
                "url": None,  # Papers don't have URLs in this format yet
                "published_date": None,
                "snippet": citation["section"]
            })

    if not payload.get("tools_used"):
        payload["tools_used"] = tools_used
        if rag_context:
            payload["tools_used"].insert(0, "rag_retrieval")
    
    if not payload.get("citations"):
        payload["citations"] = auto_citations

    # Ensure each Argument.sources only has valid URLs
    import re as _re
    if payload.get("key_arguments"):
        for arg in payload["key_arguments"]:
            if isinstance(arg.get("sources"), list) and arg["sources"]:
                arg["sources"] = [s for s in arg["sources"] if isinstance(s, str) and _re.match(r'https?://', s)]

    print("\n=== Structured Response ===")
    print(payload)


if __name__ == "__main__":
    import sys
    use_rag = "--no-rag" not in sys.argv
    main(use_rag=use_rag)
