#!/usr/bin/env python3

import asyncio
import os
import ssl
from chunk import Chunk
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import log_header, log_info, log_success
from vector_store_gateway import VectorStoreGateway

load_dotenv()

# Configure SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

index_name = os.environ["INDEX_NAME"]

vector_store = VectorStoreGateway()
tavily_crawl = TavilyCrawl()
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)


def chunk_document(document: Document) -> list[Chunk]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_text(document.page_content)

    print(f"Created {len(chunks)} chunks")
    source = document.metadata["source"]
    return [Chunk(content=chunk, source=source) for chunk in chunks]


def crawl_url(url: str, max_depth: int = 2) -> List[Document]:
    log_header(f"TavilyCrawl: Starting crawl in {url}")
    crawl_result: Dict[str, Any] = tavily_crawl.invoke(
        {
            "url": url,
            "max_depth": max_depth,
            "extract_depth": "basic",
            "categories": ["Documentation"],
            # "instruct": "Focus on information about AI agents and their applications.",
        }
    )

    crawl_results = crawl_result.get("results", [])

    documents: List[Document] = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in crawl_results
    ]

    log_success(f"TavilyCrawl: Completed crawl in {url}")
    log_success(f"Documents retrieved: {len(documents)}")


def extract_site_content(url: str) -> List[Document]:
    log_header(f"TavilyMap: Starting mapping {url}")

    site_map: Dict[str, Any] = tavily_map.invoke({"url": url})

    urls = site_map.get("results", [])[:20]
    log_success(f"TavilyMap: Found {len(urls)} URLs in the sitemap")
    log_info(f"URLs: {urls}")

    log_header(f"TavilyExtract: Starting extraction {url}")

    extraction_result: Dict[str, Any] = tavily_extract.invoke(
        {
            "urls": urls,
            "extract_depth": "basic",
        }
    )

    extraction_results = extraction_result.get("results", [])

    documents: List[Document] = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in extraction_results
    ]

    log_success(f"TavilyExtract: Completed extraction for {url}")
    log_success(f"Documents retrieved: {len(documents)}")

    return documents


async def main():
    source = "https://python.langchain.com"
    documents = extract_site_content(source)

    log_header(f"Chunking documents")
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document))

    log_success(f"Got {len(chunks)} chunks from {len(documents)} documents")

    log_header(f"Embedding chunks")
    embeddings = vector_store.embed_chunks([chunk.content for chunk in chunks])
    log_success(f"Embedded {len(embeddings)} chunks")

    log_header(f"Storing vectors")
    vector_store.store_vectors(index_name, chunks, embeddings)
    log_success(f"Stored {len(embeddings)} vectors in index '{index_name}'")


if __name__ == "__main__":
    log_header("Starting Ingestion Process")

    log_header("Initializing Vector Store Index")
    vector_store.init_index(index_name)

    asyncio.run(main())
