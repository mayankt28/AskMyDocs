import certifi
import asyncio
import os
import ssl
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from pinecone import Pinecone

from logger import (log_error, log_header, log_info, log_success, log_warning, Colors)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True, retry_min_seconds=10)

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index = pc.Index('langchain-doc-index')
vectorstore = PineconeVectorStore(index = index, embedding = embeddings)
tavily_crawl = TavilyCrawl()

async def main():
    '''Main asyn function to orchestrate the entire process'''
    log_header('DOCUMENTATION INGESTION PIPELINE')

    log_info(
        message='TavilyCrawl: Starting to crawl documentation from "https://python.langchain.com/"',
        color= Colors.PURPLE
    )

    res = tavily_crawl.invoke(
        {
            'url': 'https://python.langchain.com/',
            'max_depth': 1,
            'extract_depth': 'advanced'
        }
    )

    all_docs = [Document(page_content=result['raw_content'], metadata={'source': result['url']}) for result in res['results']]

    log_success(
        f'TavilyCrawl: Successfully crawled {len(all_docs)} URLs from the documentation site'
    )

if __name__ == "__main__":
    asyncio.run(main())