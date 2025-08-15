from dotenv import load_dotenv
from typing import List, Dict, Any
import os

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    index = pc.Index('langchain-doc-index')
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index=index, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    retrieval_qa_chat_prompot = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_document_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompot)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_document_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        'query': result['input'],
        'result': result['answer'],
        'source_documents': result['context'],
    }
    return new_result


if __name__ == '__main__':
    result = run_llm("What is langchain chain?")
    print(result['result'])
