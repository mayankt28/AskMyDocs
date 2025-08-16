# 🧊 AskMyDocs

**AskMyDocs** is an intelligent question-answering application powered by **LangChain**, **Pinecone**, and **Streamlit**.
It allows you to **upload, crawl, and query documentation** in natural language while providing **cited sources** for transparency.

---

## 🎯 Features

* 🌐 **Automated Web Crawling** with [Tavily](https://tavily.com) – fetch documentation pages at scale.
* ✂️ **Document Chunking** with LangChain `RecursiveCharacterTextSplitter` for optimal embeddings.
* 🧠 **Semantic Search** powered by **Pinecone Vector Store** and **OpenAI Embeddings**.
* 💬 **Conversational Retrieval** – context-aware Q\&A with chat history support.
* ⚡ **Asynchronous Ingestion Pipeline** – batch index documents efficiently.
* 🎨 **Interactive UI** built with Streamlit for chatting with your documents.
* 📑 **Source Attribution** – every answer is backed by document citations.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mayankt28/askmydocs.git
cd askmydocs
```

### 2. Create Virtual Environment

Using `pipenv` or `venv`:

```bash
pip install pipenv
pipenv install
```

### 3. Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### 4. Run Document Ingestion

Crawl and index documentation into Pinecone:

```bash
python ingestion.py
```

### 5. Launch Streamlit App

```bash
streamlit run app.py
```

---

## 💡 Usage

1. Enter your query in the Streamlit chat box.
2. The app retrieves context-aware results from Pinecone.
3. The LLM (ChatGPT-style) generates an answer with **citations**.

Example:

**Prompt:**

> What is a LangChain Chain?

**Answer:**

> A Chain in LangChain is a sequence of calls to components, usually a combination of LLMs, prompts, and retrievers, that can be combined for more complex workflows.

📌 **Sources:**

1. [https://python.langchain.com/docs/get\_started/introduction](https://python.langchain.com/docs/get_started/introduction)
2. [https://python.langchain.com/docs/modules/chains](https://python.langchain.com/docs/modules/chains)

---

## 📦 Tech Stack

* [LangChain](https://github.com/langchain-ai/langchain) – Orchestrating LLM pipelines
* [Pinecone](https://www.pinecone.io) – Vector database for semantic search
* [OpenAI](https://platform.openai.com/) – Embeddings + Chat LLM
* [Tavily](https://tavily.com) – Web crawling for docs
* [Streamlit](https://streamlit.io) – Interactive frontend

---
