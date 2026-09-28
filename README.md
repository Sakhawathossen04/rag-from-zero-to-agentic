# 🚀 RAG Learning Lab

> **Learn RAG from fundamentals to advanced implementations — with Bangla + English notes, code examples, experiments, and projects.**

This repository documents our journey of learning **Retrieval-Augmented Generation (RAG)** from scratch and implementing each concept step by step.

We are primarily learning from the **CampusX RAG / Generative AI / LangChain / LangGraph video series**, then converting the concepts into our own:

- 📖 Bangla explanations
- 📘 English notes
- 💻 Python implementations
- 🧪 Experiments
- 🏗️ Mini projects
- 🤖 Advanced RAG systems

The goal is not just to watch tutorials, but to **understand, implement, experiment, and build production-oriented RAG systems**.

---

# 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture where a Large Language Model retrieves relevant information from an external knowledge source before generating its final response.

Instead of depending only on the knowledge stored inside the LLM:

```text
User Question
      ↓
Retriever
      ↓
Relevant Documents
      ↓
LLM + Context
      ↓
Final Answer
```

This helps build AI applications that can work with:

- Private documents
- PDFs
- Websites
- YouTube transcripts
- Research papers
- Company knowledge bases
- Source code
- Databases
- Custom datasets

---

# 🎯 Repository Goals

By completing this repository, we aim to understand the complete RAG pipeline:

```text
Data
 ↓
Document Loading
 ↓
Text Cleaning
 ↓
Chunking
 ↓
Embedding
 ↓
Vector Database
 ↓
Retrieval
 ↓
Prompt Construction
 ↓
LLM
 ↓
Answer
```

Then gradually move toward:

```text
Basic RAG
   ↓
Advanced Retrieval
   ↓
RAG Evaluation
   ↓
Corrective RAG
   ↓
Self-RAG
   ↓
Agentic RAG
```

---

# 📚 Learning Roadmap

## 01 — RAG Fundamentals

Topics:

- What is RAG?
- Why do we need RAG?
- Problems with normal LLMs
- LLM knowledge cutoff
- Hallucination
- External knowledge
- RAG architecture
- Retrieval vs Generation
- RAG workflow

Folder:

```text
01-rag-fundamentals/
```

---

## 02 — Document Loaders

Learn how to load data from different sources using LangChain.

Topics:

- Text Loader
- PDF Loader
- CSV Loader
- Web Loader
- Directory Loader
- YouTube Transcript Loader
- Document objects
- Metadata

Example:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/sample.pdf")

documents = loader.load()

print(documents[0].page_content)
```

Folder:

```text
02-document-loaders/
```

---

## 03 — Text Splitters

Large documents cannot normally be passed directly into an LLM.

Therefore, documents need to be divided into smaller pieces called **chunks**.

Topics:

- Why chunking is needed
- Chunk size
- Chunk overlap
- CharacterTextSplitter
- RecursiveCharacterTextSplitter
- Semantic splitting
- Good vs bad chunking

Example:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)
```

Folder:

```text
03-text-splitters/
```

---

## 04 — Embeddings

Embeddings convert text into numerical vector representations.

Example:

```text
"Artificial Intelligence"

↓

[0.23, -0.51, 0.81, 0.14, ...]
```

These vectors allow machines to compare the **semantic similarity** between texts.

Topics:

- What are embeddings?
- Semantic similarity
- Vector representation
- Embedding models
- OpenAI Embeddings
- Hugging Face embeddings
- Sentence Transformers

Folder:

```text
04-embeddings/
```

---

## 05 — Vector Stores

Vector databases store embeddings and allow us to search for semantically similar documents.

Topics:

- Vector databases
- Similarity search
- Cosine similarity
- FAISS
- Chroma
- Pinecone
- Vector indexing
- Metadata filtering

Example:

```python
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)
```

Folder:

```text
05-vector-stores/
```

---

## 06 — Retrievers

Retrievers find the most relevant documents for a user's query.

```text
User Query
    ↓
Embedding
    ↓
Vector Search
    ↓
Top-K Documents
```

Topics:

- Retriever interface
- Similarity search
- Top-K retrieval
- MMR retrieval
- Multi-query retrieval
- Contextual compression
- Metadata filtering
- Retriever configuration

Example:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)

docs = retriever.invoke(
    "What is Retrieval-Augmented Generation?"
)
```

Folder:

```text
06-retrievers/
```

---

# 🤖 07 — Building a Complete RAG Pipeline

At this stage we combine everything.

```text
Question
   ↓
Retriever
   ↓
Relevant Context
   ↓
Prompt
   ↓
LLM
   ↓
Answer
```

Example architecture:

```text
PDF
 │
 ▼
Document Loader
 │
 ▼
Text Splitter
 │
 ▼
Embedding Model
 │
 ▼
Vector Database
 │
 ▼
Retriever
 │
 ▼
Prompt + Context
 │
 ▼
LLM
 │
 ▼
Answer
```

Folder:

```text
07-basic-rag/
```

---

# 🎥 08 — YouTube RAG Chatbot

Build a chatbot that can answer questions from YouTube videos.

Pipeline:

```text
YouTube Video
      ↓
Transcript
      ↓
Text Splitter
      ↓
Embeddings
      ↓
Vector Store
      ↓
Retriever
      ↓
LLM
      ↓
Answer
```

Possible features:

- YouTube URL input
- Automatic transcript extraction
- Semantic search
- Question answering
- Source chunks
- Chat history

Folder:

```text
08-youtube-rag/
```

---

# 🔗 09 — RAG using LangGraph

Traditional RAG pipelines are mostly linear.

LangGraph allows us to create more intelligent workflows.

Example:

```text
START
  ↓
Retrieve
  ↓
Check Documents
  ↓
Generate Answer
  ↓
Validate
  ↓
END
```

Topics:

- LangGraph basics
- Nodes
- Edges
- State
- Conditional routing
- Graph-based RAG
- Agentic workflows

Folder:

```text
09-langgraph-rag/
```

---

# 🛠️ 10 — Corrective RAG — CRAG

Traditional RAG can fail when the retriever finds irrelevant documents.

**Corrective RAG (CRAG)** attempts to detect this problem and correct the retrieval process.

Conceptual workflow:

```text
User Question
      ↓
Retrieve Documents
      ↓
Evaluate Relevance
      ↓
 ┌───────────────┐
 │ Relevant?     │
 └───────────────┘
    ↓ Yes    ↓ No
 Generate    Search Again
    ↓            ↓
    └──────┬─────┘
           ↓
       Final Answer
```

Topics:

- Retrieval grading
- Document relevance
- Query rewriting
- Web search fallback
- Hallucination reduction
- Corrective retrieval

Folder:

```text
10-corrective-rag/
```

---

# 🧠 11 — Self-RAG

Self-RAG allows the model to evaluate parts of its own RAG process.

Possible workflow:

```text
Question
   ↓
Need Retrieval?
   ↓
Retrieve
   ↓
Is Context Relevant?
   ↓
Generate
   ↓
Is Answer Supported?
   ↓
Final Answer
```

Topics:

- Self-reflection
- Retrieval decisions
- Context grading
- Answer grounding
- Hallucination checks
- Response quality evaluation

Folder:

```text
11-self-rag/
```

---

# 🤖 12 — Agentic RAG

Agentic RAG gives an AI system greater control over how information is retrieved.

Instead of:

```text
Retrieve → Generate
```

an agent may decide:

```text
Should I search?

↓ YES

Which source?

↓
Vector DB / Web / SQL / API

↓

Do I have enough information?

↓ NO

Search Again

↓

Generate Answer
```

Topics:

- Tool calling
- Agentic retrieval
- Dynamic routing
- Multi-step reasoning
- Query planning
- LangGraph agents
- Multi-source RAG

Folder:

```text
12-agentic-rag/
```

---

# 📊 13 — RAG Evaluation

Building a RAG application is not enough.

We also need to measure whether the system is actually working.

Important evaluation dimensions:

```text
Retrieval Quality
        +
Generation Quality
        +
Groundedness
        +
Answer Relevance
```

Topics:

- Context Precision
- Context Recall
- Faithfulness
- Answer Relevance
- Hallucination detection
- Retrieval evaluation
- RAGAS
- Evaluation datasets

Folder:

```text
13-rag-evaluation/
```

---

# 🏗️ Repository Structure

```text
rag-learning-lab/
│
├── README.md
│
├── requirements.txt
├── .env.example
├── .gitignore
│
├── 01-rag-fundamentals/
│   ├── README.md
│   ├── notes-en.md
│   └── notes-bn.md
│
├── 02-document-loaders/
│   ├── README.md
│   ├── notes-en.md
│   ├── notes-bn.md
│   └── examples/
│
├── 03-text-splitters/
│
├── 04-embeddings/
│
├── 05-vector-stores/
│
├── 06-retrievers/
│
├── 07-basic-rag/
│
├── 08-youtube-rag/
│
├── 09-langgraph-rag/
│
├── 10-corrective-rag/
│
├── 11-self-rag/
│
├── 12-agentic-rag/
│
├── 13-rag-evaluation/
│
├── projects/
│   ├── pdf-chatbot/
│   ├── youtube-chatbot/
│   ├── research-paper-assistant/
│   └── multi-source-rag/
│
├── experiments/
│
├── datasets/
│
└── docs/
```

---

# 🌐 Bangla + English Learning Format

Each topic should ideally contain three parts.

### 🇬🇧 English Notes

```text
notes-en.md
```

Used for:

- Definitions
- Technical explanations
- Architecture
- Interview preparation
- Documentation

### 🇧🇩 Bangla Notes

```text
notes-bn.md
```

Used for explaining difficult concepts in simple Bangla.

Example:

> RAG-এর মূল ধারণা হচ্ছে LLM-কে সরাসরি উত্তর দিতে না দিয়ে আগে relevant information খুঁজে দেওয়া। এরপর সেই information context হিসেবে LLM-কে দেওয়া হয়।

### 💻 Code

```text
examples/
```

Every important concept should have its own executable implementation.

---

# 🧪 Experiments

We will not only copy tutorial code.

We will experiment with different RAG configurations.

For example:

```text
Chunk Size:
300 vs 500 vs 1000

Chunk Overlap:
0 vs 100 vs 200

Retriever:
Similarity vs MMR

Top-K:
k=2 vs k=4 vs k=10

Embedding:
OpenAI vs HuggingFace

Vector Store:
FAISS vs Chroma
```

The results will be documented so that we understand **why one RAG configuration works better than another**.

---

# 🛠️ Tech Stack

Main technologies used throughout this repository:

```text
Python
LangChain
LangGraph
OpenAI API
Hugging Face
Sentence Transformers
FAISS
ChromaDB
Pinecone
RAGAS
Jupyter Notebook
FastAPI
Streamlit / Gradio
```

Different modules may use different technologies depending on the experiment.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/rag-learning-lab.git
```

Enter the repository:

```bash
cd rag-learning-lab
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create:

```text
.env
```

Example:

```env
OPENAI_API_KEY=your_api_key
LANGCHAIN_API_KEY=your_api_key
PINECONE_API_KEY=your_api_key
```

Never upload API keys to GitHub.

The repository should only contain:

```text
.env.example
```

---

# 📈 Learning Progress

| Module | Topic | Status |
|---|---|---|
| 01 | RAG Fundamentals | ⬜ |
| 02 | Document Loaders | ⬜ |
| 03 | Text Splitters | ⬜ |
| 04 | Embeddings | ⬜ |
| 05 | Vector Stores | ⬜ |
| 06 | Retrievers | ⬜ |
| 07 | Basic RAG Pipeline | ⬜ |
| 08 | YouTube RAG Chatbot | ⬜ |
| 09 | RAG with LangGraph | ⬜ |
| 10 | Corrective RAG | ⬜ |
| 11 | Self-RAG | ⬜ |
| 12 | Agentic RAG | ⬜ |
| 13 | RAG Evaluation | ⬜ |

Legend:

```text
⬜ Not Started
🟡 Learning
🧪 Experimenting
✅ Completed
```

---

# 🚀 Projects We Plan to Build

After completing the fundamentals, we plan to build practical projects.

### 📄 PDF Research Assistant

Upload research papers and ask questions based on their content.

### 🎥 YouTube Knowledge Assistant

Ask questions directly from educational YouTube videos.

### 📚 Multi-PDF Knowledge Base

Create a searchable knowledge system from multiple documents.

### 🔍 Research Paper RAG

Retrieve information across multiple academic papers and return answers with citations.

### 🌐 Hybrid RAG

Combine:

```text
Vector Search
+
Keyword Search
+
Web Search
```

### 🤖 Agentic Research Assistant

An AI agent capable of selecting different information sources depending on the question.

---

# 🔬 Future Advanced Topics

After completing the core roadmap, this repository may explore:

- Hybrid Search
- BM25
- Reciprocal Rank Fusion
- Re-ranking
- Cross Encoders
- Query Expansion
- Query Transformation
- HyDE
- Multi-Query Retrieval
- Parent Document Retriever
- Contextual Compression
- Semantic Chunking
- Graph RAG
- Knowledge Graph RAG
- Multimodal RAG
- SQL RAG
- Web RAG
- Long-context RAG
- RAG caching
- RAG observability
- RAG evaluation
- Hallucination detection
- Citation generation
- Production RAG architecture
- Agentic RAG
- Multi-Agent RAG

---

# 🧩 Our Learning Principle

We follow:

```text
Watch
  ↓
Understand
  ↓
Write Notes
  ↓
Implement
  ↓
Break It
  ↓
Experiment
  ↓
Improve
  ↓
Build Project
```

The objective is not to memorize LangChain APIs.

The objective is to understand:

> **Why does RAG work, when does it fail, and how can we design a better retrieval system?**

---

# 📺 Primary Learning Resource

Our initial learning path follows the **CampusX RAG Playlist**, covering topics such as:

1. Retrieval-Augmented Generation fundamentals
2. Document Loaders
3. Text Splitters
4. Vector Stores
5. Retrievers
6. YouTube RAG Chatbot
7. RAG using LangGraph
8. Corrective RAG
9. Self-RAG
10. Advanced RAG concepts

The repository contains our **own notes, implementations, experiments, observations, and projects** created while learning these concepts.

---

# ⚠️ Disclaimer

This repository is created for **learning, experimentation, and research purposes**.

The educational videos and original teaching materials belong to their respective creators.

We do not intend to reproduce course material verbatim.

Instead, this repository contains our own:

- Understanding
- Notes
- Code
- Experiments
- Implementations
- Projects

built while studying the concepts.

---

# 🤝 Contribution

This is primarily a learning repository, but improvements and suggestions are welcome.

You can contribute through:

```text
Bug fixes
Better explanations
New RAG experiments
Evaluation methods
Retrieval techniques
Example projects
Documentation improvements
```

---

# ⭐ Final Goal

At the end of this repository, we want to be capable of designing systems like:

```text
                     ┌─────────────┐
                     │ User Query  │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │ Query Router│
                     └──────┬──────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   Vector DB            Web Search          Database
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                     ┌──────▼──────┐
                     │  Re-ranker  │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │ Context     │
                     │ Validation  │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │     LLM     │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │ Evaluation  │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐
                     │Final Answer │
                     └─────────────┘
```

From **Basic RAG → Advanced RAG → Self-Correcting RAG → Agentic RAG → Production RAG**.

---

## ⭐ Learning in Public

We are building this repository step by step while learning.

Mistakes, experiments, improvements, and architectural changes are part of the journey.

```text
Learn → Build → Evaluate → Improve → Repeat
```

**Happy Learning & Building RAG 🚀**
