# ReAct Agent with Custom Tools & RAG System
## 1. Overview
This project implements a production-ready **ReAct (Reasoning + Acting) Agent** powered by **Qwen3-32B** via Groq, built entirely on the modern LangGraph stack. The agent dynamically selects from a set of heterogeneous tools — including a RAG pipeline over a PDF, a live weather API, Wikipedia search, and a full Swagger-generated API toolkit — to answer complex, multi-step user queries.

---

## 2. Workflow

### 2.1 LLM Setup
- Used **Qwen3-32B** via **Groq API** for fast, free inference.
- Configured with `temperature=0` for deterministic, reliable tool selection.
- Connected to **LangSmith** for full run tracing and debugging.
- API key is loaded securely from a `.env` file using `python-dotenv`:

```env
# .env
api_key=your_groq_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=your_project_name
```

```python
from dotenv import load_dotenv
import os

load_dotenv('.env')

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.0,
    api_key=os.getenv('api_key')
)
```

### 2.2 Swagger → Tools Pipeline
- Fetched a live **Swagger 2.0** spec from the PetStore API.
- Built a custom `swagger_to_tools()` parser to convert Swagger 2.0 paths into **OpenAI-style tool definitions** — bypassing LangChain's OpenAPI toolkit which only supports OpenAPI 3.x.
- Wrapped each generated tool as a `StructuredTool` with a dynamically created Pydantic schema.
- Built a universal `execute_tool()` executor that handles path params, query params, and request body automatically.

### 2.3 RAG Pipeline
- Loaded the **"Attention is All You Need"** transformer paper directly from ArXiv as a real PDF.
- Chunked the document using `RecursiveCharacterTextSplitter` (chunk size: 500, overlap: 50).
- Embedded chunks locally using **HuggingFace `all-MiniLM-L6-v2`** — no embedding API key required.
- Stored vectors in **ChromaDB** for fast similarity search.
- Wrapped the retriever as a `@tool` with a clear docstring to guide the agent's routing logic.

### 2.4 Custom Tools
Implemented and registered the following tools alongside the Swagger and RAG tools:

- **`search_transformer_paper`** — RAG over the transformer paper PDF; cites page numbers.
- **`search_wikipedia`** — General knowledge queries via Wikipedia.
- **`get_current_temperature`** — Live weather lookup by city name.

### 2.5 Agent Construction
- Combined all tools: `all_tools = [search_transformer_paper, search_wikipedia, get_current_temperature] + swagger_lc_tools`
- Built the agent using `create_react_agent` from **LangGraph** — replacing the entire deprecated stack (`AgentExecutor`, `OpenAIFunctionsAgentOutputParser`, `RunnablePassthrough`, `MessagesPlaceholder`, `format_to_openai_functions`).
- Wrote a structured system prompt that maps each tool to its intended use case and enforces rules like confirming destructive API calls.

### 2.6 Memory
- Integrated **`MemorySaver`** (LangGraph checkpointer) for persistent conversation memory across turns.
- Memory is scoped per user via `thread_id` in the config — enabling multi-user session isolation.

### 2.7 Observability
- Configured **LangSmith** tracing via `LANGCHAIN_TRACING_V2` env vars.
- All agent runs — LLM calls, tool invocations, results, token usage, and latency — are visible in the LangSmith dashboard under the configured project.

---

## 3. Tech Stack

- Python
- LangGraph (`create_react_agent`, `MemorySaver`)
- LangChain Core (`@tool`, `StructuredTool`, `SystemMessage`, `ToolMessage`)
- LangChain Groq (`ChatGroq`)
- LangChain Community (document loaders, ChromaDB, HuggingFace embeddings)
- ChromaDB
- HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- Groq API (Qwen3-32B)
- LangSmith (tracing & observability)
- Pydantic v2
- Requests
- python-dotenv (secure API key management via `.env`)

---

## 4. Architecture

```
User Query
     │
     ▼
 Qwen3-32B (via Groq)
     │
     ├── search_transformer_paper  →  ChromaDB RAG  →  ArXiv PDF
     │
     ├── search_wikipedia          →  Wikipedia API
     │
     ├── get_current_temperature   →  Weather API
     │
     └── swagger_lc_tools          →  PetStore REST API
               │
               ▼
         execute_tool()
         (path / query / body params)
     │
     ▼
 MemorySaver (thread_id scoped)
     │
     ▼
 Final Answer + LangSmith Trace
```

---

## 5. Key Design Decisions

### 5.1 Swagger 2.0 Support
LangChain's built-in `OpenAPIToolkit` only supports OpenAPI 3.x. A custom parser was written to handle Swagger 2.0 natively — extracting paths, methods, parameters (path/query/body), and operationIds to generate valid tool schemas without any spec conversion.

### 5.2 Tool Routing via Docstrings
The agent's tool selection is driven entirely by tool docstrings, not the system prompt. The system prompt focuses only on persona, rules, and edge cases — keeping it short and maintainable.

### 5.3 Modern Stack Only
All deprecated LangChain patterns were replaced:

| Deprecated | Replaced By |
|---|---|
| `AgentExecutor` | `create_react_agent` |
| `OpenAIFunctionsAgentOutputParser` | `llm.bind_tools()` |
| `ConversationBufferMemory` | `MemorySaver` + `thread_id` |
| `MessagesPlaceholder(agent_scratchpad)` | LangGraph state |
| `format_to_openai_functions` | `bind_tools()` |
| `RunnablePassthrough.assign` | LangGraph edges |
| `AgentFinish` | `response.tool_calls` check |

---

## 6. Conclusion
This project demonstrates a full-stack, production-aligned ReAct agent that integrates heterogeneous tool types — REST APIs, RAG, and live data sources — under a unified LangGraph architecture. It intentionally avoids all deprecated LangChain patterns in favor of the current recommended stack, with observability and multi-user memory built in from the ground up.
