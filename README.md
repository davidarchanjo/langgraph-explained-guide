# LangGraph Explained Guide

A collection of practical, self-contained Python examples demonstrating the most important concepts in **LangGraph**.

This repository is the companion codebase for my [**LangGraph Explained Guide**](), where I walk through the core concepts of LangGraph with explanations and practical examples. Every major example in the guide has a corresponding runnable Python implementation in this repository, allowing you to experiment with the code while reading.

Whether you're new to LangGraph or looking for concise examples of specific features, this repository is designed to serve as a hands-on learning resource for building stateful, controllable, and production-ready AI workflows.

## Features
- State management
- Nodes and edges
  - Normal edges
  - Conditional routing
- Cycles and loops
- Parallel execution
- Persistence and checkpointing
  - In-memory checkpointing
  - SQLite persistent checkpointing
- Human-in-the-Loop (HITL)
- Subgraphs
- Complete AI workflows
- Multi-agent systems
  - Supervisor agent pattern
- Production features
  - Error handling and retries
  - Streaming graph output
  - Async execution
  - Runtime configuration
  - Persistent checkpointing with SQLite

---

## Requirements
- Python 3.10+
- OpenAI API key (optional)

Install the dependencies:
```bash
pip install -r requirements.txt
```

---

## Configure the LLM
The examples use LangChain's `init_chat_model` interface to initialize the language model:

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="gpt-4o",
    model_provider="openai"
)
```

If using OpenAI, configure your API key:
```bash
export OPENAI_API_KEY=<your-api-key>
```

Alternatively, create a `.env` file in the project root:
```dotenv
OPENAI_API_KEY=<your-api-key>
```

Some examples also demonstrate how model configuration can be loaded from environment variables and passed to the graph at runtime.

---

## Examples

### 01 - Hello World
The simplest possible LangGraph application.

Concepts:
- State
- Nodes
- Edges
- `StateGraph`
- `START` and `END`
- Graph compilation and invocation

---

### 02 - Edges
Demonstrates how nodes are connected and how execution flows through a graph.

Examples:
- Normal edges
- Conditional routing

Concepts:
- Sequential execution
- Conditional edges
- Routing functions
- Dynamic execution paths

---

### 03 - Cycles and Loops
Demonstrates how LangGraph workflows can loop back to previously executed nodes.

Concepts:
- Cyclic workflows
- Conditional routing
- Retry loops
- Iteration limits
- State-driven execution

---

### 04 - Parallel Execution
Demonstrates how multiple tasks can be dynamically dispatched and executed in parallel.

Concepts:
- Parallel branches
- Dynamic fan-out
- `Send`
- State reducers
- Combining parallel results

---

### 05 - In-Memory Checkpointer
Demonstrates how LangGraph can persist state between multiple graph invocations using an in-memory checkpointer.

Concepts:
- `MemorySaver`
- Checkpointing
- Thread IDs
- Conversation memory
- Stateful graph execution

---

### 06 - Human-in-the-Loop
Demonstrates how graph execution can be paused before an action is performed, allowing a human to review and approve the operation before the workflow continues.

Concepts:
- Human approval
- Graph interrupts
- State inspection
- State updates
- Pausing and resuming execution

---

### 07 - Subgraphs
Demonstrates how complex workflows can be divided into smaller, reusable graphs.

Concepts:
- Subgraphs
- Parent graphs
- Modular workflow design
- State sharing between graphs
- Graph composition

---

### 08 - Complete AI Workflow
Combines multiple LangGraph concepts into a more complete AI workflow for research, writing, review, and human approval.

Concepts:
- Workflow planning
- Parallel execution
- Dynamic task dispatch
- AI-generated research
- Draft generation
- Review and revision loops
- Conditional routing
- Checkpointing
- Human-in-the-Loop

---

### 09 - Supervisor Agent
Demonstrates a multi-agent architecture where a supervisor agent coordinates specialized worker agents and determines which agent should execute next.

Concepts:
- Multi-agent systems
- Supervisor pattern
- Specialized worker agents
- LLM-based routing
- Conditional edges
- Cyclic agent workflows
- Human-in-the-Loop
- `interrupt`
- `Command`
- Checkpointing

---

### 10 - Production Features
Examples covering features and patterns useful when taking LangGraph applications from development to production.

Includes:
- Error handling and retries
- Streaming graph output
- Async execution
- Runtime configuration
- Persistent checkpointing with SQLite

Concepts:
- Retry policies
- Graph streaming
- Asynchronous graph execution
- Runtime configuration with `RunnableConfig`
- Environment-based configuration
- Durable state persistence
- SQLite checkpointing

---

## Recommended Testing Order
1. Hello World
2. Edges
3. Cycles and Loops
4. Parallel Execution
5. In-Memory Checkpointer
6. Human-in-the-Loop
7. Subgraphs
8. Complete AI Workflow
9. Supervisor Agent
10. Production Features

Following this sequence provides a gradual introduction from basic graph construction and state management to more advanced concepts such as persistence, cyclic workflows, Human-in-the-Loop, multi-agent architectures, and production-oriented features.

---

## Technologies
- Python
- LangChain
- LangChain OpenAI
- LangGraph
- SQLite
- python-dotenv

---

## License
This repository is intended for educational purposes. You are free to use, modify, and distribute the code for personal or educational purposes. However, please acknowledge the source and the author.