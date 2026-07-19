from typing import TypedDict, List, Annotated
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import operator

# Step 1: Initialize the LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai")

# Step 2: Define the States
class State(TypedDict):
    topic: str
    results: Annotated[List[str], operator.add]

class BranchState(TypedDict):
    topic: str
    task: str

# Step 3: Define the Nodes
def write_section(state: BranchState) -> dict:
    response = llm.invoke(
        f"For a blog post about '{state['topic']}', write {state['task']}. "
        "Keep it concise."
    )
    return {"results": [f"{state['task']}:\n{response.content}"]}

def dispatch_tasks(state: State):
    tasks = ["a catchy title", "a 5-point outline", "an engaging introduction"]
    return [
        Send("write_section", {"topic": state["topic"], "task": task})
        for task in tasks
    ]

# Step 4: Define the Edges
builder = StateGraph(State)

builder.add_node("write_section", write_section)

builder.add_conditional_edges(
    START,
    dispatch_tasks,
    ["write_section"],
)

builder.add_edge("write_section", END)

graph = builder.compile()

# Step 5: Run the graph
result = graph.invoke({
    "topic": "LangGraph for AI Developers",
    "results": [],
})

# Print the result
for section in result["results"]:
    print(section)
    print("---")