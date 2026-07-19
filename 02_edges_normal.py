from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

# Step 1: Initialize the LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai")

# Step 2: Define the State
class State(TypedDict):
    topic: str
    title: str
    body: str
    conclusion: str

# Step 3: Define the Nodes
def generate_title(state: State) -> dict:
    response = llm.invoke(
        f"Write a blog title about {state['topic']} in under 30 characters."
    )
    return {"title": response.content.strip()} # type: ignore

def write_body(state: State) -> dict:
    response = llm.invoke(
        f"Write a blog post titled '{state['title']}' in under 100 words."
    )
    return {"body": response.content.strip()} # type: ignore

def write_conclusion(state: State) -> dict:
    response = llm.invoke(
        f"Write a conclusion for this blog post in under 20 words:\n{state['body']}"
    )
    return {"conclusion": response.content.strip()} # type: ignore

# Step 4: Build the Graph
builder = StateGraph(State)

builder.add_node("generate_title", generate_title)
builder.add_node("write_body", write_body)
builder.add_node("write_conclusion", write_conclusion)

builder.add_edge(START, "generate_title") # or `graph_builder.set_entry_point("generate_title")`
builder.add_edge("generate_title", "write_body")
builder.add_edge("write_body", "write_conclusion")
builder.add_edge("write_conclusion", END)

graph = builder.compile()

# Step 5: Run the Graph
result = graph.invoke({
    "topic": "LangGraph",
    "title": "",
    "body": "",
    "conclusion": "",
})

# Print the result
print("\n--------TITLE---------\n")
print(result["title"])
print("\n---------BODY---------\n")
print(result["body"])
print("\n------CONCLUSION------\n")
print(result["conclusion"])