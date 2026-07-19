from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END


# Step 1: Initialize the LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai")


# Step 2: Define the State
class State(TypedDict):
    topic: str
    outline: str
    article: str


# Step 3: Define the Nodes
def generate_outline(state: State) -> dict:
    response = llm.invoke(
        f"Create a concise 3-point outline about: {state['topic']}"
    )
    return {"outline": response.content}

def write_article(state: State) -> dict:
    response = llm.invoke(
        f"Write a short article about '{state['topic']}' "
        f"using this outline:\n\n{state['outline']}"
    )
    return {"article": response.content}


# Step 4: Build the Graph
graph_builder = StateGraph(State)

graph_builder.add_node("generate_outline", generate_outline)
graph_builder.add_node("write_article", write_article)

graph_builder.add_edge(START, "generate_outline")
graph_builder.add_edge("generate_outline", "write_article")
graph_builder.add_edge("write_article", END)

graph = graph_builder.compile()


# Step 5: Stream the Graph Execution
for event in graph.stream(
    {
        "topic": "LangGraph for AI Developers",
        "outline": "",
        "article": ""
    },
    stream_mode="updates"
):
    print(event)