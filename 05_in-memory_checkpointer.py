from typing import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Step 1: Initialize the LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai")

# Step 2: Define the State
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

# Step 3: Define the Node
def chat(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Step 4: Define the Edge and Build the Graph
graph_builder = StateGraph(ChatState)
graph_builder.add_node("chat", chat)
graph_builder.add_edge(START, "chat")  # or `graph_builder.set_entry_point("chat")`
graph_builder.add_edge("chat", END)

# Step 5: Compile the Graph assigning a Checkpointer with Memory
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "session_001"}}

# Testing the graph
## Turn 1
result1 = graph.invoke(
    {"messages": [HumanMessage(content="My name is David.")]},
    config=config # type: ignore
)
print(result1["messages"][-1].content)
# Output: Hello David! It's nice to meet you. How can I help you today?

## Turn 2 — the graph remembers the full conversation
result2 = graph.invoke(
    {"messages": [HumanMessage(content="What is my name?")]},
    config=config # type: ignore
)
print(result2["messages"][-1].content)
# Output: Your name is David.