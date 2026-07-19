from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

# Step 1: Define the State
class State(TypedDict):
    question: str
    answer: str

# Step 2: Initialize the LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai")

# Step 3: Define a Node
def answer_question(state: State) -> dict:
    response = llm.invoke(state["question"])
    return {"answer": response.content}

# Step 4: Build the Graph
graph_builder = StateGraph(State)
graph_builder.add_node("answer_question", answer_question)
graph_builder.add_edge(START, "answer_question") # or `graph_builder.set_entry_point("answer_question")`
graph_builder.add_edge("answer_question", END)
graph = graph_builder.compile()

# Step 5: Run the Graph
result = graph.invoke({"question": "What is LangGraph in one sentence?"}) # type: ignore

# Print the result
print(graph.get_graph().draw_ascii())
print(result["answer"])