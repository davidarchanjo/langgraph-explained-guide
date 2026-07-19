from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

# Step 1: Initialize the LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai")


# Step 2: Define the State
class State(TypedDict):
    question: str
    answer: str
    error: str


# Step 3: Define the Node
def answer_question(state: State) -> dict:
    try:
        response = llm.invoke(state["question"])
        return {
            "answer": response.content,
            "error": ""
        }
    except Exception as e:
        return {
            "answer": "",
            "error": str(e)
        }


# Step 4: Build the Graph
graph_builder = StateGraph(State)

graph_builder.add_node(
    "answer_question",
    answer_question,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,
        max_interval=10.0
    )
)

graph_builder.add_edge(START, "answer_question")
graph_builder.add_edge("answer_question", END)

graph = graph_builder.compile()


# Step 5: Run the Graph
result = graph.invoke({
    "question": "What is LangGraph?",
    "answer": "",
    "error": ""
})


# Print the result
if result["error"]:
    print(f"Error: {result['error']}")
else:
    print(result["answer"])