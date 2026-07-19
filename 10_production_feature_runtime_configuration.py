import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# Step 1: Load the environment variables
load_dotenv()


# Step 2: Define the State
class State(TypedDict):
    question: str
    answer: str


# Step 3: Define a Configurable Node
def answer_question(state: State, config: RunnableConfig) -> dict:
    configurable = config.get("configurable", {})

    model_name = configurable.get("model", "gpt-4o")
    model_provider = configurable.get("provider", "openai")
    temperature = configurable.get("temperature", 0.7)

    llm = init_chat_model(
        model=model_name,
        model_provider=model_provider,
        temperature=temperature
    )

    response = llm.invoke(state["question"])

    return {"answer": response.content}


# Step 4: Build the Graph
graph_builder = StateGraph(State)

graph_builder.add_node("answer_question", answer_question)
graph_builder.add_edge(START, "answer_question")
graph_builder.add_edge("answer_question", END)

graph = graph_builder.compile()


# Step 5: Define the Runtime Configuration. It could have come from an API call or a database
config = {
    "configurable": {
        "model": os.getenv("MODEL", "gpt-4o"),
        "provider": os.getenv("MODEL_PROVIDER", "openai"),
        "temperature": float(os.getenv("MODEL_TEMPERATURE", "0.7"))
    }
}


# Step 6: Run the Graph
result = graph.invoke(
    {
        "question": "Explain AI agents in one paragraph.",
        "answer": ""
    },
    config=config  # type: ignore
)


# Print the result
print(result["answer"])