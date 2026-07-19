import asyncio
from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

# Step 1: Initialize the LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai")


# Step 2: Define the State
class State(TypedDict):
    question: str
    answer: str


# Step 3: Define an Async Node
async def answer_question(state: State) -> dict:
    response = await llm.ainvoke(state["question"])
    return {"answer": response.content}


# Step 4: Build the Graph
graph_builder = StateGraph(State)
graph_builder.add_node("answer_question", answer_question)
graph_builder.add_edge(START, "answer_question")
graph_builder.add_edge("answer_question", END)
graph = graph_builder.compile()


# Step 5: Run the Graph Asynchronously
async def main():
    result = await graph.ainvoke({
        "question": "What is LangGraph?",
        "answer": ""
    })

    print(result["answer"])


asyncio.run(main())