import os
import sqlite3
from typing import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# Step 1: Configure SQLite Connection
DB_PATH = "assets/data/checkpoints.db"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
connection = sqlite3.connect(DB_PATH, check_same_thread=False)

try:
  # Step 2: Initialize the LLM
  llm = init_chat_model(model="gpt-4o", model_provider="openai")


  # Step 3: Define the State
  class ChatState(TypedDict):
      messages: Annotated[list, add_messages]


  # Step 4: Define the Node
  def chat(state: ChatState) -> dict:
      response = llm.invoke(state["messages"])
      return {"messages": [response]}


  # Step 5: Build the Graph
  graph_builder = StateGraph(ChatState)

  graph_builder.add_node("chat", chat)
  graph_builder.add_edge(START, "chat")
  graph_builder.add_edge("chat", END)


  # Step 6: Configure SQLite Checkpointing
  checkpointer = SqliteSaver(connection)
  graph = graph_builder.compile(checkpointer=checkpointer)

  config = {
      "configurable": {
          "thread_id": "production_session_001"
      }
  }

  # NOTE: To observe checkpoint persistence across application restarts:
  #   1. On the first run, comment out Step 8 and execute only Step 7.
  #   2. On the second run, comment out Step 7, uncomment Step 8, and run the application again.
  # Because both executions use the same SQLite database and thread_id, the graph will
  # restore the conversation state saved during the first run and remember your name.

  # Step 7: First Conversation Turn
  result1 = graph.invoke(
      {
          "messages": [
              HumanMessage(content="My name is David.")
          ]
      },
      config=config  # type: ignore
  )
  print(result1["messages"][-1].content)


  # Step 8: Second Conversation Turn
  result2 = graph.invoke(
      {
          "messages": [
              HumanMessage(content="What is my name?")
          ]
      },
      config=config  # type: ignore
  )
  print(result2["messages"][-1].content)
finally:
  # Step 9: Close the Database Connection
  connection.close()