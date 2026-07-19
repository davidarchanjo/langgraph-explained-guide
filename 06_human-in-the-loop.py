from typing import TypedDict, Optional
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

# Step 1: Initialize the LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai")

# Step 2: Define the State
class State(TypedDict):
    user_request: str
    proposed_action: str
    human_approved: Optional[bool]
    result: str

# Step 3: Define the Nodes
def propose_action(state: State) -> dict:
    response = llm.invoke(
        f"A user wants to: '{state['user_request']}'\n\n"
        f"Propose a specific action to fulfill this request. "
        f"Be concise and specific."
    )
    return {"proposed_action": response.content}

def execute_action(state: State) -> dict:
    if not state.get("human_approved"):
        return {"result": "Action was rejected by the human reviewer."}

    response = llm.invoke(
        f"Execute the following action and describe what was done:\n"
        f"{state['proposed_action']}"
    )
    return {"result": response.content}

# Step 4: Build the Graph
graph_builder = StateGraph(State)
graph_builder.add_node("propose_action", propose_action)
graph_builder.add_node("execute_action", execute_action)

graph_builder.add_edge(START, "propose_action") # or `graph_builder.set_entry_point("propose_action")`
graph_builder.add_edge("propose_action", "execute_action")
graph_builder.add_edge("execute_action", END)

# Step 5: Compile the Graph assigning a Checkpointer with Memory,
# and an interrupt before the action is executed
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_action"])

config = {"configurable": {"thread_id": "approval_001"}}


# Testing the graph
## Step 1: Run until the interrupt
print("=== Running until human review point ===")
graph.invoke(
    {
        "user_request": "Delete all files older than 30 days from the archive folder",
        "proposed_action": "",
        "human_approved": None,
        "result": ""
    },
    config=config # type: ignore
)

# Show the proposed action to the human
current_state = graph.get_state(config) # type: ignore
proposed = current_state.values["proposed_action"]
print(f"\nProposed Action:\n{proposed}")

## Step 2: Human reviews and provides input
print("\n=== Waiting for human approval ===")
human_decision = input("Do you approve this action? (yes/no): ").strip().lower()
approved = human_decision == "yes"

## Step 3: Update state with human decision and resume
graph.update_state(config, {"human_approved": approved}) # type: ignore

print("\n=== Resuming graph after human input ===")
final_result = graph.invoke(None, config=config) # type: ignore
print(f"\nResult: {final_result['result']}")