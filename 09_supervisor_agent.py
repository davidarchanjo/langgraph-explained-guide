from typing import TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

# Step 1: Initialize the LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai")


# Step 2: Define the state
class AgentState(TypedDict):
    task: str
    research: str
    draft: str
    feedback: str
    final: str
    next_agent: str
    iteration: int


# Step 3: Define Supervisor Agent
def supervisor(state: AgentState) -> dict:
    iteration = state.get("iteration", 0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a supervisor managing a team of AI agents.\n"
         "Available agents: researcher, writer, reviewer, human_review\n\n"
         "Rules:\n"
         "- Start with 'researcher' if there is no research yet\n"
         "- Use 'writer' after research is complete\n"
         "- Use 'reviewer' after a draft exists\n"
         "- Use 'human_review' after review or after 3 iterations\n\n"
         "Respond with only the agent name to call next."),
        ("human",
         "Task: {task}\n"
         "Research done: {has_research}\n"
         "Draft done: {has_draft}\n"
         "Last feedback: {feedback}\n"
         "Iterations: {iteration}\n\n"
         "Which agent should work next?")
    ])

    response = (prompt | llm).invoke({
        "task": state["task"],
        "has_research": "yes" if state.get("research") else "no",
        "has_draft": "yes" if state.get("draft") else "no",
        "feedback": state.get("feedback", "none"),
        "iteration": iteration
    })

    return {"next_agent": response.content.strip().lower()} # type: ignore


# Step 4: Define Worker Agents
def researcher(state: AgentState) -> dict:
    response = llm.invoke(
        f"Research this task thoroughly and provide key findings:\n{state['task']}"
    )
    return {
        "research": response.content,
        "iteration": state.get("iteration", 0) + 1
    }


def writer(state: AgentState) -> dict:
    response = llm.invoke(
        f"Write a professional report for this task:\n{state['task']}\n\n"
        f"Based on this research:\n{state['research']}"
    )
    return {
        "draft": response.content,
        "iteration": state.get("iteration", 0) + 1
    }


def reviewer(state: AgentState) -> dict:
    response = llm.invoke(
        f"Review this draft critically. Be specific about what is good "
        f"and what needs improvement:\n{state['draft']}"
    )
    return {
        "feedback": response.content,
        "iteration": state.get("iteration", 0) + 1
    }


# Human-in-the-Loop Agent: pause the graph and request human approval
def human_review(state: AgentState) -> dict:
    approved = interrupt({
        "draft": state["draft"],
        "feedback": state["feedback"],
        "question": "Do you approve this report?"
    })

    # Route to finalizer if approved, otherwise return to the writer
    return {
        "next_agent": "finalizer" if approved else "writer"
    }


def finalizer(state: AgentState) -> dict:
    response = llm.invoke(
        f"Polish this draft into a final, publication-ready report:\n{state['draft']}"
    )
    return {"final": response.content}


# Routing: supervisor decides next agent
def route_by_supervisor(state: AgentState) -> str:
    next_agent = state.get("next_agent", "researcher")
    valid_agents = ["researcher", "writer", "reviewer", "human_review"]
    return next_agent if next_agent in valid_agents else "human_review"


# Routing: human decision determines whether to finalize or revise
def route_after_human_review(state: AgentState) -> str:
    return state["next_agent"]


# Step 5: Build the Multi-Agent Graph
graph_builder = StateGraph(AgentState)

graph_builder.add_node("supervisor",   supervisor)
graph_builder.add_node("researcher",   researcher)
graph_builder.add_node("writer",       writer)
graph_builder.add_node("reviewer",     reviewer)
graph_builder.add_node("human_review", human_review)
graph_builder.add_node("finalizer",    finalizer)

graph_builder.add_edge(START, "supervisor") # or `graph_builder.set_entry_point("supervisor")`

graph_builder.add_conditional_edges(
    "supervisor",
    route_by_supervisor,
    {
        "researcher":   "researcher",
        "writer":       "writer",
        "reviewer":     "reviewer",
        "human_review": "human_review"
    }
)

# All workers report back to supervisor (except human_review and finalizer)
graph_builder.add_edge("researcher", "supervisor")
graph_builder.add_edge("writer",     "supervisor")
graph_builder.add_edge("reviewer",   "supervisor")

# Human approval routes to finalizer; rejection routes back to writer
graph_builder.add_conditional_edges(
    "human_review",
    route_after_human_review,
    {
        "writer":    "writer",
        "finalizer": "finalizer"
    }
)

graph_builder.add_edge("finalizer", END)


# Step 6: Compile the Graph with a checkpointer
# A checkpointer is required to pause and resume execution with interrupt()
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# The same thread_id must be used when starting and resuming the graph
config = {"configurable": {"thread_id": "report_001"}}


# Step 7: Run the graph until the Human-in-the-Loop interrupt
result = graph.invoke({
    "task": "Write a comprehensive overview of LangGraph for a technical blog",
    "research": "",
    "draft": "",
    "feedback": "",
    "final": "",
    "next_agent": "",
    "iteration": 0
}, config=config) # type: ignore


# Step 8: Display the information sent by the Human-in-the-Loop interrupt
interrupt_data = result["__interrupt__"][0].value

print("=== HUMAN REVIEW ===")
print("\nDraft:")
print(interrupt_data["draft"])

print("\nAI Reviewer Feedback:")
print(interrupt_data["feedback"])


# Step 9: Collect the human decision
decision = input("\nApprove report? (yes/no): ").strip().lower()
approved = decision == "yes"


# Step 10: Resume the graph with the human decision
# Command(resume=...) becomes the return value of interrupt()
result = graph.invoke(
    Command(resume=approved),
    config=config # type: ignore
)

# Step 11: Print the final result
print("\n=== FINAL OUTPUT ===")
print(result["final"])