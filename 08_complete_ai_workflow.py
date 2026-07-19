from typing import TypedDict, List, Annotated, Optional
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
import operator

# Step 1: Initialize the LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai")

# Step 2: Define the workflow state
class WorkflowState(TypedDict):
    topic: str
    research_angles: List[str]
    research_results: Annotated[List[str], operator.add]
    combined_research: str
    draft: str
    review_feedback: Optional[str]
    revision_count: int
    human_approved: Optional[bool]
    final_report: str
    
# Step 2: Define the nodes of the workflow
## Node 1: Plan research angles
def plan_research(state: WorkflowState) -> dict:
    response = llm.invoke(
        f"List exactly 3 distinct research angles for the topic: '{state['topic']}'. "
        f"Return them as a simple numbered list."
    )
    angles = [
        line.strip()
        for line in response.content.split("\n") # type: ignore
        if line.strip() and line.strip()[0].isdigit()
    ]
    return {"research_angles": angles[:3]}

## Node 2: Research one angle (used in parallel)
def research_angle(state: dict) -> dict:
    response = llm.invoke(
        f"Research this specific angle about '{state['topic']}':\n"
        f"{state['angle']}\n\nProvide 3-4 key insights."
    )
    return {"research_results": [f"[{state['angle']}]\n{response.content}"]}

## Dispatcher: fan out to parallel research nodes
def dispatch_research(state: WorkflowState):
    return [
        Send("research_angle", {"topic": state["topic"], "angle": angle})
        for angle in state["research_angles"]
    ]

# Node 3: Combine parallel research results
def combine_research(state: WorkflowState) -> dict:
    combined = "\n\n".join(state["research_results"])
    response = llm.invoke(
        f"Synthesize these research findings into a cohesive summary:\n\n{combined}"
    )
    return {"combined_research": response.content}

## Node 4: Write the draft
def write_draft(state: WorkflowState) -> dict:
    feedback = state.get("review_feedback", "")
    revision = state.get("revision_count", 0)

    if feedback and revision > 0:
        prompt = (
            f"Revise this draft about '{state['topic']}' based on feedback.\n\n"
            f"Original Draft:\n{state['draft']}\n\n"
            f"Feedback:\n{feedback}\n\nWrite the improved version."
        )
    else:
        prompt = (
            f"Write a professional report about '{state['topic']}' "
            f"using this research:\n\n{state['combined_research']}"
        )

    response = llm.invoke(prompt)
    return {
        "draft": response.content,
        "revision_count": revision + 1,
        "review_feedback": None
    }

## Node 5: Review the draft
def review_draft(state: WorkflowState) -> dict:
    response = llm.invoke(
        f"Review this report and respond with either:\n"
        f"'APPROVED' — if complete and high quality\n"
        f"'REVISE: <feedback>' — if improvements are needed\n\n"
        f"Report:\n{state['draft']}"
    )
    feedback_text = response.content

    if "APPROVED" in feedback_text:
        return {"review_feedback": None}
    else:
        return {"review_feedback": feedback_text}

## Node 6: Finalize the report
def finalize_report(state: WorkflowState) -> dict:
    return {"final_report": state["draft"]}

## Routing: after AI review
def route_after_review(state: WorkflowState) -> str:
    if state.get("revision_count", 0) >= 3:
        return "finalize"
    if state.get("review_feedback"):
        return "revise"
    return "finalize"
  
# Step 3: Build the workflow graph
graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("plan_research",    plan_research)
graph_builder.add_node("research_angle",   research_angle) # type: ignore
graph_builder.add_node("combine_research", combine_research)
graph_builder.add_node("write_draft",      write_draft)
graph_builder.add_node("review_draft",     review_draft)
graph_builder.add_node("finalize_report",  finalize_report)

graph_builder.add_edge(START, "plan_research") # `graph_builder.set_entry_point("plan_research")`

graph_builder.add_conditional_edges(
    "plan_research",
    dispatch_research,
    ["research_angle"]
)

graph_builder.add_edge("research_angle", "combine_research")
graph_builder.add_edge("combine_research", "write_draft")
graph_builder.add_edge("write_draft", "review_draft")

graph_builder.add_conditional_edges(
    "review_draft",
    route_after_review,
    {
        "revise":   "write_draft",
        "finalize": "finalize_report"
    }
)

graph_builder.add_edge("finalize_report", END)

memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["finalize_report"]
)

# Step 4: Run the workflow
config = {"configurable": {"thread_id": "complete_workflow_001"}}

print("=== PHASE 1: AI Research and Writing ===")
graph.invoke(
    {
        "topic": "The impact of AI agents on software development",
        "research_angles": [],
        "research_results": [],
        "combined_research": "",
        "draft": "",
        "review_feedback": None,
        "revision_count": 0,
        "human_approved": None,
        "final_report": ""
    },
    config=config # type: ignore
)

current_state = graph.get_state(config) # type: ignore
print("\n=== PHASE 2: Human Review ===")
print("Draft Report:")
print(current_state.values["draft"])
print(f"\nRevisions made by AI: {current_state.values['revision_count']}")

decision = input("\nApprove this report? (yes/no): ").strip().lower()
graph.update_state(config, {"human_approved": decision == "yes"}) # type: ignore

print("\n=== PHASE 3: Finalizing ===")
final = graph.invoke(None, config=config) # type: ignore
print("\n=== FINAL REPORT ===")
print(final["final_report"])