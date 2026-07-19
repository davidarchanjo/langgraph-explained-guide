from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

llm = init_chat_model(model="gpt-4o", model_provider="openai")

# ─────────────────────────────────────────
# Subgraph 1: Research Subgraph
# ─────────────────────────────────────────
class ResearchState(TypedDict):
    topic: str
    research_notes: str
    key_facts: str

def gather_notes(state: ResearchState) -> dict:
    response = llm.invoke(
        f"Research '{state['topic']}' and provide detailed notes."
    )
    return {"research_notes": response.content}

def extract_key_facts(state: ResearchState) -> dict:
    response = llm.invoke(
        f"Extract the 5 most important facts from these notes:\n{state['research_notes']}"
    )
    return {"key_facts": response.content}

research_builder = StateGraph(ResearchState)
research_builder.add_node("gather_notes", gather_notes)
research_builder.add_node("extract_key_facts", extract_key_facts)
research_builder.add_edge(START, "gather_notes") # or `research_builder.set_entry_point("gather_notes")`
research_builder.add_edge("gather_notes", "extract_key_facts")
research_builder.add_edge("extract_key_facts", END)

research_subgraph = research_builder.compile()

# ─────────────────────────────────────────
# Subgraph 2: Writing Subgraph
# ─────────────────────────────────────────
class WritingState(TypedDict):
    topic: str
    key_facts: str
    draft: str
    polished_report: str

def write_draft(state: WritingState) -> dict:
    response = llm.invoke(
        f"Write a report about '{state['topic']}' using these facts:\n{state['key_facts']}"
    )
    return {"draft": response.content}

def polish_report(state: WritingState) -> dict:
    response = llm.invoke(
        f"Polish and improve this report for clarity and flow:\n{state['draft']}"
    )
    return {"polished_report": response.content}

writing_builder = StateGraph(WritingState)
writing_builder.add_node("write_draft", write_draft)
writing_builder.add_node("polish_report", polish_report)
writing_builder.add_edge(START, "write_draft")  # or `writing_builder.set_entry_point("write_draft")`
writing_builder.add_edge("write_draft", "polish_report")
writing_builder.add_edge("polish_report", END)

writing_subgraph = writing_builder.compile()

# ─────────────────────────────────────────
# Parent Graph: Orchestrates both subgraphs
# ─────────────────────────────────────────
class ParentState(TypedDict):
    topic: str
    research_notes: str
    key_facts: str
    draft: str
    polished_report: str

parent_builder = StateGraph(ParentState)

parent_builder.add_node("research", research_subgraph)
parent_builder.add_node("writing", writing_subgraph)

parent_builder.add_edge(START, "research") # or `parent_builder.set_entry_point("research")`
parent_builder.add_edge("research", "writing")
parent_builder.add_edge("writing", END)

parent_graph = parent_builder.compile()

result = parent_graph.invoke({
    "topic": "The future of AI agents",
    "research_notes": "",
    "key_facts": "",
    "draft": "",
    "polished_report": ""
})

print("=== FINAL REPORT ===")
print(result["polished_report"])