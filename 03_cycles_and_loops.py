from typing import TypedDict, Optional
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

# Step 1: Define the language model
llm = init_chat_model(model="gpt-4o", model_provider="openai")

# Step 2: Define the state
class State(TypedDict):
    task: str
    code: str
    error: Optional[str]
    attempts: int
    final_code: str

# Step 3: Define the nodes
def generate_code(state: State) -> dict:
    if state.get("error"):
        prompt = (
            f"Fix this Python code:\n{state['code']}\n\n"
            f"Error: {state['error']}\n"
            "Return only valid Python code without Markdown."
        )
    else:
        prompt = (
            f"Write a Python function that {state['task']}. "
            "Return only valid Python code without Markdown."
        )

    code = llm.invoke(prompt).content.strip() # type: ignore
    code = code.removeprefix("```python").removeprefix("```").removesuffix("```").strip()

    return {
        "code": code,
        "attempts": state.get("attempts", 0) + 1,
        "error": None,
    }

def validate_code(state: State) -> dict:
    try:
        compile(state["code"], "<string>", "exec")
        return {"error": None, "final_code": state["code"]}
    except SyntaxError as e:
        return {"error": str(e), "final_code": ""}

def route(state: State) -> str:
    if not state.get("error") or state["attempts"] >= 3:
        return "done"
    return "retry"

# Step 4: Define the edges
builder = StateGraph(State)
builder.add_node("generate", generate_code)
builder.add_node("validate", validate_code)
builder.add_edge(START, "generate") # or `graph_builder.set_entry_point("generate")`
builder.add_edge("generate", "validate")
builder.add_conditional_edges(
    "validate",
    route,
    {"retry": "generate", "done": END},
)

graph = builder.compile()

# Step 5: Run the graph
result = graph.invoke({
    "task": "calculate the factorial of a number recursively",
    "code": "",
    "error": None,
    "attempts": 0,
    "final_code": "",
})

# Print the result
print("Final Code:")
print(result["final_code"])
print(f"Generated in {result['attempts']} attempt(s)")