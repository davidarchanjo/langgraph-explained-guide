from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

# Step 1: Initialize the LLM
llm = init_chat_model(model="gpt-4o", model_provider="openai")

# Step 2: Define the State
class State(TypedDict):
    user_input: str
    intent: str
    response: str

# Step 3: Define the Nodes
def classify_intent(state: State) -> dict:
    prompt = (
        "Classify the user's primary intent as exactly one word: "
        "question, complaint, or compliment. "
        "A message expressing dissatisfaction is a complaint, even if phrased as a question.\n\n"
        f"Message: {state['user_input']}"
    )
    intent = llm.invoke(prompt).content.strip().lower() # type: ignore
    return {"intent": intent}

def handle_question(state: State) -> dict:
    response = llm.invoke(f"Answer helpfully: {state['user_input']}")
    return {"response": response.content}

def handle_complaint(state: State) -> dict:
    response = llm.invoke(f"Respond empathetically: {state['user_input']}")
    return {"response": response.content}

def handle_compliment(state: State) -> dict:
    response = llm.invoke(f"Respond warmly: {state['user_input']}")
    return {"response": response.content}

# Step 4: Define the Conditional Edge
def route_by_intent(state: State) -> str:
    return state["intent"]

# Step 5: Define the Graph
builder = StateGraph(State)

builder.add_node("classify", classify_intent)
builder.add_node("question", handle_question)
builder.add_node("complaint", handle_complaint)
builder.add_node("compliment", handle_compliment)

builder.add_edge(START, "classify") # or `graph_builder.set_entry_point("classify")`

builder.add_conditional_edges(
    "classify",
    route_by_intent,
    {
        "question": "question",
        "complaint": "complaint",
        "compliment": "compliment",
    },
)

builder.add_edge("question", END)
builder.add_edge("complaint", END)
builder.add_edge("compliment", END)

graph = builder.compile()

# Step 6: Run the Graph
result = graph.invoke({
    "user_input": "Why is my order still not delivered?",
    "intent": "",
    "response": "",
})

# Print the result
print(f"Intent: {result['intent']}")
print(f"Response: {result['response']}")