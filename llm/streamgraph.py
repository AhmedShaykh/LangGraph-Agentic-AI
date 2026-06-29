from langchain_core.messages import BaseMessage, HumanMessage;
from langgraph.graph import StateGraph, START, END;
from langchain_mistralai import ChatMistralAI;
from typing import TypedDict, Annotated;
from dotenv import load_dotenv;
from rich import print;
import operator;
import asyncio;

load_dotenv();

class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add];

llm = ChatMistralAI(model="mistral-large-latest", streaming=True);

def chatbot_node(state: State) -> dict:

    return {"messages": [llm.invoke(state["messages"])]};

def dummy_node(state: State) -> State:

    return state; # Return Same State In Double

builder = StateGraph(State);

builder.add_node("chatbot", chatbot_node);

builder.add_node("dummy", dummy_node);

builder.add_edge(START, "chatbot");

builder.add_edge("chatbot", "dummy");

builder.add_edge("dummy", END)

graph = builder.compile();

print("=== Method 1: stream_mode=updates ===");

for event in graph.stream(
    {
        "messages": [
            HumanMessage(
                content="List 3 benefits of LangGraph in one line each"
            )
        ]
    },
    stream_mode="updates" # Returns only each node's output.
):
    for node_name, output in event.items():

        print(f"  [{node_name}] {output["messages"]}");

print("\n=== Method 2: stream_mode=values ===");

for snapshot in graph.stream(
    {
        "messages": [
            HumanMessage(
                content="Say hello in 3 languages"
            )
        ]
    },
    stream_mode="values" # Returns the full state after each node.
):
    print(f"State has {len(snapshot["messages"])} message(s) now");

    print(snapshot["messages"]);

async def token_stream():

    print("\n========== astream_events() ==========\n");

    print("🤖 Bot: ", end="", flush=True);

    async for event in graph.astream_events( # Streams all graph events in real time
        {
            "messages": [
                HumanMessage(
                    content="Count from 1 to 5 slowly, one per line"
                )
            ]
        },
        version="v2" # Uses the latest event format.
    ):

        if event["event"] == "on_chat_model_stream":

            chunk = event["data"]["chunk"].content;

            if chunk:

                print(chunk, end="", flush=True);

        print(event);

asyncio.run(token_stream());