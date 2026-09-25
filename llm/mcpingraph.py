from langchain_core.messages import HumanMessage, SystemMessage;
from langchain_mcp_adapters.client import MultiServerMCPClient;
from langgraph.graph import StateGraph, MessagesState, START;
from langgraph.prebuilt import ToolNode, tools_condition;
from datetime import datetime, timezone, timedelta;
from langchain_core.tools import tool;
from langchain_groq import ChatGroq;
from dotenv import load_dotenv;
import requests;
import asyncio;
import os;

load_dotenv();

@tool
def get_weather(city: str) -> str:

    """Get current weather for a city using OpenWeatherMap API."""

    api_key = os.getenv("OPENWEATHER_API_KEY");

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric";

    response = requests.get(url);

    if response.status_code != 200:

        return f"Error fetching weather for {city}: {response.text}";

    data = response.json();

    description = data["weather"][0]["description"].title();

    temp = data["main"]["temp"];

    feels_like = data["main"]["feels_like"];

    temp_min = data["main"]["temp_min"];
    
    temp_max = data["main"]["temp_max"];

    humidity = data["main"]["humidity"];

    pressure = data["main"]["pressure"];

    wind_speed = data["wind"]["speed"];

    visibility = data.get("visibility", 0) // 1000;

    timezone_offset = data["timezone"];

    tz = timezone(timedelta(seconds=timezone_offset));

    sunrise = datetime.fromtimestamp(data["sys"]["sunrise"], tz=tz).strftime("%I:%M %p");

    sunset = datetime.fromtimestamp(data["sys"]["sunset"], tz=tz).strftime("%I:%M %p");

    return (
        f"🌍 Weather in {city.title()}\n"
        f"{'─' * 30}\n"
        f"🌤  Condition   : {description}\n"
        f"🌡  Temperature : {temp}°C (Feels like {feels_like}°C)\n"
        f"🔼  High / Low  : {temp_max}°C / {temp_min}°C\n"
        f"💧  Humidity    : {humidity}%\n"
        f"🔵  Pressure    : {pressure} hPa\n"
        f"💨  Wind Speed  : {wind_speed} m/s\n"
        f"👁   Visibility  : {visibility} km\n"
        f"🌅  Sunrise     : {sunrise}\n"
        f"🌇  Sunset      : {sunset}\n"
    );

SYSTEM_PROMPT = """
                You are a helpful assistant with weather and calculator tools. When you use the get_weather tool, 
                return its output EXACTLY as-is, do not rephrase or summarize it.7
                """;

def call_llm(state: MessagesState, llm_with_tools):

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"];

    response = llm_with_tools.invoke(messages);

    return {"messages": [response]};

async def main():

    groq_api_key = os.getenv("GROQ_API_KEY");

    weather_api = os.getenv("OPENWEATHER_API_KEY");

    if not groq_api_key:

        raise ValueError("GROQ_API_KEY not found in .env or environment variables.");

    if not weather_api:

        raise ValueError("OPENWEATHER_API_KEY not found in .env or environment variables.");

    client = MultiServerMCPClient(
        {
            "calculator": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "mcp_server_calculator"]
            }
        }
    );

    mcp_tools = await client.get_tools();

    all_tools = mcp_tools + [get_weather];

    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=groq_api_key
    );

    llm_with_tools = llm.bind_tools(all_tools);

    graph_builder = StateGraph(MessagesState);

    graph_builder.add_node("llm", lambda state: call_llm(state, llm_with_tools));

    graph_builder.add_node("tools", ToolNode(all_tools));

    graph_builder.add_edge(START, "llm");

    graph_builder.add_conditional_edges("llm", tools_condition);

    graph_builder.add_edge("tools", "llm");

    graph = graph_builder.compile();

    print("MCP Agent Ready! Type exit to quit.\n");

    while True:

        user_input = input("You: ").strip();

        if user_input.lower() in ["exit", "quit"]:

            print("Goodbye!");

            break;

        if not user_input:

            continue;

        result = await graph.ainvoke( # Asynchronous Invoke
            {"messages": [HumanMessage(content=user_input)]}
        );

        final_message = result["messages"][-1];

        print(f"\nAgent: {final_message.content}\n");

if __name__ == "__main__":

    asyncio.run(main());