from typing import TypedDict, Literal, Annotated
import operator
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from tavily import TavilyClient
import os

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, operator.add]


# System prompt - DEFINE THIS FIRST
SYSTEM_PROMPT = """You are a helpful assistant with access to ONLY these tools:
1. get_weather - for weather queries
2. calculate - for mathematical calculations

IMPORTANT RULES:
- ONLY use the tools listed above
- If you don't have a tool for the task, say "I don't have the right tool for this"
- DO NOT try to call any other tools like brave_search, web_search, etc.
- For general knowledge questions, say you need to search the web
"""


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Use this when user asks about weather or temperature."""
    weather_db = {
        "accra": "28°C, Sunny",
        "london": "15°C, Cloudy",
        "lagos": "30°C, Hot",
        "paris": "18°C, Rainy"
    }
    return weather_db.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations. ONLY use for math operations like 2+2, 50*3, etc."""
    try:
        result = eval(expression)
        return f"The answer is {result}"
    except Exception as e:
        return f"Cannot calculate: {str(e)}"


def perform_web_search(query: str) -> str:
    """Search the web using Tavily"""
    print("\n TAVILY FALLBACK: Searching the web...")
    try:
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily_client.search(query, max_results=3)

        results = []
        for result in response.get('results', []):
            title = result.get('title', '')
            content = result.get('content', '')
            results.append(f"• {title}: {content}")

        return "\n\n".join(results) if results else "No results found"
    except Exception as e:
        return f"Search failed: {str(e)}"


# Bind tools
tools = [get_weather, calculate]
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent(state: State) -> State:
    """The LLM decides which tool to call or admits it needs more info"""
    messages = state["messages"]

    # Add system prompt if not already there
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm_with_tools.invoke(messages)

    if (not hasattr(response, "tool_calls") or not response.tool_calls):
        content_lower = response.content.lower()

        # Trigger web search if LLM says it doesn't know
        if any(phrase in content_lower for phrase in
               ["i don't have", "i need to search", "don't have the right tool",
                "i cannot", "i'm not sure", "no information"]):
            print("\n LLM indicated it needs web search...")
            original_query = [m for m in messages if isinstance(m, HumanMessage)][-1].content
            search_result = perform_web_search(original_query)

            # Let LLM answer with search results
            final_response = llm.invoke([
                SystemMessage(content="Answer the user's question based on the web search results below."),
                HumanMessage(content=f"Question: {original_query}\n\nWeb search results:\n{search_result}")
            ])

            return {"messages": [response, final_response]}

    return {"messages": [response]}


def should_continue(state: State) -> Literal["tools", "end"]:
    """If LLM called a tool, go to tools node. Otherwise end."""
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


def create_graph():
    workflow = StateGraph(State)

    tool_node = ToolNode(tools)

    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    workflow.add_edge("tools", "agent")

    return workflow.compile()


if __name__ == "__main__":
    app = create_graph()

    test_queries = [
        "What is the weather in Accra?",
        "Calculate 25 + 75",
        "Who is the president of Ghana?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"{'=' * 60}")

        result = app.invoke({
            "messages": [HumanMessage(content=query)]
        })

        final_message = result["messages"][-1]
        print(f"\n Answer: {final_message.content}")