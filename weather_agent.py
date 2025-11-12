from typing import TypedDict, Literal
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


# 1. STATE
class State(TypedDict):
    messages: list


# 2. DEFINE TOOLS - LLM will see these and choose what to use
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
    """Perform mathematical calculations. Use when user asks to calculate or do math."""
    try:
        result = eval(expression)
        return f"The answer is {result}"
    except Exception as e:
        return f"Cannot calculate: {str(e)}"


@tool
def search_web(query: str) -> str:
    """Search the web for current information. Use this for general questions, facts, or anything not covered by other tools."""
    # THIS IS WHERE TAVILY WOULD GO
    # For now, mock response
    return f"Search results for '{query}': This is where Tavily would return real web results about your question."


# 3. BIND TOOLS TO LLM - This is the injection!
tools = [get_weather, calculate, search_web]
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)  # <-- THE MAGIC HAPPENS HERE


# 4. AGENT NODE - LLM decides which tool to call
def agent(state: State) -> State:
    """The LLM looks at the query and decides: call a tool or respond directly"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# 5. ROUTER - Check if LLM wants to use tools or is done
def should_continue(state: State) -> Literal["tools", "end"]:
    """If LLM called a tool, go to tools node. Otherwise end."""
    last_message = state["messages"][-1]

    # If LLM made tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise we're done
    return "end"


# 6. BUILD GRAPH
def create_graph():
    workflow = StateGraph(State)

    # Create tool node that executes whatever tool the LLM chose
    tool_node = ToolNode(tools)

    # Add nodes
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)

    # Start with agent
    workflow.set_entry_point("agent")

    # After agent, check if we need tools or are done
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    # After tools execute, go back to agent to respond with results
    workflow.add_edge("tools", "agent")

    return workflow.compile()


# 7. RUN IT
if __name__ == "__main__":
    app = create_graph()

    test_queries = [
        "What's the weather in Lagos?",
        "Calculate 100 + 250",
        "Who is the president of Ghana?",  # Will use search_web (Tavily fallback)
        "What is 5 times 8?"
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"{'=' * 60}")

        result = app.invoke({
            "messages": [HumanMessage(content=query)]
        })

        # Print the final answer
        final_message = result["messages"][-1]
        print(f"Answer: {final_message.content}")

        # Show which tool was used (if any)
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"Tool used: {msg.tool_calls[0]['name']}")