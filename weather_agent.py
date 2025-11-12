from typing import TypedDict, Annotated, Optional, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

class AgentState(TypedDict):
    """
    The state object that gts passed between all nodes.
    Each node can read from and update these fields.
    """

    # Messages between user and agent  (accumulates with operator.add)
    messages: Annotated[List[BaseMessage], operator.add]

    # The original user query
    user_query: str

    # Track which route we took
    route_taken: str

    # Results from different sources
    weather_result: Optional[str]
    calculator_result: Optional[str]
    datetime_result: Optional[str]
    tavily_result: Optional[str]

    final_answer: str

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0
)



def router_node(state: AgentState)-> AgentState:
    """
    Router Node: Analyzes the user query and decides which tool/path to take.
    """
    user_query = state["user_query"].lower()

    if any(word in user_query for word in ["weather", "temperature", "forecast","rain","sunny"]):
        route = "weather"
    elif any(word in user_query for word in ["calculate", "convert", "plus", "minus", "multiply", "divide", "%"]):
        route = "calculator"
    elif any(word in user_query for word in ["time", "date", "timezone", "what day", "current time"]):
        route = "datetime"
    else:
        route = "tavily"
    print(f" -> Routing to: {route.upper()}")


    # Update state with the route decision
    return {
        "route_taken": route,
        "messages": [SystemMessage(content=f"Routing query to {route} handler")]
    }

if __name__ == '__main__':
    test_queries = [
        "What's is the weather in Ghana?",
        "Convert 75 fahrenheit to Celsius",
        "What time is it in Nigeria?",
        "5 star hotels in Ghana",
        "What is the population of Nigeria"
    ]

    for query in test_queries:
        test_state = AgentState(
            messages = [],
            user_query = query,
            route_taken="",
            weather_result=None,
            calculator_result=None,
            datetime_result=None,
            tavily_result=None,
            final_answer=''
        )

        result = router_node(test_state)
        print(f"\nQuery: '{query}'")
        print(f"Route : {result['route_taken']}")