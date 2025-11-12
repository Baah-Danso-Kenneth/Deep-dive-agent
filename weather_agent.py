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


def weather_node(state: AgentState) -> AgentState:
    """
    Weather Node: Fetches real weather data for a city
    Uses openWeatherMap API
    """
    user_query = state["user_query"]

    query_lower = user_query.lower()

    city = None
    if " in " in query_lower:
        parts = query_lower.split(" in ")
        if len(parts) > 1:
            city = parts[1].split()[0].strip('?.,!')
    elif " for " in query_lower:
        parts = query_lower.split(" for ")
        if len(parts) > 1:
            city =parts[1].split()[0].strip('?.,!')
    else:
        words = query_lower.split()
        for word in reversed(words):
            if word not in ["weather", "temperature"]:
                city = word.strip('?.,!')
            break

    if not city:
        return {
            "weather_result": "Could not identify city",
            "messages": [SystemMessage(content="Failed to to extract city name")]
        }
    print(f" -> Detected city: {city.capitalize()}")

    mock_weather_data = {
        "london": {"temp": 15, "condition": "Cloudy", "humidity": 75},
        "paris": {"temp": 18, "condition": "Sunny", "humidity": 60},
        "tokyo": {"temp": 22, "condition": "Clear", "humidity": 55},
        "new york": {"temp": 12, "condition": "Rainy", "humidity": 80},
        "ghana": {"temp": 28, "condition": "Partly Cloudy", "humidity": 70},
        "accra": {"temp": 28, "condition": "Hot and Humid", "humidity": 75},
        "nigeria": {"temp": 30, "condition": "Sunny", "humidity": 65},
        "lagos": {"temp": 29, "condition": "Humid", "humidity": 80},
    }

    weather = mock_weather_data.get(city.lower(), {"temp": 20, "condition": "Clear", "humidity": 50})
    weather_result = (
        f"Weather in {city.capitalize()}: "
        f"{weather['temp']}°C, {weather['condition']}, "
        f"Humidity: {weather['humidity']}%"
    )

    print(f"   -> {weather_result}")
    return {
        "weather_result": weather_result,
        "messages": [SystemMessage(content=f"weather data retrieved for {city}")]
    }




if __name__ == '__main__':
    weather_queries = [
        "What's is the weather in Ghana?",
        "Convert 75 fahrenheit to Celsius",
        "What time is it in Nigeria?",
        "5 star hotels in Ghana",
        "What is the population of Nigeria"
    ]

    for query in weather_queries:
        print(f"\nQuery: '{query}'")

        test_state = AgentState(
            messages = [],
            user_query = query,
            route_taken="weather",
            weather_result=None,
            calculator_result=None,
            datetime_result=None,
            tavily_result=None,
            final_answer=''
        )

        routed = router_node(test_state)
        print(f"Route : {routed['route_taken']}")

        if routed['route_taken'] == 'weather':
            result = weather_node((test_state))
            print(f" Result:  {result['weather_result']}")