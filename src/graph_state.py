"""State schema for the LangGraph agent."""

from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    State of the agent that contains the list of messages.
    'add_messages' is a reducer that appends new messages to the existing list.
    """
    messages: Annotated[list, add_messages]
