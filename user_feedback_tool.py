from langgraph.types import interrupt, Command
from typing import Optional, TypedDict, Annotated
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

@tool
def ask_user_for_input(query: str, options: Optional[list[str]] = None, current_prompt: str = "", tools_called: set = None) -> Command:
    """
    A flexible tool that can ask the user any question with optional multiple choice options.
    
    Instead of modifying the prompt, this tool adds the interaction as faux AI-User messages
    to preserve conversation context and provide better understanding for the AI.
    
    Args:
        query: The question or prompt to show the user (e.g., "What tone should the essay have?")
        options: Optional list of choices for the user (e.g., ["Formal", "Informal", "Persuasive"])
        current_prompt: The current essay prompt (unchanged)
        tools_called: Set of tools that have already been called
        
    Returns:
        Command to update state with new messages and route back to agent
    """
    if tools_called is None:
        tools_called = set()
        
    # Create interrupt payload
    interrupt_payload = {"query": query}
    if options:
        interrupt_payload["options"] = options
        
    # Get user input through interrupt
    user_input = interrupt(interrupt_payload)
    
    # Create faux messages representing this interaction
    # This preserves the conversation flow for better AI understanding
    ai_question = AIMessage(content=f"I need to clarify something: {query}")
    if options:
        ai_question.content += f" Please choose from: {', '.join(options)}"
    
    user_response = HumanMessage(content=user_input)
    
    # Track this interaction
    updated_tools_called = tools_called.copy()
    updated_tools_called.add(f"user_input_{len(tools_called)}")
    
    # Return a Command that adds the conversation messages to state
    return Command(
        update={
            "messages": [ai_question, user_response],
            "tools_called": updated_tools_called
        },
        goto="agent"
    )