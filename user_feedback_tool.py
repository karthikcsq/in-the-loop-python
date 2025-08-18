from langgraph.types import interrupt, Command
from typing import Optional, TypedDict, Annotated
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

@tool
def ask_user_for_input(
    query: str,
    options: Optional[list[str]] = None,
    current_prompt: str = "",
) -> Command:
    """
    A flexible tool that can ask the user any question with optional multiple choice options.
    
    Instead of modifying the prompt, this tool adds the interaction as faux AI-User messages
    to preserve conversation context and provide better understanding for the AI.
    
    Args:
        query: The question or prompt to show the user (e.g., "What tone should the essay have?")
        options: Optional list of choices for the user (e.g., ["Formal", "Informal", "Persuasive"])
    current_prompt: The current essay prompt (unchanged)
        
    Returns:
        Command to update state with new messages and route back to agent
    """
        
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
    
    # Return a Command that adds the conversation messages to state
    return Command(update={"messages": [ai_question, user_response]}, goto="agent")
    
@tool
def common_essay_queries() -> str:
    """
    Provides a set of common questions that can be asked to clarify essay requirements.
    """
    return """
    1. What is the main topic or thesis of the essay?
    2. Who is the target audience for the essay?
    3. What is the desired length or word count?
    4. Are there any specific sources or references to include?
    5. What tone or style should the essay adopt (e.g., formal, informal, persuasive)?
    6. Are there any particular arguments or points to emphasize?
    """

@tool
def common_code_queries() -> str:
    """
    Provides a set of common questions that can be asked to clarify coding requirements.
    """
    return """
    1. What programming language should be used?
    2. Are there any specific libraries or frameworks to consider?
    3. What is the expected input and output format?
    4. Are there any performance or optimization requirements?
    """