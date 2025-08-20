from langgraph.types import interrupt, Command
from typing import Optional, Dict
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

@tool
def ask_user_for_input(
    query: str,
    options: Optional[Dict[str, str]] = None,
) -> Command:
    """
    A flexible tool that can ask the user any question with optional multiple choice options.
    
    Instead of modifying the prompt, this tool adds the interaction as faux AI-User messages
    to preserve conversation context and provide better understanding for the AI.
    
    Args:
        query: The question or prompt to show the user (e.g., "What tone should the essay have?")
        options: Optional dictionary of options for multiple choice. Values are the option descriptions (optional).
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
        # If options is a dict of label->description
        if isinstance(options, dict):

            ai_question.content += f" Please choose from: {', '.join(f'{op}: {description}' for op, description in options.items())}"
        else:
            # Fallback for list[str]
            ai_question.content += f" Please choose from: {', '.join(options)}"

    user_response = HumanMessage(content=user_input)
    
    # Return a Command that adds the conversation messages to state
    return Command(update={"messages": [ai_question, user_response]}, goto="agent")
    
@tool
def common_essay_queries() -> dict:
    """
    Provides a structured list of common questions to clarify essay requirements.
    Returns a JSON-serializable object with optional sample options per question.
    """
    return {
        "type": "common_essay_queries",
        "questions": [
            {
                "id": "topic",
                "question": "What is the main topic or thesis of the essay?",
            },
            {
                "id": "audience",
                "question": "Who is the target audience for the essay?",
                "sample_options": {
                    "General audience": "Non-technical, broad readership",
                    "Academic": "Scholarly tone; citations expected",
                    "Professional": "Business/professional readers",
                    "Technical": "Engineers or domain specialists"
                },
            },
            {
                "id": "length",
                "question": "What is the desired length or word count?",
                "sample_options": {
                    "Short (300–500 words)": "Concise overview",
                    "Medium (800–1200 words)": "Standard assignment length",
                    "Long (1500–2500 words)": "In-depth analysis"
                },
            },
            {
                "id": "sources",
                "question": "Are there any specific sources or references to include?",
            },
            {
                "id": "tone",
                "question": "What tone or style should the essay adopt?",
                "sample_options": {
                    "Formal": "Objective, academic tone",
                    "Informal": "Conversational, approachable",
                    "Persuasive": "Argues a position",
                    "Expository": "Explains a concept clearly"
                },
            },
            {
                "id": "perspective",
                "question": "What perspective should be used?",
                "sample_options": {
                    "First person": "I/we",
                    "Third person": "He/she/they; objective narrator"
                },
            },
            {
                "id": "citation",
                "question": "Do you prefer a citation style?",
                "sample_options": {
                    "APA": "American Psychological Association",
                    "MLA": "Modern Language Association",
                    "Chicago": "Notes and bibliography"
                },
            },
            {
                "id": "focus",
                "question": "Are there particular arguments or points to emphasize?",
            },
        ],
    }

@tool
def common_code_queries() -> dict:
    """
    Provides a structured list of common questions to clarify coding requirements.
    Returns a JSON-serializable object with optional sample options per question.
    """
    return {
        "type": "common_code_queries",
        "questions": [
            {
                "id": "language",
                "question": "What programming language should be used?",
                "sample_options": {
                    "Python": "Great for quick scripting and data tasks",
                    "JavaScript": "Browser and Node.js support",
                    "TypeScript": "JS with types for better maintainability",
                    "Go": "Fast, static binaries, good for CLIs",
                    "Rust": "Performance and safety; steeper learning curve"
                },
            },
            {
                "id": "deliverable",
                "question": "What form should the deliverable take?",
                "sample_options": {
                    "Function": "Reusable function with signature",
                    "Script": "Standalone script you can run",
                    "Patch": "Diff against existing code",
                    "API endpoint": "HTTP handler or route",
                    "CLI": "Command-line tool"
                },
            },
            {
                "id": "frameworks",
                "question": "Any specific libraries or frameworks to consider?",
            },
            {
                "id": "io",
                "question": "What is the expected input and output format?",
                "sample_options": {
                    "StdIn/StdOut": "Read from stdin and print to stdout",
                    "Function args/return": "Pure function signature",
                    "HTTP JSON": "Accept and return JSON over HTTP"
                },
            },
            {
                "id": "constraints",
                "question": "Any performance, complexity, or environment constraints?",
            },
            {
                "id": "tests",
                "question": "Should I include tests? If so, a preferred framework?",
                "sample_options": {
                    "Yes, minimal": None,
                    "Yes, thorough": None,
                    "No tests": None
                },
            },
        ],
    }