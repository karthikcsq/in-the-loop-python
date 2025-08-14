import os
from typing import Optional, TypedDict, Annotated

import dotenv
from langgraph.graph import StateGraph, END, add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from user_feedback_tool import ask_user_for_input

# For debugging and visualization
import json
from datetime import datetime

dotenv.load_dotenv()


class State(TypedDict, total=False):
    essay_prompt: str
    draft: Optional[str]
    messages: Annotated[list[BaseMessage], add_messages]
    tools_called: set[str]


def log_step(step_name: str, state: State, extra_info: str = ""):
    """Helper function to log what's happening at each step."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n [{timestamp}] {step_name}")
    print(f"   Essay Prompt: {state.get('essay_prompt', 'None')}")
    print(f"   Has Draft: {'Yes' if state.get('draft') else 'No'}")
    
    # Show conversation history
    messages = state.get('messages', [])
    conversation_pairs = len([m for m in messages if hasattr(m, 'content') and 'I need to clarify' in str(m.content)])
    print(f"   User Interactions: {conversation_pairs}")
    
    tools_called = state.get('tools_called', set())
    print(f"   Tools Called: {list(tools_called) if tools_called else 'None'}")
    if extra_info:
        print(f"   Info: {extra_info}")
    print("-" * 60)


# Use the single flexible tool
tools = [ask_user_for_input]
tool_node = ToolNode(tools)


def agent_node(state: State):
    """Agent that decides whether to use tools or proceed to drafting."""
    log_step("AGENT_NODE", state, "Analyzing prompt and deciding on tools")
    
    # Get the set of tools already called
    tools_called = state.get('tools_called', set())
    
    # Get existing conversation messages (includes user interactions)
    existing_messages = state.get('messages', [])
    
    # Initialize ChatOpenAI with tools bound
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0
    ).bind_tools(tools)

    # Create messages using LangChain message types
    tools_called_list = list(tools_called) if tools_called else []
    interaction_count = len([m for m in existing_messages if hasattr(m, 'content') and 'I need to clarify' in str(m.content)])
    
    # Build the conversation with system message + original request + any user interactions
    messages = [
        SystemMessage(content=(
            "You are a planning agent that ensures essays have the right guidance. "
            "Your job is to analyze the essay prompt and conversation history to determine if you need more information.\n\n"
            "AVAILABLE TOOL:\n"
            "ask_user_for_input: A flexible tool that can ask the user any question with optional multiple choice options.\n"
            "Use this tool to gather missing information like:\n"
            "- Tone (formal, informal, persuasive, etc.)\n"
            "- Word count or length requirements\n"
            "- Target audience (students, professionals, general public, etc.)\n"
            "- Specific focus areas or requirements\n"
            "- Any other clarifications needed\n\n"
            "WHEN to call the tool:\n"
            "- The prompt is vague or incomplete\n"
            "- Important essay parameters are missing\n"
            "- You need clarification on requirements\n\n"
            "WHEN NOT to call the tool:\n"
            "- You have sufficient information from the conversation history\n"
            "- The user has already provided adequate guidance\n"
            "- The prompt and previous interactions give you enough context\n\n"
            "Review the conversation history below. If you have enough information to write a good essay, respond with 'READY'. "
            "If you need more information, use the ask_user_for_input tool."
        )),
        HumanMessage(content=f"Write an essay: {state.get('essay_prompt', '')}")
    ]
    
    # Add any previous conversation messages (user interactions)
    messages.extend(existing_messages)

    # Call the model
    response = model.invoke(messages)
    
    # Check if model chose to use tools
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name == "ask_user_for_input":
                # Call the flexible tool with the agent's parameters
                return ask_user_for_input.invoke({
                    "query": tool_args.get("query", "Please provide more information"),
                    "options": tool_args.get("options"),
                    "current_prompt": state.get("essay_prompt", ""),
                    "tools_called": tools_called
                })
    
    # No tools needed, proceed to draft
    return Command(goto="draft")


def draft_node(state: State):
    """Generate essay using the original prompt plus conversation history for context."""
    log_step("DRAFT_NODE", state, "Generating essay draft with conversation context")
    
    # Initialize ChatOpenAI
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    )
    
    # Get existing conversation messages (includes user interactions)
    existing_messages = state.get('messages', [])
    
    # Create messages using conversation history for better context
    sys_prompt = (
        "You are an expert essay writer. Based on the original request and the conversation history, "
        "write a high-quality essay that incorporates all the user's requirements and preferences. "
        "The conversation history shows what the user wants in terms of tone, audience, length, focus, etc. "
        "Use this information to write a tailored essay."
    )
    
    user_prompt = state.get("essay_prompt") or "Write a short essay."

    # Build the complete message history for the essay writer
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Original request: {user_prompt}")
    ]
    
    # Add the conversation history so the essay writer knows user preferences
    if existing_messages:
        messages.append(HumanMessage(content="Here's our conversation about your requirements:"))
        messages.extend(existing_messages)
        messages.append(HumanMessage(content="Now please write the essay incorporating all the above requirements."))
    else:
        messages.append(HumanMessage(content="Please write the essay based on the request above."))

    # Call the model
    response = model.invoke(messages)
    
    content = response.content if response.content else ""
    draft = f"Essay:\n\n{content}"
    return {"draft": draft}


def build_app():
    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("draft", draft_node)

    builder.set_entry_point("agent")
    builder.add_edge("draft", END)

    return builder.compile(checkpointer=MemorySaver())


def main():
    app = build_app()
    thread_id = "essay-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    app.get_graph().draw_mermaid()  # Visualize the graph
    # Start the run with a simple prompt - the agent will ask for details
    initial_state = {
        "essay_prompt": "Write me an essay about climate change.",
        "tools_called": set()
    }
    result = app.invoke(initial_state, config=config)

    # Generic interrupt/resume loop - the agent can now ask any question
    while isinstance(result, dict) and "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        query = payload.get("query", "Input")
        options = payload.get("options")
        
        print(f"\n🤖 Agent asks: {query}")
        if options:
            print(f"   Available options: {', '.join(options)}")
        
        user_value = input(f"\n> ").strip()
        if not user_value and options:
            user_value = options[0]  # Default to first option if provided
        elif not user_value:
            user_value = "No preference"
            
        print(f"   ✅ You answered: {user_value}")
        
        result = app.invoke(Command(resume=user_value), config=config)

    # Print the final draft.
    if isinstance(result, dict) and "draft" in result:
        print("\n" + "="*50)
        print("📝 FINAL ESSAY")
        print("="*50)
        print(result["draft"])
    else:
        print("Unexpected result:", result)


if __name__ == "__main__":
    main()

