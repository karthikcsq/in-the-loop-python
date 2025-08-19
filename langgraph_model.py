import os
from typing import Optional, TypedDict, Annotated

import dotenv
from langgraph.graph import StateGraph, END, add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from user_feedback_tool import (
    ask_user_for_input,
    common_essay_queries,
    common_code_queries,
)

# For debugging and visualization
import json
from datetime import datetime

dotenv.load_dotenv()


class State(TypedDict, total=False):
    essay_prompt: str
    draft: Optional[str]
    messages: Annotated[list[BaseMessage], add_messages]
    task_type: str  # e.g., "essay" | "code" | other


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
    
    if extra_info:
        print(f"   Info: {extra_info}")
    print("-" * 60)


# Register available tools
tools = [ask_user_for_input, common_essay_queries, common_code_queries]
# Note: We handle tool execution manually below to support interrupt-based flows.


def agent_node(state: State):
    """Agent that decides whether to use tools or proceed to drafting."""
    log_step("AGENT_NODE", state, "Analyzing prompt and deciding on tools")
    
    # Get existing conversation messages (includes user interactions)
    existing_messages = state.get('messages', [])
    
    # Initialize ChatOpenAI with tools bound
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0
    ).bind_tools(tools)
    
    # Build the conversation with system message + original request + any user interactions
    messages = [
        SystemMessage(content=(
            "You are a planning agent that ensures essays have the right guidance. "
            "Your job is to analyze the essay prompt and conversation history to determine if you need more information.\n\n"
            "AVAILABLE TOOLS:\n"
            "- ask_user_for_input: Ask the user any question with optional multiple choice options.\n"
            "- common_essay_queries: Returns a helpful list of questions to clarify essay requirements.\n"
            "- common_code_queries: Returns a helpful list of questions to clarify coding requirements.\n"
            "WHEN to call the tool:\n"
            "- The prompt is vague or incomplete\n"
            "- Important essay parameters are missing\n"
            "- You need clarification on requirements\n\n"
            "WHEN NOT to call the tool:\n"
            "- You have sufficient information from the conversation history\n"
            "- The user has already provided adequate guidance\n"
            "- The prompt and previous interactions give you enough context\n\n"
            "If task type ('task_type' in state) is provided by the user, use it. Otherwise you may ask clarifying questions, "
            "but do not attempt to set or guess state keys yourself."
            "Review the conversation history below. If you have enough information to produce the final output, respond with 'READY'. "
            "If you need more information, use the ask_user_for_input tool."
        )),
        HumanMessage(content=f"User request: {state.get('essay_prompt', '')}")
    ]
    
    # Add any previous conversation messages (user interactions)
    messages.extend(existing_messages)

    # Call the model
    response = model.invoke(messages)
    
    # Check if model chose to use tools
    if response.tool_calls:
        messages_to_add: list[BaseMessage] = []
        tool_messages: list[ToolMessage] = []
        # Include the assistant message that initiated the tool calls for proper context
        messages_to_add.append(response)
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {}) or {}
            tool_call_id = tool_call.get("id") or "tool_call"

            if tool_name == "ask_user_for_input":
                # Use the interrupt-capable tool to gather user input
                params = {
                    "query": tool_args.get("query", "Please provide more information"),
                    "options": tool_args.get("options"),
                    "current_prompt": state.get("essay_prompt", ""),
                }
                return ask_user_for_input.invoke(params)
            elif tool_name == "common_essay_queries":
                # Invoke helper tool and add its result to the conversation
                result = common_essay_queries.invoke({})
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
            elif tool_name == "common_code_queries":
                result = common_code_queries.invoke({})
                tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
            else:
                # Unknown tool – inform the model
                tool_messages.append(ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_call_id))

        if tool_messages:
            # Loop back to the agent with tool outputs appended to history
            messages_to_add.extend(tool_messages)
            return Command(update={"messages": messages_to_add}, goto="agent")
    
    # No tools needed, proceed to draft
    return Command(goto="final_output")


def final_output_node(state: State):
    """Generate final output using the request plus conversation history."""
    log_step("FINAL_OUTPUT_NODE", state, "Generating task-aware output with conversation context")
    
    # Initialize ChatOpenAI
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    )
    
    # Get existing conversation messages (includes user interactions)
    existing_messages = state.get('messages', [])
    
    # Create messages using conversation history for better context
    raw_task_type = (state.get("task_type") or "essay").strip().lower()
    task_type = "code" if raw_task_type.startswith("cod") else ("essay" if raw_task_type.startswith("essay") else "other")
    if task_type == "code":
        sys_prompt = (
            "You are an expert software engineer. Based on the request and conversation history, "
            "produce the final code or patch needed. Provide minimal, self-contained output. "
            "If options or constraints were given, adhere to them. Prefer clarity and correctness." 
        )
        default_prompt = "Write a small, self-contained code snippet that satisfies the request."
    elif task_type == "essay":
        sys_prompt = (
            "You are an expert essay writer. Based on the original request and the conversation history, "
            "write a high-quality essay that incorporates the user's requirements and preferences. "
            "The conversation history shows tone, audience, length, and focus. Write a tailored essay."
        )
        default_prompt = "Write a short essay."
    else:
        sys_prompt = (
            "You are a highly capable assistant. Based on the request and conversation history, "
            "produce the requested final output in a clear, concise, and actionable form appropriate to the task. "
            "Use structured formatting (bullets, steps, tables) when it improves clarity."
        )
        default_prompt = "Produce the requested output."

    user_prompt = state.get("essay_prompt") or default_prompt

    # Build the complete message history for the essay writer
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Original request: {user_prompt}")
    ]
    
    # Add the conversation history so the writer/engineer knows user preferences
    if existing_messages:
        messages.append(HumanMessage(content="Here's our conversation about your requirements:"))
        messages.extend(existing_messages)
        if task_type == "code":
            messages.append(HumanMessage(content="Now produce the final code or patch incorporating all the above requirements."))
        elif task_type == "essay":
            messages.append(HumanMessage(content="Now please write the essay incorporating all the above requirements."))
        else:
            messages.append(HumanMessage(content="Now produce the final output incorporating all the above requirements."))
    else:
        if task_type == "code":
            messages.append(HumanMessage(content="Please produce the final code based on the request above."))
        elif task_type == "essay":
            messages.append(HumanMessage(content="Please write the essay based on the request above."))
        else:
            messages.append(HumanMessage(content="Please produce the final output based on the request above."))

    # Call the model
    response = model.invoke(messages)
    
    content = response.content if response.content else ""
    if task_type == "code":
        final = f"Code Output:\n\n{content}"
    elif task_type == "essay":
        final = f"Essay:\n\n{content}"
    else:
        final = f"Final Output:\n\n{content}"
    return {"draft": final}


def build_app():
    builder = StateGraph(State)
    builder.add_node("agent", agent_node)
    builder.add_node("final_output", final_output_node)

    builder.set_entry_point("agent")
    builder.add_edge("final_output", END)

    return builder.compile(checkpointer=MemorySaver())


def main():
    app = build_app()
    thread_id = "essay-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    app.get_graph().draw_mermaid()  # Visualize the graph
    # Start the run with a simple prompt - the agent will ask for details
    # initial_state = {
    #     "essay_prompt": "Write me an essay about climate change.",
    #     "task_type": "essay",  # set to "code" for coding tasks
    # }
    initial_state = {
        "essay_prompt": "Write me code to reverse a string.",
        "task_type": "code"
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

    # Print the final output.
    if isinstance(result, dict) and "draft" in result:
        print("\n" + "="*50)
        print("📝 FINAL OUTPUT")
        print("="*50)
        print(result["draft"])
    else:
        print("Unexpected result:", result)


if __name__ == "__main__":
    main()

