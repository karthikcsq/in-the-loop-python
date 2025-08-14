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

# For debugging and visualization
import json
from datetime import datetime

dotenv.load_dotenv()


def create_tool_with_interrupt(name: str, query: str, options: Optional[list[str]] = None, prompt_mod: str = "") -> tool:
    """
    Function wrapper that creates a tool with an interrupt and prompt modification.
    
    Args:
        name: The name of the tool to create
        query: The query to display to the user during the interrupt
        options: Optional list of options to present to the user
        prompt_mod: String to modify the prompt with (will be appended to the prompt)
    
    Returns:
        A LangChain tool function that can be used in the graph
    """
    
    def dynamic_tool(current_prompt: str, tools_called: set) -> Command:
        f"""Apply {name} to the essay prompt.
        
        This tool prompts the user to provide input for {name} and modifies the prompt accordingly.
        """
        
        # Create interrupt payload
        interrupt_payload = {"query": query}
        if options:
            interrupt_payload["options"] = options
            
        # Get user input through interrupt
        user_input = interrupt(interrupt_payload)
        
        # Update the prompt
        base_prompt = current_prompt.rstrip()
        if base_prompt and not base_prompt.endswith("."):
            base_prompt += "."
            
        # Apply the prompt modification
        if prompt_mod:
            new_prompt = f"{base_prompt} {prompt_mod.format(user_input=user_input)}".strip()
        else:
            new_prompt = f"{base_prompt} {user_input}".strip()
        
        # Add this tool to the set of called tools
        updated_tools_called = tools_called.copy()
        updated_tools_called.add(name)
        
        # Return a Command that updates state and routes back to agent
        return Command(
            update={
                "essay_prompt": new_prompt,
                "tools_called": updated_tools_called
            },
            goto="agent"
        )
    
    # Set the proper docstring for the function
    dynamic_tool.__doc__ = f"""Apply {name} to the essay prompt.
    
    This tool prompts the user to provide input for {name} and modifies the prompt accordingly.
    
    Args:
        current_prompt: The current essay prompt to modify
        tools_called: Set of tools that have already been called
        
    Returns:
        Command to update state and route back to agent
    """
    
    # Set the tool name dynamically
    dynamic_tool.__name__ = name
    
    # Create the tool with explicit description
    return tool(
        description=f"Apply {name} to the essay prompt. This tool prompts the user to provide input for {name} and modifies the prompt accordingly."
    )(dynamic_tool)


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
    tools_called = state.get('tools_called', set())
    print(f"   Tools Called: {list(tools_called) if tools_called else 'None'}")
    if extra_info:
        print(f"   Info: {extra_info}")
    print("-" * 60)


@tool
def set_tone(current_prompt: str, tools_called: set) -> Command:
    """Apply a specific tone to the essay prompt to improve writing quality.
    
    This tool should be used when the essay prompt lacks tone guidance and would benefit 
    from specifying how the essay should be written. It will prompt the user to select 
    from: Formal, Informal, Persuasive, Friendly, or Neutral tone.
    
    Use this tool when:
    - The prompt doesn't specify any tone or writing style
    - The topic would benefit from a particular tone
    - The essay needs clearer direction on how it should sound
    """
    tone = interrupt({
        "query": "Select a tone for the essay",
        # "options": ["Formal", "Informal", "Persuasive", "Friendly", "Neutral"],
    })
    
    # Update the prompt directly within the tool
    base_prompt = current_prompt.rstrip()
    if base_prompt and not base_prompt.endswith("."):
        base_prompt += "."
    new_prompt = f"{base_prompt} Use a {tone} tone.".strip()
    
    # Add this tool to the set of called tools
    updated_tools_called = tools_called.copy()
    updated_tools_called.add("set_tone")
    
    # Return a Command that updates state and routes back to agent
    return Command(
        update={
            "essay_prompt": new_prompt,
            "tools_called": updated_tools_called
        },
        goto="agent"
    )


# Example usage of the create_tool_with_interrupt function wrapper
# Create a tone setting tool using the wrapper
set_tone_dynamic = create_tool_with_interrupt(
    name="set_tone_dynamic",
    query="Select a tone for the essay",
    options=["Formal", "Informal", "Persuasive", "Friendly", "Neutral"],
    prompt_mod="Use a {user_input} tone."
)

# Create a word count tool using the wrapper
set_word_count = create_tool_with_interrupt(
    name="set_word_count",
    query="How many words should the essay be?",
    prompt_mod="The essay should be approximately {user_input} words long."
)

# Create a target audience tool using the wrapper
set_target_audience = create_tool_with_interrupt(
    name="set_target_audience", 
    query="Who is the target audience for this essay?",
    options=["General public", "Academic audience", "Children", "Professionals", "Students"],
    prompt_mod="Write this essay for {user_input}."
)


# Define tools list for the agent
tools = [set_tone_dynamic, set_word_count, set_target_audience]
tool_node = ToolNode(tools)


def agent_node(state: State):
    """Agent that decides whether to use tools or proceed to drafting."""
    log_step("AGENT_NODE", state, "Analyzing prompt and deciding on tools")
    
    # Get the set of tools already called
    tools_called = state.get('tools_called', set())
    
    # Initialize ChatOpenAI with tools bound
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0
    ).bind_tools(tools)

    # Create messages using LangChain message types
    tools_called_list = list(tools_called) if tools_called else []
    tools_called_str = f"Tools already called: {tools_called_list}" if tools_called_list else "No tools have been called yet."
    
    messages = [
        SystemMessage(content=(
            "You are a planning agent that ensures essays have the right guidance. "
            "Your job is to analyze the essay prompt and determine if it needs improvements.\n\n"
            "AVAILABLE TOOLS:\n"
            "- set_tone: Apply a specific tone (formal, informal, persuasive, friendly, neutral)\n"
            "- set_tone_dynamic: Dynamic version of tone setting with user input\n"
            "- set_word_count: Specify the target word count for the essay\n"
            "- set_target_audience: Specify who the essay is written for\n\n"
            "WHEN TO CALL TOOLS:\n"
            "- set_tone/set_tone_dynamic: When prompt lacks tone specification\n"
            "- set_word_count: When prompt doesn't specify essay length\n"
            "- set_target_audience: When prompt doesn't specify who it's for\n\n"
            "WHEN NOT to call tools:\n"
            "- The prompt already contains the relevant information\n"
            "- A tool has already been called (avoid calling the same tool twice)\n"
            "- The prompt is very specific and complete\n\n"
            f"{tools_called_str}\n\n"
            "If you decide to call a tool that hasn't been called yet, do it. If the prompt already has adequate guidance, respond with 'READY'."
        )),
        HumanMessage(content=f"Complete this task: {state.get('essay_prompt', '')}")
    ]

    # Call the model
    response = model.invoke(messages)
    
    # Check if model chose to use tools
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            
            # Check if this tool has already been called
            if tool_name in tools_called:
                print(f"   Tool {tool_name} already called, skipping...")
                continue
                
            # Handle all tools generically using the tool registry
            for tool_func in tools:
                if tool_func.name == tool_name:
                    return tool_func.invoke({
                        "current_prompt": state.get("essay_prompt", ""),
                        "tools_called": tools_called
                    })
    
    # No tools needed, proceed to draft
    return Command(goto="draft")


def draft_node(state: State):
    """Call ChatOpenAI using the (possibly updated) essay_prompt to produce a draft."""
    log_step("DRAFT_NODE", state, "Generating essay draft")
    
    # Initialize ChatOpenAI
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    )
    
    # Create messages using LangChain message types
    sys_prompt = (
        "You are an expert essay writer. Write a concise, clear essay. "
        "Aim for 3-5 short paragraphs, avoid fluff, and keep it factual."
    )
    user_prompt = state.get("essay_prompt") or "Write a short essay."

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt)
    ]

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
    # Start the run
    initial_state = {
        "essay_prompt": "Write me an essay about global warming",
        "tools_called": set()
    }
    result = app.invoke(initial_state, config=config)

    # Generic interrupt/resume loop to support any future tools
    while isinstance(result, dict) and "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        query = payload.get("query", "Input")
        options = payload.get("options")
        
        print(f"   Query: {query}")
        if options:
            print("   Options:", ", ".join(options))
        
        user_value = input(f"\n{query} > ").strip() or "Neutral"
        print(f"   User selected: {user_value}")
        
        result = app.invoke(Command(resume=user_value), config=config)

    # Print the final draft.
    if isinstance(result, dict) and "draft" in result:
        print("\n--- Essay ---\n")
        print(result["draft"])
    else:
        print(result)


if __name__ == "__main__":
    main()

