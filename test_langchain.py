import os
from typing import Optional, TypedDict, Annotated

import dotenv
from langgraph.graph import StateGraph, END, add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from openai import OpenAI

# For debugging and visualization
import json
from datetime import datetime

dotenv.load_dotenv()


class State(TypedDict, total=False):
    essay_prompt: str
    draft: Optional[str]
    messages: Annotated[list[BaseMessage], add_messages]


def log_step(step_name: str, state: State, extra_info: str = ""):
    """Helper function to log what's happening at each step."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n [{timestamp}] {step_name}")
    print(f"   Essay Prompt: {state.get('essay_prompt', 'None')}")
    print(f"   Has Draft: {'Yes' if state.get('draft') else 'No'}")
    if extra_info:
        print(f"   Info: {extra_info}")
    print("-" * 60)


@tool
def set_tone() -> str:
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
        "options": ["Formal", "Informal", "Persuasive", "Friendly", "Neutral"],
        "type": "set_tone",
    })
    return f"Tone '{tone}' will be applied to the essay prompt."


# Define tools list for the agent
tools = [set_tone]
tool_node = ToolNode(tools)


def agent_node(state: State):
    """Agent that decides whether to use tools or proceed to drafting."""
    log_step("AGENT_NODE", state, "Analyzing prompt and deciding on tools")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please set it in your environment or .env file.")

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Let the model decide whether to use tools
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a planning agent that ensures essays have the right tone and style. "
                "Your job is to analyze the essay prompt and determine if it needs a tone specification.\n\n"
                "WHEN TO CALL set_tone tool:\n"
                "- The prompt does NOT explicitly mention any tone (formal, informal, persuasive, friendly, neutral, etc.)\n"
                "- The prompt does NOT specify writing style or voice\n"
                "- The prompt would benefit from a specific tone to be more effective\n"
                "- The topic or context suggests a particular tone would be appropriate\n\n"
                "WHEN NOT to call tools:\n"
                "- The prompt already contains tone words like 'formal', 'informal', 'persuasive', 'friendly', 'casual', 'academic', etc.\n"
                "- The prompt specifies a writing style like 'professional', 'conversational', 'argumentative', etc.\n"
                "- The prompt is very specific about how to write\n\n"
                "Examples:\n"
                "- 'Write about climate change' → NEEDS tone (call set_tone)\n"
                "- 'Write a formal essay about climate change' → Has tone (don't call tools)\n"
                "- 'Write a persuasive piece on recycling' → Has tone (don't call tools)\n"
                "- 'Explain quantum physics' → NEEDS tone (call set_tone)\n\n"
                "If you decide to call set_tone, do it. If the prompt already has adequate tone guidance, respond with 'READY'."
            )
        },
        {
            "role": "user",
            "content": f"Analyze this essay prompt and decide if it needs a tone: '{state.get('essay_prompt', '')}'"
        }
    ]

    # Define tools for OpenAI
    tools_def = [
        {
            "type": "function",
            "function": {
                "name": "set_tone",
                "description": "Apply a specific tone to the essay prompt. Use this when the prompt lacks tone guidance and would benefit from specifying how the essay should be written (formal, informal, persuasive, friendly, or neutral).",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False}
            }
        }
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools_def,
        tool_choice="auto",
        temperature=0
    )

    message = completion.choices[0].message
    
    # If model chose to use a tool, call it
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "set_tone":
                # Call our tool function and handle the interrupt
                tone = interrupt({
                    "query": "Select a tone for the essay", 
                    "options": ["Formal", "Informal", "Persuasive", "Friendly", "Neutral"],
                    "type": "set_tone"
                })
                
                # Update the prompt directly
                base_prompt = state.get("essay_prompt", "").rstrip()
                if base_prompt and not base_prompt.endswith("."):
                    base_prompt += "."
                new_prompt = f"{base_prompt} Use a {tone} tone.".strip()
                
                return Command(update={"essay_prompt": new_prompt}, goto="agent")
    
    # No tools needed, proceed to draft
    return Command(goto="draft")


def draft_node(state: State):
    """Call OpenAI using the (possibly updated) essay_prompt to produce a draft."""
    log_step("DRAFT_NODE", state, "Generating essay draft")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please set it in your environment or .env file.")

    client = OpenAI()
    sys_prompt = (
        "You are an expert essay writer. Write a concise, clear essay. "
        "Aim for 3-5 short paragraphs, avoid fluff, and keep it factual."
    )
    user_prompt = state.get("essay_prompt") or "Write a short essay."

    completion = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = completion.choices[0].message.content if completion.choices else ""
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
    initial_state = {"essay_prompt": "Write me an essay about global warming"}
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

