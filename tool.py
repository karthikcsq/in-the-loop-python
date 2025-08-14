from langgraph.types import interrupt, Command
from typing import Optional, TypedDict, Annotated
from langchain_core.tools import tool


class UserFeedbackTool:
    """Tool for collecting user feedback on a prompt."""
    
    def __init__(self,
                 name: str,
                 description: str,
                 query: str,
                 options: Optional[list[str]] = None,
                 prompt_mod: str = ""):
        self.name = name
        self.description = description
        self.query = query
        self.options = options
        self.prompt_mod = prompt_mod
        self.tool = self.create_tool_with_interrupt()
    

    def create_tool_with_interrupt(self) -> tool:
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
        
        def dynamic_tool(current_prompt: str) -> Command:
            
            # Create interrupt payload
            interrupt_payload = {"query": self.query}
            if self.options:
                interrupt_payload["options"] = self.options

            # Get user input through interrupt
            user_input = interrupt(interrupt_payload)
            
            # Update the prompt
            base_prompt = current_prompt.rstrip()
            if base_prompt and not base_prompt.endswith("."):
                base_prompt += "."
                
            # Apply the prompt modification
            if self.prompt_mod:
                new_prompt = f"{base_prompt} {self.prompt_mod.format(user_input=user_input)}".strip()
            else:
                new_prompt = f"{base_prompt} {user_input}".strip()
            
            # Add this tool to the set of called tools
            updated_tools_called = tools_called.copy()
            updated_tools_called.add(self.name)

            # Return a Command that updates state and routes back to agent
            return Command(
                update={
                    "essay_prompt": new_prompt,
                    "tools_called": updated_tools_called
                },
                goto="agent"
            )
        
        # Set the proper docstring for the function
        dynamic_tool.__doc__ = f"""{self.description}

        Args:
            current_prompt: The current essay prompt to modify
            
        Returns:
            Command to update state and route back to agent
        """
        
        # Set the tool name dynamically
        dynamic_tool.__name__ = self.name

        # Create the tool with explicit description
        return tool(
            description=f"Apply {self.name} to the essay prompt. This tool prompts the user to provide input for {self.name} and modifies the prompt accordingly."
        )(dynamic_tool)