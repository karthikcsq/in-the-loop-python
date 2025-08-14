# In-The-Loop Essay Agent 🤖✍️

A LangGraph-based AI agent that writes essays with interactive user input using a revolutionary **faux messages approach** for better context understanding.

## 🚀 Key Innovation: Faux Messages Approach

Instead of modifying the essay prompt, our agent now uses **faux AI-User conversation messages** to preserve context. This provides dramatically better results because:

### 🧠 Why Faux Messages Are Superior

1. **Better Context**: The AI sees the full conversation flow, not just a modified prompt
2. **Natural Understanding**: Mimics how AI is trained on conversations
3. **Preserved Original Intent**: The original prompt stays clean and focused
4. **Rich Context**: AI can reference previous interactions naturally
5. **Scalable**: Can handle unlimited interactions without prompt bloat

### 🔄 How It Works

```
Original Prompt: "Write an essay about climate change"

Instead of modifying to: "Write an essay about climate change. Use formal tone. Target students. 500 words."

We create conversation history:
🤖 AI: "I need to clarify something: What tone should the essay have?"
👤 User: "Formal and academic"
🤖 AI: "I need to clarify something: Who is your target audience?"
👤 User: "College students"
🤖 AI: "I need to clarify something: How long should it be?"
👤 User: "500 words"
```

The essay writer then sees this natural conversation and produces much better results!

## 🛠️ Technical Implementation

### State Structure
```python
class State(TypedDict, total=False):
    essay_prompt: str                    # Original prompt (unchanged)
    draft: Optional[str]                 # Final essay
    messages: Annotated[list[BaseMessage], add_messages]  # Conversation history!
    tools_called: set[str]              # Interaction tracking
```

### The Flexible Tool
```python
@tool
def ask_user_for_input(
    query: str,                          # Question to ask
    options: Optional[list[str]] = None, # Optional choices
    current_prompt: str = "",            # Original prompt (unchanged)
    tools_called: set = None             # History tracking
) -> Command:
    """Creates faux AI-User messages instead of modifying the prompt."""
    
    # Get user input
    user_input = interrupt({"query": query, "options": options})
    
    # Create conversation messages
    ai_question = AIMessage(content=f"I need to clarify something: {query}")
    user_response = HumanMessage(content=user_input)
    
    # Add to conversation history
    return Command(
        update={"messages": [ai_question, user_response]},
        goto="agent"
    )
```

### Enhanced Agent Node
```python
def agent_node(state: State):
    # Get conversation history
    existing_messages = state.get('messages', [])
    
    # Build complete conversation context
    messages = [
        SystemMessage(content="Analyze the conversation history..."),
        HumanMessage(content=f"Write an essay: {state['essay_prompt']}")
    ]
    
    # Add previous interactions for context
    messages.extend(existing_messages)
    
    # AI decides based on full conversation
    response = model.invoke(messages)
```

### Enhanced Draft Node
```python
def draft_node(state: State):
    # Use conversation history for essay writing
    messages = [
        SystemMessage(content="Write essay based on conversation history..."),
        HumanMessage(content=f"Original: {state['essay_prompt']}")
    ]
    
    # Include conversation for context
    if existing_messages:
        messages.append(HumanMessage(content="Our conversation:"))
        messages.extend(existing_messages)
        messages.append(HumanMessage(content="Write the essay with these requirements."))
    
    response = model.invoke(messages)
```

## 📂 Project Structure

```
├── langgraph_model.py        # Main agent with faux messages approach
├── user_feedback_tool.py     # Flexible tool creating conversation messages
├── test_flexible_tool.py     # Test demonstrating the conversation approach
└── README.md                 # This file
```

## 🔧 Installation

1. **Set up environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install langgraph langchain-openai python-dotenv
   ```

2. **Configure OpenAI API**:
   ```env
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_TEMPERATURE=0.7
   ```

## 🚀 Usage

### Interactive Mode
```bash
python langgraph_model.py
```

Sample interaction:
```
🤖 Agent asks: What tone should the essay have?
   Available options: Formal, Informal, Academic, Persuasive
> Academic

🤖 Agent asks: Who is your target audience?
   Available options: Students, Professionals, General Public
> Students

🤖 Agent asks: How long should the essay be?
> 800 words

🤖 Agent asks: Any specific focus areas?
> Environmental solutions and policy recommendations

📝 FINAL ESSAY
[High-quality essay incorporating all conversation context]
```

### Testing Mode
```bash
python test_flexible_tool.py
```

## 🆚 Before vs After Comparison

### Before: Prompt Modification Approach ❌
```
Original: "Write about climate change"
Modified: "Write about climate change. Use academic tone. Target students. 800 words. Focus on solutions."

Problems:
- Loses conversational context
- Feels unnatural to the AI
- Can become unwieldy with many modifications
- No memory of how decisions were made
```

### After: Faux Messages Approach ✅
```
Original: "Write about climate change" (unchanged)
Conversation:
- AI: "What tone?" → User: "Academic"  
- AI: "Who's the audience?" → User: "Students"
- AI: "How long?" → User: "800 words"
- AI: "Focus areas?" → User: "Solutions"

Benefits:
- Natural conversation flow
- AI understands decision context
- Unlimited interactions possible
- Better essay quality
```

## ✨ Key Benefits

1. **🎯 Better Results**: Essays that truly reflect user intent
2. **🔄 Natural Flow**: Conversational interactions feel more human
3. **📈 Scalable**: Handle any number of clarifications without issues
4. **🧠 Smart Context**: AI understands the "why" behind requirements
5. **🚀 Future-Proof**: Works with any AI model trained on conversations
6. **🔧 Maintainable**: Cleaner, more intuitive code structure

## 🔬 Example Conversation Analysis

The AI now sees rich context like:
```
Messages: [
  AIMessage("I need to clarify: What tone should the essay have?"),
  HumanMessage("Academic and formal"),
  AIMessage("I need to clarify: Who is your target audience?"), 
  HumanMessage("Graduate students studying environmental science"),
  AIMessage("I need to clarify: How long should it be?"),
  HumanMessage("1200-1500 words"),
  AIMessage("I need to clarify: Any specific aspects to focus on?"),
  HumanMessage("Focus on recent technological solutions and policy frameworks")
]
```

This gives the essay writer incredible context to work with!

---

**🎉 The faux messages approach represents a significant leap forward in AI agent design, providing more natural interactions and superior results! 🎯**
