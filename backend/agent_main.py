###############################################
# 1. Environment Setup
###############################################
# Run these in your terminal or in requirements.txt
# pip install langgraph langchain-core langchain-groq pydantic requests python-dotenv fastapi uvicorn websockets

# .env file or export in terminal
# export GROQ_API_KEY="your_groq_api_key_here"

###############################################
# 2. Imports and Pydantic Models
###############################################
import os
import re
import json
import uuid
import asyncio
import requests
import datetime
from typing import List, Optional, TypedDict, Union, Dict, Any
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fancy_logger.console_logger import get_fancy_logger  # Custom logger for better output

load_dotenv()

# Configure logging
logger = get_fancy_logger()

API_BASE_URL = "http://localhost:8000"  # Change as needed
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set. Please set it in your .env file or export it in your terminal.")
    exit()

class CreateTodoInput(BaseModel):
    title: str
    description: Optional[str] = ""
    criticality: Optional[str] = Field("medium", description="Task criticality: low, medium, high")
    urgency: Optional[str] = Field("normal", description="Task urgency: normal or urgent")
    due_date: str  # Natural language (LLM inferred), e.g. "tomorrow"

class UpdateTodoInput(BaseModel):
    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    criticality: Optional[str] = None
    urgency: Optional[str] = None
    due_date: Optional[str] = None
    status: Optional[str] = None

class DeleteTodoInput(BaseModel):
    id: str

class ListTodosInput(BaseModel):
    pass  # Placeholder if you add filters later

###############################################
# 3. Agent State Management
###############################################
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    session_id: str

# Global dictionary to store agent states by session ID
agent_states: Dict[str, AgentState] = {}

###############################################
# 4. To-Do API Tool Definitions
###############################################
def get_headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

def parse_date(natural_date: str) -> str:
    try:
        today = datetime.date.today()
        if natural_date.lower() == "tomorrow":
            return str(today + datetime.timedelta(days=1))
        elif natural_date.lower() == "today":
            return str(today)
        elif natural_date.lower() == "yesterday":
            return str(today - datetime.timedelta(days=1))
        # Add more custom parsing rules here
        return str(datetime.datetime.strptime(natural_date, "%Y-%m-%d").date())
    except Exception as e:
        return str(today)

@tool("add_todo", return_direct=True)
def add_todo(input: CreateTodoInput, config: RunnableConfig) -> str:
    """Create a new to-do item."""
    logger.info(f"Adding To-Do with input: {input}")
    try:
        # Get token from the current state context
        token = config.get("configurable", {}).get("user_token")
        if not token:
            raise ValueError("No authentication context available")
        
        payload = input.dict()
        payload["due_date"] = parse_date(input.due_date)
        res = requests.post(f"{API_BASE_URL}/todos", json=payload, headers=get_headers(token))
        res.raise_for_status()
        return f"✅ To-Do '{input.title}' added successfully!"
    except Exception as e:
        logger.error(f"Failed to add To-Do: {e}")
        return f"❌ Failed to add To-Do: {e}"

@tool("list_todos", return_direct=True)
def list_todos(config: RunnableConfig) -> str:
    """Fetch all To-Do items for the current user."""
    try:
        # Get token from the current state context
        token = config.get("configurable", {}).get("user_token")
        if not token:
            raise ValueError("No authentication context available")
        
        res = requests.get(f"{API_BASE_URL}/todos", headers=get_headers(token))
        res.raise_for_status()
        todos = res.json()
        
        if not todos:
            return "📝 You have no todos yet. Would you like to create one?"
        
        formatted_todos = []
        for todo in todos:
            status_emoji = {"open": "🔵", "in-progress": "🟡", "done": "✅"}
            priority_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
            
            formatted_todos.append(
                f"{status_emoji.get(todo['status'], '🔵')} **{todo['title']}**\n"
                f"   Priority: {priority_emoji.get(todo['criticality'], '🟡')} {todo['criticality']}\n"
                f"   Due: {todo['due_date']}\n"
                f"   Status: {todo['status']}\n"
                f"   ID: {todo['id']}"
            )
        
        return "📋 **Your Current Todos:**\n\n" + "\n\n".join(formatted_todos)
    except Exception as e:
        logger.error(f"Error fetching To-Dos: {e}")
        return f"❌ Error fetching To-Dos: {e}"

@tool("update_todo", return_direct=True)
def update_todo(input: UpdateTodoInput, config: RunnableConfig) -> str:
    """Update an existing To-Do item by ID."""
    try:
        # Get token from the current state context
        token = config.get("configurable", {}).get("user_token")
        if not token:
            raise ValueError("No authentication context available")
        
        payload = input.dict(exclude_none=True)
        if "due_date" in payload:
            payload["due_date"] = parse_date(payload["due_date"])
        
        res = requests.patch(f"{API_BASE_URL}/todos/{input.id}", json=payload, headers=get_headers(token))
        res.raise_for_status()
        return f"✅ To-Do with ID {input.id} updated successfully!"
    except Exception as e:
        logger.error(f"Failed to update To-Do: {e}")
        return f"❌ Failed to update To-Do: {e}"

@tool("delete_todo", return_direct=True)
def delete_todo(input: DeleteTodoInput, config: RunnableConfig) -> str:
    """Delete a todo item only if it is provided explicitly in the user input."""
    try:
        # Get token from the current state context
        token = config.get("configurable", {}).get("user_token")
        if not token:
            raise ValueError("No authentication context available")
        
        res = requests.delete(f"{API_BASE_URL}/todos/{input.id}", headers=get_headers(token))
        res.raise_for_status()
        return f"✅ To-Do with ID {input.id} deleted successfully!"
    except Exception as e:
        logger.error(f"Error deleting To-Do: {e}")
        return f"❌ Error deleting To-Do: {e}"

###############################################
# 5. Groq LLM Setup
###############################################
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
).bind_tools([add_todo, list_todos, update_todo, delete_todo])

###############################################
# 6. LangGraph Agent Setup
###############################################
system_prompt = '''
You are a helpful and friendly task management AI assistant. 
Your job is to help users manage their To-Do list using the tools provided.

Guidelines:
- Always be helpful, concise, and friendly
- When creating tasks, extract title, priority (criticality), urgency, and due date
- If information is missing, ask the user politely
- Use emojis to make responses more engaging
- Provide clear confirmations when tasks are created, updated, or deleted
- When listing tasks, format them nicely with status indicators
- Be proactive in suggesting actions (e.g., "Would you like me to mark this as done?")

Remember: You can create, list, update, and delete todos. Always confirm actions with the user.
'''

###############################################
# 6. Agent Definition Using ReAct
###############################################
def run_agent(state: AgentState) -> AgentState:
    logger.debug(f"Running ReAct agent with state: {state}")
    agent = create_react_agent(
        llm, 
        tools=[add_todo, list_todos, update_todo, delete_todo],
        prompt=system_prompt
        )
    response = agent.invoke(state)
    updated_state = {
        "messages": state["messages"] + [response]
    }
    return updated_state


def input_handler(state: AgentState) -> AgentState:
    logger.debug("Handling user input")
    # user_input = input("\nEnter your command (or 'exit' to quit): ")
    user_input = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    state.setdefault("messages", []).append(HumanMessage(content=user_input))
    return {"messages": state["messages"] + [HumanMessage(content=user_input)]}

def agent_node(state: AgentState) -> AgentState:
    logger.debug("Agent reasoning on state")
    agent = create_react_agent(llm, tools=[add_todo, list_todos, update_todo, delete_todo])
    response = agent.invoke(state)
    state.setdefault("messages", []).append(AIMessage(content=response["messages"][-1].content))
    return state

def check_continue(state: AgentState) -> str:
    last_user_msg = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
    if isinstance(last_user_msg, HumanMessage):
        text = last_user_msg.content.strip().lower()
        if re.search(r"\b(bye|exit|quit|no)\b", text):
            logger.debug(f"Exit command detected in: {text}")
            return END
    logger.debug("Continuing conversation")
    return "agent"

def final_response(state: AgentState) -> AgentState:
    logger.debug("Generating final response to user")
    last_msg = state["messages"][-1]
    logger.info(last_msg.content)
    return state  # Already handled inside agent_node

###############################################
# 7. LangGraph Graph Compilation
###############################################
# --- Build Graph ---
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("input_handler", input_handler)
builder.add_node("agent", agent_node)
builder.add_node("final_response", final_response)

# Add edges
builder.set_entry_point("input_handler")
builder.add_edge("input_handler", "agent")
builder.add_edge("agent", "final_response")
builder.add_edge("final_response", END)

# Compile the graph
graph = builder.compile()

###############################################
# 7. FastAPI WebSocket Setup
###############################################
app = FastAPI(title="Todo Agent WebSocket API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in agent_states:
            del agent_states[session_id]
        logger.info(f"WebSocket disconnected for session: {session_id}")

    async def send_message(self, session_id: str, message: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            user_token = message_data.get("token", "")
            
            if not user_token:
                await manager.send_message(session_id, json.dumps({
                    "type": "error",
                    "message": "Authentication token required"
                }))
                continue
            
            # Initialize or get existing agent state
            if session_id not in agent_states:
                agent_states[session_id] = AgentState(
                    messages=[],
                    session_id=session_id
                )
            
            # Add user message to state
            agent_states[session_id]["messages"].append(HumanMessage(content=user_message))
            
            # Send typing indicator
            await manager.send_message(session_id, json.dumps({
                "type": "typing",
                "message": "AI is thinking..."
            }))
            
            try:
                config = {
                    "configurable": {
                        "thread_id": session_id,
                        "user_token": user_token
                    }
                }
                # Create agent and process message
                # Stream responses from the compiled graph
                async for chunk in graph.astream(
                    input=agent_states[session_id],
                    config=config
                ):
                    logger.debug(f"Agent created for session {session_id} with state: {agent_states[session_id]}")
                    # Send each chunk as it's generated
                    if "agent" in chunk:
                        agent_message = chunk["agent"]["messages"][-1]
                        await websocket.send_text(json.dumps({
                            "message": agent_message.content,
                            "session_id": session_id,
                            "type": "partial"
                        }))
                    
                    # Send final response
                    if "tools" in chunk:
                        tool_message = chunk["tools"]["messages"][-1]
                        await websocket.send_text(json.dumps({
                            "message": tool_message.content,
                            "session_id": session_id,
                            "type": "final"
                        }))                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await manager.send_message(session_id, json.dumps({
                    "type": "error",
                    "message": f"Sorry, I encountered an error: {str(e)}"
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)

@app.get("/")
async def root():
    return {"message": "Todo Agent WebSocket API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_sessions": len(agent_states)}

###############################################
# 8. Run the server
###############################################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Using port 8001 for WebSocket server