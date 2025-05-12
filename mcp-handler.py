import asyncio
import os
import json
import logging
from typing import Dict, List
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from mcp_use import MCPAgent, MCPClient
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Set up logging with verbosity and file output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/app.log")  # Use /tmp for Render
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('uvicorn').setLevel(logging.DEBUG)
logging.getLogger('mcp_use').setLevel(logging.DEBUG)
logging.getLogger('langchain_anthropic').setLevel(logging.DEBUG)

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://payment-ol-mcp.onrender.com/sse")
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "60"))  # Configurable timeout

if not ANTHROPIC_API_KEY:
    logger.error("ANTHROPIC_API_KEY not set in environment variables")
    raise ValueError("ANTHROPIC_API_KEY is required")

# System prompt
SYSTEM_PROMPT = """
You are a Paytm MCP Assistant, an AI agent powered by the Paytm MCP Server, which enables secure access to Paytm's Payments and Business Payments APIs. Your role is to automate payment workflows using the available tools: `create_link`, `fetch_link`, and `fetch_transaction`. Follow these steps for every request:

1. **Understand the Request**:
   - Analyze the user's prompt to identify the intended task (e.g., create a payment link, fetch link details, or check transaction status).
   - Determine which tool to use based on the task:
     - `create_link`: To create a new payment link (e.g., "Create a ₹500 payment link").
     - `fetch_link`: To retrieve details of an existing payment link (e.g., "Get details for link ID xyz").
     - `fetch_transaction`: To fetch transaction details for a link or transaction ID (e.g., "Check transaction status for link ID xyz").
   - Extract all relevant parameters from the prompt (e.g., amount, email, link ID, transaction ID).

2. **Check Tool Parameters**:
   - Refer to the tool's schema provided by the MCP server to identify required and optional parameters.
   - If a required parameter is missing, explicitly ask the user for it with a clear question, referencing the original request to maintain context (e.g., "You requested a ₹500 payment link. Please provide the email address to send the payment link.").
   - Use provided parameters and any previous responses to fill optional fields (e.g., set `send_email` to true by default for `create_link`).

3. **Call the Tool**:
   - Invoke the selected tool with the extracted or user-provided parameters.
   - Only include parameters that the tool's schema accepts. Map user-provided terms (e.g., "recipient name") to appropriate fields (e.g., `description`) or omit if not supported.

4. **Validate the Output**:
   - For `create_link`: Ensure the returned URL starts with "paytm.me/". Confirm the email was sent if requested.
   - For `fetch_link`: Verify the response contains valid link details (e.g., link ID, status).
   - For `fetch_transaction`: Confirm the response includes transaction details (e.g., status, amount).
   - If the output is invalid, report the issue and retry with corrected parameters if possible.

5. **Handle Missing Parameters**:
   - If a tool call fails due to missing required parameters, ask the user for the missing information, referencing the original request (e.g., "You requested to check transactions for a link. Please provide the link ID.").
   - Incorporate the new input and previous context to retry the tool call, ensuring all previously provided parameters are retained.

6. **Provide a Polished Response**:
   - Summarize the action taken in a structured format using bullet points or numbered lists.
   - Example response format:
     - Action: Created payment link
     - Details: Amount: ₹{amount}, Link: {url}, Email: {email}
     - Next Steps: {next_steps}
   - If an error occurs, explain the issue clearly and suggest next steps (e.g., "Invalid link ID. Please provide a valid ID or create a new link.").
   - If requesting user input, format the question clearly:
     - Question: {question referencing original request}

7. **Maintain Context**:
   - Use previous responses and user inputs to inform subsequent tool calls, ensuring continuity in the workflow.
   - When asking for missing parameters, restate the original request to confirm intent (e.g., "You requested a ₹500 payment link for lunch. Please provide the email address.").

Be concise, proactive, and user-friendly. If unsure about the task or parameters, ask clarifying questions to ensure accuracy, referencing the original request to maintain context.
"""

# FastAPI app
app = FastAPI(title="Paytm MCP Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CHATBOT_ORIGIN", "*")],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    status: str
    session_id: str

# In-memory session storage with attempt tracking
sessions: Dict[str, Dict] = {}

# MCPClient and MCPAgent
config = {
    "mcpServers": {
        "http": {
            "url": MCP_SERVER_URL
        }
    }
}

try:
    logger.info("Initializing MCPClient")
    client = MCPClient.from_dict(config)
    logger.info("MCPClient initialized successfully")
except Exception as e:
    logger.exception("Failed to initialize MCPClient")
    raise

try:
    logger.info("Initializing ChatAnthropic")
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=ANTHROPIC_API_KEY,
        temperature=0.7,
        default_headers={"anthropic-beta": "tools-2024-05-16"},
        model_kwargs={"system": SYSTEM_PROMPT},
    )
    logger.info("ChatAnthropic initialized successfully")
except Exception as e:
    logger.exception("Failed to initialize ChatAnthropic")
    raise

try:
    logger.info("Initializing MCPAgent")
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=30,
        verbose=True,
    )
except Exception as e:
    logger.exception("Failed to initialize MCPAgent")
    raise

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Preload tools
    async def preload_tools():
        try:
            await agent.initialize()
            tools_response = await client.send_request('tools/list')
            logger.info(f"MCP Server tools: {json.dumps(tools_response.get('result', {}).get('tools', []), indent=2)}")
        except Exception as e:
            logger.error(f"Failed to fetch tools: {str(e)}")
    await preload_tools()
    logger.info("MCPAgent initialized successfully with preloaded tools")
    
    yield
    
    # Shutdown: Clean up
    if client.sessions:
        await client.close_all_sessions()
        logger.info("Closed all MCP client sessions")

app.lifespan = lifespan

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {"status": "healthy", "message": "FastAPI app is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chatbot POST requests."""
    logger.info(f"Received request: session_id={request.session_id}, message={request.message}")

    try:
        # Generate session_id if not provided
        session_id = request.session_id or str(uuid.uuid4())
        if session_id not in sessions:
            sessions[session_id] = {
                "conversation_history": [],
                "original_prompt": request.message,
                "attempts": 0,
                "last_active": asyncio.get_event_loop().time()
            }

        session = sessions[session_id]
        session["conversation_history"].append({"role": "user", "content": request.message})
        session["last_active"] = asyncio.get_event_loop().time()
        logger.debug(f"Session {session_id}: history: {json.dumps(session['conversation_history'], indent=2)}")

        max_attempts = 3
        while session["attempts"] < max_attempts:
            try:
                # Prepare input with context
                input_content = session["conversation_history"][-1]["content"]
                if session["attempts"] > 0:
                    input_content = (f"Original request: {session['original_prompt']}\n" +
                                     "\n".join(f"Provided: {msg['content']}" for msg in session["conversation_history"] if msg["role"] == "user"))

                logger.debug(f"Session {session_id}: Running agent with input: {input_content}")
                result = await asyncio.wait_for(agent.run(input_content, max_steps=30), timeout=AGENT_TIMEOUT)
                result_text = result[0]["text"] if isinstance(result, list) and result and "text" in result[0] else str(result)
                logger.info(f"Session {session_id}: Agent response: {result_text}")

                # Check for missing parameters
                if "please provide" in result_text.lower() or "missing" in result_text.lower():
                    missing_param = result_text.split("please provide ")[-1].split(".")[0].strip() if "please provide" in result_text.lower() else "required parameter"
                    session["attempts"] += 1
                    session["conversation_history"].append({"role": "assistant", "content": result_text})
                    sessions[session_id] = session
                    return ChatResponse(response=result_text, status="missing_parameter", session_id=session_id)

                # Success
                session["conversation_history"].append({"role": "assistant", "content": result_text})
                session["attempts"] = 0
                session["last_active"] = asyncio.get_event_loop().time()
                sessions[session_id] = session
                return ChatResponse(response=result_text, status="success", session_id=session_id)

            except asyncio.TimeoutError:
                logger.error(f"Session {session_id}: Agent execution timed out")
                raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
            except TypeError as e:
                logger.exception(f"Session {session_id}: Pickling error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Pickling error: {str(e)}. Please try again or contact support.")
            except Exception as e:
                logger.exception(f"Session {session_id}: Agent execution failed: {str(e)}")
                error_str = str(e).lower()
                try:
                    # Fetch tool schema to identify missing parameters dynamically
                    tools_response = await client.send_request('tools/list')
                    tools = tools_response.get('result', {}).get('tools', [])
                    missing_param = None
                    for tool in tools:
                        schema = tool.get('schema', {})
                        required = schema.get('required', [])
                        for param in required:
                            if param.lower() in error_str:
                                missing_param = param
                                break
                        if missing_param:
                            break
                    missing_param = missing_param or ("email" if "email" in error_str else "link_id" if "link_id" in error_str else "transaction_id")
                except Exception as schema_error:
                    logger.error(f"Failed to fetch tool schema: {str(schema_error)}")
                    missing_param = "required parameter"
                response_text = f"You requested: {session['original_prompt']}. Please provide the {missing_param}."
                session["attempts"] += 1
                session["conversation_history"].append({"role": "assistant", "content": response_text})
                sessions[session_id] = session
                return ChatResponse(response=response_text, status="missing_parameter", session_id=session_id)

        # Max attempts reached
        error_msg = "Maximum attempts reached. Please provide all required parameters and try again."
        logger.error(f"Session {session_id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    except Exception as e:
        logger.exception(f"Session {session_id}: Request processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.on_event("startup")
async def cleanup_sessions():
    """Periodically clean up old sessions (runs every hour)."""
    async def cleanup():
        while True:
            current_time = asyncio.get_event_loop().time()
            expired_sessions = [
                sid for sid, s in sessions.items()
                if (current_time - s["last_active"]) > 86400  # 24 hours
                or len(s["conversation_history"]) == 0
            ]
            for sid in expired_sessions:
                del sessions[sid]
                logger.info(f"Cleaned up session: {sid}")
            await asyncio.sleep(3600)  # Run hourly
    asyncio.create_task(cleanup())