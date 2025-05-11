import asyncio
import json
import uuid
import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from mcp_use import MCPAgent, MCPClient

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")]
)
logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.DEBUG)
logging.getLogger('mcp_use').setLevel(logging.DEBUG)
logging.getLogger('langchain_anthropic').setLevel(logging.DEBUG)

# Flask app
app = Flask(__name__)
CORS(app)

# Load .env
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY missing in environment variables")

# Prompt
SYSTEM_PROMPT = """
You are a Paytm MCP Assistant, an AI agent powered by the Paytm MCP Server, which enables secure access to Paytm's Payments and Business Payments APIs. Your role is to automate payment workflows using the available tools: `create_link`, `fetch_link`, and `fetch_transaction`. Follow these steps for every request:

1. **Understand the Request**:
   - Analyze the user's prompt to identify the intended task (e.g., create a payment link, fetch link details, or check transaction status).
   - Determine which tool to use based on the task:
     - `create_link`: To create a new payment link (e.g., "Create a ₹500 payment link").
     - `fetch_link`: To retrieve details of an existing payment link (e.g., "Get details for link ID XYZ").
     - `fetch_transaction`: To fetch transaction details for a link or transaction ID (e.g., "Check transaction status for link ID XYZ").
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

# MCP config
config = {
    "mcpServers": {
        "http": {
            "url": "https://payment-ol-mcp.onrender.com/sse"
        }
    }
}

# Init client + Claude
logger.info("Initializing MCPClient")
client = MCPClient.from_dict(config)
logger.info("Initializing ChatAnthropic")
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=ANTHROPIC_API_KEY,
    temperature=0.7,
    default_headers={"anthropic-beta": "tools-2024-05-16"},
    model_kwargs={"system": SYSTEM_PROMPT},
)
agent = MCPAgent(llm=llm, client=client, max_steps=30, verbose=True)
sessions = {}
max_attempts = 3

# ✅ Delayed async initialization
initialized = False
@app.before_first_request
def lazy_initialize():
    global initialized
    if not initialized:
        logger.info("🔧 Lazy loading tools from MCP...")
        asyncio.run(agent.initialize())
        initialized = True
        logger.info("✅ Tools loaded")

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route("/process_request", methods=["POST"])
def process_request():
    session_id = None
    try:
        data = request.get_json()
        if not data or "user_input" not in data:
            return jsonify({"error": "Missing user_input"}), 400

        session_id = data.get("session_id", str(uuid.uuid4()))
        session = sessions.get(session_id, {
            "conversation_history": [],
            "original_prompt": data["user_input"],
            "attempts": 0
        })

        session["conversation_history"].append({"role": "user", "content": data["user_input"]})
        prev_inputs = [m["content"] for m in session["conversation_history"] if m["role"] == "user"][1:]
        input_content = session["original_prompt"] + "\n" + "\n".join(f"Provided: {p}" for p in prev_inputs)

        result = asyncio.run(asyncio.wait_for(agent.run(input_content, max_steps=30), timeout=45))
        result_text = result[0]["text"] if isinstance(result, list) and "text" in result[0] else str(result)

        if "please provide" in result_text.lower():
            session["attempts"] += 1
            if session["attempts"] >= max_attempts:
                return jsonify({"status": "error", "message": "Too many attempts", "session_id": session_id})
            sessions[session_id] = session
            return jsonify({"status": "missing_parameter", "message": result_text, "session_id": session_id})

        session["conversation_history"].append({"role": "assistant", "content": result_text})
        sessions[session_id] = session
        return jsonify({"status": "success", "message": result_text, "session_id": session_id})

    except asyncio.TimeoutError:
        return jsonify({"status": "error", "message": "Agent timeout"}), 504
    except Exception as e:
        logger.exception("Unhandled exception")
        return jsonify({"status": "error", "message": str(e), "session_id": session_id}), 500

@app.teardown_appcontext
def shutdown(exception=None):
    try:
        if client.sessions:
            asyncio.run(client.close_all_sessions())
    except Exception as e:
        logger.exception("Error closing sessions")

if __name__ == "__main__":
    app.run(debug=True)

