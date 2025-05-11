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

# Set up logging with maximum verbosity
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure Flask/Werkzeug logs are captured
logging.getLogger('werkzeug').setLevel(logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for chatbot access

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    logger.error("ANTHROPIC_API_KEY not set in environment variables")
    raise ValueError("ANTHROPIC_API_KEY is required")

# System prompt (unchanged)
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

# MCP server config
config = {
    "mcpServers": {
        "http": {
            "url": "https://payment-ol-mcp.onrender.com/sse"
        }
    }
}

# Initialize MCPClient
try:
    client = MCPClient.from_dict(config)
except Exception as e:
    logger.exception("Failed to initialize MCPClient")
    raise

# Initialize Claude LLM
try:
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=ANTHROPIC_API_KEY,
        temperature=0.7,
        default_headers={"anthropic-beta": "tools-2024-05-16"},
        model_kwargs={"system": SYSTEM_PROMPT},
    )
except Exception as e:
    logger.exception("Failed to initialize ChatAnthropic")
    raise

# Create MCPAgent
try:
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=30,
        verbose=True,
    )
except Exception as e:
    logger.exception("Failed to initialize MCPAgent")
    raise

# In-memory session storage (use Redis for production)
sessions = {}

# Maximum attempts for missing parameters
max_attempts = 3

@app.route("/health", methods=["GET"])
def health_check():
    """Check if the app is running and dependencies are loaded."""
    try:
        return jsonify({
            "status": "healthy",
            "message": "Flask app is running",
            "dependencies": {
                "flask": "2.3.3",
                "mcp_use": "loaded",
                "anthropic": "loaded"
            }
        }), 200
    except Exception as e:
        logger.exception("Health check failed")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/process_request", methods=["POST"])
async def process_request():
    """Handle user requests from chatbot."""
    try:
        request_json = request.get_json()
        if not request_json or "user_input" not in request_json:
            logger.error("Invalid request: missing user_input")
            return jsonify({"status": "error", "message": "Missing user_input in request"}), 400

        # Get or create session ID
        session_id = request_json.get("session_id", str(uuid.uuid4()))
        session_data = sessions.get(session_id, {
            "conversation_history": [],
            "original_prompt": request_json["user_input"],
            "attempts": 0
        })

        # Append user input to conversation history
        session_data["conversation_history"].append({"role": "user", "content": request_json["user_input"]})

        # Construct input_content with all previous user inputs
        previous_inputs = [msg["content"] for msg in session_data["conversation_history"] if msg["role"] == "user"][1:]  # Skip original prompt
        input_content = (request_json.get("original_prompt", session_data["original_prompt"]) +
                         "\n" + "\n".join(f"Provided: {inp}" for inp in previous_inputs) +
                         "\nProvided: " + request_json["user_input"])
        logger.debug(f"Session {session_id}: input_content: {input_content}")

        try:
            # Run agent with timeout
            result = await asyncio.wait_for(agent.run(input_content, max_steps=30), timeout=30.0)
            result_text = result[0]["text"] if isinstance(result, list) and result and "text" in result[0] else str(result)
            logger.info(f"Session {session_id}: Agent response: {result_text}")

            # Check for missing parameters
            if "please provide" in result_text.lower() or "missing" in result_text.lower():
                missing_param = result_text.split("please provide ")[-1].split(".")[0].strip() if "please provide" in result_text.lower() else "required parameter"
                session_data["attempts"] += 1
                session_data["conversation_history"].append({"role": "assistant", "content": result_text})

                if session_data["attempts"] >= max_attempts:
                    logger.warning(f"Session {session_id}: Max attempts reached")
                    sessions[session_id] = session_data
                    return jsonify({
                        "status": "error",
                        "message": "Maximum attempts reached. Please start a new request with all required parameters.",
                        "session_id": session_id
                    })

                sessions[session_id] = session_data
                return jsonify({
                    "status": "missing_parameter",
                    "message": result_text,
                    "missing_param": missing_param,
                    "session_id": session_id,
                    "original_prompt": session_data["original_prompt"]
                })

            # Success response
            session_data["conversation_history"].append({"role": "assistant", "content": result_text})
            sessions[session_id] = session_data
            return jsonify({
                "status": "success",
                "message": result_text,
                "session_id": session_id
            })

        except asyncio.TimeoutError:
            logger.error(f"Session {session_id}: Agent execution timed out")
            return jsonify({"status": "error", "message": "Request timed out. Please try again.", "session_id": session_id}), 504
        except Exception as e:
            logger.exception(f"Session {session_id}: Agent execution failed: {str(e)}")
            if "missing required parameter" in str(e).lower():
                missing_param = str(e).split("missing required parameter")[-1].strip()
                session_data["attempts"] += 1
                session_data["conversation_history"].append({"role": "assistant", "content": f"Missing required parameter: {missing_param}"})

                if session_data["attempts"] >= max_attempts:
                    logger.warning(f"Session {session_id}: Max attempts reached")
                    sessions[session_id] = session_data
                    return jsonify({
                        "status": "error",
                        "message": "Maximum attempts reached. Please start a new request with all required parameters.",
                        "session_id": session_id
                    })

                sessions[session_id] = session_data
                return jsonify({
                    "status": "missing_parameter",
                    "message": f"You requested: {session_data['original_prompt']}. Please provide the {missing_param}.",
                    "missing_param": missing_param,
                    "session_id": session_id,
                    "original_prompt": session_data["original_prompt"]
                })

            return jsonify({"status": "error", "message": f"Error: {str(e)}", "session_id": session_id}), 500

    except Exception as e:
        logger.exception(f"Session {session_id}: Request processing failed: {str(e)}")
        return jsonify({"status": "error", "message": f"Internal error: {str(e)}"}), 500

@app.teardown_appcontext
def cleanup_sessions(exception=None):
    """Close MCP client sessions on app shutdown."""
    try:
        if client.sessions:
            asyncio.run(client.close_all_sessions())
            logger.info("Closed all MCP client sessions")
    except Exception as e:
        logger.exception("Failed to close MCP client sessions")

if __name__ == "__main__":
    app.run(debug=True)