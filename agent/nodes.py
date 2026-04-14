import json
import re
from typing import Any
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

OLLAMA_MODEL = "llama3.2:latest"

DEFAULT_STATE: dict[str, Any] = {
    "audio_path": None,          # Path or bytes of audio input
    "transcription": "",         # Raw text from Whisper
    "intent": "",                # Detected intent string
    "intent_params": {},         # Extracted parameters for the tool
    "confirmation_required": False,
    "confirmed": None,           # True / False / None (pending)
    "tool_output": "",           # Result from tool execution
    "error": "",                 # Any error message
    "history": [],               # List of past action dicts
}


INTENT_SYSTEM_PROMPT = """You are an intent classifier for a voice-controlled AI agent.
Given a transcribed user command, classify it into exactly one of these intents:
  - create_file
  - write_code_to_file
  - summarize_text
  - general_chat

CRITICAL RULES FOR PARAMETER EXTRACTION:

For write_code_to_file:
  - ALWAYS infer a filename with proper extension based on language (e.g., .py, .js, .html)
  - Extract the programming language from context
  - Extract what code to build from the description
  - Example: "write python code for retry" → {"filename": "retry.py", "language": "python", "description": "retry decorator function"}

For create_file:
  - ALWAYS infer a meaningful filename with extension (e.g., .txt, .md, .json)
  - Extract any provided content or use empty string
  - Example: "create a text file with notes" → {"filename": "notes.txt", "content": ""}

Parameter schemas:
  create_file:        {"filename": "<name.ext>", "content": "<optional text>"}
  write_code_to_file: {"filename": "<name.ext>", "language": "<lang>", "description": "<what to build>"}
  summarize_text:     {"text": "<text to summarize>"}
  general_chat:       {"message": "<the user message>"}

Respond ONLY with valid JSON like:
{
  "intent": "<intent>",
  "params": { ... }
}
No extra text, no markdown."""


def classify_intent(state: dict) -> dict:
    """Call Ollama to classify the transcribed text into an intent + params."""
    transcription = state.get("transcription", "")
    if not transcription:
        return {**state, "error": "No transcription to classify."}

    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=f'User said: "{transcription}"'),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    raw = re.sub(r"^```[\w]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        parsed = json.loads(raw)
        intent = parsed.get("intent", "general_chat")
        params = parsed.get("params", {"message": transcription})
    except json.JSONDecodeError:
        intent = "general_chat"
        params = {"message": transcription}

    confirmation_required = intent in ("create_file", "write_code_to_file")

    return {
        **state,
        "intent": intent,
        "intent_params": params,
        "confirmation_required": confirmation_required,
        "confirmed": None if confirmation_required else True,
        "error": "",
    }


def execute_tool(state: dict) -> dict:
    """Run the appropriate tool based on detected intent."""
    from agent.tools import TOOLS  # local import to avoid circular

    intent = state.get("intent", "general_chat")
    params = state.get("intent_params", {})

    tool_fn = TOOLS.get(intent, TOOLS["general_chat"])

    try:
        result = tool_fn.invoke(params)
    except Exception as e:
        result = f"❌ Tool execution failed: {type(e).__name__}: {str(e)}\nParams: {params}"

    history = state.get("history", [])
    history = history + [
        {
            "transcription": state.get("transcription"),
            "intent": intent,
            "params": params,
            "output": result,
        }
    ]

    return {**state, "tool_output": result, "history": history, "error": ""}
