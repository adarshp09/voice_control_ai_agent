import os
import re
import json
import csv
import datetime
from pathlib import Path
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

OLLAMA_MODEL = "llama3.2:latest"

FILE_TYPE_MAP = {
    ".json": "json",
    ".ipynb": "notebook",
    ".csv": "csv",
    ".txt": "text",
    ".md": "markdown",
    ".py": "python",
    ".js": "javascript",
    ".html": "html",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".log": "log",
}


def _safe_path(filename: str) -> Path:
    """Ensure path stays inside output/ directory."""
    safe_name = re.sub(r"[^\w\-_\. ]", "_", filename)
    safe_name = safe_name.strip().replace(" ", "_")
    path = (OUTPUT_DIR / safe_name).resolve()
    if not str(path).startswith(str(OUTPUT_DIR.resolve())):
        raise ValueError("Path traversal attempt detected.")
    return path


def _detect_file_type(filename: str) -> str:
    """Detect file type based on extension."""
    ext = Path(filename).suffix.lower()
    return FILE_TYPE_MAP.get(ext, "text")


def _format_json_or_notebook(content: str) -> str:
    """Format JSON and notebook files with pretty-printing."""
    data = json.loads(content)
    return json.dumps(data, indent=2)


def _format_markdown(content: str) -> str:
    """Format markdown files with consistent line endings."""
    return content.strip() + "\n"


def _format_python(content: str) -> str:
    """Format Python files with basic cleanup."""
    return content.strip() + "\n"


FORMATTERS = {
    "json": _format_json_or_notebook,
    "notebook": _format_json_or_notebook,
    "markdown": _format_markdown,
    "python": _format_python,
    "javascript": _format_python,  # Similar formatting rules
}


def _format_content_by_type(filename: str, content: str) -> tuple[str, str]:
    """
    Format content based on file type using built-in formatters.
    Returns: (formatted_content, file_type)
    """
    file_type = _detect_file_type(filename)
    
    if not content.strip():
        return content, file_type
    
    formatter = FORMATTERS.get(file_type, lambda c: c)
    try:
        return formatter(content), file_type
    except (json.JSONDecodeError, ValueError):
        return content, file_type


@tool
def create_file(filename: str, content: str = "") -> str:
    """
    Create a file in the output/ directory with intelligent type detection.
    Args:
        filename: Name of the file to create (e.g. 'notes.txt', 'data.json').
        content: Optional initial content for the file.
    Returns:
        Success message with the file path and file type info.
    """
    path = _safe_path(filename)
    formatted_content, file_type = _format_content_by_type(filename, content)
    path.write_text(formatted_content)
    return f"✅ File created: {path}\n   Type: {file_type}"


@tool
def write_code_to_file(filename: str, language: str, description: str) -> str:
    """
    Generate code using Ollama and save it to a file in output/.
    Automatically detects file type from extension.
    Args:
        filename: Name of the file (e.g. 'retry.py').
        language: Programming language (e.g. 'python', 'javascript').
        description: Description of the code to generate.
    Returns:
        Success message with the file path, file type, and generated code preview.
    """
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)
    prompt = (
        f"Write a complete, working {language} program/function for: {description}\n"
        "Return ONLY the code, no explanations, no markdown fences."
    )
    response = llm.invoke(prompt)
    code = response.content.strip()
    code = re.sub(r"^```[\w]*\n?", "", code)
    code = re.sub(r"\n?```$", "", code)

    path = _safe_path(filename)
    formatted_content, file_type = _format_content_by_type(filename, code)
    path.write_text(formatted_content)
    preview = formatted_content[:300] + ("..." if len(formatted_content) > 300 else "")
    return f"✅ Code saved to: {path}\n   Type: {file_type}\n\n```{language}\n{preview}\n```"


@tool
def summarize_text(text: str) -> str:
    """
    Summarize a piece of text using Ollama.
    Args:
        text: The text to summarize.
    Returns:
        A concise summary of the text.
    """
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.3)
    prompt = (
        f"Summarize the following text concisely in 3-5 bullet points:\n\n{text}"
    )
    response = llm.invoke(prompt)
    return response.content.strip()


@tool
def general_chat(message: str) -> str:
    """
    Handle general conversational queries using Ollama.
    Args:
        message: The user's message or question.
    Returns:
        A conversational response.
    """
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.7)
    response = llm.invoke(message)
    return response.content.strip()


TOOLS = {
    "create_file": create_file,
    "write_code_to_file": write_code_to_file,
    "summarize_text": summarize_text,
    "general_chat": general_chat,
}
