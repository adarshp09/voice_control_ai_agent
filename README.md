# Voice-Controlled Local AI Agent

A local-first voice agent that transcribes audio, classifies user intent, asks for confirmation before file writes, and executes safe local tools through a Streamlit interface.

## What It Does

- Accepts audio from a microphone or uploaded audio file.
- Converts speech to text using a local Whisper model.
- Classifies the request into one of four intents using Ollama through LangGraph/LangChain.
- Executes safe local actions and displays the result in the UI.

Supported intents:
- `create_file`
- `write_code_to_file`
- `summarize_text`
- `general_chat`

## Tech Stack

- UI: Streamlit
- Speech-to-text: `distil-whisper/distil-small.en` via Hugging Face Transformers
- Agent flow: LangGraph
- Tool interface: LangChain tools
- Local LLM: Ollama with `llama3.2:latest`

## Architecture

```text
Audio Input (microphone or upload)
        |
        v
Local STT (Distil-Whisper)
        |
        v
Transcribed Text
        |
        v
Intent Classification (Ollama)
        |
        v
Human Confirmation for File Operations
        |
        v
Tool Execution
  - create_file
  - write_code_to_file
  - summarize_text
  - general_chat
        |
        v
Streamlit UI Output
```

## Safety

- All file writes are restricted to the `output/` directory.
- File-writing actions require explicit user confirmation before execution.
- Unclear audio is blocked with a warning instead of being executed blindly.

## Project Structure

```text
voice-agent/
├── app.py
├── README.md
├── requirements.txt
├── agent/
│   ├── graph.py
│   ├── nodes.py
│   └── tools.py
├── stt/
│   └── whisper_stt.py
└── output/
    └── .gitkeep
```

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama and pull the model

```bash
ollama pull llama3.2:latest
ollama serve
```

### 4. Run the app

```bash
streamlit run app.py
```

## Example Commands

- "Create a file called notes.txt"
- "Write Python code for a retry decorator and save it to retry.py"
- "Summarize this text for me"
- "What can you do?"

## Demo Flow

1. Record audio or upload an audio file.
2. The app transcribes the audio locally.
3. The agent classifies the intent.
4. If the action writes a file, the app asks for confirmation.
5. The tool executes and the result appears in the UI.

## Hardware Notes

This project was built to run locally on Apple Silicon. Whisper uses PyTorch with MPS when available, and Ollama runs locally for intent classification and response generation.

If MPS is unavailable, the speech model falls back to CPU, which is slower but still functional.

## Submission Notes

This project satisfies the core assignment requirements:
- microphone input and uploaded audio input
- local speech-to-text
- local intent classification
- safe tool execution
- web UI showing transcription, intent, action, and output

Bonus features included:
- human-in-the-loop confirmation before file writes
- session history in the UI
- quality checks for unclear audio
- text input mode for easier debugging and demo backup

## Troubleshooting

If transcription is slow on first run:
- the speech model may still be downloading or warming up

If Ollama is not responding:
- make sure `ollama serve` is running
- verify the model exists with `ollama list`

If microphone recording does not work:
- install `streamlit-mic-recorder`
- allow microphone access in the browser