import streamlit as st
import tempfile
import os
import time
from pathlib import Path
from stt.whisper_stt import transcribe_audio_array, transcribe_bytes

st.set_page_config(
    page_title="Voice Agent",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13151c !important;
    border-right: 1px solid #1e2130;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.5px;
}

/* Cards */
.agent-card {
    background: #13151c;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

.agent-card-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5a6080;
    margin-bottom: 10px;
}

.agent-card-body {
    font-size: 15px;
    color: #c8cce0;
    line-height: 1.6;
}

/* Intent badge */
.intent-badge {
    display: inline-block;
    background: #1a2540;
    border: 1px solid #2a3f6f;
    color: #6eb3ff;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 20px;
    margin-top: 6px;
}

/* Status dots */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 8px;
}
.dot-green  { background: #4ade80; box-shadow: 0 0 6px #4ade80; }
.dot-yellow { background: #facc15; box-shadow: 0 0 6px #facc15; }
.dot-red    { background: #f87171; box-shadow: 0 0 6px #f87171; }
.dot-blue   { background: #60a5fa; box-shadow: 0 0 6px #60a5fa; }

/* Confirm box */
.confirm-box {
    background: #1a1400;
    border: 1px solid #7c5900;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
}

/* History row */
.history-row {
    background: #13151c;
    border-left: 3px solid #2a3f6f;
    padding: 10px 16px;
    margin-bottom: 8px;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
}

/* Output area */
.output-area {
    background: #0a0c10;
    border: 1px solid #1e2130;
    border-radius: 8px;
    padding: 16px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    white-space: pre-wrap;
    color: #a8ffbd;
    min-height: 60px;
}

/* File list */
.file-chip {
    display: inline-block;
    background: #131f12;
    border: 1px solid #1e4d1c;
    color: #86efac;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 4px;
    margin: 3px;
}

/* Streamlit button overrides */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 8px !important;
}

/* Divider */
hr { border-color: #1e2130; }
</style>
""", unsafe_allow_html=True)


def init_session():
    defaults = {
        "history": [],
        "pending_state": None,
        "whisper_loaded": False,
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


with st.sidebar:
    st.markdown("## 🎙️ Voice Agent")
    st.markdown("---")

    st.markdown("### ⚙️ Model Info")
    st.markdown("""
    <div class="agent-card">
        <div class="agent-card-header">STT Model</div>
        <div class="agent-card-body">Whisper <b>small</b><br><span style="font-size:12px;color:#5a6080;">HuggingFace · Local · MPS</span></div>
    </div>
    <div class="agent-card">
        <div class="agent-card-header">LLM</div>
        <div class="agent-card-body">llama3.2:3b<br><span style="font-size:12px;color:#5a6080;">Ollama · Local · Metal</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Supported Intents")
    for intent, icon in [
        ("create_file", "📄"),
        ("write_code_to_file", "💻"),
        ("summarize_text", "📝"),
        ("general_chat", "💬"),
    ]:
        st.markdown(f"`{icon} {intent}`")

    st.markdown("---")

    # Output files
    st.markdown("### 📁 Output Files")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    files = list(output_dir.iterdir())
    if files:
        for f in sorted(files):
            st.markdown(f'<span class="file-chip">📄 {f.name}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#5a6080;font-size:13px;">No files yet</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.pending_state = None
        st.session_state.last_result = None
        st.rerun()


st.markdown("# 🎙️ Voice-Controlled AI Agent")
st.markdown('<p style="color:#5a6080;font-size:14px;">Powered by Whisper · LangGraph · Ollama llama3.2:3b</p>', unsafe_allow_html=True)
st.markdown("---")

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 🎤 Audio Input")

    input_mode = st.radio(
        "Input mode",
        ["🎙️ Record from Microphone", "📤 Upload Audio File", "📝 Type Text"],
        horizontal=True,
        label_visibility="collapsed",
    )

    transcription_text = ""
    audio_ready = False

    if input_mode == "🎙️ Record from Microphone":
        st.info("🎙️ Allow microphone access and click to record.")
        try:
            from streamlit_mic_recorder import mic_recorder
            audio_data = mic_recorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                just_once=False,
                use_container_width=True,
                key="recorder"
            )
            
            if audio_data:
                raw_audio = audio_data.get("bytes", b"")
                sample_rate = int(audio_data.get("sample_rate", 16000))

                with st.spinner("Transcribing audio..."):
                    try:
                        import numpy as np

                        audio_array = np.frombuffer(raw_audio, dtype=np.int16)
                        stt_result = transcribe_audio_array(audio_array, sample_rate=sample_rate, return_details=True)
                        result_text = stt_result.get("text", "")


                        if stt_result.get("ok"):
                            transcription_text = result_text
                            audio_ready = True
                            st.success("✅ Audio transcribed successfully")
                        else:
                            st.warning(
                                "⚠️ Audio was unclear for reliable command execution. "
                                f"Reason: {stt_result.get('reason', 'unknown')}. Please retry with a clearer clip."
                            )
                    except Exception as e:
                        st.error(f"❌ Transcription failed: {e}")
        except ImportError as e:
            st.error(f"streamlit-mic-recorder not installed: {e}")
            st.info("Install with: pip install streamlit-mic-recorder")
        except Exception as e:
            st.error(f"Microphone setup failed: {e}")
            st.info("Try using '📤 Upload Audio File' or '📝 Type Text' mode instead.")

    elif input_mode == "📤 Upload Audio File":
        uploaded = st.file_uploader(
            "Upload .wav, .mp3, .m4a, or .ogg",
            type=["wav", "mp3", "m4a", "ogg"],
            label_visibility="visible",
        )
        if uploaded:
            st.audio(uploaded)
            if st.button("▶ Transcribe & Run", type="primary", use_container_width=True):
                with st.spinner("Loading Whisper model..."):
                    audio_bytes = uploaded.getvalue()
                with st.spinner("Transcribing audio..."):
                    try:

                        stt_result = transcribe_bytes(audio_bytes, return_details=True)
                        transcription_text = stt_result.get("text", "")


                        if stt_result.get("ok"):
                            audio_ready = True
                        else:
                            audio_ready = False
                            st.warning(
                                "⚠️ Audio was unclear for reliable command execution. "
                                f"Reason: {stt_result.get('reason', 'unknown')}. Please retry with a clearer clip."
                            )
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")

    else:  # Text mode
        debug_text = st.text_area(
            "Enter command text directly:",
            placeholder='e.g. "Create a Python file with a retry decorator"',
            height=100,
        )
        if st.button("▶ Run Agent", type="primary", use_container_width=True):
            if debug_text.strip():
                transcription_text = debug_text.strip()
                audio_ready = True
            else:
                st.warning("Please enter some text.")

    # ── Transcription display ─────────────────────────────────────────────────
    if transcription_text and audio_ready:
        st.markdown("---")
        
        # Pipeline visualization
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;font-family:'Space Mono',monospace;font-size:12px;">
            <div style="padding:6px 12px;background:#1a2540;border:1px solid #2a3f6f;border-radius:6px;color:#6eb3ff;">🎤 Transcribe</div>
            <div style="color:#5a6080;">→</div>
            <div style="padding:6px 12px;background:#1a2540;border:1px solid #2a3f6f;border-radius:6px;color:#6eb3ff;">🧠 Classify</div>
            <div style="color:#5a6080;">→</div>
            <div style="padding:6px 12px;background:#1a2540;border:1px solid #2a3f6f;border-radius:6px;color:#6eb3ff;">⚠️ Confirm</div>
            <div style="color:#5a6080;">→</div>
            <div style="padding:6px 12px;background:#1a2540;border:1px solid #2a3f6f;border-radius:6px;color:#6eb3ff;">⚡ Execute</div>
            <div style="color:#5a6080;">→</div>
            <div style="padding:6px 12px;background:#1a2540;border:1px solid #2a3f6f;border-radius:6px;color:#6eb3ff;">📁 Result</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="agent-card">
            <div class="agent-card-header">📄 Transcription</div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="agent-card-body">"{transcription_text}"</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Run classification
        with st.spinner("🧠 Classifying intent..."):
            from agent.graph import run_classification, run_execution
            state = run_classification(transcription_text, st.session_state.history)

        intent = state.get("intent", "unknown")
        params = state.get("intent_params", {})
        needs_confirm = state.get("confirmation_required", False) and state.get("confirmed") is None

        # Intent display
        intent_icons = {
            "create_file": "📄",
            "write_code_to_file": "💻",
            "summarize_text": "📝",
            "general_chat": "💬",
        }
        icon = intent_icons.get(intent, "❓")
        
        # Format params for display
        params_display = ""
        for k, v in params.items():
            val_preview = str(v)[:50] + "..." if len(str(v)) > 50 else str(v)
            params_display += f"<br><span style='color:#86b4ef;'>{k}:</span> <span style='color:#a8ffbd;'>{val_preview}</span>"
        
        st.markdown(f"""
        <div class="agent-card" style="margin-top:12px;">
            <div class="agent-card-header">🎯 Detected Intent</div>
            <div class="intent-badge">{icon} {intent}</div>
            <div class="agent-card-body" style="margin-top:12px;font-size:13px;color:#c8cce0;">
                <span style="color:#5a6080;">Parameters:</span>
                <div style="margin-top:8px;padding:8px;background:#0a0c10;border-left:2px solid #2a3f6f;border-radius:4px;font-family:'Space Mono',monospace;font-size:12px;">
                    {params_display}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if needs_confirm:
            st.session_state.pending_state = state
        else:
            # Execute directly (non-file ops)
            with st.spinner("⚡ Executing..."):
                final_state = run_execution(state)
            st.session_state.last_result = final_state
            st.session_state.history = final_state.get("history", [])
            st.session_state.pending_state = None


with col_output:
    st.markdown("### ⚡ Output")

    # ── Human-in-the-loop confirmation ────────────────────────────────────────
    if st.session_state.pending_state is not None:
        ps = st.session_state.pending_state
        intent = ps.get("intent", "")
        params = ps.get("intent_params", {})

        st.markdown(f"""
        <div class="confirm-box">
            <div style="font-family:'Space Mono',monospace;font-size:12px;color:#facc15;letter-spacing:1px;margin-bottom:10px;">
                ⚠️ CONFIRMATION REQUIRED
            </div>
            <div style="font-size:15px;margin-bottom:8px;">
                The agent wants to perform a <b>file operation</b>:
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:13px;color:#fde68a;">
                Intent: {intent}<br>
                Params: {params}
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve", use_container_width=True, type="primary"):
                from agent.graph import run_execution
                with st.spinner("⚡ Executing..."):
                    try:
                        final_state = run_execution(ps)
                        st.session_state.last_result = final_state
                        st.session_state.history = final_state.get("history", [])
                    except Exception as e:
                        st.error(f"❌ Execution failed: {str(e)}")
                        st.session_state.last_result = {
                            "intent": ps.get("intent"),
                            "tool_output": "",
                            "error": f"Execution error: {str(e)}"
                        }
                st.session_state.pending_state = None
                st.rerun()
        with c2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.pending_state = None
                st.info("Action cancelled by user.")
                st.rerun()

    # ── Last result display ───────────────────────────────────────────────────
    if st.session_state.last_result:
        r = st.session_state.last_result
        output = r.get("tool_output", "")
        error = r.get("error", "")
        intent = r.get("intent", "")

        if error:
            st.markdown(f"""
            <div class="agent-card">
                <div class="agent-card-header">❌ Execution Error</div>
                <div class="output-area" style="color:#f87171;">{error}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Success with intent-specific display
            st.markdown("""
            <div class="agent-card">
                <div class="agent-card-header">✅ Execution Complete</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show output in a code block for file operations
            if "Code saved to:" in output or "File created:" in output:
                st.markdown(output, unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="output-area">{output}</div>', unsafe_allow_html=True)

    elif st.session_state.pending_state is None:
        st.markdown("""
        <div class="agent-card" style="text-align:center;padding:40px;">
            <div style="font-size:40px;margin-bottom:12px;">🎙️</div>
            <div style="color:#5a6080;font-size:14px;">Upload audio or type a command to get started.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── History ───────────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📋 Session History")
        for i, item in enumerate(reversed(st.session_state.history), 1):
            intent_icons = {"create_file": "📄", "write_code_to_file": "💻", "summarize_text": "📝", "general_chat": "💬"}
            ic = intent_icons.get(item.get("intent", ""), "❓")
            st.markdown(f"""
            <div class="history-row">
                <span style="color:#5a6080;font-family:'Space Mono',monospace;font-size:11px;">#{len(st.session_state.history) - i + 1}</span>
                &nbsp;<span class="intent-badge" style="font-size:10px;">{ic} {item.get('intent','?')}</span><br>
                <span style="color:#c8cce0;font-size:13px;">"{item.get('transcription','')}"</span>
            </div>
            """, unsafe_allow_html=True)
