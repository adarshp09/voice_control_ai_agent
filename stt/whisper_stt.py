import os
import re
import tempfile

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = "distil-whisper/distil-small.en"

_pipe = None


def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    # if torch.cuda.is_available():
    #     return "cuda"
    return "cpu"


def load_model():
    """Load Whisper model (cached after first load)."""
    global _pipe
    if _pipe is not None:
        return _pipe

    device = _get_device()
    torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    _pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return _pipe


def _basic_quality_check(text: str, duration_s: float, rms: float):
    """
    Cheap quality gate to reject likely hallucinations / unusable input.
    This keeps obviously bad transcripts (e.g. "you!" or "!") from entering intent routing.
    """
    cleaned = text.strip()
    words = re.findall(r"[A-Za-z0-9']+", cleaned.lower())
    known_hallucinations = {
        "you",
        "you!",
        "thanks for watching",
        "thank you",
        "thank you for watching",
        "bye",
        "bye.",
        "!",
    }

    if not cleaned:
        return False, "empty transcript"
    if rms < 0.006:
        return False, "audio too quiet"
    if cleaned.lower() in known_hallucinations and duration_s >= 0.8:
        return False, "likely hallucination"
    if len(words) <= 1 and duration_s >= 1.0:
        return False, "transcript too short for clip duration"
    if len(cleaned) < 3:
        return False, "transcript too short"
    return True, "ok"


def _decode_once(pipe, audio_array: np.ndarray, use_beam: bool = False) -> str:
    """Decode one audio array with optional beam search fallback."""
    kwargs = {}
    if use_beam:
        kwargs = {"generate_kwargs": {"num_beams": 5, "do_sample": False}}

    result = pipe(
        {"array": audio_array.astype(np.float32), "sampling_rate": 16000},
        **kwargs,
    )
    return result["text"].strip()


def _normalize_peak(audio_array: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio_array)))
    if peak <= 1e-8:
        return audio_array
    return (0.95 * (audio_array / peak)).astype(np.float32)


def _score_text(text: str) -> int:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    penalty = 0
    if text.strip() in {"!", ".", "?", "you", "you!"}:
        penalty = 100
    return (len(words) * 20) + len(text) - penalty


def transcribe_file(audio_path: str) -> str:
    """Transcribe an audio file (wav/mp3) and return text."""
    pipe = load_model()
    result = pipe(audio_path)
    return result["text"].strip()


def transcribe_audio_array(audio_array: np.ndarray, sample_rate: int = 16000, return_details: bool = False):
    """Transcribe an in-memory mono audio array."""
    pipe = load_model()

    if audio_array is None:
        details = {
            "text": "",
            "ok": False,
            "reason": "audio array is None",
            "duration_s": 0.0,
            "rms": 0.0,
        }
        return details if return_details else ""

    audio_array = np.asarray(audio_array)
    if audio_array.size == 0:
        details = {
            "text": "",
            "ok": False,
            "reason": "empty audio array",
            "duration_s": 0.0,
            "rms": 0.0,
        }
        return details if return_details else ""

    if audio_array.dtype.kind in {"i", "u"}:
        max_val = max(abs(np.iinfo(audio_array.dtype).min), np.iinfo(audio_array.dtype).max)
        audio_array = audio_array.astype(np.float32) / float(max_val)
    else:
        audio_array = audio_array.astype(np.float32)

    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1 if audio_array.shape[0] > audio_array.shape[-1] else 0)

    if sample_rate != 16000:
        import librosa

        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    duration_s = float(len(audio_array) / sample_rate)
    rms = float(np.sqrt(np.mean(np.square(audio_array)))) if audio_array.size else 0.0

    texts = []
    candidates = [audio_array, _normalize_peak(audio_array)]
    try:
        import librosa

        trimmed, _ = librosa.effects.trim(audio_array, top_db=28)
        if trimmed.size > 0:
            candidates.append(_normalize_peak(trimmed))
    except Exception:
        pass

    for cand in candidates:
        for use_beam in (False, True):
            try:
                txt = _decode_once(pipe, cand, use_beam=use_beam)
                if txt:
                    texts.append(txt)
            except Exception:
                continue

    text = max(texts, key=_score_text) if texts else ""
    ok, reason = _basic_quality_check(text, duration_s=duration_s, rms=rms)

    details = {
        "text": text,
        "ok": ok,
        "reason": reason,
        "duration_s": round(duration_s, 3),
        "rms": round(rms, 6),
    }
    return details if return_details else text


def transcribe_bytes(audio_bytes: bytes, return_details: bool = False):
    pipe = load_model()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.write(audio_bytes)
    tmp.close()

    try:
        import librosa

        try:
            audio_array, sr = librosa.load(tmp_path, sr=16000, mono=True)
        except Exception as exc:
            details = {
                "text": "",
                "ok": False,
                "reason": f"invalid or unsupported audio: {type(exc).__name__}",
                "duration_s": 0.0,
                "rms": 0.0,
            }
            return details if return_details else ""

        if audio_array.size == 0:
            details = {
                "text": "",
                "ok": False,
                "reason": "empty audio array",
                "duration_s": 0.0,
                "rms": 0.0,
            }
            return details if return_details else ""

        return transcribe_audio_array(audio_array, sample_rate=sr, return_details=return_details)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
