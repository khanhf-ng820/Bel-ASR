#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asr_bench.py - Vietnamese ASR benchmark for HistoTalk.

Runs several ASR backends on the SAME audio (clean, and with noise mixed in at fixed SNRs),
then reports WER/CER with bootstrap 95% CIs, latency (mean/SD/median/p95), real-time factor,
per-group breakdowns (region, speaker, gender, ...) and paired bootstrap tests between systems.

Backends
  qwen3-asr       Qwen3-ASR 0.6B / 1.7B via the `qwen-asr` package (local GPU)
  faster-whisper  Whisper via faster-whisper (beam search, or greedy + VAD as deployed)
  hf-whisper      Any Whisper-style HF model via transformers
                  (e.g. vinai/PhoWhisper-medium, openai/whisper-large-v3-turbo)
  gemini          Gemini API via `google-genai` (audio goes to the cloud; latency includes network)
  dummy           Self-test backend: returns the reference with random word errors

Subcommands
  build-vivos     Build a manifest CSV from a VIVOS-style folder (prompts.txt + wav files)
  run             Transcribe a manifest with one backend (resumable, one JSONL line per utterance)
  summarize       Aggregate all runs into summary.csv / summary.md (+ comparisons)
  estimate        Project the run time of the full manifest from a short trial (useful on CPU)

CPU
  Add --device cpu. dtype becomes fp32 and all cores are used (override with --cpu-threads N).
  faster-whisper uses int8 on CPU, so give CPU runs their own --tag (WER can differ slightly from GPU fp16).

Example
  python asr_bench.py build-vivos --root data/vivos/test --out manifests/vivos_test.csv
  python asr_bench.py run --manifest manifests/vivos_test.csv --backend qwen3-asr \
      --model Qwen/Qwen3-ASR-1.7B --tag qwen3-asr-1.7b --snr 20 10 5 0 --noise-dir noises/
  python asr_bench.py summarize --results results --group-by gender --baseline whisper-medium-beam5
"""

import argparse
import csv
import hashlib
import inspect
import io
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

SR = 16000
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
LANG_NAMES = {"vi": "Vietnamese", "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean"}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =============================================================================
# Vietnamese text normalization (applied identically to reference and hypothesis)
# =============================================================================
NORM = {"numbers": True, "variants": True}

_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_SCALES = ["", "nghìn", "triệu", "tỷ"]
_TONE = "\u0300\u0301\u0303\u0309\u0323"  # huyền, sắc, ngã, hỏi, nặng (combining)
# Old vs new tone placement: hòa/hoà, khỏe/khoẻ, thúy/thuý -> canonical new style
_RE_OA = re.compile(r"([oO])([" + _TONE + r"])([aAeE])$")
_RE_UY = re.compile(r"([uU])([" + _TONE + r"])([yY])$")
# Spelling / reading variants mapped to one form on BOTH sides (so they never count as errors)
_VARIANTS = {"ngàn": "nghìn", "lẻ": "linh", "tỉ": "tỷ", "mốt": "một", "tư": "bốn",
             "lăm": "năm", "nhăm": "năm"}
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _canon_tone(word):
    d = unicodedata.normalize("NFD", word)
    d = _RE_OA.sub(r"\1\3\2", d)
    d = _RE_UY.sub(r"\1\3\2", d)
    return unicodedata.normalize("NFC", d)


def _read_triple(n, full):
    h, t, u = n // 100, (n // 10) % 10, n % 10
    w = []
    if h or full:
        w += [_DIGITS[h], "trăm"]
    if t == 0:
        if u:
            if h or full:
                w.append("linh")
            w.append(_DIGITS[u])
    elif t == 1:
        w.append("mười")
        if u == 5:
            w.append("lăm")
        elif u:
            w.append(_DIGITS[u])
    else:
        w += [_DIGITS[t], "mươi"]
        if u == 1:
            w.append("mốt")
        elif u == 4:
            w.append("tư")
        elif u == 5:
            w.append("lăm")
        elif u:
            w.append(_DIGITS[u])
    return w


def number_to_words(n):
    """Read a non-negative integer in Vietnamese (up to 999 tỷ). Returns None if too large."""
    if n == 0:
        return "không"
    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    if len(groups) > len(_SCALES):
        return None
    top = len(groups) - 1
    w = []
    for i in range(top, -1, -1):
        g = groups[i]
        if g == 0:
            continue
        w += _read_triple(g, full=(i < top))
        if _SCALES[i]:
            w.append(_SCALES[i])
    return " ".join(w)


def _digits_reading(s):
    return " ".join(_DIGITS[int(c)] for c in s if c.isdigit())


def _int_words(s):
    if len(s) > 1 and s.startswith("0"):  # phone numbers / codes: digit by digit
        return _digits_reading(s)
    w = number_to_words(int(s))
    return w if w is not None else _digits_reading(s)


def _verbalize_token(tok):
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", tok):      # 1.288 / 1.000.000 (VN thousands separator)
        return _int_words(tok.replace(".", ""))
    if re.fullmatch(r"\d+,\d+", tok):                  # 3,14 -> ba phẩy mười bốn
        a, b = tok.split(",")
        return f"{_int_words(a)} phẩy {_int_words(b)}"
    if re.fullmatch(r"\d+", tok):
        return _int_words(tok)
    return " ".join(_int_words(p) for p in re.split(r"[.,]", tok) if p)


def normalize_vi(text):
    t = unicodedata.normalize("NFC", text or "").lower()
    t = t.replace("%", " phần trăm ")
    if NORM["numbers"]:
        t = _NUM_RE.sub(lambda m: " " + _verbalize_token(m.group(0)) + " ", t)
    # punctuation / symbols -> space; keep letters, digits, combining marks
    t = "".join(ch if (ch.isspace() or unicodedata.category(ch)[0] in "LNM") else " " for ch in t)
    words = [_canon_tone(w) for w in t.split()]
    if NORM["variants"]:
        words = [_VARIANTS.get(w, w) for w in words]
    return " ".join(words)


# =============================================================================
# Edit distance with substitution / deletion / insertion counts
# =============================================================================
def align_counts(ref, hyp):
    """Levenshtein alignment between token lists. Returns (S, D, I)."""
    m = len(hyp)
    prev = [(j, 0, 0, j) for j in range(m + 1)]  # (cost, S, D, I)
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)] + [None] * m
        ri = ref[i - 1]
        for j in range(1, m + 1):
            c, s, d, ins = prev[j - 1]
            best = (c, s, d, ins) if ri == hyp[j - 1] else (c + 1, s + 1, d, ins)
            c, s, d, ins = prev[j]
            if c + 1 < best[0]:
                best = (c + 1, s, d + 1, ins)
            c, s, d, ins = cur[j - 1]
            if c + 1 < best[0]:
                best = (c + 1, s, d, ins + 1)
            cur[j] = best
        prev = cur
    _, S, D, I = prev[m]
    return S, D, I


def score_pair(ref_text, hyp_text):
    ref_n, hyp_n = normalize_vi(ref_text), normalize_vi(hyp_text)
    rw, hw = ref_n.split(), hyp_n.split()
    S, D, I = align_counts(rw, hw)
    rc, hc = list(ref_n), list(hyp_n)  # CER over characters incl. spaces (jiwer-compatible)
    cS, cD, cI = align_counts(rc, hc)
    return {"ref_norm": ref_n, "hyp_norm": hyp_n, "n_ref_words": len(rw), "S": S, "D": D, "I": I,
            "n_ref_chars": len(rc), "cS": cS, "cD": cD, "cI": cI}


# =============================================================================
# Audio + noise
# =============================================================================
def _read_wav_stdlib(path):
    """PCM WAV reader without soundfile (fallback). Returns (frames x channels float32, sr)."""
    import wave
    with wave.open(path if hasattr(path, "read") else str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    elif sw == 3:
        a = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        x = (np.where(v >= 1 << 23, v - (1 << 24), v)).astype(np.float32) / 8388608.0
    elif sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width {sw} in {path}")
    return x.reshape(-1, ch), sr


def load_audio(path):
    """Any audio file -> mono float32 at 16 kHz. Uses soundfile; falls back to stdlib for PCM WAV."""
    try:
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except ImportError:
        if Path(path).suffix.lower() != ".wav":
            raise SystemExit(f"pip install soundfile  (needed to read {Path(path).suffix} files)")
        x, sr = _read_wav_stdlib(path)
    x = x.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.ascontiguousarray(x, dtype=np.float32)


def write_wav16(target, y):
    """Write mono 16 kHz PCM_16 WAV to a path or a binary file object."""
    try:
        import soundfile as sf
        sf.write(target if not isinstance(target, Path) else str(target), y, SR, format="WAV",
                 subtype="PCM_16")
        return
    except ImportError:
        pass
    import wave
    pcm = (np.clip(y, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()
    with wave.open(str(target) if isinstance(target, (str, Path)) else target, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def stable_seed(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def safe_name(s):
    return re.sub(r"[^\w.-]", "_", str(s))


class NoiseBank:
    """Deterministic noise mixing: the same (utterance, SNR, seed) always gets the same noise."""

    def __init__(self, noise_dir, seed=0):
        self.files = sorted(p for p in Path(noise_dir).rglob("*") if p.suffix.lower() in AUDIO_EXTS)
        if not self.files:
            raise SystemExit(f"No audio files found in --noise-dir {noise_dir}")
        self.seed = seed
        listing = "\n".join(str(p.relative_to(noise_dir)) for p in self.files)
        self.set_id = hashlib.md5(f"{listing}|{seed}".encode()).hexdigest()[:8]

    @lru_cache(maxsize=32)
    def _noise(self, i):
        return load_audio(self.files[i])

    def mix(self, speech, snr_db, key):
        rng = np.random.default_rng(stable_seed(key, snr_db, self.seed))
        i = int(rng.integers(len(self.files)))
        noise = self._noise(i)
        if len(noise) == 0:
            raise RuntimeError(f"empty noise file {self.files[i]}")
        if len(noise) < len(speech):
            noise = np.tile(noise, int(math.ceil(len(speech) / len(noise))))
        start = int(rng.integers(0, len(noise) - len(speech) + 1))
        n = noise[start:start + len(speech)].astype(np.float64)
        s = speech.astype(np.float64)
        ps, pn = np.mean(s ** 2) + 1e-12, np.mean(n ** 2) + 1e-12
        y = s + n * np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))
        peak = np.max(np.abs(y)) if len(y) else 0.0
        if peak > 0.99:  # rescale whole mixture: avoids clipping, keeps the SNR
            y *= 0.99 / peak
        return y.astype(np.float32), {"noise_file": self.files[i].name, "offset": start}


def get_audio(row, cond, noise, cache_root):
    clean = load_audio(row["audio_path"])
    if cond == "clean":
        return clean, None
    path = cache_root / noise.set_id / f"snr{cond:g}" / f"{safe_name(row['utt_id'])}.wav"
    info_path = path.with_suffix(".json")
    if not path.exists():
        y, info = noise.mix(clean, cond, row["utt_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_wav16(path, y)
        info_path.write_text(json.dumps(info), encoding="utf-8")
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info["cached_wav"] = str(path)
    return load_audio(path), info  # every backend reads the identical cached file


# =============================================================================
# Backends
# =============================================================================
def _torch_dtype(name):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _split_device(device):
    if device.startswith("cuda"):
        return "cuda", int(device.split(":")[1]) if ":" in device else 0
    return "cpu", 0


def _set_torch_threads(args):
    if args.device == "cpu" and args.cpu_threads:
        import torch
        torch.set_num_threads(args.cpu_threads)


def _hw_label(args):
    """Short hardware description stored with every prediction (shown in the summary table)."""
    if args.backend == "gemini":
        return "cloud API"
    if args.backend == "dummy":
        return "dummy"
    if args.device.startswith("cuda"):
        gpus = _gpu_info()
        if gpus:
            idx = _split_device(args.device)[1]
            return "GPU: " + (gpus[idx] if idx < len(gpus) else gpus[0]).split(",")[0].strip()
        return "GPU"
    return f"CPU: {_cpu_name()} ({args.cpu_threads} threads)"


class Backend:
    uses_torch_cuda = False

    def __init__(self, args):
        self.args = args

    def prepare(self, audio, row):
        """Turn 16 kHz float32 audio into the backend's input. Not timed."""
        return audio

    def transcribe(self, payload):
        raise NotImplementedError

    def transcribe_timed(self, payload):
        self._sync()
        t0 = time.perf_counter()
        text = self.transcribe(payload)
        self._sync()
        return text, time.perf_counter() - t0, {}

    def _sync(self):
        if self.uses_torch_cuda:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def describe(self):
        return {}


class Qwen3ASRBackend(Backend):
    def __init__(self, args):
        super().__init__(args)
        from qwen_asr import Qwen3ASRModel
        self.model_id = args.model or "Qwen/Qwen3-ASR-1.7B"
        if args.qwen_vllm:
            self.model = Qwen3ASRModel.LLM(model=self.model_id, gpu_memory_utilization=args.gpu_mem_util,
                                           max_inference_batch_size=1, max_new_tokens=args.max_new_tokens)
        else:
            _set_torch_threads(args)
            kw = dict(dtype=_torch_dtype(args.dtype), device_map=args.device,
                      max_inference_batch_size=1, max_new_tokens=args.max_new_tokens)
            if args.attn:
                kw["attn_implementation"] = args.attn
            self.model = Qwen3ASRModel.from_pretrained(self.model_id, **kw)
            self.uses_torch_cuda = args.device.startswith("cuda")
        self.language = None if args.language == "auto" else LANG_NAMES.get(args.language, args.language)

    def prepare(self, audio, row):
        return (audio, SR)

    def transcribe(self, payload):
        return self.model.transcribe(audio=payload, language=self.language)[0].text

    def describe(self):
        return {"model": self.model_id, "language": self.language, "vllm": bool(self.args.qwen_vllm),
                "dtype": self.args.dtype, "batch_size": 1}


class FasterWhisperBackend(Backend):
    def __init__(self, args):
        super().__init__(args)
        from faster_whisper import WhisperModel
        dev, idx = _split_device(args.device)
        self.compute_type = args.compute_type or ("float16" if dev == "cuda" else "int8")
        self.model_id = args.model or "medium"
        kw = dict(device=dev, device_index=idx, compute_type=self.compute_type)
        if args.cpu_threads:
            kw["cpu_threads"] = args.cpu_threads
        self.model = WhisperModel(self.model_id, **kw)
        self.kw = dict(language=None if args.language == "auto" else args.language,
                       beam_size=args.beam, vad_filter=args.vad, condition_on_previous_text=False)
        if args.temperature is not None:
            self.kw["temperature"] = args.temperature

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(audio, **self.kw)
        return " ".join(s.text.strip() for s in segments)  # consuming the generator = decoding

    def describe(self):
        return {"model": self.model_id, "compute_type": self.compute_type, **self.kw}


class HFWhisperBackend(Backend):
    def __init__(self, args):
        super().__init__(args)
        from transformers import pipeline
        if not args.model:
            raise SystemExit("--model is required for hf-whisper (e.g. vinai/PhoWhisper-medium)")
        cuda = args.device.startswith("cuda")
        _set_torch_threads(args)
        dtype = _torch_dtype(args.dtype if cuda else "fp32")
        key = "dtype" if "dtype" in inspect.signature(pipeline).parameters else "torch_dtype"
        self.pipe = pipeline("automatic-speech-recognition", model=args.model, device=args.device,
                             **{key: dtype})
        self.uses_torch_cuda = cuda
        self.gen = {"num_beams": args.beam}
        if args.language != "auto":
            self.gen.update(language=args.language, task="transcribe")
        self._checked = False

    def transcribe(self, audio):
        try:
            out = self.pipe({"raw": audio, "sampling_rate": SR}, generate_kwargs=dict(self.gen))
        except (ValueError, TypeError):
            if self._checked or "language" not in self.gen:
                raise
            # some fine-tuned checkpoints reject the language token: retry once without it
            self.gen.pop("language", None)
            self.gen.pop("task", None)
            print("  [hf-whisper] model rejected language/task kwargs; continuing without them")
            out = self.pipe({"raw": audio, "sampling_rate": SR}, generate_kwargs=dict(self.gen))
        self._checked = True
        return out["text"]

    def describe(self):
        return {"model": self.args.model, "generate_kwargs": self.gen, "dtype": self.args.dtype}


DEFAULT_GEMINI_PROMPT = (
    "Hãy chép lại nguyên văn lời nói tiếng Việt trong đoạn âm thanh. "
    "Chỉ trả về đúng những gì được nói, không giải thích, không dịch, không tóm tắt, "
    "không sửa câu. Nếu không có lời nói, trả về chuỗi rỗng."
)


def _retryable(e):
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if code in (408, 429, 500, 502, 503, 504):
        return True
    s = str(e)
    return any(k in s for k in ("429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "503", "500",
                                "DEADLINE_EXCEEDED", "timed out", "Timeout"))


class GeminiBackend(Backend):
    def __init__(self, args):
        super().__init__(args)
        from google import genai
        from google.genai import types
        if not args.model:
            raise SystemExit("--model is required for gemini (use the exact model id from the paper)")
        self.types = types
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=key) if key else genai.Client()
        cfg = {"temperature": 0.0}
        tc = {}
        if args.thinking_budget is not None:
            tc["thinking_budget"] = args.thinking_budget
        if args.thinking_level:
            tc["thinking_level"] = args.thinking_level
        if tc:
            cfg["thinking_config"] = types.ThinkingConfig(**tc)
        self.config = types.GenerateContentConfig(**cfg)
        self.prompt = (Path(args.prompt_file).read_text(encoding="utf-8").strip()
                       if args.prompt_file else DEFAULT_GEMINI_PROMPT)
        self.min_interval = 60.0 / args.rpm if args.rpm else 0.0
        self._last = 0.0

    def prepare(self, audio, row):
        buf = io.BytesIO()
        write_wav16(buf, audio)
        return buf.getvalue()

    def transcribe_timed(self, payload):
        part = self.types.Part.from_bytes(data=payload, mime_type="audio/wav")
        retries = 0
        while True:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)  # rate limiting happens outside the timed region
            self._last = time.monotonic()
            t0 = time.perf_counter()
            try:
                resp = self.client.models.generate_content(
                    model=self.args.model, contents=[self.prompt, part], config=self.config)
                lat = time.perf_counter() - t0  # only the successful attempt is timed
            except Exception as e:  # noqa: BLE001
                if retries < self.args.max_retries and _retryable(e):
                    retries += 1
                    time.sleep(min(60.0, 2.0 ** retries) + random.random())
                    continue
                raise
            extra = {"retries": retries}
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                extra.update(prompt_tokens=getattr(um, "prompt_token_count", None),
                             output_tokens=getattr(um, "candidates_token_count", None),
                             thoughts_tokens=getattr(um, "thoughts_token_count", None))
            try:
                text = resp.text or ""
            except Exception:  # noqa: BLE001
                text = ""
            if not text:
                try:
                    extra["finish_reason"] = str(resp.candidates[0].finish_reason)
                except Exception:  # noqa: BLE001
                    pass
            return text, lat, extra

    def describe(self):
        return {"model": self.args.model, "prompt": self.prompt, "temperature": 0.0,
                "thinking_budget": self.args.thinking_budget, "thinking_level": self.args.thinking_level,
                "note": "latency includes network round-trip"}


class DummyBackend(Backend):
    """Self-test only: corrupts the reference with random word errors."""

    def prepare(self, audio, row):
        return (len(audio), row["text"], row["utt_id"])

    def transcribe(self, payload):
        n, ref, uid = payload
        p = self.args.dummy_error_rate
        rng = random.Random(stable_seed("dummy", uid, n, p))
        out = []
        for w in ref.split():
            r = rng.random()
            if r < p / 3:
                continue
            if r < 2 * p / 3:
                out.append("xxx")
            elif r < p:
                out += [w, "ừ"]
            else:
                out.append(w)
        time.sleep(0.002)
        return " ".join(out)


BACKENDS = {"qwen3-asr": Qwen3ASRBackend, "faster-whisper": FasterWhisperBackend,
            "hf-whisper": HFWhisperBackend, "gemini": GeminiBackend, "dummy": DummyBackend}
PIP_HINTS = {"qwen3-asr": "-U qwen-asr", "faster-whisper": "faster-whisper",
             "hf-whisper": "transformers torch accelerate", "gemini": "google-genai"}


# =============================================================================
# Manifest
# =============================================================================
_ID_KEYS, _PATH_KEYS, _TEXT_KEYS = ("utt_id", "id"), ("audio_path", "path", "audio", "file"), \
    ("text", "transcript", "sentence")


def read_manifest(path):
    """CSV with columns utt_id, audio_path, text (+ any extra columns used for grouping).

    Optional column `condition` (e.g. clean / snr20 / snr0): the same utt_id may then appear once per
    condition, and every condition is scored separately, paired on the same sentences."""
    path = Path(path)
    rows, seen = [], set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, raw in enumerate(csv.DictReader(f)):
            r = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            uid = next((r[k] for k in _ID_KEYS if r.get(k)), f"utt{i:06d}")
            ap = next((r[k] for k in _PATH_KEYS if r.get(k)), None)
            txt = next((r[k] for k in _TEXT_KEYS if k in r), None)
            cond = r.get("condition") or None
            if not ap or txt is None:
                raise SystemExit(f"{path}: row {i + 2} needs audio_path and text columns")
            if (cond, uid) in seen:
                raise SystemExit(f"{path}: duplicate utt_id {uid}" + (f" in condition {cond}" if cond else ""))
            seen.add((cond, uid))
            p = Path(ap)
            if not p.is_absolute():
                p = path.parent / p
            fields = {k: v for k, v in r.items()
                      if k not in _ID_KEYS + _PATH_KEYS + _TEXT_KEYS + ("condition",)}
            rows.append({"utt_id": uid, "audio_path": str(p), "text": txt, "fields": fields, "condition": cond})
    if not rows:
        raise SystemExit(f"{path}: empty manifest")
    has = {r["condition"] is not None for r in rows}
    if len(has) > 1:
        raise SystemExit(f"{path}: the condition column is empty on some rows")
    return rows


def select_subset(rows, limit=None, seed=None):
    """Order/limit by sentence (utt_id), keeping every condition of each chosen sentence.

    Sentences that exist in more conditions come first, so a small --limit still covers conditions that
    only a few sentences have (e.g. dfnet_*). Within that, the order is a seeded hash (independent of
    Python's `random`, so it cannot line up with subsets drawn elsewhere with random.seed). The order is
    deterministic: a larger --limit with the same seed extends the previous subset."""
    ids = list(dict.fromkeys(r["utt_id"] for r in rows))
    if seed is not None:
        ids.sort(key=lambda u: hashlib.md5(f"{seed}:{u}".encode("utf-8")).hexdigest())
    n_cond = defaultdict(int)
    for r in rows:
        n_cond[r["utt_id"]] += 1
    ids.sort(key=lambda u: -n_cond[u])   # stable sort keeps the hash order inside each group
    if limit:
        ids = ids[:limit]
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["utt_id"]].append(r)
    return [r for u in ids for r in by_id[u]]


def cmd_build_vivos(args):
    root = Path(args.root)
    audio_root = Path(args.audio_root) if args.audio_root else root
    prompts = sorted(root.rglob("prompts.txt"))
    if not prompts:
        raise SystemExit(f"No prompts.txt under {root}")
    index, dups = {}, 0
    for p in sorted(audio_root.rglob("*")):
        if p.suffix.lower() in AUDIO_EXTS:
            if p.stem in index:
                dups += 1
                continue
            index[p.stem] = p
    if dups:
        print(f"WARNING: {dups} duplicate audio file names under {audio_root}; kept the first of each. "
              f"Use --audio-root to point at one copy (e.g. only the noisy or only the clean waves).")
    genders = {}
    for gf in root.rglob("genders.txt"):
        for line in gf.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                genders[parts[0]] = parts[1]
    rows, missing = [], 0
    for pf in prompts:
        for line in pf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            uid, _, text = line.partition(" ")
            wav = index.get(uid)
            if wav is None:
                missing += 1
                continue
            spk = uid.rsplit("_", 1)[0]
            rows.append({"utt_id": uid, "audio_path": str(wav.resolve()), "text": text.strip(),
                         "speaker": spk, "gender": genders.get(spk, ""), "split": pf.parent.name})
    if args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = sorted(rows[:args.limit], key=lambda r: r["utt_id"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "audio_path", "text", "speaker", "gender", "split"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out} ({missing} prompts without audio)")


# =============================================================================
# Run
# =============================================================================
def cond_key(c):
    return c if isinstance(c, str) else f"snr{c:g}"


def _snr_of(c):
    if isinstance(c, (int, float)):
        return float(c)
    m = re.fullmatch(r"snr(-?\d+(?:\.\d+)?)", c)
    return float(m.group(1)) if m else None


def _pkg_versions():
    from importlib import metadata
    out = {}
    for pkg in ("numpy", "scipy", "soundfile", "torch", "transformers", "qwen-asr", "faster-whisper",
                "ctranslate2", "google-genai", "vllm"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _gpu_info():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return [line.strip() for line in q.stdout.splitlines() if line.strip()] or None
    except Exception:  # noqa: BLE001
        return None


def _cpu_name():
    if sys.platform == "darwin":   # Apple Silicon: "Apple M2" etc.
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            if out:
                return out
        except Exception:  # noqa: BLE001
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def _torch_peak_mb():
    if "torch" not in sys.modules:
        return None
    import torch
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)


def _peak_rss_mb():
    """Peak RAM of this process (model weights included) - useful for CPU runs."""
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / (2 ** 20 if sys.platform == "darwin" else 1024), 1)
    except Exception:  # noqa: BLE001
        return None


def audio_duration(path):
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except ImportError:
        import wave
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())


def load_records(path, keep_errors=False):
    recs = {}
    if not path.exists():
        return recs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = (r["condition"], r["utt_id"])
            if r.get("error") and not keep_errors and k in recs and not recs[k].get("error"):
                continue
            recs[k] = r  # later lines win (re-runs / retried errors)
    return recs


def corpus_rate(recs, e_keys, n_key):
    num = sum(sum(r[k] for k in e_keys) for r in recs)
    den = sum(r[n_key] for r in recs)
    return 100.0 * num / den if den else float("nan")


def cmd_run(args):
    NORM["numbers"], NORM["variants"] = not args.no_number_norm, not args.no_variants
    on_cpu = not args.device.startswith("cuda")
    if args.dtype is None:
        args.dtype = "fp32" if on_cpu else "bf16"
    elif on_cpu and args.dtype != "fp32" and args.backend == "qwen3-asr":
        print(f"NOTE: --dtype {args.dtype} on CPU is only fast on CPUs with native bf16/fp16 support; "
              f"fp32 is the safe default.")
    if on_cpu and not args.cpu_threads:
        args.cpu_threads = os.cpu_count() or 1
    rows = read_manifest(args.manifest)
    manifest_conds = rows[0]["condition"] is not None
    if manifest_conds:
        if args.snr:
            raise SystemExit("--snr cannot be combined with a manifest that already has a condition column")
        if args.conditions:
            rows = [r for r in rows if r["condition"] in set(args.conditions)]
            if not rows:
                raise SystemExit(f"No rows with condition in {args.conditions}")
    elif args.conditions:
        raise SystemExit("--conditions needs a manifest with a condition column")
    rows = select_subset(rows, args.limit, args.shuffle_seed)
    if manifest_conds:
        conds = list(dict.fromkeys(r["condition"] for r in rows))
    else:
        conds = ([] if args.no_clean else ["clean"]) + [float(s) for s in (args.snr or [])]
    if not conds:
        raise SystemExit("Nothing to run: --no-clean without --snr")
    if not manifest_conds and any(c != "clean" for c in conds) and not args.noise_dir:
        raise SystemExit("--snr needs --noise-dir (folder of noise recordings)")

    tag = args.tag or safe_name(f"{args.backend}-{Path(args.model or 'default').name}"
                                f"{'-beam' + str(args.beam) if 'whisper' in args.backend else ''}"
                                f"{'-vad' if args.vad else ''}")
    out_root = Path(args.out)
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    done = {k for k, r in load_records(pred_path).items() if not (args.retry_errors and r.get("error"))}
    if manifest_conds:   # sentence by sentence, all its conditions together -> balanced partial results
        jobs = [(r["condition"], r) for r in rows]
        n_sent = len({r["utt_id"] for r in rows})
        print(f"[{tag}] {n_sent} sentences, {len(rows)} rows over conditions {conds}")
    else:
        jobs = [(c, r) for c in conds for r in rows]
        print(f"[{tag}] {len(rows)} utterances x {len(conds)} conditions")
    todo = [(c, r) for c, r in jobs if (cond_key(c), r["utt_id"]) not in done]
    print(f"  {len(todo)} to do, {len(jobs) - len(todo)} already done")

    if todo:
        noise = NoiseBank(args.noise_dir, args.noise_seed) if args.noise_dir else None
        t_load = time.perf_counter()
        try:
            backend = BACKENDS[args.backend](args)
        except ImportError as e:
            raise SystemExit(f"Backend '{args.backend}' needs a missing package ({e.name}). "
                             f"Install with: pip install {PIP_HINTS.get(args.backend, e.name)}")
        load_s = time.perf_counter() - t_load
        hw = _hw_label(args)
        print(f"  hardware: {hw} | model load {load_s:.1f}s")
        for r in rows[:args.warmup]:  # warm-up (CUDA kernels, caches, connections): not recorded
            try:
                backend.transcribe_timed(backend.prepare(load_audio(r["audio_path"]), r))
            except Exception as e:  # noqa: BLE001
                print(f"  warm-up failed: {type(e).__name__}: {e}")
        meta_path = out_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"runs": []}
        run_meta = {
            "started_at": now_iso(), "tag": tag, "backend": args.backend, "hardware": hw,
            "backend_info": backend.describe(), "manifest": str(Path(args.manifest).resolve()),
            "conditions": [cond_key(c) for c in conds], "n_utterances": len(rows),
            "noise_dir": args.noise_dir, "noise_set": noise.set_id if noise else None,
            "normalization": dict(NORM), "model_load_s": round(load_s, 2), "warmup": args.warmup,
            "args": {k: v for k, v in vars(args).items() if k != "func"},
            "python": sys.version.split()[0], "platform": platform.platform(), "cpu": _cpu_name(),
            "gpu_after_warmup": _gpu_info(), "packages": _pkg_versions(),
        }
        cache_root = out_root / "_noisy_cache"
        n_err, streak, t_start = 0, 0, time.perf_counter()
        with open(pred_path, "a", encoding="utf-8") as f:
            for k, (c, r) in enumerate(todo, 1):
                ck = cond_key(c)
                rec = {"utt_id": r["utt_id"], "condition": ck, "snr": _snr_of(c),
                       "fields": r["fields"], "time": now_iso(), "hw": hw}
                try:
                    audio, ninfo = get_audio(r, "clean" if manifest_conds else c, noise, cache_root)
                    rec["duration_s"] = round(len(audio) / SR, 3)
                    rec["noise"] = ninfo
                    hyp, lat, extra = backend.transcribe_timed(backend.prepare(audio, r))
                    rec.update(latency_s=round(lat, 4), rtf=round(lat / max(rec["duration_s"], 1e-6), 4),
                               extra=extra, error=None)
                    streak = 0
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # noqa: BLE001
                    hyp, n_err, streak = "", n_err + 1, streak + 1
                    rec.update(latency_s=None, rtf=None, error=f"{type(e).__name__}: {e}"[:500])
                rec["ref"], rec["hyp"] = r["text"], hyp
                rec["hyp_had_digits"] = bool(re.search(r"\d", hyp))
                rec.update(score_pair(r["text"], hyp))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if args.max_consecutive_errors and streak >= args.max_consecutive_errors:
                    print(f"  STOPPED: {streak} errors in a row (last: {rec['error']}).\n"
                          f"  Fix the cause (quota, missing model, ...) and re-run with --retry-errors.")
                    break
                if k % args.log_every == 0 or k == len(todo):
                    el = time.perf_counter() - t_start
                    print(f"  {k}/{len(todo)}  errors={n_err}  elapsed={el:.0f}s  "
                          f"eta={el / k * (len(todo) - k):.0f}s")
        run_meta.update(finished_at=now_iso(), n_errors=n_err, torch_peak_alloc_mb=_torch_peak_mb(),
                        peak_rss_mb=_peak_rss_mb(), gpu_at_end=_gpu_info())
        meta["runs"].append(run_meta)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    recs = [r for r in load_records(pred_path).values() if not r.get("error")]
    for c in conds:
        rs = [r for r in recs if r["condition"] == cond_key(c)]
        if rs:
            lat = [r["latency_s"] for r in rs]
            print(f"  {cond_key(c):>8}: n={len(rs)}  WER={corpus_rate(rs, ('S', 'D', 'I'), 'n_ref_words'):.2f}%"
                  f"  CER={corpus_rate(rs, ('cS', 'cD', 'cI'), 'n_ref_chars'):.2f}%"
                  f"  latency median={np.median(lat):.3f}s")


# =============================================================================
# Summarize
# =============================================================================
def _aggregate(num, den, clusters):
    num, den = np.asarray(num, float), np.asarray(den, float)
    if clusters is None:
        return num, den
    keys = {}
    idx = np.array([keys.setdefault(c, len(keys)) for c in clusters])
    return np.bincount(idx, num, len(keys)), np.bincount(idx, den, len(keys))


def _resample_sums(arrays, B, rng, chunk=200):
    n = len(arrays[0])
    outs = [[] for _ in arrays]
    for start in range(0, B, chunk):
        idx = rng.integers(0, n, size=(min(chunk, B - start), n))
        for o, a in zip(outs, arrays):
            o.append(a[idx].sum(axis=1))
    return [np.concatenate(o) for o in outs]


def bootstrap_ci(num, den, B, rng, clusters=None):
    num, den = _aggregate(num, den, clusters)
    if len(num) < 2 or den.sum() == 0:
        return float("nan"), float("nan")
    sn, sd = _resample_sums([num, den], B, rng)
    vals = 100.0 * sn / np.maximum(sd, 1e-12)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(err_a, err_b, den, B, rng, clusters=None):
    a, d = _aggregate(err_a, den, clusters)
    b, _ = _aggregate(err_b, den, clusters)
    sa, sb, sd = _resample_sums([a, b, d], B, rng)
    delta = 100.0 * (sa - sb) / np.maximum(sd, 1e-12)
    p = min(1.0, 2 * min(np.mean(delta <= 0), np.mean(delta >= 0)))
    return float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5)), float(p)


def _cond_sort(c):
    """clean first, then snrNN from high to low SNR, then any other condition name."""
    if c == "clean":
        return (0, 0.0, c)
    m = re.fullmatch(r"snr(-?\d+(?:\.\d+)?)", c)
    return (1, -float(m.group(1)), c) if m else (2, 0.0, c)


def _fmt(x, nd=2):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def stats_row(tag, cond, group, value, rs, args, rng):
    ok = [r for r in rs if not r.get("error")]
    clusters = None
    if args.bootstrap_unit == "speaker":
        clusters = [r["fields"].get("speaker") or r["utt_id"] for r in ok]
    lat = np.array([r["latency_s"] for r in ok], float) if ok else np.array([])
    wer_e = [r["S"] + r["D"] + r["I"] for r in ok]
    cer_e = [r["cS"] + r["cD"] + r["cI"] for r in ok]
    nw, nc = sum(r["n_ref_words"] for r in ok), sum(r["n_ref_chars"] for r in ok)
    wer = 100.0 * sum(wer_e) / nw if nw else float("nan")
    cer = 100.0 * sum(cer_e) / nc if nc else float("nan")
    wl, wh = bootstrap_ci(wer_e, [r["n_ref_words"] for r in ok], args.bootstrap, rng, clusters) if ok else (float("nan"),) * 2
    cl, ch = bootstrap_ci(cer_e, [r["n_ref_chars"] for r in ok], args.bootstrap, rng, clusters) if ok else (float("nan"),) * 2
    pct = lambda k: 100.0 * sum(r[k] for r in ok) / nw if nw else float("nan")  # noqa: E731
    return {
        "system": tag, "condition": cond, "group": group, "value": value,
        "hardware": " + ".join(sorted({r.get("hw") or "?" for r in ok})) or "?",
        "n_utt": len(ok), "n_errors": len(rs) - len(ok), "n_ref_words": nw,
        "wer": wer, "wer_ci_low": wl, "wer_ci_high": wh,
        "sub": pct("S"), "del": pct("D"), "ins": pct("I"),
        "cer": cer, "cer_ci_low": cl, "cer_ci_high": ch,
        "lat_mean": float(lat.mean()) if len(lat) else float("nan"),
        "lat_sd": float(lat.std(ddof=1)) if len(lat) > 1 else float("nan"),
        "lat_median": float(np.median(lat)) if len(lat) else float("nan"),
        "lat_p95": float(np.percentile(lat, 95)) if len(lat) else float("nan"),
        "rtf_mean": float(np.mean([r["rtf"] for r in ok])) if ok else float("nan"),
        "audio_mean_s": float(np.mean([r["duration_s"] for r in ok])) if ok else float("nan"),
        "digits_in_hyp": sum(1 for r in ok if r.get("hyp_had_digits")),
    }


def cmd_summarize(args):
    NORM["numbers"], NORM["variants"] = not args.no_number_norm, not args.no_variants
    root = Path(args.results)
    dirs = [d for d in sorted(root.iterdir()) if (d / "predictions.jsonl").exists()]
    if args.tags:
        dirs = [d for d in dirs if d.name in set(args.tags)]
    if not dirs:
        raise SystemExit(f"No */predictions.jsonl under {root}")
    systems = {}
    for d in dirs:
        recs = list(load_records(d / "predictions.jsonl", keep_errors=True).values())
        if args.renormalize:
            for r in recs:
                r.update(score_pair(r["ref"], r["hyp"]))
        hws = {r.get("hw") or "?" for r in recs if not r.get("error")}
        if len(hws) > 1:
            print(f"WARNING: {d.name} mixes predictions from different hardware {sorted(hws)}; "
                  f"its latency numbers are not comparable. Re-run on one machine or use separate tags.")
        systems[d.name] = recs
    rng = np.random.default_rng(args.seed)
    rows = []
    for tag, recs in systems.items():
        by_cond = defaultdict(list)
        for r in recs:
            by_cond[r["condition"]].append(r)
        for cond in sorted(by_cond, key=_cond_sort):
            rs = by_cond[cond]
            rows.append(stats_row(tag, cond, "ALL", "ALL", rs, args, rng))
            for g in args.group_by or []:
                sub = defaultdict(list)
                for r in rs:
                    sub[r["fields"].get(g) or "?"].append(r)
                for val in sorted(sub):
                    rows.append(stats_row(tag, cond, g, val, sub[val], args, rng))

    out = Path(args.out) if args.out else root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = ["# ASR benchmark summary", "",
          f"Generated {now_iso()}. WER/CER are corpus-level (sum of errors / sum of reference tokens). "
          f"95% CI: {args.bootstrap} bootstrap resamples over {args.bootstrap_unit}s. "
          f"Normalization: NFC, lowercase, punctuation removed, tone placement canonicalized"
          f"{', numbers verbalized' if NORM['numbers'] else ''}"
          f"{', reading variants merged' if NORM['variants'] else ''}. "
          "Latency is per utterance at batch size 1, warm-up excluded.", "",
          "## Overall", "",
          "| System | Hardware | Condition | N | WER % [95% CI] | CER % [95% CI] | S / D / I % | Latency median / p95 (s) | RTF | Failed |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["group"] == "ALL":
            md.append(f"| {r['system']} | {r['hardware']} | {r['condition']} | {r['n_utt']} | {_fmt(r['wer'])} "
                      f"[{_fmt(r['wer_ci_low'])}, {_fmt(r['wer_ci_high'])}] | {_fmt(r['cer'])} "
                      f"[{_fmt(r['cer_ci_low'])}, {_fmt(r['cer_ci_high'])}] | {_fmt(r['sub'], 1)} / "
                      f"{_fmt(r['del'], 1)} / {_fmt(r['ins'], 1)} | {_fmt(r['lat_median'], 3)} / "
                      f"{_fmt(r['lat_p95'], 3)} | {_fmt(r['rtf_mean'], 3)} | {r['n_errors']} |")
    for g in args.group_by or []:
        md += ["", f"## By {g}", "", f"| System | Condition | {g} | N | WER % [95% CI] | CER % |",
               "|---|---|---|---|---|---|"]
        for r in rows:
            if r["group"] == g:
                md.append(f"| {r['system']} | {r['condition']} | {r['value']} | {r['n_utt']} | "
                          f"{_fmt(r['wer'])} [{_fmt(r['wer_ci_low'])}, {_fmt(r['wer_ci_high'])}] | "
                          f"{_fmt(r['cer'])} |")

    pairs = [tuple(p) for p in (args.compare or [])]
    if args.baseline:
        pairs += [(t, args.baseline) for t in systems if t != args.baseline]
    if pairs:
        md += ["", "## Paired comparisons (same utterances)", "",
               "Δ = WER(A) − WER(B); negative means A is better. p: two-sided paired bootstrap.", "",
               "| Condition | A | B | N | WER A | WER B | Δ [95% CI] | p |", "|---|---|---|---|---|---|---|---|"]
        for a, b in pairs:
            if a not in systems or b not in systems:
                print(f"WARNING: comparison {a} vs {b}: unknown system")
                continue
            ra = {(r["condition"], r["utt_id"]): r for r in systems[a] if not r.get("error")}
            rb = {(r["condition"], r["utt_id"]): r for r in systems[b] if not r.get("error")}
            for cond in sorted({k[0] for k in ra} & {k[0] for k in rb}, key=_cond_sort):
                keys = sorted(k for k in ra if k[0] == cond and k in rb)
                if len(keys) < 2:
                    continue
                ea = [ra[k]["S"] + ra[k]["D"] + ra[k]["I"] for k in keys]
                eb = [rb[k]["S"] + rb[k]["D"] + rb[k]["I"] for k in keys]
                den = [ra[k]["n_ref_words"] for k in keys]
                cl = ([ra[k]["fields"].get("speaker") or k[1] for k in keys]
                      if args.bootstrap_unit == "speaker" else None)
                lo, hi, p = paired_bootstrap(ea, eb, den, args.bootstrap, rng, cl)
                wa, wb = 100 * sum(ea) / sum(den), 100 * sum(eb) / sum(den)
                p_txt = f"< {1 / args.bootstrap:.4f}" if p < 1 / args.bootstrap else f"{p:.4f}"
                md.append(f"| {cond} | {a} | {b} | {len(keys)} | {wa:.2f} | {wb:.2f} | "
                          f"{wa - wb:+.2f} [{lo:+.2f}, {hi:+.2f}] | {p_txt} |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nWrote {out / 'summary.csv'} and {out / 'summary.md'}")


def cmd_estimate(args):
    """Project how long a run takes, from the speed measured on the utterances already done."""
    rows = read_manifest(args.manifest)
    manifest_conds = rows[0]["condition"] is not None
    if args.conditions and manifest_conds:
        rows = [r for r in rows if r["condition"] in set(args.conditions)]
    subset = select_subset(rows, args.limit, args.shuffle_seed)
    dur, bad = {}, 0   # (condition or None, utt_id) -> seconds
    for r in subset:
        try:
            dur[(r["condition"], r["utt_id"])] = audio_duration(r["audio_path"])
        except Exception:  # noqa: BLE001
            bad += 1
    total = sum(dur.values())
    n_sent = len({u for _, u in dur})
    scope = f"first {args.limit} sentences (seed {args.shuffle_seed})" if args.limit else "whole manifest"
    print(f"Target: {scope}: {n_sent} sentences, {len(dur)} rows, {total / 3600:.2f} h of audio"
          f" (mean {total / max(len(dur), 1):.1f} s/row)" + (f", {bad} unreadable" if bad else ""))
    root = Path(args.results)
    dirs = [d for d in sorted(root.iterdir()) if (d / "predictions.jsonl").exists()] if root.exists() else []
    if args.tags:
        dirs = [d for d in dirs if d.name in set(args.tags)]
    if not dirs:
        print(f"No results under {root} yet - run each system with a small --limit first.")
        return
    fmt = lambda s: f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min"  # noqa: E731
    print(f"\n{'system':<32}{'hardware':<30}{'RTF':>7}{'done':>7}{'per 100 rows':>14}{'target':>10}{'remaining':>11}")
    grand = 0.0
    for d in dirs:
        recs = [r for r in load_records(d / "predictions.jsonl").values()
                if not r.get("error") and r.get("latency_s") is not None]
        if not recs:
            continue
        rtf = sum(r["latency_s"] for r in recs) / max(sum(r["duration_s"] for r in recs), 1e-9)
        done = {(r["condition"], r["utt_id"]) for r in recs}
        if manifest_conds:
            items = list(dur)
        else:   # synthetic conditions: every condition already present x every sentence
            conds = sorted({r["condition"] for r in recs}, key=_cond_sort)
            items = [(c, u) for c in conds for (_, u) in dur]
        d_of = lambda c, u: dur.get((c, u), dur.get((None, u), 0.0))  # noqa: E731
        target = sum(d_of(c, u) for c, u in items) * rtf
        left = [(c, u) for c, u in items if (c, u) not in done]
        rem = sum(d_of(c, u) for c, u in left) * rtf
        per100 = 100 * total / max(len(dur), 1) * rtf
        rpm = None
        meta_path = d / "meta.json"
        if meta_path.exists():
            runs = json.loads(meta_path.read_text(encoding="utf-8")).get("runs", [])
            rpm = runs[-1].get("args", {}).get("rpm") if runs else None
        if rpm:  # the rate limit is a floor on wall time
            per100, target, rem = max(per100, 6000 / rpm), max(target, len(items) * 60 / rpm), \
                max(rem, len(left) * 60 / rpm)
        grand += rem
        hw = " + ".join(sorted({r.get("hw") or "?" for r in recs}))
        print(f"{d.name[:31]:<32}{hw[:29]:<30}{rtf:>7.3f}{len(recs):>7}{fmt(per100):>14}{fmt(target):>10}{fmt(rem):>11}")
    print(f"\nRemaining for all listed systems run one after another: {fmt(grand)} "
          f"(model loading and downloads not included).")


# =============================================================================
# CLI
# =============================================================================
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-vivos", help="manifest from a VIVOS-style folder")
    b.add_argument("--root", required=True, help="folder containing prompts.txt (searched recursively)")
    b.add_argument("--audio-root", help="where to look for audio files (default: --root)")
    b.add_argument("--out", required=True)
    b.add_argument("--limit", type=int, help="random subset size")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_build_vivos)

    r = sub.add_parser("run", help="transcribe a manifest with one backend")
    r.add_argument("--manifest", required=True)
    r.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    r.add_argument("--model", help="model id or local path")
    r.add_argument("--tag", help="name of this system in the results (default: derived)")
    r.add_argument("--out", default="results")
    r.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    r.add_argument("--dtype", choices=["bf16", "fp16", "fp32"],
                   help="default: bf16 on GPU, fp32 on CPU")
    r.add_argument("--language", default="vi", help="language code, or 'auto' for detection")
    g = r.add_argument_group("whisper")
    g.add_argument("--beam", type=int, default=5, help="beam size (1 = greedy)")
    g.add_argument("--vad", action="store_true", help="faster-whisper VAD filter")
    g.add_argument("--temperature", type=float, help="faster-whisper: fixed temperature (disables fallback)")
    g.add_argument("--compute-type", help="faster-whisper: float16 / int8_float16 / int8 ...")
    g.add_argument("--cpu-threads", type=int, help="CPU threads for torch / CTranslate2 (default: all cores)")
    g = r.add_argument_group("qwen3-asr")
    g.add_argument("--max-new-tokens", type=int, default=256)
    g.add_argument("--attn", help="e.g. flash_attention_2")
    g.add_argument("--qwen-vllm", action="store_true", help="use the vLLM backend of qwen-asr")
    g.add_argument("--gpu-mem-util", type=float, default=0.7)
    g = r.add_argument_group("gemini")
    g.add_argument("--rpm", type=float, help="max requests per minute (free-tier friendly)")
    g.add_argument("--max-retries", type=int, default=5)
    g.add_argument("--thinking-budget", type=int, help="e.g. 0 to disable thinking (2.5-style models)")
    g.add_argument("--thinking-level", help="e.g. low (3.x-style models)")
    g.add_argument("--prompt-file", help="custom transcription prompt (UTF-8 text)")
    g = r.add_argument_group("noise")
    g.add_argument("--snr", type=float, nargs="+", help="SNRs in dB, e.g. 20 10 5 0")
    g.add_argument("--noise-dir", help="folder of noise recordings (MUSAN/DEMAND/your museum)")
    g.add_argument("--noise-seed", type=int, default=0)
    g.add_argument("--no-clean", action="store_true", help="skip the clean condition")
    g = r.add_argument_group("run control")
    g.add_argument("--limit", type=int)
    g.add_argument("--shuffle-seed", type=int)
    g.add_argument("--warmup", type=int, default=3)
    g.add_argument("--retry-errors", action="store_true", help="re-run utterances that failed before")
    g.add_argument("--max-consecutive-errors", type=int, default=10,
                   help="stop the run after this many failures in a row (0 = never)")
    g.add_argument("--conditions", nargs="+",
                   help="only these values of the manifest's condition column (e.g. clean snr0)")
    g.add_argument("--log-every", type=int, default=25)
    g.add_argument("--no-number-norm", action="store_true")
    g.add_argument("--no-variants", action="store_true")
    g.add_argument("--dummy-error-rate", type=float, default=0.1)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("summarize", help="aggregate results/*/predictions.jsonl")
    s.add_argument("--results", default="results")
    s.add_argument("--tags", nargs="*", help="only these systems")
    s.add_argument("--group-by", nargs="*", help="manifest columns, e.g. region speaker gender")
    s.add_argument("--bootstrap", type=int, default=1000)
    s.add_argument("--bootstrap-unit", choices=["utt", "speaker"], default="utt",
                   help="'speaker' resamples whole speakers (needs a speaker column)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--compare", nargs=2, action="append", metavar=("A", "B"))
    s.add_argument("--baseline", help="compare every other system against this one")
    s.add_argument("--renormalize", action="store_true", help="re-score from raw text with current rules")
    s.add_argument("--no-number-norm", action="store_true")
    s.add_argument("--no-variants", action="store_true")
    s.add_argument("--out", help="output folder (default: <results>/summary)")
    s.set_defaults(func=cmd_summarize)

    e = sub.add_parser("estimate", help="project run time for the full manifest from a short trial run")
    e.add_argument("--manifest", required=True)
    e.add_argument("--results", default="results")
    e.add_argument("--tags", nargs="*")
    e.add_argument("--limit", type=int, help="estimate for the first N sentences (same selection as run)")
    e.add_argument("--shuffle-seed", type=int)
    e.add_argument("--conditions", nargs="+")
    e.set_defaults(func=cmd_estimate)
    return ap


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":  # required by vLLM's multiprocessing
    main()
