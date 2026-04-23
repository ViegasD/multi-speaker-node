"""
MultiSpeakerLipSync — single ComfyUI node that:
  1. Calls ElevenLabs TTS for each speaker (defined as JSON list).
  2. Detects N faces in frame 0, sorts them left-to-right.
  3. For each speaker i: crops the time slice where speaker i is talking to
     speaker i's face bbox, runs LatentSync on the crop, pastes back.
  4. Concatenates per-speaker audio (with a silence gap) into a single track.

The whole multi-speaker pipeline collapses into a single node so the user can
declare an arbitrary number of speakers via one JSON widget — something that is
not expressible with vanilla ComfyUI graphs (no runtime iteration).
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import traceback
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import requests


# ---------------------------------------------------------------------------
# Lazy imports — face detection (facexlib comes with LatentSyncWrapper) and
# the LatentSyncNode itself are only required at execution time so that the
# node can still be loaded even if those packs are not installed yet.
# ---------------------------------------------------------------------------

def _load_latentsync_node():
    """Locate the ComfyUI-LatentSyncWrapper package and return its node class."""
    try:
        # Most reliable: import via ComfyUI's loaded NODE_CLASS_MAPPINGS.
        import nodes  # ComfyUI's module

        cls = nodes.NODE_CLASS_MAPPINGS.get("LatentSyncNode")
        if cls is not None:
            return cls
    except Exception:
        pass

    # Fallback: search custom_nodes for the wrapper directory.
    here = os.path.dirname(os.path.abspath(__file__))
    custom_nodes_dir = os.path.dirname(here)
    for name in os.listdir(custom_nodes_dir):
        cand = os.path.join(custom_nodes_dir, name)
        nodes_py = os.path.join(cand, "nodes.py")
        if os.path.isdir(cand) and "latentsync" in name.lower() and os.path.exists(nodes_py):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            import importlib.util

            spec = importlib.util.spec_from_file_location("_latentsync_wrapper_nodes", nodes_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, "LatentSyncNode")
    raise ImportError(
        "ComfyUI-LatentSyncWrapper not found. Install it under custom_nodes/ first."
    )


def _load_face_detector(device):
    """Use facexlib (LatentSync's own dependency) for robust face detection."""
    from facexlib.detection import init_detection_model

    return init_detection_model("retinaface_resnet50", half=False, device=device)


# ---------------------------------------------------------------------------
# ElevenLabs REST call — kept dependency-free (no SDK).
# ---------------------------------------------------------------------------

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


def _tts_elevenlabs(api_key: str, voice_id: str, text: str, model_id: str,
                    stability: float, similarity_boost: float,
                    style: float, use_speaker_boost: bool) -> bytes:
    if not api_key:
        raise ValueError("ElevenLabs api_key is empty.")
    if not voice_id:
        raise ValueError("ElevenLabs voice_id is empty.")
    if not text or not text.strip():
        raise ValueError("Speaker text is empty.")

    resp = requests.post(
        ELEVENLABS_TTS_URL.format(voice_id=voice_id),
        headers={
            "xi-api-key": api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        },
        json={
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": use_speaker_boost,
            },
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs TTS failed ({resp.status_code}): {resp.text[:500]}"
        )
    return resp.content


def _decode_mp3_to_waveform(mp3_bytes: bytes, target_sr: int = 24000
                            ) -> Tuple[torch.Tensor, int]:
    """Decode mp3 bytes to mono float32 waveform [1, T] at target_sr."""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes)
        path = f.name
    try:
        wav, sr = torchaudio.load(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    if wav.shape[0] > 1:  # stereo -> mono
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


# ---------------------------------------------------------------------------
# Helpers — face detection, video looping, paste-back compositing.
# ---------------------------------------------------------------------------

def _detect_faces_left_to_right(detector, frame_uint8: np.ndarray, k: int
                                ) -> List[Tuple[int, int, int, int]]:
    """Return up to k bboxes (x1,y1,x2,y2) sorted left-to-right by center x."""
    # facexlib expects BGR. Our frames are RGB.
    bgr = frame_uint8[:, :, ::-1]
    raw = detector.detect_faces(bgr, 0.85)
    if raw is None or len(raw) == 0:
        raise RuntimeError("No faces detected in the first frame.")

    faces = []
    for row in raw:
        x1, y1, x2, y2 = [int(v) for v in row[:4]]
        faces.append((x1, y1, x2, y2))
    faces.sort(key=lambda b: (b[0] + b[2]) / 2.0)
    if len(faces) < k:
        raise RuntimeError(
            f"Detected only {len(faces)} face(s); workflow needs {k}."
        )
    return faces[:k]


def _pad_bbox(bbox, pad_pct: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * pad_pct)
    py = int(bh * pad_pct)
    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(w, x2 + px)
    ny2 = min(h, y2 + py)
    return nx1, ny1, nx2, ny2


def _ensure_video_length(frames: torch.Tensor, target_n: int) -> torch.Tensor:
    """Loop or truncate frames to exactly target_n frames (dim 0)."""
    n = frames.shape[0]
    if n == target_n:
        return frames
    if n > target_n:
        return frames[:target_n]
    # loop forward+backward (ping-pong) so the joint is smooth.
    pong = torch.cat([frames, frames.flip(0)], dim=0)
    out = pong
    while out.shape[0] < target_n:
        out = torch.cat([out, pong], dim=0)
    return out[:target_n]


def _silence(seconds: float, sr: int) -> torch.Tensor:
    n = int(round(seconds * sr))
    return torch.zeros(1, max(n, 0), dtype=torch.float32)


# ---------------------------------------------------------------------------
# The node.
# ---------------------------------------------------------------------------

class MultiSpeakerLipSync:
    """Single ComfyUI node implementing the full multi-speaker pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "elevenlabs_api_key": ("STRING", {"default": "", "multiline": False}),
                "elevenlabs_model": ("STRING", {"default": "eleven_multilingual_v2"}),
                "speakers_json": ("STRING", {
                    "default": json.dumps([
                        {"text": "Hello, my name is Alice.",
                         "voice_id": "EXAVITQu4vr4xnSDxMaL"},
                        {"text": "Nice to meet you, Alice.",
                         "voice_id": "21m00Tcm4TlvDq8ikWAM"},
                    ], indent=2),
                    "multiline": True,
                }),
                "gap_seconds": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.05}),
                "face_padding": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "stability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "similarity_boost": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "style": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "use_speaker_boost": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1247, "min": 0, "max": 2**31 - 1}),
                "lips_expression": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                "inference_steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
            },
        }

    CATEGORY = "audio/lipsync"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "run"

    # ------------------------------------------------------------------
    def run(self, images, fps, elevenlabs_api_key, elevenlabs_model,
            speakers_json, gap_seconds, face_padding,
            stability, similarity_boost, style, use_speaker_boost,
            seed, lips_expression, inference_steps):
        try:
            return self._run_impl(
                images, fps, elevenlabs_api_key, elevenlabs_model,
                speakers_json, gap_seconds, face_padding,
                stability, similarity_boost, style, use_speaker_boost,
                seed, lips_expression, inference_steps,
            )
        except Exception as e:
            print(f"[MultiSpeakerLipSync] ERROR: {e}")
            traceback.print_exc()
            raise

    def _run_impl(self, images, fps, api_key, model_id, speakers_json,
                  gap_seconds, face_padding, stability, similarity_boost,
                  style, use_speaker_boost, seed, lips_expression,
                  inference_steps):

        # ---------- 1. Parse speaker definitions ----------
        try:
            speakers = json.loads(speakers_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"speakers_json is not valid JSON: {e}")
        if not isinstance(speakers, list) or not speakers:
            raise ValueError("speakers_json must be a non-empty JSON array.")
        speakers = [s for s in speakers if (s.get("text") or "").strip()]
        if not speakers:
            raise ValueError("All speaker entries are empty.")
        n_speakers = len(speakers)
        print(f"[MultiSpeakerLipSync] {n_speakers} speaker(s) defined.")

        # ---------- 2. Synthesize each speaker's audio ----------
        AUDIO_SR = 24000  # LatentSync internally resamples to 16k anyway.
        per_speaker_wavs: List[torch.Tensor] = []
        for i, spk in enumerate(speakers):
            print(f"[MultiSpeakerLipSync] TTS speaker {i+1}/{n_speakers}...")
            mp3 = _tts_elevenlabs(
                api_key=api_key,
                voice_id=spk["voice_id"],
                text=spk["text"],
                model_id=model_id,
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                use_speaker_boost=use_speaker_boost,
            )
            wav, _ = _decode_mp3_to_waveform(mp3, target_sr=AUDIO_SR)
            per_speaker_wavs.append(wav)

        # ---------- 3. Build full timeline ----------
        gap_wav = _silence(gap_seconds, AUDIO_SR)
        timeline_parts = []
        speaker_ranges_sec: List[Tuple[float, float]] = []
        cursor_s = 0.0
        for i, wav in enumerate(per_speaker_wavs):
            if i > 0:
                timeline_parts.append(gap_wav)
                cursor_s += gap_seconds
            start_s = cursor_s
            timeline_parts.append(wav)
            dur_s = wav.shape[1] / AUDIO_SR
            cursor_s += dur_s
            speaker_ranges_sec.append((start_s, cursor_s))
        full_wav = torch.cat(timeline_parts, dim=1)
        total_seconds = full_wav.shape[1] / AUDIO_SR
        total_frames = int(round(total_seconds * fps))
        print(f"[MultiSpeakerLipSync] total audio={total_seconds:.2f}s, "
              f"target frames={total_frames} @ {fps} fps")

        # ---------- 4. Adjust video length ----------
        # `images` from ComfyUI is float32 [N,H,W,3] in [0,1].
        if images.dim() != 4 or images.shape[-1] != 3:
            raise ValueError(f"Unexpected images shape: {tuple(images.shape)}")
        frames = _ensure_video_length(images.detach().cpu(), total_frames)
        H, W = frames.shape[1], frames.shape[2]

        # ---------- 5. Detect faces in frame 0 ----------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detector = _load_face_detector(device)
        first_frame = (frames[0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
        bboxes = _detect_faces_left_to_right(detector, first_frame, n_speakers)
        bboxes = [_pad_bbox(b, face_padding, W, H) for b in bboxes]
        print(f"[MultiSpeakerLipSync] face bboxes (L->R): {bboxes}")

        # ---------- 6. Per-speaker LatentSync ----------
        LatentSyncNode = _load_latentsync_node()
        ls_node = LatentSyncNode()

        for i, ((s_start, s_end), bbox) in enumerate(
                zip(speaker_ranges_sec, bboxes)):
            f0 = max(0, int(round(s_start * fps)))
            f1 = min(total_frames, int(round(s_end * fps)))
            if f1 <= f0:
                continue

            x1, y1, x2, y2 = bbox
            crop = frames[f0:f1, y1:y2, x1:x2, :].clone()
            print(f"[MultiSpeakerLipSync] speaker {i+1}: frames [{f0},{f1}) "
                  f"crop {crop.shape[2]}x{crop.shape[1]}")

            # Slice the speaker's own audio out of the timeline to feed LatentSync.
            a0 = int(round(s_start * AUDIO_SR))
            a1 = int(round(s_end * AUDIO_SR))
            spk_wav = full_wav[:, a0:a1].clone()
            audio_dict = {
                "waveform": spk_wav.unsqueeze(0),  # [B,C,T] expected
                "sample_rate": AUDIO_SR,
            }

            synced_imgs, _ = ls_node.inference(
                images=crop,
                audio=audio_dict,
                seed=seed + i,
                lips_expression=lips_expression,
                inference_steps=inference_steps,
            )
            synced = synced_imgs.detach().cpu()
            # LatentSync may return slightly different frame count or size —
            # resize/clip to original crop dims.
            if synced.shape[0] != (f1 - f0):
                synced = _ensure_video_length(synced, f1 - f0)
            if synced.shape[1] != (y2 - y1) or synced.shape[2] != (x2 - x1):
                # NHWC -> NCHW for interpolation, then back.
                t = synced.permute(0, 3, 1, 2)
                t = torch.nn.functional.interpolate(
                    t, size=(y2 - y1, x2 - x1),
                    mode="bilinear", align_corners=False,
                )
                synced = t.permute(0, 2, 3, 1).contiguous()
            frames[f0:f1, y1:y2, x1:x2, :] = synced.to(frames.dtype)

        # ---------- 7. Pack audio output ----------
        audio_out = {
            "waveform": full_wav.unsqueeze(0),  # [B,C,T]
            "sample_rate": AUDIO_SR,
        }
        return (frames, audio_out)
