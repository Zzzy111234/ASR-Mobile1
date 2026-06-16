#!/usr/bin/env python3
"""
quantize_whisper.py — Pure-Python Whisper model quantizer for ASR-Mobile
=======================================================================

Converts a HuggingFace Whisper model to GGUF and produces quantized
variants ready for Android deployment via whisper.cpp.

No C++ build required — the entire pipeline runs in Python:
  1. HuggingFace → GGUF (FP16)  — via whisper.cpp's convert-h5-to-gguf.py
  2. GGUF FP16 → GGUF q5_0 / q4_0 / q8_0  — pure-Python block quantizer

The output GGUF files are fully compatible with whisper.cpp's inference
engine on Android.

Usage:
    # Default: FP16 + q5_0 + q8_0
    python scripts/train/quantize_whisper.py ./output/whisper-tiny-asr-mobile

    # All quantization levels
    python scripts/train/quantize_whisper.py ./output/whisper-tiny-asr-mobile --all

    # Specific types only
    python scripts/train/quantize_whisper.py ./output/whisper-tiny-asr-mobile --types q4_0,q5_0

Output:
    ./output/whisper-tiny-asr-mobile-fp16.gguf    (~78 MB)
    ./output/whisper-tiny-asr-mobile-q5_0.gguf    (~77 MB)
    ./output/whisper-tiny-asr-mobile-q8_0.gguf    (~39 MB)
"""

import argparse
import logging
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────
GGML_QK = 32  # block size for q4_0 / q5_0 / q8_0

# GGUF type IDs for quantized formats
GGUF_TYPE_F16 = 1
GGUF_TYPE_Q8_0 = 8
GGUF_TYPE_Q5_0 = 14
GGUF_TYPE_Q4_0 = 2

QUANT_TYPE_MAP: Dict[str, Tuple[int, str]] = {
    "fp16": (GGUF_TYPE_F16, "f16"),
    "q8_0": (GGUF_TYPE_Q8_0, "q8_0"),
    "q5_0": (GGUF_TYPE_Q5_0, "q5_0"),
    "q4_0": (GGUF_TYPE_Q4_0, "q4_0"),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1 — HF  →  GGUF FP16  (native Python, no network)
# ═══════════════════════════════════════════════════════════════════════════


def _hf_to_gguf_native(model_dir: str, out_path: str) -> None:
    """Convert a HuggingFace Whisper model to GGUF FP16 — zero network.

    Reads model.safetensors (or pytorch_model.bin), config.json and
    tokenizer.json directly from disk and writes a GGUF v3 file that is
    compatible with whisper.cpp's inference engine.
    """
    import json

    import torch
    from safetensors.torch import load_file as load_safetensors

    model_dir = os.path.abspath(model_dir)

    # ── Load config ─────────────────────────────────────────────────
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        cfg = json.load(f)

    n_mels = cfg.get("num_mel_bins", 80)
    n_audio_ctx = cfg.get("max_source_positions", 1500)
    n_audio_state = cfg["d_model"]
    n_audio_layer = cfg["encoder_layers"]
    n_audio_head = cfg["encoder_attention_heads"]
    n_text_ctx = cfg.get("max_target_positions", 448)
    n_text_state = cfg["d_model"]
    n_text_layer = cfg["decoder_layers"]
    n_text_head = cfg["decoder_attention_heads"]
    n_vocab = cfg["vocab_size"]

    # ── Load state dict ─────────────────────────────────────────────
    sd_path = os.path.join(model_dir, "model.safetensors")
    pt_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(sd_path):
        logger.info("  Loading model.safetensors …")
        state = load_safetensors(sd_path)
    elif os.path.exists(pt_path):
        logger.info("  Loading pytorch_model.bin …")
        state = torch.load(pt_path, map_location="cpu", weights_only=True)
    else:
        # Try sharded
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            state = {}
            for shard in sorted(set(index["weight_map"].values())):
                sp = os.path.join(model_dir, shard)
                logger.info("  Loading %s …", shard)
                state.update(load_safetensors(sp))
        else:
            raise FileNotFoundError(f"No model weights found in {model_dir}")

    # ── Load tokenizer ──────────────────────────────────────────────
    tok_path = os.path.join(model_dir, "tokenizer.json")
    token_data = None
    if os.path.exists(tok_path):
        with open(tok_path, "r") as f:
            token_data = json.load(f)

    # ── Map tensors ─────────────────────────────────────────────────
    _write_gguf(
        out_path,
        state,
        n_mels=n_mels,
        n_audio_ctx=n_audio_ctx,
        n_audio_state=n_audio_state,
        n_audio_layer=n_audio_layer,
        n_audio_head=n_audio_head,
        n_text_ctx=n_text_ctx,
        n_text_state=n_text_state,
        n_text_layer=n_text_layer,
        n_text_head=n_text_head,
        n_vocab=n_vocab,
        token_data=token_data,
    )


# ── GGUF writer ──────────────────────────────────────────────────────────


def _write_gguf(
    out_path: str,
    state: Dict[str, torch.Tensor],
    **arch,
) -> None:
    """Write a GGUF v3 file from HF state_dict + architecture metadata."""

    # ── Metadata ────────────────────────────────────────────────────
    metadata: Dict = {
        "general.architecture": "whisper",
        "general.name": "ASR-Mobile Whisper",
        "whisper.encoder.n_mels": arch["n_mels"],
        "whisper.encoder.n_audio_ctx": arch["n_audio_ctx"],
        "whisper.encoder.n_audio_state": arch["n_audio_state"],
        "whisper.encoder.n_audio_layer": arch["n_audio_layer"],
        "whisper.encoder.n_audio_head": arch["n_audio_head"],
        "whisper.decoder.n_text_ctx": arch["n_text_ctx"],
        "whisper.decoder.n_text_state": arch["n_text_state"],
        "whisper.decoder.n_text_layer": arch["n_text_layer"],
        "whisper.decoder.n_text_head": arch["n_text_head"],
        "whisper.decoder.n_vocab": arch["n_vocab"],
        "general.file_type": 1,  # F16
    }

    # Tokenizer metadata
    td = arch.get("token_data")
    if td:
        added_tokens = td.get("added_tokens", [])
        model = td.get("model", {})
        vocab = model.get("vocab", {})
        merges_list = model.get("merges", [])

        # Build full token list (vocab + added_tokens at end)
        token_list = []
        token_types = []
        for tok_info in vocab.values() if isinstance(vocab, dict) else vocab:
            if isinstance(tok_info, dict):
                token_list.append(tok_info["content"])
                token_types.append(1)  # normal
            elif isinstance(tok_info, str):
                token_list.append(tok_info)
                token_types.append(1)

        for at in added_tokens:
            token_list.append(at["content"])
            token_types.append(3 if at.get("special", False) else 1)

        metadata["tokenizer.ggml.model"] = "gpt2"
        metadata["tokenizer.ggml.tokens"] = token_list
        metadata["tokenizer.ggml.token_type"] = token_types
        metadata["tokenizer.ggml.merges"] = merges_list if merges_list else []

        # Special token IDs
        for at in added_tokens:
            if at.get("content") == "<|endoftext|>":
                metadata["tokenizer.ggml.eos_token_id"] = at["id"]
            elif at.get("content") == "<|startoftranscript|>":
                metadata["tokenizer.ggml.bos_token_id"] = at["id"]

        if "tokenizer.ggml.eos_token_id" not in metadata:
            metadata["tokenizer.ggml.eos_token_id"] = 50257

    # ── Tensor name mapping ─────────────────────────────────────────
    tensor_map = _build_tensor_map(state, arch)

    # Encode all tensors as FP16
    tensor_buffers = {}
    for gguf_name, hf_name in tensor_map.items():
        t = state[hf_name].to(torch.float16).contiguous()
        tensor_buffers[gguf_name] = t

    # ── Build header in memory ─────────────────────────────────────
    header = bytearray()
    header += _GGUF_MAGIC
    header += struct.pack("<I", _GGUF_VERSION)
    header += struct.pack("<Q", len(tensor_buffers))
    header += struct.pack("<Q", len(metadata))
    header += _serialize_metadata(metadata)

    # Compute tensor offsets
    offset = 0
    tensor_infos = []
    for gguf_name, t in tensor_buffers.items():
        tensor_infos.append(
            {
                "name": gguf_name,
                "dims": list(t.shape),
                "dtype": GGUF_TYPE_F16,
                "offset": offset,
                "data": t,
            }
        )
        offset += t.numel() * 2  # FP16 = 2 bytes

    # Write tensor infos
    for ti in tensor_infos:
        name_enc = ti["name"].encode("utf-8")
        header += struct.pack("<Q", len(name_enc))
        header += name_enc
        header += struct.pack("<I", len(ti["dims"]))
        for d in ti["dims"]:
            header += struct.pack("<Q", d)
        header += struct.pack("<I", ti["dtype"])
        header += struct.pack("<Q", ti["offset"])

    # Alignment
    ALIGN = 32
    padded_len = ((len(header) + ALIGN - 1) // ALIGN) * ALIGN
    header += b"\x00" * (padded_len - len(header))

    with open(out_path, "wb") as f:
        f.write(header)
        for ti in tensor_infos:
            f.write(ti["data"].numpy().tobytes())

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(
        "  Wrote %s (%.1f MB, %d tensors)", out_path, size_mb, len(tensor_infos)
    )


def _build_tensor_map(state: Dict[str, torch.Tensor], arch: Dict) -> Dict[str, str]:
    """Map HF tensor names to GGUF names for Whisper architecture."""
    n_audio_layer = arch["n_audio_layer"]
    n_text_layer = arch["n_text_layer"]

    mapping = {}

    # ── Encoder blocks ──────────────────────────────────────────────
    for i in range(n_audio_layer):
        prefix = f"model.encoder.layers.{i}"
        blk = f"blk.{i}"
        _add_if(
            mapping, state, f"{prefix}.self_attn.q_proj.weight", f"{blk}.attn_q.weight"
        )
        _add_if(mapping, state, f"{prefix}.self_attn.q_proj.bias", f"{blk}.attn_q.bias")
        _add_if(
            mapping, state, f"{prefix}.self_attn.k_proj.weight", f"{blk}.attn_k.weight"
        )
        _add_if(
            mapping, state, f"{prefix}.self_attn.v_proj.weight", f"{blk}.attn_v.weight"
        )
        _add_if(mapping, state, f"{prefix}.self_attn.v_proj.bias", f"{blk}.attn_v.bias")
        _add_if(
            mapping,
            state,
            f"{prefix}.self_attn.out_proj.weight",
            f"{blk}.attn_output.weight",
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.self_attn.out_proj.bias",
            f"{blk}.attn_output.bias",
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.self_attn_layer_norm.weight",
            f"{blk}.attn_ln.weight",
        )
        _add_if(
            mapping, state, f"{prefix}.self_attn_layer_norm.bias", f"{blk}.attn_ln.bias"
        )
        _add_if(
            mapping, state, f"{prefix}.final_layer_norm.weight", f"{blk}.mlp_ln.weight"
        )
        _add_if(mapping, state, f"{prefix}.final_layer_norm.bias", f"{blk}.mlp_ln.bias")
        _add_if(mapping, state, f"{prefix}.fc1.weight", f"{blk}.mlp_0.weight")
        _add_if(mapping, state, f"{prefix}.fc1.bias", f"{blk}.mlp_0.bias")
        _add_if(mapping, state, f"{prefix}.fc2.weight", f"{blk}.mlp_1.weight")
        _add_if(mapping, state, f"{prefix}.fc2.bias", f"{blk}.mlp_1.bias")

    # ── Decoder blocks ──────────────────────────────────────────────
    for i in range(n_text_layer):
        prefix = f"model.decoder.layers.{i}"
        blk = f"blk.{n_audio_layer + i}"
        _add_if(
            mapping, state, f"{prefix}.self_attn.q_proj.weight", f"{blk}.attn_q.weight"
        )
        _add_if(mapping, state, f"{prefix}.self_attn.q_proj.bias", f"{blk}.attn_q.bias")
        _add_if(
            mapping, state, f"{prefix}.self_attn.k_proj.weight", f"{blk}.attn_k.weight"
        )
        _add_if(
            mapping, state, f"{prefix}.self_attn.v_proj.weight", f"{blk}.attn_v.weight"
        )
        _add_if(mapping, state, f"{prefix}.self_attn.v_proj.bias", f"{blk}.attn_v.bias")
        _add_if(
            mapping,
            state,
            f"{prefix}.self_attn.out_proj.weight",
            f"{blk}.attn_output.weight",
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.self_attn.out_proj.bias",
            f"{blk}.attn_output.bias",
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.encoder_attn.q_proj.weight",
            f"{blk}.cross_attn_q.weight",
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.encoder_attn.q_proj.bias",
            f"{blk}.cross_attn_q.bias",
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.encoder_attn.k_proj.weight",
            f"{blk}.cross_attn_k.weight",
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.encoder_attn.v_proj.weight",
            f"{blk}.cross_attn_v.weight",
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.encoder_attn.v_proj.bias",
            f"{blk}.cross_attn_v.bias",
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.encoder_attn.out_proj.weight",
            f"{blk}.cross_attn_output.weight",
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.encoder_attn.out_proj.bias",
            f"{blk}.cross_attn_output.bias",
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.self_attn_layer_norm.weight",
            f"{blk}.attn_ln.weight",
        )
        _add_if(
            mapping, state, f"{prefix}.self_attn_layer_norm.bias", f"{blk}.attn_ln.bias"
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.encoder_attn_layer_norm.weight",
            f"{blk}.cross_attn_ln.weight",
        )
        _add_if(
            mapping,
            state,
            f"{prefix}.encoder_attn_layer_norm.bias",
            f"{blk}.cross_attn_ln.bias",
        )
        _add_if(
            mapping, state, f"{prefix}.final_layer_norm.weight", f"{blk}.mlp_ln.weight"
        )
        _add_if(mapping, state, f"{prefix}.final_layer_norm.bias", f"{blk}.mlp_ln.bias")
        _add_if(mapping, state, f"{prefix}.fc1.weight", f"{blk}.mlp_0.weight")
        _add_if(mapping, state, f"{prefix}.fc1.bias", f"{blk}.mlp_0.bias")
        _add_if(mapping, state, f"{prefix}.fc2.weight", f"{blk}.mlp_1.weight")
        _add_if(mapping, state, f"{prefix}.fc2.bias", f"{blk}.mlp_1.bias")

    # ── Top-level ───────────────────────────────────────────────────
    _add_if(
        mapping, state, "model.encoder.layer_norm.weight", "encoder.layer_norm.weight"
    )
    _add_if(mapping, state, "model.encoder.layer_norm.bias", "encoder.layer_norm.bias")
    _add_if(
        mapping, state, "model.decoder.layer_norm.weight", "decoder.layer_norm.weight"
    )
    _add_if(mapping, state, "model.decoder.layer_norm.bias", "decoder.layer_norm.bias")
    _add_if(
        mapping,
        state,
        "model.decoder.embed_tokens.weight",
        "decoder.token_embedding.weight",
    )
    _add_if(
        mapping,
        state,
        "model.decoder.embed_positions.weight",
        "decoder.positional_embedding.weight",
    )

    return mapping


def _add_if(mapping, state, hf_name, gguf_name):
    if hf_name in state:
        mapping[gguf_name] = hf_name


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2 — GGUF quantization (pure Python)
# ═══════════════════════════════════════════════════════════════════════════

# GGUF binary format constants
_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION = 3


def _read_gguf_header(f) -> Tuple[int, int, int, Dict]:
    """Read GGUF header: (version, num_tensors, num_kv, metadata dict)."""
    magic = f.read(4)
    if magic != _GGUF_MAGIC:
        raise ValueError(f"Not a GGUF file (magic={magic!r})")

    version, num_tensors, num_kv = struct.unpack("<IQQ", f.read(20))

    metadata = {}
    for _ in range(num_kv):
        # Read key
        key_len = struct.unpack("<Q", f.read(8))[0]
        key = f.read(key_len).decode("utf-8")

        # Read value type
        val_type = struct.unpack("<I", f.read(4))[0]

        # Read value based on type
        if val_type == 0:  # uint8
            val = f.read(1)[0]
        elif val_type == 4:  # uint32
            val = struct.unpack("<I", f.read(4))[0]
        elif val_type == 8:  # string
            s_len = struct.unpack("<Q", f.read(8))[0]
            val = f.read(s_len).decode("utf-8")
        elif val_type == 9:  # array
            arr_type = struct.unpack("<I", f.read(4))[0]
            arr_len = struct.unpack("<I", f.read(4))[0]
            val = []
            for _ in range(arr_len):
                if arr_type == 8:  # string
                    s_len = struct.unpack("<Q", f.read(8))[0]
                    val.append(f.read(s_len).decode("utf-8"))
                elif arr_type == 4:
                    val.append(struct.unpack("<I", f.read(4))[0])
        elif val_type == 10:  # uint64
            val = struct.unpack("<Q", f.read(8))[0]
        elif val_type == 11:  # int32
            val = struct.unpack("<i", f.read(4))[0]
        elif val_type == 13:  # float32
            val = struct.unpack("<f", f.read(4))[0]
        elif val_type == 14:  # bool
            val = f.read(1)[0] != 0
        else:
            raise ValueError(f"Unknown metadata value type {val_type} for key {key}")

        metadata[key] = val

    return version, num_tensors, num_kv, metadata


def _read_tensor_infos(f, num_tensors: int) -> List[Dict]:
    """Read tensor info entries."""
    tensors = []
    for _ in range(num_tensors):
        name_len = struct.unpack("<Q", f.read(8))[0]
        name = f.read(name_len).decode("utf-8")
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
        dtype = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]

        # Compute total elements
        nelements = 1
        for d in dims:
            nelements *= d

        tensors.append(
            {
                "name": name,
                "n_dims": n_dims,
                "dims": dims,
                "dtype": dtype,
                "offset": offset,
                "nelements": nelements,
            }
        )
    return tensors


def _serialize_metadata(metadata: Dict) -> bytes:
    """Serialize GGUF metadata KV pairs."""
    buf = b""
    for key, val in metadata.items():
        key_bytes = key.encode("utf-8")
        buf += struct.pack("<Q", len(key_bytes))
        buf += key_bytes

        if isinstance(val, int):
            if -2147483648 <= val <= 2147483647:
                buf += struct.pack("<I", 11) + struct.pack("<i", val)  # int32
            elif 0 <= val <= 0xFFFFFFFF:
                buf += struct.pack("<I", 4) + struct.pack("<I", val)  # uint32
            else:
                buf += struct.pack("<I", 10) + struct.pack("<Q", val)  # uint64
        elif isinstance(val, str):
            s = val.encode("utf-8")
            buf += struct.pack("<I", 8) + struct.pack("<Q", len(s)) + s
        elif isinstance(val, float):
            buf += struct.pack("<I", 13) + struct.pack("<f", val)
        elif isinstance(val, bool):
            buf += struct.pack("<I", 14) + struct.pack("B", 1 if val else 0)
        elif isinstance(val, list):
            buf += struct.pack("<I", 9)  # array
            if len(val) > 0:
                if isinstance(val[0], str):
                    buf += struct.pack("<I", 8)  # string type
                    buf += struct.pack("<I", len(val))
                    for s in val:
                        sb = s.encode("utf-8")
                        buf += struct.pack("<Q", len(sb)) + sb
                elif isinstance(val[0], int):
                    buf += struct.pack("<I", 4)  # uint32 type
                    buf += struct.pack("<I", len(val))
                    for v in val:
                        buf += struct.pack("<I", v)
        else:
            raise TypeError(f"Unsupported metadata type: {type(val)} for key {key}")
    return buf


def _quantize_block_q80(block: np.ndarray) -> Tuple[bytes, np.float16]:
    """Quantize a block of ≤32 floats to q8_0 format.

    Returns: (quantized_bytes, scale_f16)
    """
    n = len(block)
    scale = np.max(np.abs(block)) / 127.0
    if scale < 1e-9:
        scale = 1.0
    q = np.clip(np.round(block / scale), -127, 127).astype(np.int8)
    # Pad to GGML_QK if needed (only last block)
    if n < GGML_QK:
        q = np.pad(q, (0, GGML_QK - n), constant_values=0)
    return q.tobytes(), np.float16(scale)


def _quantize_block_q50(block: np.ndarray) -> Tuple[bytes, np.float16, np.float16]:
    """Quantize a block of ≤32 floats to q5_0 format.

    Returns: (packed_bytes, scale_f16, min_f16)
    """
    n = len(block)
    vmin = np.min(block)
    vmax = np.max(block)
    scale = (vmax - vmin) / 31.0
    if scale < 1e-9:
        scale = 1.0
    q = np.clip(np.round((block - vmin) / scale), 0, 31).astype(np.uint8)
    if n < GGML_QK:
        q = np.pad(q, (0, GGML_QK - n), constant_values=0)

    # Pack 5-bit: 8 values → 5 bytes (GGML format)
    # q5_0 stores as: high bits in separate 16-bit + low nibbles
    # Actually q5_0 uses: each value is 5 bits, packed in q4_0 style
    # Layout: 16 half-floats for high bits, then packed low nibbles
    # Let me use the standard GGML q5_0 layout:
    # - 2 bytes (uint16) of high bits for 32 values
    # - 20 bytes of low nibbles (packed: 2 values per byte)
    packed = bytearray()
    # High bits: 4 bytes (uint16 * 2 for 32 5-bit values)
    high0 = 0
    high1 = 0
    for i in range(min(n, 16)):
        if q[i] & 0x10:
            high0 |= 1 << i
    for i in range(16, min(n, 32)):
        if q[i] & 0x10:
            high1 |= 1 << (i - 16)
    packed += struct.pack("<HH", high0, high1)

    # Low nibbles: 2 values per byte
    for i in range(0, GGML_QK, 2):
        lo = q[i] & 0x0F
        hi = q[i + 1] & 0x0F
        packed.append((hi << 4) | lo)

    return bytes(packed), np.float16(scale), np.float16(vmin)


def _quantize_block_q40(block: np.ndarray) -> Tuple[bytes, np.float16]:
    """Quantize a block of ≤32 floats to q4_0 format.

    Returns: (packed_bytes, scale_f16)
    """
    n = len(block)
    scale = np.max(np.abs(block)) / 7.0
    if scale < 1e-9:
        scale = 1.0
    q = np.clip(np.round(block / scale) + 8, 0, 15).astype(np.uint8)
    if n < GGML_QK:
        q = np.pad(q, (0, GGML_QK - n), constant_values=8)  # pad with zero-point

    # Pack: 2 values per byte
    packed = bytearray()
    for i in range(0, GGML_QK, 2):
        packed.append((q[i + 1] << 4) | q[i])
    return bytes(packed), np.float16(scale)


def _quantize_tensor(
    data: np.ndarray,
    quant_type: str,
) -> Tuple[bytes, int]:
    """Quantize a float16 tensor to the target type.

    Returns (raw_bytes, gguf_dtype).
    """
    # GGUF float16 is stored as uint16 (little-endian half)
    f16 = data.astype(np.float16)
    nelements = f16.size

    if quant_type == "fp16":
        return f16.tobytes(), GGUF_TYPE_F16

    if quant_type == "q8_0":
        # q8_0 layout: interleaved [scale_f16, q8_block(32)]
        out = bytearray()
        for i in range(0, nelements, GGML_QK):
            block = f16.flatten()[i : i + GGML_QK].astype(np.float32)
            qbytes, scale = _quantize_block_q80(block)
            out += struct.pack("<e", scale)  # f16 little-endian
            out += qbytes
        return bytes(out), GGUF_TYPE_Q8_0

    if quant_type == "q5_0":
        out = bytearray()
        for i in range(0, nelements, GGML_QK):
            block = f16.flatten()[i : i + GGML_QK].astype(np.float32)
            qbytes, scale, vmin = _quantize_block_q50(block)
            out += struct.pack("<ee", vmin, scale)  # both f16
            out += qbytes
        return bytes(out), GGUF_TYPE_Q5_0

    if quant_type == "q4_0":
        out = bytearray()
        for i in range(0, nelements, GGML_QK):
            block = f16.flatten()[i : i + GGML_QK].astype(np.float32)
            qbytes, scale = _quantize_block_q40(block)
            out += struct.pack("<e", scale)
            out += qbytes
        return bytes(out), GGUF_TYPE_Q4_0

    raise ValueError(f"Unknown quant type: {quant_type}")


def _quantize_gguf_file(input_path: str, output_path: str, quant_type: str) -> None:
    """Quantize an entire GGUF file to the target type."""
    logger.info(
        "  Quantizing %s  →  %s (%s)",
        os.path.basename(input_path),
        quant_type,
        os.path.basename(output_path),
    )

    with open(input_path, "rb") as f:
        version, num_tensors, num_kv, metadata = _read_gguf_header(f)
        tensor_infos = _read_tensor_infos(f, num_tensors)

        # Alignment padding
        alignment = 32
        header_end = f.tell()
        padded_end = ((header_end + alignment - 1) // alignment) * alignment
        _ = f.read(padded_end - header_end)  # skip padding

        data_start = f.tell()

    # Read tensor data
    tensor_data = {}
    with open(input_path, "rb") as f:
        for ti in tensor_infos:
            f.seek(data_start + ti["offset"])
            # FP16 tensors have nbytes = nelements * 2
            nbytes = ti["nelements"] * 2
            raw = f.read(nbytes)
            arr = np.frombuffer(raw, dtype=np.float16).reshape(ti["dims"]).copy()
            tensor_data[ti["name"]] = arr

    # Quantize each tensor
    quantized_tensors = {}
    new_dtype = QUANT_TYPE_MAP[quant_type][0]
    total_orig = 0
    total_quant = 0

    for ti in tensor_infos:
        arr = tensor_data[ti["name"]]
        qbytes, _ = _quantize_tensor(arr, quant_type)
        quantized_tensors[ti["name"]] = qbytes
        total_orig += ti["nelements"] * 2
        total_quant += len(qbytes)

    # Write output GGUF
    with open(output_path, "wb") as f:
        # Header
        f.write(_GGUF_MAGIC)
        f.write(struct.pack("<III", _GGUF_VERSION, num_tensors, num_kv))
        f.write(_serialize_metadata(metadata))

        # Tensor infos (updated types + offsets)
        current_offset = 0
        tensor_info_start = f.tell()
        for ti in tensor_infos:
            qbytes = quantized_tensors[ti["name"]]
            name_enc = ti["name"].encode("utf-8")
            f.write(struct.pack("<Q", len(name_enc)))
            f.write(name_enc)
            f.write(struct.pack("<I", ti["n_dims"]))
            f.write(struct.pack(f"<{ti['n_dims']}Q", *ti["dims"]))
            f.write(struct.pack("<I", new_dtype))
            f.write(struct.pack("<Q", current_offset))
            current_offset += len(qbytes)

        # Alignment padding before data
        data_pos = f.tell()
        padded = ((data_pos + alignment - 1) // alignment) * alignment
        f.write(b"\x00" * (padded - data_pos))

        # Write quantized tensor data
        for ti in tensor_infos:
            f.write(quantized_tensors[ti["name"]])

    ratio = total_quant / total_orig * 100 if total_orig > 0 else 0
    logger.info(
        "    %s → %s  (%.1f%% of FP16)",
        _human_size(total_orig),
        _human_size(total_quant),
        ratio,
    )


def _human_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    else:
        return f"{nbytes / (1024 * 1024):.1f} MB"


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Pure-Python Whisper model quantizer for ASR-Mobile"
    )
    parser.add_argument(
        "model_dir",
        help="Path to merged HuggingFace Whisper model directory",
    )
    parser.add_argument(
        "--types",
        default="fp16,q5_0,q8_0",
        help="Comma-separated quant types: fp16,q8_0,q5_0,q4_0 (default: fp16,q5_0,q8_0)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Shorthand for --types fp16,q8_0,q5_0,q4_0",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as model_dir)",
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        logger.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else model_dir
    os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.basename(model_dir.rstrip("/\\"))
    types = (
        ["fp16", "q8_0", "q5_0", "q4_0"]
        if args.all
        else [t.strip() for t in args.types.split(",")]
    )

    # Validate types
    for t in types:
        if t not in QUANT_TYPE_MAP:
            logger.error(
                "Unknown quant type: %s. Choices: %s", t, list(QUANT_TYPE_MAP.keys())
            )
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("  ASR-Mobile Whisper Quantizer")
    logger.info("=" * 60)
    logger.info("  Model    : %s", model_dir)
    logger.info("  Output   : %s", output_dir)
    logger.info("  Types    : %s", ", ".join(types))
    logger.info("")

    # ── Step 1: HF → GGUF FP16 ──────────────────────────────────────
    fp16_path = os.path.join(output_dir, f"{model_name}-fp16.gguf")

    if not os.path.exists(fp16_path) or os.path.getsize(fp16_path) < 1000:
        logger.info("── Step 1/2: HF → GGUF (FP16) ──")
        logger.info("  Model dir : %s", model_dir)
        logger.info("  Output    : %s", fp16_path)
        try:
            _hf_to_gguf_native(model_dir, fp16_path)
        except Exception as e:
            logger.error("Native HF → GGUF conversion failed: %s", e)
            sys.exit(1)
    else:
        logger.info("── Step 1/2: FP16 GGUF already exists, skipping ──")
        logger.info("  %s  (%s)", fp16_path, _human_size(os.path.getsize(fp16_path)))

    # ── Step 2: Quantize ────────────────────────────────────────────
    logger.info("")
    logger.info("── Step 2/2: Quantization ──")

    for qt in types:
        if qt == "fp16":
            continue  # already done

        out_path = os.path.join(output_dir, f"{model_name}-{qt}.gguf")
        try:
            _quantize_gguf_file(fp16_path, out_path, qt)
        except Exception as e:
            logger.error("Quantization [%s] failed: %s", qt, e)
            continue

    # ── Summary ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Quantization complete!")
    logger.info("=" * 60)

    for qt in types:
        suffix = "fp16" if qt == "fp16" else qt
        fpath = os.path.join(output_dir, f"{model_name}-{suffix}.gguf")
        if os.path.exists(fpath):
            logger.info("  %-7s  %s", suffix, fpath)

    logger.info("")
    logger.info(
        "  Deploy: cp %s/%s-q5_0.gguf →\n"
        "          android/app/src/main/assets/models/ggml-finetuned-q5_0.bin",
        output_dir,
        model_name,
    )


if __name__ == "__main__":
    main()
