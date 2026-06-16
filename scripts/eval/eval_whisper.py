"""
ASR-Mobile Whisper Evaluation — Accuracy & Performance Benchmark
================================================================

Two-part evaluation for the fine-tuned Whisper model:

  Part 1 — Multilingual Accuracy
    * WER (Word Error Rate)  per language on google/fleurs test split
    * CER (Character Error Rate) per language
    * Compares fine-tuned model vs original base model
    * Bar-chart visualisations

  Part 2 — Performance Benchmark
    * Inference latency (P50 / P95 / mean, ms)
    * Throughput (samples/second)
    * Real-Time Factor (RTF)
    * Sweeps: CPU vs GPU, batch sizes, audio durations
    * Violin / box / bar visualisations

Usage:
    # Full evaluation (accuracy + performance)
    python scripts/eval/eval_whisper.py --model ./output/whisper-tiny-asr-mobile

    # Accuracy only
    python scripts/eval/eval_whisper.py --model ./output/whisper-tiny-asr-mobile --accuracy-only

    # Performance only
    python scripts/eval/eval_whisper.py --model ./output/whisper-tiny-asr-mobile --perf-only

    # Compare against base model
    python scripts/eval/eval_whisper.py --model ./output/whisper-tiny-asr-mobile --baseline openai/whisper-tiny

Output:
    ./eval_output/
    ├── accuracy_lang.png          # WER / CER per language (bar chart)
    ├── accuracy_comparison.png    # fine-tuned vs baseline comparison
    ├── performance_latency.png    # latency distribution
    ├── performance_rtf.png        # Real-Time Factor per duration bucket
    └── report.json                # full structured results
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jiwer
import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ── Optional plotting ───────────────────────────────────────────────────
try:
    import matplotlib

    matplotlib.use("Agg")  # headless-friendly
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # noqa

# ── HF mirror support (mainland China) ──────────────────────────────────
_HF_MIRRORS: Dict[str, str] = {
    "hf-mirror": "https://hf-mirror.com",
    "modelscope": "https://www.modelscope.cn",
}


def _setup_hf_mirror(mirror: Optional[str]) -> None:
    if mirror and mirror in _HF_MIRRORS:
        os.environ.setdefault("HF_ENDPOINT", _HF_MIRRORS[mirror])
    if "HF_ENDPOINT" not in os.environ:
        mirror_env = os.environ.get("HF_MIRROR")
        if mirror_env and mirror_env in _HF_MIRRORS:
            os.environ.setdefault("HF_ENDPOINT", _HF_MIRRORS[mirror_env])
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        logging.info("HF_ENDPOINT already set to %s (honouring existing env)", endpoint)


# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MEL_FRAMES = 3000  # Whisper expects 30 s of mel frames
FLEURS_LANG_MAP: Dict[str, str] = {
    "en": "en_us",
    "fr": "fr_fr",
    "zh-CN": "cmn_hans_cn",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EvalConfig:
    model_path: str
    baseline_model: Optional[str] = None
    output_dir: str = "./eval_output"
    languages: List[str] = field(default_factory=lambda: ["en", "fr", "zh-CN"])
    # Accuracy knobs
    max_eval_samples: int = 200  # per language
    streaming: bool = False
    # Performance knobs
    perf_batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    perf_audio_durations: List[float] = field(default_factory=lambda: [5.0, 15.0, 30.0])
    perf_warmup: int = 5
    perf_repeats: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers — data loading
# ═══════════════════════════════════════════════════════════════════════════


def load_eval_dataset(
    lang: str,
    split: str = "test",
    max_samples: int = 200,
    streaming: bool = False,
):
    fleurs_lang = FLEURS_LANG_MAP.get(lang, lang)
    logger.info("  Loading fleurs %s/%s …", fleurs_lang, split)
    ds = load_dataset(
        "google/fleurs",
        fleurs_lang,
        split=split,
        streaming=streaming,
        trust_remote_code=True if not streaming else False,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    # Normalise transcript column name
    if "transcription" in ds.features:
        ds = ds.rename_column("transcription", "transcript")
    elif "sentence" in ds.features:
        ds = ds.rename_column("sentence", "transcript")
    if max_samples and max_samples > 0:
        ds = (
            ds.select(range(min(max_samples, len(ds))))
            if not streaming
            else ds.take(max_samples)
        )
    return ds


# ═══════════════════════════════════════════════════════════════════════════
#  Part 1 — Multilingual Accuracy
# ═══════════════════════════════════════════════════════════════════════════


def _transcribe(
    model, processor, audio_array: np.ndarray, device: str, lang: str
) -> str:
    """Run Whisper inference — model generates language/task tokens
    freely from audio, matching the label structure used in training.
    Using forced_decoder_ids (via generate(language=...)) shifts the
    decoder context off-by-one vs training and degrades LoRA performance."""
    inputs = processor(audio_array, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    mel = inputs.input_features.to(device)
    original_len = mel.shape[-1]
    if original_len < N_MEL_FRAMES:
        pad = N_MEL_FRAMES - original_len
        mel = torch.nn.functional.pad(mel, (0, pad))
        attn_mask = (
            torch.cat(
                [
                    torch.ones(original_len, dtype=torch.long),
                    torch.zeros(pad, dtype=torch.long),
                ]
            )
            .unsqueeze(0)
            .to(device)
        )
    else:
        attn_mask = None

    with torch.no_grad():
        generated_ids = model.generate(
            mel,
            attention_mask=attn_mask,
            max_length=225,
        )
    return processor.decode(generated_ids[0], skip_special_tokens=True)


def _normalize_text(text: str, lang: str, tokenizer) -> str:
    """Apply Whisper's text normalizer for fair WER/CER comparison."""
    if not text:
        return ""
    if lang == "en":
        try:
            return tokenizer.normalize(text).strip()
        except (AttributeError, FileNotFoundError):
            pass
    return tokenizer.basic_normalize(text).strip()


def _wer_safe(refs: List[str], preds: List[str]) -> float:
    """WER that ignores empty references (jiwer raises otherwise)."""
    pairs = [(r, p) for r, p in zip(refs, preds) if r.strip()]
    if not pairs:
        return 0.0
    r, p = zip(*pairs)
    return jiwer.wer(list(r), list(p))


def _cer_safe(refs: List[str], preds: List[str]) -> float:
    pairs = [(r, p) for r, p in zip(refs, preds) if r.strip()]
    if not pairs:
        return 0.0
    r, p = zip(*pairs)
    return jiwer.cer(list(r), list(p))


def _compute_metrics(
    predictions: List[str], references: List[str], lang: str, tokenizer
) -> Dict[str, float]:
    """Compute raw + normalized WER/CER.

    For Chinese, WER is computed by inserting a space between every character
    (without this jiwer treats each sentence as one token and WER becomes
    a meaningless binary signal). CER is character-based by construction so
    it's already meaningful for Chinese.
    """
    raw_wer = _wer_safe(references, predictions)
    raw_cer = _cer_safe(references, predictions)

    norm_refs = [_normalize_text(r, lang, tokenizer) for r in references]
    norm_preds = [_normalize_text(p, lang, tokenizer) for p in predictions]

    if lang.startswith("zh"):
        # Treat each character as a token for WER on Chinese
        norm_refs_wer = [" ".join(list(r.replace(" ", ""))) for r in norm_refs]
        norm_preds_wer = [" ".join(list(p.replace(" ", ""))) for p in norm_preds]
    else:
        norm_refs_wer = norm_refs
        norm_preds_wer = norm_preds

    return {
        "wer": _wer_safe(norm_refs_wer, norm_preds_wer),
        "cer": _cer_safe(norm_refs, norm_preds),
        "wer_raw": raw_wer,
        "cer_raw": raw_cer,
    }


def _format_wer(val: float) -> str:
    return f"{val * 100:.2f}%"


def run_accuracy_eval(
    cfg: EvalConfig,
    model,
    processor,
    model_label: str,
) -> Dict[str, Any]:
    """Evaluate WER/CER per language.  Returns dict lang -> {wer, cer}."""
    logger.info("=" * 60)
    logger.info(" PART 1 — Multilingual Accuracy  [%s]", model_label)
    logger.info("=" * 60)

    results: Dict[str, Any] = {}
    for lang in cfg.languages:
        logger.info("--- Language: %s ---", lang)
        ds = load_eval_dataset(
            lang,
            split="test",
            max_samples=cfg.max_eval_samples,
            streaming=cfg.streaming,
        )
        preds, refs = [], []
        for example in tqdm(ds, desc=f"  eval {lang}"):
            audio_arr = example["audio"]["array"]
            ref = example["transcript"].strip() or " "
            hyp = _transcribe(model, processor, audio_arr, cfg.device, lang)
            preds.append(hyp)
            refs.append(ref)

        metrics = _compute_metrics(preds, refs, lang, processor.tokenizer)
        results[lang] = metrics
        logger.info(
            "  %-6s  WER %s (raw %s)  |  CER %s (raw %s)",
            lang,
            _format_wer(metrics["wer"]),
            _format_wer(metrics["wer_raw"]),
            _format_wer(metrics["cer"]),
            _format_wer(metrics["cer_raw"]),
        )
    return results


# ── Visualisation: per-language bar chart ────────────────────────────────


def _plot_accuracy_per_lang(
    results: Dict[str, Any],
    out_path: str,
    title_suffix: str = "",
):
    if not HAS_MPL:
        logger.warning("matplotlib not available — skipping accuracy plot")
        return

    langs = list(results.keys())
    wer_vals = [results[l]["wer"] * 100 for l in langs]
    cer_vals = [results[l]["cer"] * 100 for l in langs]

    x = np.arange(len(langs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, wer_vals, width, label="WER", color="#E74C3C")
    bars2 = ax.bar(x + width / 2, cer_vals, width, label="CER", color="#3498DB")

    for b in bars1:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.3,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for b in bars2:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.3,
            f"{b.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Error Rate (%)")
    ax.set_title(f"WER / CER per Language{title_suffix}")
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.legend()
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  Saved  %s", out_path)


def _plot_accuracy_comparison(
    ft_results: Dict[str, Any],
    base_results: Optional[Dict[str, Any]],
    out_path: str,
):
    """Two-panel chart: WER (left) and CER (right), baseline vs fine-tuned
    side-by-side with absolute deltas annotated."""
    if not HAS_MPL or base_results is None:
        return

    langs = list(ft_results.keys())
    metrics = [("wer", "WER"), ("cer", "CER")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(langs))
    width = 0.36

    for ax, (key, label) in zip(axes, metrics):
        ft_vals = [ft_results[l][key] * 100 for l in langs]
        base_vals = [base_results[l][key] * 100 for l in langs]

        b_bars = ax.bar(
            x - width / 2,
            base_vals,
            width,
            label="Baseline",
            color="#95A5A6",
            edgecolor="white",
        )
        f_bars = ax.bar(
            x + width / 2,
            ft_vals,
            width,
            label="Fine-tuned",
            color="#27AE60",
            edgecolor="white",
        )

        # Value labels on each bar
        for bar in list(b_bars) + list(f_bars):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(base_vals + ft_vals) * 0.01,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Delta annotations above each pair
        top = max(base_vals + ft_vals)
        for i, (b, f) in enumerate(zip(base_vals, ft_vals)):
            delta = b - f  # positive => improvement
            arrow = "↓" if delta > 0 else ("↑" if delta < 0 else "→")
            color = "#27AE60" if delta > 0 else ("#E74C3C" if delta < 0 else "#7F8C8D")
            ax.text(
                i,
                top * 1.10,
                f"{arrow}{abs(delta):.1f}",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

        ax.set_ylabel(f"{label} (%)")
        ax.set_title(f"{label} — Baseline vs Fine-tuned (normalized)")
        ax.set_xticks(x)
        ax.set_xticklabels(langs)
        ax.set_ylim(0, top * 1.25)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Accuracy Comparison (text-normalized; ↓ = improvement)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved  %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Part 2 — Performance Benchmark
# ═══════════════════════════════════════════════════════════════════════════


def _synthetic_audio(duration_sec: float) -> np.ndarray:
    """Generate a silent audio clip at `duration_sec` (16 kHz mono)."""
    n = int(SAMPLE_RATE * duration_sec)
    return np.zeros(n, dtype=np.float32)


@torch.no_grad()
def _measure_latency(
    model,
    processor,
    audio: np.ndarray,
    device: str,
    warmup: int = 3,
    repeats: int = 20,
) -> Dict[str, float]:
    """Measure encode + decode latency (ms) for a single sample."""
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    mel = inputs.input_features.to(device)
    if mel.shape[-1] < N_MEL_FRAMES:
        mel = torch.nn.functional.pad(mel, (0, N_MEL_FRAMES - mel.shape[-1]))

    # Warmup
    for _ in range(warmup):
        _ = model.generate(mel, max_length=225)

    timings: List[float] = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.generate(mel, max_length=225)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)

    arr = np.array(timings)
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(np.std(arr)),
    }


def run_performance_bench(cfg: EvalConfig, model, processor) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info(" PART 2 — Performance Benchmark  [device=%s]", cfg.device.upper())
    logger.info("=" * 60)

    model = model.to(cfg.device)
    model.eval()

    results: Dict[str, Any] = {
        "device": cfg.device,
        "latency_by_duration": {},
        "throughput_by_batch": {},
    }

    # ── 1. Latency vs audio duration (batch=1) ──────────────────────
    logger.info("--- Latency vs audio duration (batch=1) ---")
    for dur in cfg.perf_audio_durations:
        audio = _synthetic_audio(dur)
        stats = _measure_latency(
            model,
            processor,
            audio,
            cfg.device,
            warmup=cfg.perf_warmup,
            repeats=cfg.perf_repeats,
        )
        rtf = stats["mean_ms"] / (dur * 1000)  # RTF = proc_time / audio_len
        stats["rtf"] = float(rtf)
        results["latency_by_duration"][f"{dur:.0f}s"] = stats
        logger.info(
            "  %5.0f s  →  mean %6.1f ms  |  P95 %6.1f ms  |  RTF %.3f",
            dur,
            stats["mean_ms"],
            stats["p95_ms"],
            rtf,
        )

    # ── 2. Throughput vs batch size (30 s audio) ─────────────────────
    logger.info("--- Throughput vs batch size (30 s) ---")
    audio_30s = _synthetic_audio(30.0)
    for bs in cfg.perf_batch_sizes:
        # Build batch
        mels = []
        for _ in range(bs):
            inputs = processor(
                audio_30s, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            mel = inputs.input_features
            if mel.shape[-1] < N_MEL_FRAMES:
                mel = torch.nn.functional.pad(mel, (0, N_MEL_FRAMES - mel.shape[-1]))
            mels.append(mel)
        batch_mel = torch.cat(mels, dim=0).to(cfg.device)

        # Warmup
        for _ in range(min(cfg.perf_warmup, 3)):
            _ = model.generate(batch_mel, max_length=225)

        # Timed runs
        timings = []
        for _ in range(cfg.perf_repeats):
            if cfg.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.generate(batch_mel, max_length=225)
            if cfg.device == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)

        elapsed = np.mean(timings)
        throughput = bs / elapsed  # samples / sec
        latency_per_sample = (elapsed * 1000) / bs
        results["throughput_by_batch"][str(bs)] = {
            "throughput_sps": float(throughput),
            "latency_per_sample_ms": float(latency_per_sample),
        }
        logger.info(
            "  batch %2d  →  %6.1f samples/s  |  %6.1f ms/sample",
            bs,
            throughput,
            latency_per_sample,
        )

    return results


# ── Performance visualisations ───────────────────────────────────────────


def _plot_latency_by_duration(results: Dict, out_path: str):
    if not HAS_MPL:
        return

    data = results.get("latency_by_duration", {})
    if not data:
        return
    durations = list(data.keys())
    means = [data[d]["mean_ms"] for d in durations]
    p95s = [data[d]["p95_ms"] for d in durations]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: bar
    x = np.arange(len(durations))
    w = 0.35
    axes[0].bar(x - w / 2, means, w, label="Mean", color="#3498DB")
    axes[0].bar(x + w / 2, p95s, w, label="P95", color="#E67E22")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Inference Latency by Audio Duration")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(durations)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.35)

    # Right: RTF line
    rtfs = [data[d]["rtf"] for d in durations]
    axes[1].plot(durations, rtfs, "o-", color="#8E44AD", markersize=8, linewidth=2)
    axes[1].axhline(y=1.0, color="#E74C3C", linestyle="--", alpha=0.6, label="RTF=1.0")
    axes[1].set_ylabel("Real-Time Factor (RTF)")
    axes[1].set_title("RTF by Audio Duration")
    axes[1].legend()
    axes[1].grid(alpha=0.35)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  Saved  %s", out_path)


def _plot_throughput(results: Dict, out_path: str):
    if not HAS_MPL:
        return

    data = results.get("throughput_by_batch", {})
    if not data:
        return
    batches = [int(b) for b in data.keys()]
    sps = [data[str(b)]["throughput_sps"] for b in batches]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(b) for b in batches], sps, color="#1ABC9C", edgecolor="white")
    for i, v in enumerate(sps):
        ax.text(i, v + max(sps) * 0.01, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (samples / s)")
    ax.set_title("Throughput vs Batch Size (30 s audio)")
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  Saved  %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


def _load_model(model_path: str, device: str):
    """Load a Whisper model + processor, patching broken generation_config."""
    logger.info("Loading model: %s", model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)

    # Fix generation_config that may have been broken during LoRA training
    gc = model.generation_config
    gc.forced_decoder_ids = None
    gc.suppress_tokens = None
    gc.task = None  # let model detect task from audio (trained with <|transcribe|>)
    gc.max_length = 225

    # Suppress BPE tokenization warning
    processor.tokenizer.clean_up_tokenization_spaces = False

    model = model.to(device)
    model.eval()
    return model, processor


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Whisper: accuracy + performance"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path or HF id of the (merged) fine-tuned model",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional base model for comparison (e.g. openai/whisper-tiny)",
    )
    parser.add_argument(
        "--output", default="./eval_output", help="Output directory for plots + json"
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        default=["en", "fr", "zh-CN"],
        help="Languages to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max test samples per language",
    )
    parser.add_argument(
        "--mirror",
        default=None,
        choices=["hf-mirror", "modelscope"],
        help="HF mirror for mainland China",
    )
    parser.add_argument(
        "--accuracy-only",
        action="store_true",
        help="Run only the accuracy evaluation",
    )
    parser.add_argument(
        "--perf-only",
        action="store_true",
        help="Run only the performance benchmark",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream data instead of downloading",
    )
    args = parser.parse_args()

    # Mirror
    _setup_hf_mirror(args.mirror)
    os.makedirs(args.output, exist_ok=True)

    run_both = not args.accuracy_only and not args.perf_only

    cfg = EvalConfig(
        model_path=args.model,
        baseline_model=args.baseline,
        output_dir=args.output,
        languages=args.lang,
        max_eval_samples=args.max_samples,
        streaming=args.stream,
    )

    report: Dict[str, Any] = {
        "model": cfg.model_path,
        "baseline": cfg.baseline_model,
        "languages": cfg.languages,
        "device": cfg.device,
    }

    # ── Load fine-tuned model ──────────────────────────────────────
    ft_model, processor = _load_model(cfg.model_path, cfg.device)

    # ── Part 1: Accuracy ───────────────────────────────────────────
    if run_both or args.accuracy_only:
        raw_results_ft = run_accuracy_eval(cfg, ft_model, processor, "Fine-tuned")
        report["accuracy"] = {"fine_tuned": raw_results_ft}
        _plot_accuracy_per_lang(
            raw_results_ft,
            os.path.join(cfg.output_dir, "accuracy_lang.png"),
            title_suffix=" (Fine-tuned)",
        )

        # Optional baseline comparison
        if cfg.baseline_model:
            base_model, base_proc = _load_model(cfg.baseline_model, cfg.device)
            base_model = base_model.to(cfg.device)
            raw_results_base = run_accuracy_eval(cfg, base_model, base_proc, "Baseline")
            report["accuracy"]["baseline"] = raw_results_base
            _plot_accuracy_per_lang(
                raw_results_base,
                os.path.join(cfg.output_dir, "accuracy_baseline.png"),
                title_suffix=" (Baseline)",  # <-- fix: properly use title_suffix
            )
            _plot_accuracy_comparison(
                raw_results_ft,
                raw_results_base,
                os.path.join(cfg.output_dir, "accuracy_comparison.png"),
            )
            del base_model  # free memory

    # ── Part 2: Performance ────────────────────────────────────────
    if run_both or args.perf_only:
        perf_results = run_performance_bench(cfg, ft_model, processor)
        report["performance"] = perf_results
        _plot_latency_by_duration(
            perf_results,
            os.path.join(cfg.output_dir, "performance_latency.png"),
        )
        _plot_throughput(
            perf_results,
            os.path.join(cfg.output_dir, "performance_throughput.png"),
        )

    # ── Save report ───────────────────────────────────────────────
    report_path = os.path.join(cfg.output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", report_path)

    # ── Summary ───────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("  EVALUATION COMPLETE")
    logger.info("=" * 60)

    if "accuracy" in report:
        acc = report["accuracy"]
        has_base = "baseline" in acc

        if has_base:
            logger.info("  Metrics are text-normalized (Whisper normalizer).")
            logger.info(
                "  %-6s  %10s  %10s  %8s  |  %10s  %10s  %8s",
                "Lang",
                "Base WER",
                "FT WER",
                "ΔWER",
                "Base CER",
                "FT CER",
                "ΔCER",
            )
            logger.info("  " + "-" * 76)
            for lang in cfg.languages:
                base_w = acc["baseline"].get(lang, {}).get("wer", 0)
                ft_w = acc["fine_tuned"].get(lang, {}).get("wer", 0)
                base_c = acc["baseline"].get(lang, {}).get("cer", 0)
                ft_c = acc["fine_tuned"].get(lang, {}).get("cer", 0)
                dw = base_w - ft_w
                dc = base_c - ft_c
                logger.info(
                    "  %-6s  %10s  %10s  %s%6.2f%%  |  %10s  %10s  %s%6.2f%%",
                    lang,
                    _format_wer(base_w),
                    _format_wer(ft_w),
                    "↓" if dw > 0 else "↑",
                    abs(dw) * 100,
                    _format_wer(base_c),
                    _format_wer(ft_c),
                    "↓" if dc > 0 else "↑",
                    abs(dc) * 100,
                )
            logger.info("  (↓ = improvement)")
        else:
            for lang in cfg.languages:
                ft = acc["fine_tuned"].get(lang, {})
                logger.info(
                    "  %-6s  WER %s  |  CER %s",
                    lang,
                    _format_wer(ft.get("wer", 0)),
                    _format_wer(ft.get("cer", 0)),
                )

    if "performance" in report:
        perf = report["performance"]
        logger.info("")
        logger.info("  Device: %s", perf["device"])
        logger.info("  %5s  %8s  %8s  %6s", "Audio", "Mean(ms)", "P95(ms)", "RTF")
        logger.info("  " + "-" * 36)
        for d, s in perf["latency_by_duration"].items():
            logger.info(
                "  %5s  %8.1f  %8.1f  %6.3f",
                d,
                s["mean_ms"],
                s["p95_ms"],
                s["rtf"],
            )

    logger.info("")
    logger.info("  All plots → %s/", cfg.output_dir)


if __name__ == "__main__":
    main()
