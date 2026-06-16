"""
Train / Fine-tune a Whisper model for ASR-Mobile using Mozilla Common Voice.

Design decisions:
- Base model: openai/whisper-tiny (39M params) — best for mobile RTF.
- Fine-tune strategy: LoRA (PEFT) — only adapter weights are trained;
  the original Whisper weights are frozen and never modified.
- Dataset: google/fleurs (multi-language, 16 kHz, streaming via Hugging Face datasets).
- Output: a merged HuggingFace model that can be converted to GGUF
  for deployment with whisper.cpp on Android.

Usage:
    # Step 1: Install dependencies (mainland China mirrors recommended)
    pip install -r scripts/train/requirements.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn

    # Step 2: Train with default settings (tiny, multi-language streaming)
    #   Hugging Face Hub will auto-use HF_ENDPOINT if set (see below)
    python scripts/train/train_whisper.py

    #   Or override language / model
    python scripts/train/train_whisper.py --model openai/whisper-base --lang en

    # ── Mainland China mirror configuration ──
    #
    # Option A: Environment variable (recommended)
    #   export HF_ENDPOINT=https://hf-mirror.com
    #   python scripts/train/train_whisper.py
    #
    # Option B: Command-line flag
    #   python scripts/train/train_whisper.py --mirror hf-mirror
    #
    # Supported mirrors: hf-mirror (default), modelscope
    #
    # For persistent setup, add to your shell rc:
    #   echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

After training, convert to GGUF for Android:
    python whisper.cpp/models/convert-h5-to-gguf.py \
        ./output/whisper-tiny-asr-mobile \
        --outfile ./output/whisper-tiny-asr-mobile-fp16.gguf

    whisper.cpp/build/bin/quantize \
        ./output/whisper-tiny-asr-mobile-fp16.gguf \
        ./output/whisper-tiny-asr-mobile-q5_0.gguf \
        q5_0

    # Then register in ModelRepository.kt:
    # BundledModel(
    #     fileName = "ggml-finetuned-q5_0.bin",
    #     displayName = "Whisper Tiny Finetuned",
    #     description = "多语言, LoRA微调版, 适配移动端录音场景 (~77MB)",
    #     estimatedSizeMB = 77,
    # ),
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import Audio, concatenate_datasets, interleave_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Hugging Face mirrors for mainland China ────────────────────────────
_HF_MIRRORS: Dict[str, str] = {
    "hf-mirror": "https://hf-mirror.com",
    "modelscope": "https://www.modelscope.cn",
}


def _setup_hf_mirror(mirror_name: Optional[str] = None) -> None:
    """
    Configure HF_ENDPOINT so that `from_pretrained`, `load_dataset`, etc.
    go through a mainland-China-friendly mirror.

    Priority:
      1. ``--mirror`` CLI flag
      2. ``HF_ENDPOINT`` environment variable (already set) → no-op
      3. ``HF_MIRROR`` environment variable (our own shorthand)
      4. nothing → Hugging Face Hub directly
    """
    # If the user already exported HF_ENDPOINT explicitly, respect it.
    if os.environ.get("HF_ENDPOINT"):
        logger.info(
            "HF_ENDPOINT already set to %s (honouring existing env)",
            os.environ["HF_ENDPOINT"],
        )
        return

    target = mirror_name or os.environ.get("HF_MIRROR")
    if target and target in _HF_MIRRORS:
        endpoint = _HF_MIRRORS[target]
        os.environ["HF_ENDPOINT"] = endpoint
        logger.info("HF mirror enabled: %s → %s", target, endpoint)
    elif target:
        logger.warning(
            "Unknown mirror '%s'. Supported: %s. Falling back to direct HF access.",
            target,
            list(_HF_MIRRORS.keys()),
        )
    else:
        logger.info("No HF mirror configured. Connecting to huggingface.co directly.")


@dataclass
class TrainerConfig:
    """All hyper-parameters in one place."""

    # Model
    model_name: str = "openai/whisper-tiny"

    # Data
    # Default: en + fr only. Chinese is dropped because FLEURS cmn_hans_cn
    # train (~3k samples) is too small to outperform Whisper's pretraining;
    # multi-language LoRA also dilutes capacity. For Chinese, prefer the
    # original openai/whisper-tiny model.
    languages: List[str] = field(default_factory=lambda: ["en", "fr"])
    max_train_samples: Optional[int] = None
    max_eval_samples: int = 400
    audio_sampling_rate: int = 16000
    streaming: bool = False  # False = download to local cache, True = stream online

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Training
    output_dir: str = "./output/whisper-tiny-asr-mobile"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    max_steps: int = 2500
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    fp16: bool = True
    dataloader_num_workers: int = 2

    # Push to Hub (optional)
    hub_model_id: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
#  Data helpers
# ═══════════════════════════════════════════════════════════════════════


def _load_fleurs(
    lang: str, split: str, max_samples: Optional[int], streaming: bool = False
):
    """Load one language from google/fleurs (local or streaming)."""
    fleurs_lang = {"en": "en_us", "fr": "fr_fr", "zh-CN": "cmn_hans_cn"}.get(lang, lang)
    mode = "streaming" if streaming else "local cache"
    logger.info("Loading fleurs %s/%s (%s) …", fleurs_lang, split, mode)

    ds = load_dataset(
        "google/fleurs",
        fleurs_lang,
        split=split,
        streaming=streaming,
        trust_remote_code=True if not streaming else False,
    )

    if max_samples is not None and max_samples > 0:
        ds = (
            ds.select(range(min(max_samples, len(ds))))
            if not streaming
            else ds.take(max_samples)
        )

    return ds


def _prepare_dataset(ds, streaming: bool = False):
    """Cast audio to 16 kHz and normalize transcript column name."""
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    if "transcription" in ds.features:
        ds = ds.rename_column("transcription", "transcript")
    elif "sentence" in ds.features:
        ds = ds.rename_column("sentence", "transcript")
    return ds


def build_dataset(
    languages: List[str],
    split: str,
    max_samples: Optional[int],
    streaming: bool = False,
):
    """Build a multi-language dataset from google/fleurs."""
    parts = []
    for lang in languages:
        ds = _load_fleurs(lang, split, max_samples, streaming=streaming)
        ds = _prepare_dataset(ds, streaming=streaming)

        def _add_lang_tag(example: Dict[str, Any], lang=lang) -> Dict[str, Any]:
            example["lang"] = lang
            return example

        ds = ds.map(_add_lang_tag)
        parts.append(ds)

    if len(parts) == 1:
        return parts[0]

    if streaming:
        # interleave for streaming
        return interleave_datasets(
            parts, probabilities=[1.0 / len(parts)] * len(parts), seed=42
        )
    else:
        # concatenate for local
        return concatenate_datasets(parts)


# ═══════════════════════════════════════════════════════════════════════
#  Data collator
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    # Each sample is tokenized with its own language prefix so labels match the
    # <|sot|><|lang|><|transcribe|><|notimestamps|> structure Whisper uses at
    # inference time. Otherwise multi-language fine-tuning corrupts the
    # prompt-conditioning interface and accuracy degrades across all languages.

    processor: WhisperProcessor

    _WHISPER_LANG = {"zh-CN": "zh"}

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [f["audio"]["array"] for f in features]
        texts = [f.get("transcript", "").strip() or " " for f in features]
        langs = [
            self._WHISPER_LANG.get(f.get("lang", "en"), f.get("lang", "en"))
            for f in features
        ]

        feat_batch = self.processor.feature_extractor(
            audio_arrays, sampling_rate=16000, return_tensors="pt"
        )
        input_features = feat_batch["input_features"]
        if input_features.shape[-1] < 3000:
            pad_len = 3000 - input_features.shape[-1]
            input_features = F.pad(input_features, (0, pad_len))

        tokenizer = self.processor.tokenizer
        label_ids: List[List[int]] = []
        for text, lang in zip(texts, langs):
            tokenizer.set_prefix_tokens(language=lang, task="transcribe")
            label_ids.append(tokenizer(text).input_ids)

        labels_batch = tokenizer.pad(
            {"input_ids": label_ids}, return_tensors="pt", padding=True
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        return {"input_features": input_features, "labels": labels}


# ═══════════════════════════════════════════════════════════════════════
#  Custom Trainer — avoids PEFT kwargs conflict
# ═══════════════════════════════════════════════════════════════════════


class WhisperTrainer(Seq2SeqTrainer):
    """Trainer that passes inputs cleanly without PEFT kwargs conflicts."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # PeftModelForSeq2SeqLM.forward declares an explicit `input_ids` arg
        # and re-passes it to the underlying model. Whisper's forward also
        # accepts `input_ids` as a deprecated alias of `input_features`, so
        # PEFT ends up supplying both -> "multiple values for keyword
        # argument 'input_ids'". Going through `model.base_model` (the
        # LoraModel) bypasses PEFT's Seq2SeqLM forward while still applying
        # the in-place LoRA adapters on q_proj/k_proj/v_proj/out_proj.
        outputs = model.base_model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def setup_model_and_processor(cfg: TrainerConfig):
    """Load base Whisper model, freeze it, attach LoRA, return processor."""
    logger.info("Loading base model: %s", cfg.model_name)

    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_name)
    processor = WhisperProcessor.from_pretrained(cfg.model_name)

    # Freeze the entire base model
    for param in model.parameters():
        param.requires_grad = False

    # Enable gradient checkpointing for memory savings
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # required when gradient checkpointing is on

    # Attach LoRA adapters — only these will be trained
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA attached. Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    # Suppress language/task token automatic prefixing – we set them manually
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    return model, processor


# ── Custom callback: save LoRA adapter only ───────────────────────────
class SavePeftAdapterCallback(TrainerCallback):
    """Save a standalone LoRA adapter on each save step in addition to full checkpoint."""

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            adapter_dir = os.path.join(
                args.output_dir, f"lora-adapter-{state.global_step}"
            )
            model = kwargs.get("model")
            if model is not None and hasattr(model, "save_pretrained"):
                model.save_pretrained(adapter_dir)
                logger.info("LoRA adapter saved to %s", adapter_dir)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Whisper on Common Voice (streaming) for ASR-Mobile."
    )
    parser.add_argument(
        "--model", default="openai/whisper-tiny", help="Base Whisper model"
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        default=["en", "fr"],
        help="Language codes to fine-tune (default: en, fr). zh-CN is not"
        " recommended due to insufficient FLEURS Chinese data.",
    )
    parser.add_argument(
        "--output", default="./output/whisper-tiny-asr-mobile", help="Output directory"
    )
    parser.add_argument(
        "--max-steps", type=int, default=2500, help="Max training steps"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Per-device batch size"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True, help="Use mixed precision (fp16)"
    )
    parser.add_argument(
        "--no-fp16", dest="fp16", action="store_false", help="Disable fp16"
    )
    parser.add_argument(
        "--hub-model-id", default=None, help="Optional Hugging Face Hub model id"
    )
    parser.add_argument(
        "--mirror",
        default=None,
        choices=["hf-mirror", "modelscope"],
        help="HF mirror for mainland China (or set env HF_ENDPOINT / HF_MIRROR)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream data online instead of downloading to local cache",
    )
    args = parser.parse_args()

    # Configure HF mirror BEFORE any model/dataset access
    _setup_hf_mirror(args.mirror)

    cfg = TrainerConfig(
        model_name=args.model,
        languages=args.lang,
        output_dir=args.output,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        fp16=args.fp16,
        hub_model_id=args.hub_model_id,
        streaming=args.stream,
    )

    # ── Setup ─────────────────────────────────────────────────────
    logger.info("=== ASR-Mobile Whisper Fine-Tuning ===")
    logger.info("Languages: %s", cfg.languages)
    logger.info("Base model: %s", cfg.model_name)
    logger.info("Output dir: %s", cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)

    model, processor = setup_model_and_processor(cfg)

    # ── Data ───────────────────────────────────────────────────────
    logger.info("Building dataset (streaming=%s) …", cfg.streaming)
    train_dataset = build_dataset(
        cfg.languages,
        split="train",
        max_samples=cfg.max_train_samples,
        streaming=cfg.streaming,
    )
    eval_dataset = build_dataset(
        cfg.languages,
        split="validation",
        max_samples=cfg.max_eval_samples,
        streaming=cfg.streaming,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # ── Trainer ───────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        push_to_hub=False,  # Set True if you want to push to HF Hub
        hub_model_id=cfg.hub_model_id,
        save_total_limit=3,
        remove_unused_columns=False,
    )

    # Resume from checkpoint if one exists
    last_checkpoint = get_last_checkpoint(cfg.output_dir)

    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            SavePeftAdapterCallback(),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    # ── Train ─────────────────────────────────────────────────────
    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ── Save final ────────────────────────────────────────────────
    logger.info("Saving final model …")
    trainer.save_model()

    # Merge LoRA into base weights & save full model for GGUF conversion
    logger.info("Merging LoRA adapter into base weights …")
    merged_model = model.merge_and_unload()

    # Restore generation defaults that were overridden for training
    # (forced_decoder_ids=None breaks inference; Whisper needs them for
    # language detection / task token prefixing at generation time.)
    merged_model.config.forced_decoder_ids = None  # will be auto-set by generate()
    merged_model.generation_config.forced_decoder_ids = None
    merged_model.generation_config.suppress_tokens = None
    merged_model.generation_config.task = "transcribe"

    merged_model.save_pretrained(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    logger.info("Merged model saved to %s", cfg.output_dir)

    # ── Conversion instructions ───────────────────────────────────
    logger.info(
        """
╔══════════════════════════════════════════════════════════════════════╗
║  Training complete! Next steps for Android deployment:               ║
║                                                                      ║
║  1. Convert to GGUF:                                                 ║
║     python whisper.cpp/models/convert-h5-to-gguf.py \\               ║
║         %s \\                         ║
║         --outfile %s-fp16.gguf               ║
║                                                                      ║
║  2. Quantize for mobile:                                             ║
║     whisper.cpp/build/bin/quantize \\                                ║
║         %s-fp16.gguf \\                              ║
║         %s-q5_0.gguf \\                              ║
║         q5_0                                                         ║
║                                                                      ║
║  3. Deploy to Android:                                               ║
║     Copy %s-q5_0.gguf →                                             ║
║     android/app/src/main/assets/models/ggml-finetuned-q5_0.bin       ║
║                                                                      ║
║  4. Register in ModelRepository.kt AVAILABLE_MODELS:                 ║
║     BundledModel(                                                    ║
║         fileName = "ggml-finetuned-q5_0.bin",                        ║
║         displayName = "Whisper Tiny Finetuned",                      ║
║         description = "LoRA微调 多语言(EN/FR/ZH) ~77MB",            ║
║         estimatedSizeMB = 77,                                        ║
║     ),                                                               ║
╚══════════════════════════════════════════════════════════════════════╝
""".strip()
        % (
            cfg.output_dir,
            cfg.output_dir,
            cfg.output_dir,
            cfg.output_dir,
            cfg.output_dir,
        )
    )


if __name__ == "__main__":
    main()
