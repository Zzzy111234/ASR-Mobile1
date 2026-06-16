# Whisper Evaluation — Field Reference

This document explains every field produced by `eval_whisper.py` and how to
interpret the output plots and `report.json`.

## Running

```bash
# Full eval (accuracy + performance), compare fine-tuned vs baseline
python scripts/eval/eval_whisper.py \
    --model ./output/whisper-tiny-asr-mobile \
    --baseline openai/whisper-tiny

# Accuracy only
python scripts/eval/eval_whisper.py --model <path> --accuracy-only

# Performance only
python scripts/eval/eval_whisper.py --model <path> --perf-only

# Use a mainland-China mirror
python scripts/eval/eval_whisper.py --model <path> --mirror hf-mirror
```

Outputs land in `./eval_output/` (configurable via `--output`).

---

## Output files

| File | Contents |
|---|---|
| `report.json`                | All numbers in structured form. |
| `accuracy_lang.png`          | Per-language WER & CER for the fine-tuned model. |
| `accuracy_baseline.png`      | Same chart for the baseline (only if `--baseline` was passed). |
| `accuracy_comparison.png`    | Side-by-side WER + CER bars, fine-tuned vs baseline, with deltas. |
| `performance_latency.png`    | Mean / P95 latency per audio duration + RTF curve. |
| `performance_throughput.png` | Throughput (samples/s) at batch sizes 1 / 2 / 4 / 8. |

---

## `report.json` structure

```jsonc
{
  "model": "./output/whisper-tiny-asr-mobile",   // path/id evaluated
  "baseline": "openai/whisper-tiny",             // null if no --baseline
  "languages": ["en", "fr", "zh-CN"],            // langs evaluated
  "device": "cuda",                              // cuda | cpu
  "accuracy": { ... },                           // see below
  "performance": { ... }                         // see below
}
```

### `accuracy.<model>.<lang>`

`<model>` is `fine_tuned` and (if `--baseline` was passed) `baseline`.

| Field | Meaning |
|---|---|
| `wer`     | **Primary WER**, computed on **normalized** text. Use this for cross-model comparison. Lower is better. |
| `cer`     | **Primary CER**, computed on normalized text. Lower is better. |
| `wer_raw` | WER on **raw** strings (no normalization). Always higher; sensitive to punctuation/casing. |
| `cer_raw` | CER on raw strings. |

All values are **fractions in `[0, 1+]`** (multiply by 100 for percent).
WER can exceed 1.0 when the prediction has many more tokens than the reference.

#### Text normalization

Without normalization, scores are inflated by trivial differences like
`Hello, world!` vs `hello world`. We apply Whisper's official normalizers:

- **English (`en`)** — `WhisperTokenizer.normalize()` (the same
  `EnglishTextNormalizer` OpenAI uses in their paper):
  lowercase, expand contractions (`don't` → `do not`), spelled-out numbers
  → digits, strip punctuation, collapse whitespace.
- **Other languages** — `WhisperTokenizer.basic_normalize()`:
  Unicode NFKC, lowercase, strip punctuation, collapse whitespace.

#### Chinese-specific WER handling

`jiwer.wer` splits on whitespace. Chinese has no word boundaries, so a whole
sentence becomes a single "word" and WER becomes nearly meaningless
(reference and prediction are either fully equal or fully different).

For Chinese (`lang` starts with `zh`), we insert a space between every
character before computing WER so each character acts as a token. CER is
character-based by construction and doesn't need this fix.

**Recommendation:** for Chinese, trust **CER** as the primary signal. WER on
Chinese is approximately CER with extra noise from segmentation.

### `performance.latency_by_duration.<duration>`

Single-sample (batch=1) `generate()` latency for synthetic audio of the
given length. Measured on `cfg.device`.

| Field | Meaning |
|---|---|
| `mean_ms` | Mean wall-clock time per call, in milliseconds. |
| `p50_ms`  | Median latency. |
| `p95_ms`  | 95th-percentile latency — use this for SLA / UX budgeting. |
| `std_ms`  | Standard deviation; high values mean unstable inference. |
| `rtf`     | **Real-Time Factor** = `mean_ms / (duration_sec * 1000)`. RTF < 1.0 means transcription is faster than the audio plays back; RTF < 0.1 is comfortable for on-device streaming. |

Caveat: synthetic audio is all zeros, so decoder iterations may be shorter
than for real speech (early `<eot>`). Treat absolute numbers as an
**optimistic lower bound**; relative comparisons across devices/batch
sizes are still meaningful.

### `performance.throughput_by_batch.<batch_size>`

30-second synthetic clips run in batches of size 1, 2, 4, 8.

| Field | Meaning |
|---|---|
| `throughput_sps`         | Samples processed per second across the whole batch. |
| `latency_per_sample_ms`  | `(batch_elapsed_ms / batch_size)` — amortized cost per sample. |

Throughput typically scales sub-linearly with batch size (GPU saturates).

---

## How to read the plots

### `accuracy_comparison.png` (most important)

- Two panels: **WER** (left), **CER** (right).
- Each panel: grey bar = baseline, green bar = fine-tuned, per language.
- Number on top of each bar = absolute error rate (%).
- Arrow above each pair: `↓` (green) = fine-tuned improved over baseline,
  `↑` (red) = regression. The number next to the arrow is the absolute
  difference in percentage points.

### `accuracy_lang.png`

- Red bars = WER, blue bars = CER, one pair per language.
- Shows the absolute error level for a single model (no comparison).

### `performance_latency.png`

- **Left panel:** Mean (blue) and P95 (orange) latency per audio duration.
  Should be roughly flat — encoder cost dominates and Whisper always pads
  to 30 s of mel frames.
- **Right panel:** RTF curve. The red dashed line at `RTF=1.0` is the
  real-time threshold. Lower is better.

### `performance_throughput.png`

- Single bar chart of samples/sec per batch size. Look for the point of
  diminishing returns — the optimal batch size for your hardware.

---

## Picking the best model

1. Open `accuracy_comparison.png`.
2. Look at **CER** (right panel) — it's the most robust metric and works
   for both Latin and CJK scripts.
3. Green bars shorter than grey bars = win. Red `↑` arrows = the
   fine-tune hurt that language and you should probably ship the baseline
   for it.
4. If your fine-tuned model wins on `en`/`fr` but loses on `zh-CN`
   (the typical outcome with limited Chinese data): ship the fine-tuned
   model for `en`/`fr` and the baseline for `zh-CN`, selected at runtime
   based on the user's chosen language.

---

## Common gotchas

- **`wer_raw` > 1.0 for Chinese in older reports**: That's the original
  jiwer bug. Use `wer` (normalized) or `cer` instead.
- **Baseline EN WER ~30% looks high**: FLEURS test references contain
  punctuation, capitalization and digit formatting that Whisper outputs
  differently. After normalization (`wer` field) the number drops to a
  realistic single-digit percentage.
- **Latency much lower than your phone**: this script measures on
  `cuda` by default. For on-device numbers, pass `--device cpu` or run
  on the target hardware.
- **`max_eval_samples` defaults to 200 per language**: enough for a
  stable estimate, not enough for paper-quality numbers. Bump it via
  `--max-samples 1000` if you need tighter confidence intervals.
